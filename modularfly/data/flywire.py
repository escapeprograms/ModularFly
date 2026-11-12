"""File IO: FlyWire v783 connections (feather) + annotations (tsv) -> ConnectomeGraph."""
from pathlib import Path

import pandas as pd
import pyarrow.feather as feather
import torch

from modularfly.config import DataConfig
from modularfly.data.graph import ConnectomeGraph

PRE, POST, COUNT = "pre_pt_root_id", "post_pt_root_id", "syn_count"


def read_connections(path: Path) -> pd.DataFrame:
    """Only the three needed columns; root ids must arrive as int64 (never via float)."""
    table = feather.read_table(path, columns=[PRE, POST, COUNT])
    frame = table.to_pandas()
    for column in (PRE, POST):
        if frame[column].dtype != "int64":
            raise TypeError(f"{column} is {frame[column].dtype}, expected int64")
    return frame


def read_annotations(path: Path, cfg: DataConfig) -> pd.DataFrame:
    label_columns = (*cfg.annotation_levels, *cfg.extra_annotation_columns)
    dtypes = {"root_id": "int64", cfg.nt_confidence_column: "float64"}
    dtypes.update({column: "string" for column in label_columns})
    frame = pd.read_csv(path, sep="\t", usecols=list(dtypes), dtype=dtypes)
    if frame["root_id"].duplicated().any():
        raise ValueError("annotation table has duplicated root_id values")
    return frame


def aggregate_connections(connections: pd.DataFrame) -> pd.DataFrame:
    """Sum synapse counts over neuropils: one row per (pre, post)."""
    return connections.groupby([PRE, POST], sort=False, as_index=False)[COUNT].sum()


def threshold_connections(connections: pd.DataFrame, min_syn_count: int) -> pd.DataFrame:
    return connections[connections[COUNT] >= min_syn_count]


def drop_unannotated_edges(connections: pd.DataFrame, root_ids: pd.Series) -> tuple[pd.DataFrame, int]:
    """Explicit choice: edges whose pre or post neuron is absent from the annotation table
    are dropped (v783 connections vs. the main-branch annotation file). Returns the count."""
    known = connections[PRE].isin(root_ids) & connections[POST].isin(root_ids)
    return connections[known], int((~known).sum())


def index_edges(connections: pd.DataFrame, root_ids: pd.Series) -> tuple[torch.Tensor, torch.Tensor]:
    """Map root ids to neuron indices; every endpoint must be annotated."""
    lookup = pd.Index(root_ids)
    pre = lookup.get_indexer(connections[PRE])
    post = lookup.get_indexer(connections[POST])
    if (pre < 0).any() or (post < 0).any():
        raise ValueError("edge endpoint missing from the annotation table")
    return torch.from_numpy(pre).long(), torch.from_numpy(post).long()


def encode_labels(labels: pd.Series) -> tuple[torch.Tensor, list[str]]:
    """Sorted vocabulary; missing labels get code -1."""
    codes, uniques = pd.factorize(labels, sort=True, use_na_sentinel=True)
    return torch.from_numpy(codes).long(), [str(label) for label in uniques]


def neuron_signs(nt_labels: pd.Series, cfg: DataConfig) -> torch.Tensor:
    unknown = set(nt_labels.dropna().unique()) - set(cfg.nt_sign)
    if unknown:
        raise ValueError(f"no sign defined for transmitters {sorted(unknown)}")
    signs = nt_labels.map(cfg.nt_sign).astype("float64").fillna(cfg.missing_nt_sign)
    return torch.tensor(signs.to_numpy(), dtype=torch.float32)


def encode_annotations(annotations: pd.DataFrame, cfg: DataConfig):
    codes, vocab = {}, {}
    for column in (*cfg.annotation_levels, *cfg.extra_annotation_columns):
        codes[column], vocab[column] = encode_labels(annotations[column])
    return codes, vocab


def load_edges(connections_path: Path, root_ids: pd.Series, cfg: DataConfig):
    """Returns (pre, post, syn_count, n_dropped_unannotated)."""
    edges = aggregate_connections(read_connections(connections_path))
    edges = threshold_connections(edges, cfg.min_syn_count)
    edges, n_dropped = drop_unannotated_edges(edges, root_ids)
    pre, post = index_edges(edges, root_ids)
    syn_count = torch.tensor(edges[COUNT].to_numpy(), dtype=torch.float32)
    return pre, post, syn_count, n_dropped


def build_graph(connections_path: Path, annotations_path: Path, cfg: DataConfig):
    """Returns (graph, n_edges_dropped_for_unannotated_endpoints)."""
    annotations = read_annotations(annotations_path, cfg)
    pre, post, syn_count, n_dropped = load_edges(connections_path, annotations["root_id"], cfg)
    codes, vocab = encode_annotations(annotations, cfg)
    confidence = annotations[cfg.nt_confidence_column].fillna(0.0).to_numpy()
    graph = ConnectomeGraph(
        n_neurons=len(annotations), pre=pre, post=post, syn_count=syn_count,
        neuron_sign=neuron_signs(annotations[cfg.nt_column], cfg),
        nt_confidence=torch.tensor(confidence, dtype=torch.float32),
        annotations=codes, vocab=vocab,
        root_ids=torch.from_numpy(annotations["root_id"].to_numpy()).long(),
    )
    return graph, n_dropped
