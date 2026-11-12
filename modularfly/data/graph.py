"""ConnectomeGraph: plain tensors describing the wiring, plus save/load.

Invariant: edges are sorted by (post, pre) with no duplicate (pre, post) pair. It is
established in __post_init__, the only construction path, so every graph satisfies it.
"""
from dataclasses import dataclass, fields
from pathlib import Path

import torch

_EDGE_FIELDS = ("pre", "post", "syn_count")
_NEURON_FIELDS = ("neuron_sign", "nt_confidence", "root_ids")


@dataclass
class ConnectomeGraph:
    n_neurons: int
    pre: torch.Tensor  # LongTensor[E], edge e carries pre[e] -> post[e]
    post: torch.Tensor  # LongTensor[E]
    syn_count: torch.Tensor  # FloatTensor[E]
    neuron_sign: torch.Tensor  # FloatTensor[N], +-1
    nt_confidence: torch.Tensor  # FloatTensor[N], 0 where no prediction
    annotations: dict[str, torch.Tensor]  # LongTensor[N] code per level, -1 = missing
    vocab: dict[str, list[str]]  # code -> label, per level
    root_ids: torch.Tensor  # LongTensor[N]

    def __post_init__(self):
        self._validate_shapes()
        order = torch.argsort(edge_keys(self.post, self.pre, self.n_neurons))
        self.pre, self.post, self.syn_count = self.pre[order], self.post[order], self.syn_count[order]
        keys = edge_keys(self.post, self.pre, self.n_neurons)
        if keys.numel() > 1 and not bool((keys[1:] > keys[:-1]).all()):
            raise ValueError("graph has duplicate (pre, post) edges")

    def _validate_shapes(self):
        n_edges = self.pre.numel()
        for name in _EDGE_FIELDS:
            if getattr(self, name).shape != (n_edges,):
                raise ValueError(f"{name} must have shape ({n_edges},)")
        for name in _NEURON_FIELDS:
            if getattr(self, name).shape != (self.n_neurons,):
                raise ValueError(f"{name} must have shape ({self.n_neurons},)")
        for level, codes in self.annotations.items():
            if codes.shape != (self.n_neurons,):
                raise ValueError(f"annotation {level} must have shape ({self.n_neurons},)")
        if n_edges and (min(self.pre.min(), self.post.min()) < 0
                        or max(self.pre.max(), self.post.max()) >= self.n_neurons):
            raise ValueError("edge endpoint outside [0, n_neurons)")

    @property
    def n_edges(self) -> int:
        return self.pre.numel()

    def in_degree(self) -> torch.Tensor:
        return torch.bincount(self.post, minlength=self.n_neurons)

    def out_degree(self) -> torch.Tensor:
        return torch.bincount(self.pre, minlength=self.n_neurons)

    def with_edges(self, pre: torch.Tensor, post: torch.Tensor, syn_count: torch.Tensor) -> "ConnectomeGraph":
        """Same neurons, new edge list (re-canonicalized)."""
        neuron_part = {f.name: getattr(self, f.name) for f in fields(self) if f.name not in _EDGE_FIELDS}
        return ConnectomeGraph(pre=pre, post=post, syn_count=syn_count, **neuron_part)


def edge_keys(major: torch.Tensor, minor: torch.Tensor, n_neurons: int) -> torch.Tensor:
    """Unique int64 key per (major, minor) pair, ordered lexicographically."""
    return major * n_neurons + minor


def graph_to_dict(graph: ConnectomeGraph) -> dict:
    return {f.name: getattr(graph, f.name) for f in fields(graph)}


def graph_cache_path(processed_dir: Path, min_syn_count: int) -> Path:
    return Path(processed_dir) / f"flywire783_graph_min{min_syn_count}.pt"


def save_graph(graph: ConnectomeGraph, path: Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(graph_to_dict(graph), path)


def load_graph(path: Path) -> ConnectomeGraph:
    return ConnectomeGraph(**torch.load(path, weights_only=True))
