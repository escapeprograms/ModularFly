"""CLI: build the FlyWire v783 graph from data/raw/, cache it under data/processed/, and
print counts, E/I statistics, the spectral radius of sign * syn_count and the derived w0."""
import argparse
import time

import torch

from modularfly.config import DataConfig, ScaleConfig
from modularfly.data.flywire import build_graph
from modularfly.data.graph import ConnectomeGraph, graph_cache_path, save_graph
from modularfly.data.groups import select_neurons
from modularfly.model.fly.scaling import (ei_balance_scale, graph_spectral_radius,
                                      signed_count_weights, w0_for_radius)


def print_label_counts(graph: ConnectomeGraph, level: str) -> None:
    codes = graph.annotations[level]
    counts = torch.bincount(codes[codes >= 0], minlength=len(graph.vocab[level]))
    print(f"neurons per {level}:")
    for label, count in sorted(zip(graph.vocab[level], counts.tolist()), key=lambda x: -x[1]):
        print(f"  {label:<22} {count}")
    print(f"  {'(missing)':<22} {int((codes < 0).sum())}")


def print_io_counts(graph: ConnectomeGraph, cfg: DataConfig) -> None:
    n_in = select_neurons(graph, cfg.input_level, cfg.input_labels).numel()
    n_out = select_neurons(graph, cfg.output_level, cfg.output_labels).numel()
    print(f"input neurons  ({cfg.input_level} in {cfg.input_labels}): {n_in}")
    print(f"output neurons ({cfg.output_level} in {cfg.output_labels}): {n_out}")


def print_ei_counts(graph: ConnectomeGraph) -> None:
    excitatory = graph.neuron_sign > 0
    edge_exc = excitatory[graph.pre]
    no_nt = int((graph.annotations["top_nt"] < 0).sum())
    print(f"excitatory neurons {int(excitatory.sum())}, inhibitory neurons {int((~excitatory).sum())}")
    print(f"excitatory edges {int(edge_exc.sum())}, inhibitory edges {int((~edge_exc).sum())}")
    print(f"neurons with no NT prediction (signed as missing_nt_sign): {no_nt}")
    print(f"E/I edge ratio = inhibitory_init_scale that balances E/I edges: {ei_balance_scale(graph):.4f}")


def print_scale(graph: ConnectomeGraph, scale: ScaleConfig, device: str) -> None:
    generator = torch.Generator().manual_seed(scale.seed)
    start = time.perf_counter()
    radius = graph_spectral_radius(graph, signed_count_weights(graph), scale.power_iterations,
                                   generator, device)
    w0 = w0_for_radius(radius, scale.target_spectral_radius)
    print(f"spectral radius of sign*syn_count: {radius:.4f} "
          f"({scale.power_iterations} iterations, {time.perf_counter() - start:.1f} s)")
    print(f"w0 for target radius {scale.target_spectral_radius}: {w0:.6g}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    cfg, scale = DataConfig(), ScaleConfig()
    start = time.perf_counter()
    graph, n_dropped = build_graph(cfg.raw_dir / cfg.connections.relative_path,
                                   cfg.raw_dir / cfg.annotations.relative_path, cfg)
    print(f"built in {time.perf_counter() - start:.1f} s")
    print(f"neurons {graph.n_neurons}, edges (syn_count >= {cfg.min_syn_count}) {graph.n_edges}")
    print(f"edges dropped because an endpoint is not in the annotation table: {n_dropped}")
    path = graph_cache_path(cfg.processed_dir, cfg.min_syn_count)
    save_graph(graph, path)
    print(f"saved {path}")
    print_io_counts(graph, cfg)
    print_label_counts(graph, "flow")
    print_label_counts(graph, "super_class")
    print_ei_counts(graph)
    print_scale(graph, scale, args.device)


if __name__ == "__main__":
    main()
