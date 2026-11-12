"""CLI: one forward + backward per condition on the cached real graph; report peak VRAM,
step time, trainable parameters, which parameters got gradients, and final-step activity."""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # TODO(debt-001)

import torch

from modularfly.config import DataConfig, DynamicsConfig, ScaleConfig
from modularfly.data.graph import ConnectomeGraph, graph_cache_path, load_graph
from modularfly.data.groups import group_neurons, pair_gain_index, select_neurons
from modularfly.model.fly.connectome import ConnectomeRNN
from modularfly.model.fly.scaling import graph_spectral_radius, signed_count_weights, w0_for_radius
from modularfly.model.weights.gain_modulated import GainModulatedWeights, GroupGain
from modularfly.model.weights.signed_edge import SignedEdgeWeights
from modularfly.model.weights.synapse_count import SynapseCountWeights

VRAM_WARN_BYTES = 6 * 2**30  # Windows spills to shared memory instead of raising OOM


def condition_weights(graph: ConnectomeGraph, w0: float, scale: ScaleConfig, level: str):
    edge_group, pair_keys = pair_gain_index(graph, group_neurons(graph, level))
    generator = torch.Generator().manual_seed(scale.seed)
    return {
        "1 frozen synapse counts": SynapseCountWeights(graph, w0),
        f"2 {level}-pair gains ({pair_keys.shape[0]} gains)": GainModulatedWeights(
            SynapseCountWeights(graph, w0), GroupGain(edge_group, pair_keys.shape[0])),
        "3 signed per-edge theta": SignedEdgeWeights(graph, scale.inhibitory_init_scale, generator),
    }


def timed_step(rnn: ConnectomeRNN, drive: torch.Tensor) -> float:
    torch.cuda.synchronize()
    start = time.perf_counter()
    rnn(drive).pow(2).mean().backward()
    torch.cuda.synchronize()
    return time.perf_counter() - start


def gradient_report(rnn: ConnectomeRNN, drive: torch.Tensor) -> str:
    named = [(name, p) for name, p in rnn.named_parameters()]
    got = [name for name, p in named if p.grad is not None and float(p.grad.abs().sum()) > 0]
    n_params = sum(p.numel() for _, p in named)
    return f"trainable params {n_params}; grads on {got or 'none'}; drive grad {drive.grad is not None}"


@torch.no_grad()
def activity_report(rnn: ConnectomeRNN, drive: torch.Tensor) -> str:
    *_, h = rnn.iter_states(drive)
    a, out = h.abs(), h[rnn.output_idx].abs()
    return (f"final step: mean|h| {a.mean():.4f}, frac|h|>0.99 {(a > 0.99).float().mean():.4f}, "
            f"frac|h|<1e-3 {(a < 1e-3).float().mean():.4f}, output mean|h| {out.mean():.4f}, "
            f"output frac|h|>0.99 {(out > 0.99).float().mean():.4f}")


def run_condition(name, weights, graph, io_idx, drive, dyn) -> None:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    rnn = ConnectomeRNN(graph, weights, *io_idx, dyn).cuda()
    first = timed_step(rnn, drive)
    report = gradient_report(rnn, drive)
    rnn.zero_grad(set_to_none=True)
    second = timed_step(rnn, drive)
    peak = torch.cuda.max_memory_allocated()
    flag = "  <-- ABOVE 6 GB" if peak > VRAM_WARN_BYTES else ""
    print(f"[{name}] peak VRAM {peak / 2**30:.2f} GB{flag}; step {first:.2f} s (first), {second:.2f} s (second)")
    print(f"    {report}")
    print(f"    {activity_report(rnn, drive.detach())}")
    del rnn


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--drive-scale", type=float, default=1.0)
    parser.add_argument("--level", default="cell_type")
    args = parser.parse_args()
    data, dyn, scale = DataConfig(), DynamicsConfig(), ScaleConfig()
    graph = load_graph(graph_cache_path(data.processed_dir, data.min_syn_count))
    radius = graph_spectral_radius(graph, signed_count_weights(graph), scale.power_iterations,
                                   torch.Generator().manual_seed(scale.seed), "cuda")
    w0 = w0_for_radius(radius, scale.target_spectral_radius)
    io_idx = (select_neurons(graph, data.input_level, data.input_labels),
              select_neurons(graph, data.output_level, data.output_labels))
    print(f"N {graph.n_neurons}, E {graph.n_edges}, n_in {io_idx[0].numel()}, n_out {io_idx[1].numel()}, "
          f"T {dyn.n_steps}, batch {args.batch}, w0 {w0:.6g}, drive N(0, {args.drive_scale}^2)")
    gen = torch.Generator().manual_seed(scale.seed)
    drive = (args.drive_scale * torch.randn(args.batch, io_idx[0].numel(), generator=gen)).cuda()
    for name, weights in condition_weights(graph, w0, scale, args.level).items():
        run_condition(name, weights, graph, io_idx, drive.clone().requires_grad_(), dyn)


if __name__ == "__main__":
    main()
