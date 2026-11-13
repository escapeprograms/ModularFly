"""Channel-group interface between token channels and connectome neurons (arXiv 2610.10014).

Shapes: B = batch, C = input channels, G_in = input group size, K = classes,
G_out = output group size.

Sampling: input groups are drawn without replacement from neurons with out-degree > 0
(graph.pre), output groups from neurons with in-degree > 0 (graph.post) minus the chosen
input neurons, so every group is disjoint from every other. Drive: channel c is copied
with one amplitude onto all G_in neurons of input group c. Readout: logit k is the mean
readout of output group k divided by a temperature (no learned weights).

Pure tensor functions: no nn.Module, no I/O. The generator is a CPU torch.Generator.
"""
import torch

from modularfly.data.graph import ConnectomeGraph


def input_candidates(graph: ConnectomeGraph) -> torch.Tensor:
    """Sorted LongTensor of neurons with out-degree > 0 (graph.pre.unique())."""
    return graph.pre.unique()


def output_candidates(graph: ConnectomeGraph, exclude: torch.Tensor) -> torch.Tensor:
    """Sorted LongTensor of neurons with in-degree > 0 (graph.post.unique()) not in exclude."""
    receivers = graph.post.unique()
    return receivers[~torch.isin(receivers, exclude)]


def sample_groups(candidates: torch.Tensor, n_groups: int, group_size: int,
                  generator: torch.Generator) -> torch.Tensor:
    """[n_groups, group_size] distinct neurons drawn without replacement from candidates.
    Raises ValueError when there are fewer candidates than n_groups * group_size."""
    n = n_groups * group_size
    if n > candidates.numel():
        raise ValueError(f"need {n} neurons, only {candidates.numel()} candidates")
    order = torch.randperm(candidates.numel(), generator=generator)[:n]
    return candidates[order].view(n_groups, group_size)


def sample_interface(graph: ConnectomeGraph, n_channels: int, input_group_size: int,
                     n_classes: int, output_group_size: int,
                     generator: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    """(input_groups [C, G_in], output_groups [K, G_out]), all neurons distinct.
    Inputs are drawn first; outputs exclude them."""
    input_groups = sample_groups(input_candidates(graph), n_channels, input_group_size, generator)
    outputs = output_candidates(graph, input_groups.flatten())
    output_groups = sample_groups(outputs, n_classes, output_group_size, generator)
    return input_groups, output_groups


def group_drive(channels: torch.Tensor, group_size: int, amplitude: float) -> torch.Tensor:
    """[B, C] -> [B, C * G_in]: position c * G_in + j is channels[:, c] * amplitude."""
    return channels.repeat_interleave(group_size, dim=1) * amplitude


def group_mean_logits(readout: torch.Tensor, n_classes: int, group_size: int,
                      temperature: float) -> torch.Tensor:
    """[B, K * G_out] -> [B, K]: mean over each consecutive block of G_out, / temperature."""
    return readout.view(readout.shape[0], n_classes, group_size).mean(dim=-1) / temperature
