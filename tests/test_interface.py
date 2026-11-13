import pytest
import torch

from modularfly.model.io.interface import (
    group_drive, group_mean_logits, input_candidates, output_candidates, sample_groups,
    sample_interface,
)


def sample(graph, seed=0, input_group_size=3, output_group_size=2):
    return sample_interface(graph, 4, input_group_size, 7, output_group_size,
                            torch.Generator().manual_seed(seed))


def test_interface_shapes(small_graph):
    inputs, outputs = sample(small_graph)
    assert inputs.shape == (4, 3) and outputs.shape == (7, 2)


def test_groups_are_disjoint(small_graph):
    inputs, outputs = (g.flatten() for g in sample(small_graph))
    assert inputs.unique().numel() == 12 and outputs.unique().numel() == 14
    assert not torch.isin(inputs, outputs).any()


def test_candidates(small_graph):
    assert torch.equal(input_candidates(small_graph), small_graph.pre.unique())
    exclude = torch.tensor([0, 5, 9])
    outputs = output_candidates(small_graph, exclude)
    expected = [n for n in small_graph.post.unique().tolist() if n not in (0, 5, 9)]
    assert outputs.tolist() == expected


def test_sampling_is_seeded(small_graph):
    first, second, other = sample(small_graph, 0), sample(small_graph, 0), sample(small_graph, 1)
    assert all(torch.equal(a, b) for a, b in zip(first, second))
    assert not all(torch.equal(a, b) for a, b in zip(first, other))


def test_too_few_candidates_raise(small_graph):
    with pytest.raises(ValueError):
        sample(small_graph, input_group_size=20)  # 80 of 40
    with pytest.raises(ValueError):
        sample(small_graph, output_group_size=5)  # 35 of the 28 left after 12 inputs


def test_sample_groups_without_replacement():
    groups = sample_groups(torch.arange(10, 20), 2, 4, torch.Generator().manual_seed(0))
    assert groups.shape == (2, 4) and groups.unique().numel() == 8
    assert bool(((groups >= 10) & (groups < 20)).all())


def test_group_drive_repeats_each_channel():
    drive = group_drive(torch.arange(8.).view(2, 4), 3, 2.0)
    expected = torch.tensor([[0., 0, 0, 2, 2, 2, 4, 4, 4, 6, 6, 6],
                             [8., 8, 8, 10, 10, 10, 12, 12, 12, 14, 14, 14]])
    assert torch.equal(drive, expected)


def test_group_mean_logits_pairwise_means():
    readout = torch.arange(28.).view(2, 14)
    logits = group_mean_logits(readout, 7, 2, 0.1)
    assert logits.shape == (2, 7)
    assert torch.allclose(logits, readout.view(2, 7, 2).mean(-1) / 0.1)
    assert torch.allclose(logits[0, :2], torch.tensor([5.0, 25.0]))
