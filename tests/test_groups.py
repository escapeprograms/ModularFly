import torch

from modularfly.data.groups import (global_gain_index, group_neurons, pair_gain_index,
                                    select_neurons, sign_gain_index)

COARSER = ("supertype", "cell_sub_class", "cell_class", "super_class", "flow")


def test_select_neurons(small_graph):
    idx = select_neurons(small_graph, "flow", ["afferent"])
    assert torch.equal(idx, torch.nonzero(small_graph.annotations["flow"] == 0).flatten())


def test_group_neurons_falls_back_to_coarser_level(small_graph):
    groups = group_neurons(small_graph, "cell_type")
    ann = small_graph.annotations
    unlabeled = torch.nonzero(ann["cell_type"] < 0).flatten()
    assert unlabeled.numel() > 0
    for n in unlabeled.tolist():
        same = groups == groups[n]
        assert bool((ann["cell_type"][same] < 0).all())  # never merged with labeled types
        level = next(lv for lv in COARSER if ann[lv][n] >= 0)  # first labeled coarser level
        assert bool((ann[level][same] == ann[level][n]).all())


def test_group_neurons_labeled_neurons_group_by_label(small_graph):
    groups = group_neurons(small_graph, "cell_type")
    labeled = small_graph.annotations["cell_type"] >= 0
    for code in range(3):
        members = groups[labeled & (small_graph.annotations["cell_type"] == code)]
        assert members.unique().numel() == 1


def test_pair_gain_index_only_occupied_pairs(small_graph):
    groups = group_neurons(small_graph, "cell_class")
    edge_group, pair_keys = pair_gain_index(small_graph, groups)
    assert edge_group.shape == (small_graph.n_edges,)
    assert edge_group.unique().numel() == pair_keys.shape[0]
    assert torch.equal(pair_keys[edge_group, 0], groups[small_graph.post])
    assert torch.equal(pair_keys[edge_group, 1], groups[small_graph.pre])


def test_baseline_indices(small_graph):
    edge_group, keys = global_gain_index(small_graph)
    assert keys.numel() == 1 and bool((edge_group == 0).all())
    edge_group, keys = sign_gain_index(small_graph)
    assert torch.equal(keys[edge_group].float(), small_graph.neuron_sign[small_graph.pre])
