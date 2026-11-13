import pytest
import torch
import torch.nn.functional as F

from modularfly.model.end_to_end.group_interface import GroupInterfaceModel
from modularfly.model.weights.synapse_count import SynapseCountWeights
from tests.conftest import (
    SMALL_EMB, SMALL_INTERFACE, SMALL_P as P, make_model, make_rnn, trainable_fly,
)


def trainable_names(model):
    return sorted(name for name, p in model.named_parameters() if p.requires_grad)


def test_forward_shape(small_graph):
    model = make_model(small_graph, trainable_fly(small_graph))
    assert model(torch.randint(0, P, (5, 2))).shape == (5, P)


def test_drive_copies_each_channel_onto_its_group(small_graph):
    model = make_model(small_graph, trainable_fly(small_graph))
    tokens = torch.randint(0, P, (5, 2))
    drive = model.drive(tokens).view(5, 4, 3)
    assert torch.equal(drive, drive[:, :, :1].expand(5, 4, 3))
    assert torch.equal(drive[:, :, 0], model.embedding(tokens).flatten(1))


def test_same_token_gives_same_drive_on_both_operands(small_graph):
    model = make_model(small_graph, trainable_fly(small_graph))
    tokens = torch.arange(P).unsqueeze(1).expand(P, 2)
    drive = model.drive(tokens)
    assert torch.equal(drive[:, :6], drive[:, 6:])


def test_logits_are_group_means_over_temperature(small_graph):
    model = make_model(small_graph, trainable_fly(small_graph))
    tokens = torch.randint(0, P, (5, 2))
    with torch.no_grad():
        expected = model.rnn(model.drive(tokens)).view(5, 7, 2).mean(-1) / 0.1
        assert torch.allclose(model(tokens), expected)


def test_groups_view_rnn_buffers(small_graph):
    model = make_model(small_graph, trainable_fly(small_graph))
    assert model.input_groups.shape == (4, 3) and model.output_groups.shape == (7, 2)
    assert torch.equal(model.input_groups.flatten(), model.rnn.input_idx)
    assert torch.equal(model.output_groups.flatten(), model.rnn.output_idx)


def test_gradients_reach_embedding_and_theta(small_graph):
    model = make_model(small_graph, trainable_fly(small_graph))
    tokens = torch.tensor([[0, 1], [2, 3], [1, 0]])
    F.cross_entropy(model(tokens), torch.tensor([1, 5, 1])).backward()
    assert model.embedding.weight.grad.abs().sum() > 0
    assert model.rnn.edge_weights.theta.grad.abs().sum() > 0
    assert bool((model.embedding.weight.grad[[4, 5, 6]] == 0).all())
    assert trainable_names(model) == ["embedding.weight", "rnn.edge_weights.theta"]


def test_frozen_fly_trains_only_embedding(small_graph):
    model = make_model(small_graph, SynapseCountWeights(small_graph, 0.01))
    assert trainable_names(model) == ["embedding.weight"]


def test_mismatched_rnn_raises(small_graph):
    rnn, n_in, _ = make_rnn(small_graph, trainable_fly(small_graph))
    assert n_in != 12
    with pytest.raises(ValueError):
        GroupInterfaceModel(rnn, SMALL_EMB, SMALL_INTERFACE)
