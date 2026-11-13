import torch

from modularfly.config import ScaleConfig
from modularfly.experiments.fly_model import build_group_interface_model, random_trainable_weights
from modularfly.model.io.interface import sample_interface
from modularfly.model.weights.synapse_count import SynapseCountWeights
from tests.conftest import SMALL_DYN, SMALL_EMB, SMALL_INTERFACE


def build(graph, weights, seed=0):
    return build_group_interface_model(graph, weights, SMALL_DYN, SMALL_EMB, SMALL_INTERFACE, seed)


def trainable_names(model):
    return sorted(name for name, p in model.named_parameters() if p.requires_grad)


def test_random_weights_train_fly_and_tokens(small_graph):
    model = build(small_graph, random_trainable_weights(small_graph, ScaleConfig()))
    assert trainable_names(model) == ["embedding.weight", "rnn.edge_weights.theta"]


def test_frozen_weights_train_only_tokens(small_graph):
    model = build(small_graph, SynapseCountWeights(small_graph, 0.01))
    assert trainable_names(model) == ["embedding.weight"]


def test_same_seed_gives_equal_embedding(small_graph):
    weights = random_trainable_weights(small_graph, ScaleConfig())
    first, second = build(small_graph, weights, 3), build(small_graph, weights, 3)
    other = build(small_graph, weights, 4)
    assert torch.equal(first.embedding.weight, second.embedding.weight)
    assert not torch.equal(first.embedding.weight, other.embedding.weight)


def test_interface_fixed_across_training_seeds(small_graph):
    weights = random_trainable_weights(small_graph, ScaleConfig())
    first, other = build(small_graph, weights, 3), build(small_graph, weights, 4)
    assert torch.equal(first.rnn.input_idx, other.rnn.input_idx)
    assert torch.equal(first.rnn.output_idx, other.rnn.output_idx)
    assert not torch.equal(first.embedding.weight, other.embedding.weight)
    generator = torch.Generator().manual_seed(SMALL_INTERFACE.seed)
    expected_inputs = sample_interface(small_graph, 4, 3, 7, 2, generator)[0].flatten()
    assert torch.equal(first.rnn.input_idx, expected_inputs)
