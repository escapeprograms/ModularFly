import torch
import torch.nn.functional as F

from modularfly.config import EmbeddingConfig
from modularfly.model.end_to_end.split_input import SplitInputModel
from modularfly.model.weights.signed_edge import SignedEdgeWeights
from modularfly.model.weights.synapse_count import SynapseCountWeights
from tests.conftest import make_rnn

P = 7


def make_model(graph, weights, seed=0):
    rnn, _, _ = make_rnn(graph, weights)
    return SplitInputModel(rnn, EmbeddingConfig(modulus=P), torch.Generator().manual_seed(seed))


def trainable_fly(graph):
    return SignedEdgeWeights(graph, 1.0, torch.Generator().manual_seed(0))


def test_forward_shape(small_graph):
    model = make_model(small_graph, trainable_fly(small_graph))
    assert model(torch.randint(0, P, (5, 2))).shape == (5, P)


def test_halves_partition_input_slots(small_graph):
    model = make_model(small_graph, trainable_fly(small_graph))
    slots = model.slots
    assert slots.shape == (2, 7)
    assert slots.unique().numel() == 14 and bool((slots < 15).all())
    tokens = torch.randint(0, P, (4, 2))
    drive = model.drive(tokens)
    unused = torch.ones(15, dtype=torch.bool)
    unused[slots.flatten()] = False
    assert int(unused.sum()) == 1 and bool((drive[:, unused] == 0).all())
    emb = model.embedding(tokens)
    assert torch.equal(drive[:, slots[0]], emb[:, 0])
    assert torch.equal(drive[:, slots[1]], emb[:, 1])


def test_same_token_gives_same_drive_on_both_halves(small_graph):
    model = make_model(small_graph, trainable_fly(small_graph))
    tokens = torch.arange(P).unsqueeze(1).expand(P, 2)
    drive = model.drive(tokens)
    assert torch.equal(drive[:, model.slots[0]], drive[:, model.slots[1]])


def test_split_is_seeded(small_graph):
    weights = trainable_fly(small_graph)
    same_a, same_b = make_model(small_graph, weights, 0), make_model(small_graph, weights, 0)
    other = make_model(small_graph, weights, 1)
    assert torch.equal(same_a.slots, same_b.slots)
    assert not torch.equal(same_a.slots, other.slots)


def test_gradients_reach_embedding_unembedding_and_theta(small_graph):
    model = make_model(small_graph, trainable_fly(small_graph))
    tokens = torch.tensor([[0, 1], [2, 3], [1, 0]])
    targets = torch.tensor([1, 5, 1])
    F.cross_entropy(model(tokens), targets).backward()
    assert model.embedding.weight.grad.abs().sum() > 0
    assert model.unembedding.weight.grad.abs().sum() > 0
    assert model.rnn.edge_weights.theta.grad.abs().sum() > 0
    assert model.slots.requires_grad is False
    unused_tokens = [4, 5, 6]
    assert bool((model.embedding.weight.grad[unused_tokens] == 0).all())


def test_frozen_fly_trains_only_embeddings(small_graph):
    model = make_model(small_graph, SynapseCountWeights(small_graph, 0.01))
    names = [name for name, p in model.named_parameters() if p.requires_grad]
    assert sorted(names) == ["embedding.weight", "unembedding.weight"]
