import torch

from modularfly.data.groups import group_neurons, pair_gain_index
from modularfly.model.weights.gain_modulated import GainModulatedWeights, GroupGain
from modularfly.model.weights.signed_edge import SignedEdgeWeights
from modularfly.model.weights.synapse_count import SynapseCountWeights
from tests.conftest import SMALL_DYN, make_rnn


def condition2(graph, w0):
    edge_group, keys = pair_gain_index(graph, group_neurons(graph, "cell_type"))
    return GainModulatedWeights(SynapseCountWeights(graph, w0), GroupGain(edge_group, keys.shape[0]))


def test_condition2_equals_condition1_at_init(small_graph):
    base = SynapseCountWeights(small_graph, 0.01)
    assert torch.equal(condition2(small_graph, 0.01)(), base())
    rnn1, n_in, _ = make_rnn(small_graph, base)
    rnn2, _, _ = make_rnn(small_graph, condition2(small_graph, 0.01))
    drive = torch.randn(3, n_in)
    assert torch.equal(rnn1(drive), rnn2(drive))


def test_condition3_keeps_presynaptic_sign_after_step(small_graph):
    weights = SignedEdgeWeights(small_graph, 2.0, torch.Generator().manual_seed(0))
    rnn, n_in, _ = make_rnn(small_graph, weights)
    opt = torch.optim.Adam(weights.parameters(), lr=0.5)
    rnn(torch.randn(4, n_in)).pow(2).sum().backward()
    assert weights.theta.grad is not None
    opt.step()
    w = weights()
    nonzero = w != 0
    expected = small_graph.neuron_sign[small_graph.pre]
    assert bool((torch.sign(w[nonzero]) == expected[nonzero]).all())


def test_forward_shape_and_gradients(small_graph):
    rnn, n_in, n_out = make_rnn(small_graph, condition2(small_graph, 0.01))
    drive = torch.randn(5, n_in, requires_grad=True)
    out = rnn(drive)
    assert out.shape == (5, n_out)
    out.sum().backward()
    assert drive.grad is not None and rnn.edge_weights.gain.z.grad is not None


def test_iter_states_yields_t_states(small_graph):
    rnn, n_in, _ = make_rnn(small_graph, SynapseCountWeights(small_graph, 0.01))
    states = list(rnn.iter_states(torch.randn(2, n_in)))
    assert len(states) == SMALL_DYN.n_steps and states[0].shape == (small_graph.n_neurons, 2)


def test_window_mean_matches_forward(small_graph):
    rnn, n_in, _ = make_rnn(small_graph, SynapseCountWeights(small_graph, 0.01))
    drive = torch.randn(3, n_in)
    assert torch.equal(rnn.window_mean(drive, rnn.output_idx).t(), rnn(drive))
    idx = torch.tensor([3, 7, 11])
    last = list(rnn.iter_states(drive))[-SMALL_DYN.readout_steps:]
    manual = torch.stack([h[idx] for h in last]).mean(dim=0)
    assert torch.allclose(rnn.window_mean(drive, idx), manual)
