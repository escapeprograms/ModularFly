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


def test_drift_penalty_zero_at_init_and_positive_after_change(small_graph):
    weights = SignedEdgeWeights(small_graph, 2.0, torch.Generator().manual_seed(0))
    initial = weights.theta.detach().clone()
    assert weights.drift_penalty().item() == 0.0
    with torch.no_grad():
        weights.theta.data += 0.1
    penalty = weights.drift_penalty()
    assert penalty.requires_grad and penalty.dim() == 0
    assert torch.isclose(penalty, torch.tensor(0.01 * small_graph.n_edges), rtol=1e-4)
    assert "theta_init" in weights.state_dict()
    assert torch.equal(weights.theta_init, initial)


def test_frozen_and_gain_drift_penalties(small_graph):
    frozen = SynapseCountWeights(small_graph, 0.01).drift_penalty()
    assert frozen.item() == 0.0 and not frozen.requires_grad
    modulated = condition2(small_graph, 0.01)
    with torch.no_grad():
        modulated.gain.z.copy_(torch.linspace(-1.0, 1.0, modulated.gain.z.numel()))
    assert torch.equal(modulated.gain.drift_penalty(), (modulated.gain.z ** 2).sum())
    parts = modulated.base.drift_penalty() + modulated.gain.drift_penalty()
    assert torch.equal(modulated.drift_penalty(), parts)
