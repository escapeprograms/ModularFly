import torch

from modularfly.model.fly.propagate import SparsePropagator
from tests.conftest import dense_matrix


def test_forward_matches_dense(small_graph):
    prop = SparsePropagator(small_graph, chunk_edges=7)
    w = torch.randn(small_graph.n_edges)
    h = torch.randn(small_graph.n_neurons, 3)
    torch.testing.assert_close(prop(w, h), dense_matrix(small_graph, w) @ h)


def test_gradcheck_float64(small_graph):
    prop = SparsePropagator(small_graph, chunk_edges=37)
    w = torch.randn(small_graph.n_edges, dtype=torch.float64, requires_grad=True)
    h = torch.randn(small_graph.n_neurons, 2, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda w_, h_: prop(w_, h_), (w, h))


def test_gradients_match_dense_reference(small_graph):
    prop = SparsePropagator(small_graph, chunk_edges=16)
    w = torch.randn(small_graph.n_edges, dtype=torch.float64, requires_grad=True)
    h = torch.randn(small_graph.n_neurons, 4, dtype=torch.float64, requires_grad=True)
    g = torch.randn(small_graph.n_neurons, 4, dtype=torch.float64)
    grad_w, grad_h = torch.autograd.grad((prop(w, h) * g).sum(), (w, h))
    w_ref, h_ref = w.detach().requires_grad_(), h.detach().requires_grad_()
    dense_out = dense_matrix(small_graph, w_ref) @ h_ref
    ref_w, ref_h = torch.autograd.grad((dense_out * g).sum(), (w_ref, h_ref))
    torch.testing.assert_close(grad_w, ref_w)
    torch.testing.assert_close(grad_h, ref_h)


def test_frozen_weights_get_no_gradient(small_graph):
    prop = SparsePropagator(small_graph, chunk_edges=16)
    w = torch.randn(small_graph.n_edges)
    h = torch.randn(small_graph.n_neurons, 2, requires_grad=True)
    prop(w, h).sum().backward()
    assert w.grad is None and h.grad is not None
