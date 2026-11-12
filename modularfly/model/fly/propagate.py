"""Memory-safe sparse W @ h with gradients to the per-edge weights.

torch.sparse.mm differentiated w.r.t. sparse values materialises a dense N x N gradient,
so autograd never sees the sparse tensor's values here. SparsePropagate computes
grad_h with a precomputed transposed CSR and grad_w per edge in chunks.
State layout is [N, B].
"""
import torch
from torch import nn

from modularfly.data.graph import ConnectomeGraph


def csr_row_pointers(rows: torch.Tensor, n_rows: int) -> torch.Tensor:
    counts = torch.bincount(rows, minlength=n_rows)
    return torch.cat([counts.new_zeros(1), torch.cumsum(counts, 0)])


def csr_matmul(crow, col, values, h):
    n = crow.numel() - 1
    matrix = torch.sparse_csr_tensor(crow, col, values, size=(n, h.shape[0]), check_invariants=False)
    return matrix @ h


def edge_weight_grad(grad_out, h, pre, post, chunk_edges):
    """grad_w[e] = sum_b grad_out[post[e], b] * h[pre[e], b], chunked over edges."""
    grad_w = grad_out.new_empty(pre.numel())
    for start in range(0, pre.numel(), chunk_edges):
        stop = start + chunk_edges
        grad_w[start:stop] = (grad_out[post[start:stop]] * h[pre[start:stop]]).sum(dim=1)
    return grad_w


class SparsePropagate(torch.autograd.Function):
    """out[post] += w[e] * h[pre]; `plan` is a SparsePropagator holding the indices."""

    @staticmethod
    def forward(ctx, weights, h, plan):
        ctx.save_for_backward(weights, h)
        ctx.plan = plan
        return csr_matmul(plan.crow, plan.pre, weights.detach(), h)

    @staticmethod
    def backward(ctx, grad_out):
        weights, h = ctx.saved_tensors
        plan = ctx.plan
        grad_out = grad_out.contiguous()
        grad_w = grad_h = None
        if ctx.needs_input_grad[0]:
            grad_w = edge_weight_grad(grad_out, h, plan.pre, plan.post, plan.chunk_edges)
        if ctx.needs_input_grad[1]:
            values_t = weights.detach()[plan.transpose_perm]
            grad_h = csr_matmul(plan.crow_t, plan.col_t, values_t, grad_out)
        return grad_w, grad_h, None


class SparsePropagator(nn.Module):
    """Holds CSR indices of a graph (and its transpose) as buffers; forward(w, h) = W @ h."""

    def __init__(self, graph: ConnectomeGraph, chunk_edges: int):
        super().__init__()
        self.chunk_edges = chunk_edges
        self.register_buffer("pre", graph.pre.clone())
        self.register_buffer("post", graph.post.clone())
        self.register_buffer("crow", csr_row_pointers(graph.post, graph.n_neurons))
        perm = torch.sort(graph.pre, stable=True).indices  # (pre, post) order
        self.register_buffer("transpose_perm", perm)
        self.register_buffer("crow_t", csr_row_pointers(graph.pre, graph.n_neurons))
        self.register_buffer("col_t", graph.post[perm].clone())

    def forward(self, weights: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        return SparsePropagate.apply(weights, h, self)
