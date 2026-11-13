"""Token embedding layer for the modular-addition task.

input_embedding: one table shared by both operands, LongTensor[B, N_OPERANDS] -> [B, N_OPERANDS, dim].
N_OPERANDS: tokens per example (a, b); the one place the two-operand assumption lives.
"""
from torch import nn

N_OPERANDS = 2


def input_embedding(n_tokens: int, dim: int, init_std: float) -> nn.Embedding:
    """One embedding table of shape [n_tokens, dim], entries ~ N(0, init_std^2)."""
    embedding = nn.Embedding(n_tokens, dim)
    nn.init.normal_(embedding.weight, std=init_std)
    return embedding

