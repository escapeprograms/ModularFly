"""Token embedding and unembedding layers for the modular-addition task.

input_embedding: one table shared by both operands, LongTensor[B, 2] -> [B, 2, dim].
unembedding: output-neuron readout [B, n_out] -> logits [B, n_tokens], no bias.
"""
from torch import nn


def input_embedding(n_tokens: int, dim: int, init_std: float) -> nn.Embedding:
    """One embedding table of shape [n_tokens, dim], entries ~ N(0, init_std^2)."""
    embedding = nn.Embedding(n_tokens, dim)
    nn.init.normal_(embedding.weight, std=init_std)
    return embedding


def unembedding(n_out: int, n_tokens: int) -> nn.Linear:
    """Linear map from output-neuron readout to one logit per token, without bias."""
    return nn.Linear(n_out, n_tokens, bias=False)
