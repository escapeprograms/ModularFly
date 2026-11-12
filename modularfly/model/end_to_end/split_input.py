"""Split-input mechanism: each operand's embedding drives a fixed random half of the
input neurons.

tokens LongTensor[B, 2] -> embeddings [B, 2, n_in // 2] -> drive [B, n_in]
-> ConnectomeRNN readout [B, n_out] -> logits [B, p].
Embedding dim is n_in // 2, so each entry is the drive on one neuron. If n_in is odd,
one input neuron stays undriven.

`slots` are positions in the drive vector, not neuron ids: operand a drives the
neurons rnn.input_idx[slots[0]], operand b drives rnn.input_idx[slots[1]].
"""
import torch
from torch import nn

from modularfly.config import EmbeddingConfig
from modularfly.model.fly.connectome import ConnectomeRNN
from modularfly.model.io.embedding import input_embedding, unembedding


def random_halves(n: int, generator: torch.Generator) -> torch.Tensor:
    """Two disjoint random sets of n // 2 positions in 0..n-1: LongTensor[2, n // 2].
    Row 0 is operand a's slots, row 1 operand b's."""
    half = n // 2
    return torch.randperm(n, generator=generator)[: 2 * half].view(2, half)


class SplitInputModel(nn.Module):
    def __init__(self, rnn: ConnectomeRNN, cfg: EmbeddingConfig, generator: torch.Generator):
        super().__init__()
        n_in = rnn.input_idx.numel()
        if n_in < 2:
            raise ValueError(f"split input needs at least 2 input neurons, got {n_in}")
        self.register_buffer("slots", random_halves(n_in, generator))
        self.embedding = input_embedding(cfg.modulus, n_in // 2, cfg.init_std)
        self.unembedding = unembedding(rnn.output_idx.numel(), cfg.modulus)
        self.rnn = rnn

    def drive(self, tokens: torch.Tensor) -> torch.Tensor:
        """tokens [B, 2] -> drive [B, n_in]; the leftover slot (odd n_in) is zero."""
        emb = self.embedding(tokens)  # [B, 2, n_in // 2]
        full = emb.new_zeros(tokens.shape[0], self.rnn.input_idx.numel())
        return full.index_copy(1, self.slots.flatten(), emb.flatten(1))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """tokens [B, 2] -> logits [B, p]."""
        return self.unembedding(self.rnn(self.drive(tokens)))
