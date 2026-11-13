"""Group-interface mechanism (arXiv 2610.10014): token channels drive neuron groups and
class logits are group-mean readouts.

tokens LongTensor[B, 2] -> embedding [B, 2, d] -> channels [B, C = 2d]
-> drive [B, C * G_in] (each channel copied onto its input group)
-> ConnectomeRNN readout [B, K * G_out] -> logits [B, K] = group means / temperature.
K = p = modulus. The only trainable parameters are embedding.weight [p, d] and the RNN's
weight module; there is no learned readout.

The groups are views of the RNN's input_idx / output_idx buffers (row-major), so drive
position c * G_in + j lands on input_groups[c, j] and readout column k * G_out + j is
output_groups[k, j]; nothing is stored twice in the checkpoint.
"""
import torch
from torch import nn

from modularfly.config import EmbeddingConfig, InterfaceConfig
from modularfly.model.fly.connectome import ConnectomeRNN
from modularfly.model.io.embedding import N_OPERANDS, input_embedding
from modularfly.model.io.interface import group_drive, group_mean_logits


class GroupInterfaceModel(nn.Module):
    def __init__(self, rnn: ConnectomeRNN, emb: EmbeddingConfig, interface: InterfaceConfig):
        super().__init__()
        self.cfg = interface
        self.n_channels = N_OPERANDS * emb.channels_per_token
        self.n_classes = emb.modulus
        check_group_sizes(rnn, self.n_channels * interface.input_group_size,
                          self.n_classes * interface.output_group_size)
        self.embedding = input_embedding(emb.modulus, emb.channels_per_token, emb.init_std)
        self.rnn = rnn

    @property
    def input_groups(self) -> torch.Tensor:
        """LongTensor[C, G_in] of graph neuron ids; row c is driven by channel c."""
        return self.rnn.input_idx.view(self.n_channels, self.cfg.input_group_size)

    @property
    def output_groups(self) -> torch.Tensor:
        """LongTensor[K, G_out] of graph neuron ids; row k is averaged into logit k."""
        return self.rnn.output_idx.view(self.n_classes, self.cfg.output_group_size)

    def channels(self, tokens: torch.Tensor) -> torch.Tensor:
        """tokens [B, 2] -> [B, C]: operand a's d channels, then operand b's."""
        return self.embedding(tokens).flatten(1)

    def drive(self, tokens: torch.Tensor) -> torch.Tensor:
        """tokens [B, 2] -> drive [B, C * G_in] on rnn.input_idx."""
        return group_drive(self.channels(tokens), self.cfg.input_group_size,
                           self.cfg.input_amplitude)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """tokens [B, 2] -> logits [B, K]."""
        readout = self.rnn(self.drive(tokens))
        return group_mean_logits(readout, self.n_classes, self.cfg.output_group_size,
                                 self.cfg.temperature)


def check_group_sizes(rnn: ConnectomeRNN, n_in: int, n_out: int) -> None:
    """Raises ValueError unless the RNN has exactly n_in input and n_out output neurons."""
    got_in, got_out = rnn.input_idx.numel(), rnn.output_idx.numel()
    if got_in != n_in or got_out != n_out:
        raise ValueError(f"interface needs {n_in} input / {n_out} output neurons, "
                         f"RNN has {got_in} / {got_out}")
