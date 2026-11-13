"""Assemble the real-fly GroupInterfaceModel for one weight condition.

Experiment 1 uses random_trainable_weights (one random trainable magnitude per connection,
presynaptic signs kept). Experiment 2 adds frozen_natural_weights (SynapseCountWeights at
the w0 for the target spectral radius) beside it; build_group_interface_model is shared.
"""
import torch
from torch import nn

from modularfly.config import DynamicsConfig, EmbeddingConfig, InterfaceConfig, ScaleConfig
from modularfly.data.graph import ConnectomeGraph
from modularfly.model.end_to_end.group_interface import GroupInterfaceModel
from modularfly.model.fly.connectome import ConnectomeRNN
from modularfly.model.io.embedding import N_OPERANDS
from modularfly.model.io.interface import sample_interface
from modularfly.model.weights.signed_edge import SignedEdgeWeights


def random_trainable_weights(graph: ConnectomeGraph, scale: ScaleConfig) -> SignedEdgeWeights:
    """Experiment 1's weights: SignedEdgeWeights seeded with scale.seed, inhibitory thetas
    multiplied by scale.inhibitory_init_scale."""
    generator = torch.Generator().manual_seed(scale.seed)
    return SignedEdgeWeights(graph, scale.inhibitory_init_scale, generator)


def build_group_interface_model(graph: ConnectomeGraph, weights: nn.Module,
                                dyn: DynamicsConfig, emb: EmbeddingConfig,
                                interface: InterfaceConfig, seed: int) -> GroupInterfaceModel:
    """ConnectomeRNN on a sampled channel-group interface, wrapped in a GroupInterfaceModel.

    torch.manual_seed(seed) seeds the nn.Embedding init; the interface is sampled with its
    own generator seeded by interface.seed, so it stays fixed across training seeds. On the
    CPU; the caller moves the model to its device.
    """
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(interface.seed)
    input_groups, output_groups = sample_interface(
        graph, N_OPERANDS * emb.channels_per_token, interface.input_group_size,
        emb.modulus, interface.output_group_size, generator)
    rnn = ConnectomeRNN(graph, weights, input_groups.flatten(), output_groups.flatten(), dyn)
    return GroupInterfaceModel(rnn, emb, interface)
