"""Per-neuron Fourier analysis over the full (a, b) input grid.

Shapes: p = modulus, n = selected neurons, K = p // 2 frequencies, B = batch of pairs.

Probe pattern: a `probe` is any callable (model, tokens [B, 2]) -> [n, B] giving the
activation of a fixed neuron list for each pair. The caller binds the neurons (and
step) with functools.partial, e.g.
    partial(readout_window_mean, neuron_idx=idx)        # what the group readout averages
    partial(state_at_step, neuron_idx=idx, step=t)      # a single time step
so new probes plug in without changing the grid collection.

Chunking: each RNN state is [N, B]. On the real fly N = 139,248, so running all
p^2 = 3,481 pairs (p = 59) at once would need ~1.9 GB per step; pairs are therefore run in
slices of cfg.pair_chunk and only the selected neurons' [n, B_chunk] rows are kept.
"""
from collections.abc import Callable, Iterator
from dataclasses import dataclass

import torch

from modularfly.config import InterpretabilityConfig
from modularfly.interpretability.fourier import (
    block_energy, fourier_2d, fourier_basis, frequency_energy, sum_difference_energy,
)
from modularfly.model.end_to_end.group_interface import GroupInterfaceModel

Probe = Callable[[GroupInterfaceModel, torch.Tensor], torch.Tensor]


def all_pairs(p: int, device=None) -> torch.Tensor:
    """LongTensor[p*p, 2], row-major: row a*p + b is (a, b)."""
    x = torch.arange(p, device=device)
    return torch.cartesian_prod(x, x)


def readout_window_mean(model: GroupInterfaceModel, tokens: torch.Tensor,
                        neuron_idx: torch.Tensor) -> torch.Tensor:
    """[n, B]: neuron activity averaged over the readout window (what the group readout averages)."""
    return model.rnn.window_mean(model.drive(tokens), neuron_idx)


def state_at_step(model: GroupInterfaceModel, tokens: torch.Tensor,
                  neuron_idx: torch.Tensor, step: int) -> torch.Tensor:
    """[n, B]: h_step[neuron_idx], step 0 = first yielded state; later steps are not run."""
    n_steps = model.rnn.cfg.n_steps
    if not 0 <= step < n_steps:
        raise ValueError(f"step={step} outside 0..{n_steps - 1}")
    for t, h in enumerate(model.rnn.iter_states(model.drive(tokens))):
        if t == step:
            return h[neuron_idx]


def iter_activation_chunks(model: GroupInterfaceModel, cfg: InterpretabilityConfig,
                           probe: Probe) -> Iterator[torch.Tensor]:
    """Yields probe output [n, B_chunk] for consecutive cfg.pair_chunk slices of all_pairs."""
    pairs = all_pairs(model.embedding.num_embeddings, device=model.rnn.input_idx.device)
    with torch.no_grad():
        for start in range(0, pairs.shape[0], cfg.pair_chunk):
            yield probe(model, pairs[start:start + cfg.pair_chunk])


def activation_grid(model: GroupInterfaceModel, cfg: InterpretabilityConfig,
                    probe: Probe) -> torch.Tensor:
    """[n, p, p]: probe activation for every (a, b)."""
    p = model.embedding.num_embeddings
    activations = torch.cat(list(iter_activation_chunks(model, cfg, probe)), dim=1)
    return activations.view(-1, p, p)


@dataclass(frozen=True)
class NeuronSpectra:
    neuron_idx: torch.Tensor     # LongTensor[n] graph neuron ids
    grid: torch.Tensor           # [n, p, p] activation over (a, b)
    coeffs: torch.Tensor         # [n, p, p] 2D Fourier coefficients (component of a, component of b)
    dominant_freq: torch.Tensor  # LongTensor[n], best k in 1..K
    fve: torch.Tensor            # [n] fraction of non-constant variance explained by dominant_freq; 0 if silent
    sum_share: torch.Tensor      # [n] share of the dominant (k,k) block on w(a+b) vs w(a-b); 0 if silent


def silent_mask(nonconstant: torch.Tensor, full: torch.Tensor) -> torch.Tensor:
    """[n] bool: non-constant energy is at roundoff level of the full energy
    (a constant grid leaves ~eps residue in the non-constant coefficients)."""
    return nonconstant <= torch.finfo(full.dtype).eps * full


def masked_ratio(num: torch.Tensor, den: torch.Tensor, silent: torch.Tensor) -> torch.Tensor:
    """num / den, with 0 for silent neurons (and no NaN from den == 0)."""
    safe_den = torch.where(silent, torch.ones_like(den), den)
    return torch.where(silent, torch.zeros_like(num), num / safe_den)


def neuron_spectra(neuron_idx: torch.Tensor, grid: torch.Tensor,
                   cfg: InterpretabilityConfig) -> NeuronSpectra:
    """Pure: 2D Fourier spectrum metrics of each neuron's [p, p] activation grid."""
    basis = fourier_basis(grid.shape[-1], dtype=grid.dtype, device=grid.device)
    coeffs = fourier_2d(grid, basis)
    energy = block_energy(coeffs)
    per_k = frequency_energy(energy)  # [n, K]
    full = energy.sum(dim=(-2, -1))
    total = full - energy[..., 0, 0]
    silent = silent_mask(total, full)
    best, arg = per_k.max(dim=-1)
    sum_e, diff_e = sum_difference_energy(coeffs)  # [n, K] each
    pick = arg.unsqueeze(-1)
    dom_sum = sum_e.gather(-1, pick).squeeze(-1)
    dom_block = (sum_e + diff_e).gather(-1, pick).squeeze(-1)
    return NeuronSpectra(
        neuron_idx=neuron_idx, grid=grid, coeffs=coeffs, dominant_freq=arg + 1,
        fve=masked_ratio(best, total, silent),
        sum_share=masked_ratio(dom_sum, dom_block, silent_mask(dom_block, full)),
    )


def cluster_by_frequency(spectra: NeuronSpectra,
                         cfg: InterpretabilityConfig) -> dict[int, torch.Tensor]:
    """{k: graph ids of neurons with dominant_freq k and fve >= cfg.fve_threshold}, keys ascending."""
    kept = spectra.fve >= cfg.fve_threshold
    clusters = {}
    for k in spectra.dominant_freq[kept].unique().tolist():
        clusters[k] = spectra.neuron_idx[kept & (spectra.dominant_freq == k)]
    return clusters
