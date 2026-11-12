# modularfly/interpretability

Fourier (DFT) analysis of a modular-addition model, following Nanda et al. 2023 ("Progress measures for grokking via mechanistic interpretability"). The embedding and unembedding get a 1D transform over the input number; individual neurons get a 2D transform over the input pair `(a, b)`. Matplotlib figures show the results.

The fly version has no MLP. Here the "neurons" are hidden neurons of the `ConnectomeRNN`, read over the readout window by default or at a chosen time step.

`p` = modulus (odd), `d` = feature dim, `n` = selected neurons, `K = p // 2` frequencies, `B` = batch of pairs. Per-frequency arrays have length `K+1`; index 0 is the constant component.

| File | Job |
|---|---|
| [fourier.py](fourier.py) | Pure tensor math: the real DFT basis, 1D/2D transforms, and per-frequency norms and energies. |
| [embedding.py](embedding.py) | 1D spectrum of the embedding and unembedding weights, its dominant frequencies and Gini sparsity. |
| [neurons.py](neurons.py) | Collect each neuron's activation over all `p²` pairs, plus its 2D spectrum, best frequency and clusters. |
| [plots.py](plots.py) | Matplotlib figures (object-oriented API, no pyplot) and `save_figure`, the only file I/O in the package. |

Tunables are in [`InterpretabilityConfig`](../README.md) in `config.py`.

## fourier.py

| Signature | Returns | What it does |
|---|---|---|
| `fourier_basis(p, dtype=float32, device=None)` | `[p, p]` | Orthonormal real DFT basis (row layout under Definitions). Built in float64, then cast. Raises `ValueError` for even `p`. |
| `pair_sum(x, dim)` | size `K+1` along `dim` | `[x0, x1+x2, x3+x4, …]`: adds each frequency's cos and sin entries together. |
| `fourier_1d(x, basis)` | `[p, d]` | `basis @ x`: coefficients over the input axis of a `[p, d]` matrix. |
| `fourier_2d(grid, basis)` | `[..., p, p]` | `basis @ grid @ basis.T`; entry `(i, j)` = component `i` of `a`, component `j` of `b`. Batched over leading dims. |
| `frequency_norms(coeffs)` | `[K+1]` | L2 norm of each frequency's (cos, sin) rows of `[p, d]` coefficients, over all features. |
| `block_energy(coeffs)` | `[..., K+1, K+1]` | Squared 2D coefficients summed inside each (frequency of `a`, frequency of `b`) block. |
| `frequency_energy(energy)` | `[..., K]` | `E[k,k] + E[0,k] + E[k,0]` for `k = 1..K`: frequency `k`'s 3×3 block without the constant `(0,0)`. |
| `sum_difference_energy(coeffs)` | `(sum [..., K], diff [..., K])` | Splits each `(k,k)` 2×2 block into energy on `cos/sin(w(a+b))` and on `cos/sin(w(a−b))`. The two add up to the block energy exactly. |
| `frequency_component(coeffs, basis, k)` | `[p, d]` | The part of the signal carried by frequency `k`, mapped back to `x`. Summing over `k = 0..K` gives the original. Raises `ValueError` if `k` is outside `0..K`. |

## embedding.py

| Signature | Returns | What it does |
|---|---|---|
| `EmbeddingSpectrum` | frozen dataclass | Fields: `coeffs [p, d]` (`basis @ weight`); `freq_norms [K+1]` (index 0 = constant); `dominant` (`tuple[int, ...]`, frequencies `k ≥ 1` in ascending order); `gini` (`float`, Gini coefficient of `freq_norms[1:]`). |
| `gini(x)` | `float` | Gini coefficient of a non-negative 1D tensor (Hurley & Rickard 2009). One-hot gives `1 − 1/n`, uniform gives 0, all-zero gives 0. |
| `dominant_frequencies(freq_norms, fraction)` | `tuple[int, ...]` | Frequencies `k ≥ 1` with `freq_norms[k] ≥ fraction · max(freq_norms[1:])`, in ascending order. Never includes the constant. |
| `embedding_spectrum(weight, cfg)` | `EmbeddingSpectrum` | Spectrum of a `[p, d]` weight over its input axis. The weight is detached first; the threshold is `cfg.dominant_fraction`. |
| `model_token_spectra(model, cfg)` | `(EmbeddingSpectrum, EmbeddingSpectrum)` | Spectra of a `SplitInputModel`'s `embedding.weight [p, n_in // 2]` and `unembedding.weight [p, n_out]`. |

## neurons.py

`Probe` = any callable `(model, tokens [B, 2]) -> [n, B]` (see Definitions).

| Signature | Returns | What it does |
|---|---|---|
| `all_pairs(p, device=None)` | `LongTensor[p*p, 2]` | Every `(a, b)` in row-major order: row `a*p + b` is `(a, b)`. |
| `readout_window_mean(model, tokens, neuron_idx)` | `[n, B]` | Probe: neuron activity averaged over the readout window (what the unembedding sees), via `model.rnn.window_mean`. |
| `state_at_step(model, tokens, neuron_idx, step)` | `[n, B]` | Probe: `h_step[neuron_idx]`; step 0 is the first yielded state. Stops after that step. Raises `ValueError` if `step` is outside `0..n_steps-1`. |
| `iter_activation_chunks(model, cfg, probe)` | generator of `[n, B_chunk]` | Runs `probe` under `no_grad` on consecutive `cfg.pair_chunk` slices of `all_pairs`. |
| `activation_grid(model, cfg, probe)` | `[n, p, p]` | Concatenates the chunks and reshapes: `grid[i, a, b]` = neuron `i`'s activation on `(a, b)`. |
| `NeuronSpectra` | frozen dataclass | Fields: `neuron_idx LongTensor[n]` (graph neuron ids); `grid [n, p, p]` (activation over `(a, b)`); `coeffs [n, p, p]` (2D coefficients: component of `a`, component of `b`); `dominant_freq LongTensor[n]` (best `k` in `1..K`); `fve [n]` (fraction of non-constant energy explained by `dominant_freq`, 0 if silent); `sum_share [n]` (share of the dominant `(k,k)` block on `w(a+b)` rather than `w(a−b)`, 0 if silent). |
| `silent_mask(nonconstant, full)` | `BoolTensor[n]` | True where the non-constant energy is at most `eps · full` (roundoff level: a constant grid leaves about `eps` of residue). |
| `masked_ratio(num, den, silent)` | `[n]` | `num / den`, with 0 where `silent` (and no NaN when `den == 0`). |
| `neuron_spectra(neuron_idx, grid, cfg)` | `NeuronSpectra` | Pure: 2D DFT of each `[p, p]` grid, then the best frequency, `fve` and `sum_share` per neuron. |
| `cluster_by_frequency(spectra, cfg)` | `dict[int, LongTensor]` | `{k: graph ids}` of neurons whose `dominant_freq` is `k` and `fve ≥ cfg.fve_threshold`. Keys are in ascending order. |

## plots.py

Every plot function returns a `matplotlib.figure.Figure` and does not touch pyplot or a backend.

| Signature | Returns | What it does |
|---|---|---|
| `plot_frequency_norms(freq_norms, cfg)` | `Figure` | Bar chart of the `[K+1]` norms, one bar per `k = 0..K` (Nanda 2023 Fig. 3). Size `cfg.figure_size`. |
| `plot_input_profile(values, cfg)` | `Figure` | Bar chart of a `[p]` value for each input `x`, e.g. one embedding column. |
| `plot_activation_grid(grid, cfg)` | `Figure` | Heatmap of a `[p, p]` activation with `a` on rows (y) and `b` on columns (x), origin at the bottom left, plus a colorbar (Nanda 2023 Fig. 4 centre). Size `cfg.heatmap_size`. |
| `plot_fourier_grid(coeffs, cfg)` | `Figure` | Heatmap of the absolute 2D Fourier coefficients of a `[p, p]` grid, plus a colorbar (Nanda 2023 Fig. 4 right). |
| `plot_fve_histogram(fve, cfg)` | `Figure` | Histogram of the `[n]` per-neuron FVE on `[0, 1]` with `cfg.histogram_bins` bins (Nanda 2023 Fig. 5 left). |
| `save_figure(fig, path, cfg)` | `Path` | Writes `fig` at `cfg.dpi` with a tight bounding box, creating parent directories, and returns `path`. |
| `_bar_figure`, `_heatmap`, `_numpy` | (private) | Shared helpers: bar chart over indices, `imshow` with colorbar, and tensor → numpy. |

## Definitions

**Basis.** `fourier_basis(p)` is orthonormal over `x = 0..p-1`, with `w_k = 2πk/p`:

```
row 0      = 1/sqrt(p)                    (constant)
row 2k - 1 = sqrt(2/p) * cos(w_k x),  k = 1..K
row 2k     = sqrt(2/p) * sin(w_k x),  k = 1..K
```

`p` must be odd, so every frequency has both a cos and a sin row.

**Fraction of variance explained (FVE).** Let `E = block_energy(coeffs)`. Frequency `k`'s energy is its 3×3 block without the constant: `E[k,k] + E[0,k] + E[k,0]`. This covers terms in both `a` and `b`, terms in `b` only, and terms in `a` only. `fve` is the best frequency's energy divided by the total non-constant energy (all energy minus `E[0,0]`).

**Sum vs difference.** With `cc = cos(wa)cos(wb)`, `ss = sin(wa)sin(wb)`, `sc = sin(wa)cos(wb)`, `cs = cos(wa)sin(wb)`:

```
cos(w(a+b)) = cc - ss      cos(w(a-b)) = cc + ss
sin(w(a+b)) = sc + cs      sin(w(a-b)) = sc - cs
```

`sum_difference_energy` projects each `(k,k)` block onto these directions. `sum_share` near 1 means the neuron depends on `a + b`, the clock-like pattern of a grokked network. Near 0 means it depends on `a − b`. A neuron that depends on only one operand has an empty `(k,k)` block, so its `sum_share` is 0.

**Gini sparsity.** `gini` follows Hurley & Rickard 2009: `G = 1 − 2·Σ_k (x_(k)/‖x‖₁)·(n − k + ½)/n` with `x` sorted ascending. A spectrum concentrated on a few frequencies scores near 1, so it works as a progress measure.

**Silent neurons.** A constant grid leaves roundoff-level residue in its non-constant coefficients, which would give a meaningless best frequency. When the non-constant energy is `≤ eps · total energy`, `fve` is 0. When the dominant `(k,k)` block energy is at that level, `sum_share` is 0. No NaN is produced.

**No transpose for the unembedding.** `SplitInputModel.unembedding.weight` is an `nn.Linear` weight, already shaped `[p, n_out]`. This is the reverse of Nanda's `W_U [d, p]`, so `embedding_spectrum` applies to it as is.

**Probes.** The caller binds the neuron list (and step) with `functools.partial`:

```python
partial(readout_window_mean, neuron_idx=idx)     # what the unembedding sees
partial(state_at_step, neuron_idx=idx, step=t)   # one time step
```

New probes plug in without changing the grid collection.

**Memory (`pair_chunk`).** A state is `[N, B]`. On the real fly `N = 139,248`, so running all `p² = 12,769` pairs at once would need about 7 GB per step. Pairs therefore run in slices of `cfg.pair_chunk` (256 by default, about 143 MB per state), and only the selected neurons' `[n, B_chunk]` rows are kept.

## Putting it together

Build `model` (a `SplitInputModel`) as in [modularfly/README.md](../README.md), then:

```python
from functools import partial
from modularfly.config import InterpretabilityConfig
from modularfly.interpretability.embedding import model_token_spectra
from modularfly.interpretability.neurons import (
    activation_grid, cluster_by_frequency, neuron_spectra, readout_window_mean,
)
from modularfly.interpretability.plots import (
    plot_activation_grid, plot_fourier_grid, plot_frequency_norms, plot_fve_histogram,
    plot_input_profile, save_figure,
)

cfg = InterpretabilityConfig()
emb, unemb = model_token_spectra(model, cfg)        # emb.dominant, emb.gini, ...

probe = partial(readout_window_mean, neuron_idx=model.rnn.output_idx)
grid = activation_grid(model, cfg, probe)           # [n_out, p, p]
spectra = neuron_spectra(model.rnn.output_idx, grid, cfg)
clusters = cluster_by_frequency(spectra, cfg)       # {k: graph neuron ids}

save_figure(plot_frequency_norms(emb.freq_norms, cfg), cfg.figure_dir / "embedding_freq_norms.png", cfg)
plot_input_profile(model.embedding.weight[:, 0].detach(), cfg)
plot_activation_grid(spectra.grid[0], cfg)
plot_fourier_grid(spectra.coeffs[0], cfg)
plot_fve_histogram(spectra.fve, cfg)
```

## Next features

- **Whole-brain summary scan**: consume `iter_activation_chunks` with all `N` neurons into a memmap, keeping only the per-neuron summaries rather than the full grids.
- **aCRT sinusoid fit** on `NeuronSpectra.grid`.
- **Logit 2D DFT / restricted loss** through a probe that returns `model(tokens).t()`.
- **CLI script** to run the analysis once a checkpoint format exists.
