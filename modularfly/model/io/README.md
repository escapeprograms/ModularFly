# modularfly/model/io

The layers and functions that turn tokens into input drive and output activity into logits.

`B` = batch, `p` = modulus (number of tokens and of answers), `C` = input channels, `G_in` = input group size, `K` = classes (`= p`), `G_out` = output group size.

| File | Job |
|---|---|
| [embedding.py](embedding.py) | The token-embedding factory and `N_OPERANDS`. |
| [interface.py](interface.py) | The channel-group interface (arXiv 2610.10014): sample disjoint input/output neuron groups from a graph, copy channels onto input groups, collapse output groups into logits. Pure tensor functions, no `nn.Module`, no I/O. |

## embedding.py

| Signature | Returns | What it does |
|---|---|---|
| `N_OPERANDS` | `int` | `2`: tokens per example `(a, b)`. The one place the two-operand assumption lives (channels = `N_OPERANDS · channels_per_token`). |
| `input_embedding(n_tokens, dim, init_std)` | `nn.Embedding` | One table `[n_tokens, dim]` shared by both numbers, entries `~ N(0, init_std²)`. Maps `LongTensor[B, 2]` to `[B, 2, dim]`. |

## interface.py

| Signature | Returns | What it does |
|---|---|---|
| `input_candidates(graph)` | `LongTensor` (sorted) | Neurons with out-degree > 0: `graph.pre.unique()` (129,343 on the real fly). |
| `output_candidates(graph, exclude)` | `LongTensor` (sorted) | Neurons with in-degree > 0 (`graph.post.unique()`, 125,184 on the real fly) that are not in `exclude`. |
| `sample_groups(candidates, n_groups, group_size, generator)` | `LongTensor[n_groups, group_size]` | `n_groups · group_size` distinct candidates drawn without replacement (`randperm` on the generator). Raises `ValueError` if there are too few candidates. |
| `sample_interface(graph, n_channels, input_group_size, n_classes, output_group_size, generator)` | `(LongTensor[C, G_in], LongTensor[K, G_out])` | Input groups first from `input_candidates`, then output groups from `output_candidates` excluding the chosen inputs, so every neuron is in at most one group. The generator is a CPU `torch.Generator`. |
| `group_drive(channels, group_size, amplitude)` | `[B, C·G_in]` | `channels [B, C]` with each channel repeated `group_size` times (`repeat_interleave`) and multiplied by `amplitude`: position `c·G_in + j` is `channels[:, c] · amplitude`. |
| `group_mean_logits(readout, n_classes, group_size, temperature)` | `[B, K]` | `readout [B, K·G_out]` viewed as `[B, K, G_out]`, averaged over the last axis, divided by `temperature`. |

The groups are flattened row-major into `ConnectomeRNN`'s `input_idx` / `output_idx`, which is what makes `group_drive` and `group_mean_logits` line up with the right neurons (see [end_to_end/](../end_to_end/README.md)).
