# modularfly/model/end_to_end

Complete tokens-to-logits models built from the recurrent network and the token layers, one per input mechanism. New input mechanisms go here.

| File | Job |
|---|---|
| [group_interface.py](group_interface.py) | `GroupInterfaceModel`: the channel-group interface of arXiv 2610.10014. Token channels drive neuron groups; class logits are group-mean readouts. |

## group_interface.py

`d` = `channels_per_token`, `C = 2d` input channels, `G_in` = `input_group_size`, `K = p` = `modulus` classes, `G_out` = `output_group_size`. On the real fly with the defaults: `d = 22`, `C = 44`, `G_in = 64` (2,816 input neurons), `K = 59`, `G_out = 32` (1,888 output neurons).

### `GroupInterfaceModel(rnn, emb, interface)`

```
tokens LongTensor[B, 2] -> embedding [B, 2, d] -> channels [B, C]
  -> drive [B, C·G_in]        (channel c copied, times input_amplitude, onto input group c)
  -> rnn readout [B, K·G_out] (window mean of the output neurons)
  -> logits [B, K]            (mean of each output group / temperature)
```

| Argument | Meaning |
|---|---|
| `rnn` | A `ConnectomeRNN` whose `input_idx` holds exactly `C·G_in` neurons and `output_idx` exactly `K·G_out` (otherwise `ValueError`). Built by `experiments/fly_model.py::build_group_interface_model` on groups from [`sample_interface`](../io/README.md). |
| `emb` | An [`EmbeddingConfig`](../../README.md) (`modulus` = `p`, `channels_per_token` = `d`, `init_std`). |
| `interface` | An [`InterfaceConfig`](../../README.md) (`input_group_size`, `output_group_size`, `input_amplitude`, `temperature`). Its `seed` is used by the builder, not here. |

| Member | Returns | What it does |
|---|---|---|
| `forward(tokens)` | `[B, K]` | `group_mean_logits(rnn(drive(tokens)), K, G_out, temperature)`. `tokens` is `LongTensor[B, 2]`. |
| `drive(tokens)` | `[B, C·G_in]` | `group_drive(channels(tokens), G_in, input_amplitude)`: position `c·G_in + j` is channel `c`, landing on `rnn.input_idx[c·G_in + j]`. |
| `channels(tokens)` | `[B, C]` | `embedding(tokens).flatten(1)`: operand `a`'s `d` channels, then operand `b`'s. A new input code replaces only this method. |
| `input_groups` (property) | `LongTensor[C, G_in]` | Graph neuron ids; row `c` is driven by channel `c`. |
| `output_groups` (property) | `LongTensor[K, G_out]` | Graph neuron ids; row `k` is averaged into logit `k`. |
| `embedding` | `nn.Embedding [p, d]` | The token table from [io/embedding.py](../io/README.md), shared by both operands. |
| `rnn` | `ConnectomeRNN` | The recurrent network. |
| `cfg`, `n_channels`, `n_classes` | | The `InterfaceConfig`, `C` and `K`. |

| Function | Returns | What it does |
|---|---|---|
| `check_group_sizes(rnn, n_in, n_out)` | `None` | Raises `ValueError` unless the RNN has exactly `n_in` input and `n_out` output neurons. |

**The groups are views of `rnn.input_idx` / `rnn.output_idx`** (row-major `view(C, G_in)` and `view(K, G_out)`), not separate buffers, so nothing is stored twice in the checkpoint. `ConnectomeRNN` scatters drive position `i` onto `input_idx[i]` and returns its readout in `output_idx` order, so flattening the sampled groups and viewing them back is exact.

**Trainable parameters**: `embedding.weight [p, d]` plus the weight module's (`rnn.edge_weights.theta` for `SignedEdgeWeights`). There is no learned readout.

Experiment 2 (frozen natural weights) = pass [`SynapseCountWeights`](../weights/README.md) to the RNN; then only `embedding.weight` trains.
