# modularfly/model/weights

The pluggable modules that supply the recurrent network's connection weights.

A weight module is any `nn.Module` whose `forward()` takes no arguments and returns `FloatTensor[E]`, one weight per connection in the graph's order, and whose `drift_penalty()` returns a 0-dim tensor: the sum of squared deviation of its trainable parameters from their initial values (zero for a frozen module). The training loop adds `fly_penalty · drift_penalty()` to the loss and logs it as `theta_drift` (see [training/README.md](../../training/README.md)).

| File | Job |
|---|---|
| [synapse_count.py](synapse_count.py) | `SynapseCountWeights`: the fly's natural weights, frozen. |
| [gain_modulated.py](gain_modulated.py) | `GainModulatedWeights` and `GroupGain`: natural weights times learned gains shared by groups of connections. |
| [signed_edge.py](signed_edge.py) | `SignedEdgeWeights`: one random trainable weight per connection, signs kept. |

## synapse_count.py

| Class | `forward()` returns | Trainable | What it is |
|---|---|---|---|
| `SynapseCountWeights(graph, w0)` | `w0 · sign(pre) · syn_count` | nothing | The fly's natural weights, frozen (Shiu et al. 2024). |

`drift_penalty()` returns a 0-dim zero (no grad): there is nothing to drift.

Notes:
- `w0` is a plain float; `scripts/build_graph.py` prints a value derived from the target spectral radius.

## gain_modulated.py

| Class | `forward()` returns | Trainable | What it is |
|---|---|---|---|
| `GainModulatedWeights(base, gain)` | `base() · gain()` | whatever `gain` holds | Natural weights times learned gains (Duan et al. 2025). |

`GainModulatedWeights.drift_penalty()` returns `base.drift_penalty() + gain.drift_penalty()`.

### `GroupGain(edge_group, n_groups)`

| Member | Returns | What it does |
|---|---|---|
| `forward()` | `FloatTensor[E]` | `1 + z[edge_group]`. `z` has one learned value per group and starts at 0, so every gain starts at exactly 1. |
| `min_gain()` | scalar tensor | The smallest gain. A gain below 0 flips a connection's sign; gains are not clamped, so log this during training. |
| `drift_penalty()` | 0-dim tensor, with grad | `sum(z²)`. `z` starts at exactly 0, so this is the squared drift from init and needs no stored copy. |

`edge_group` comes from the index builders in [data/groups.py](../../data/README.md): `pair_gain_index` (cell-type pairs), `global_gain_index` (one gain) or `sign_gain_index` (excitatory vs inhibitory).

## signed_edge.py

| Class | `forward()` returns | Trainable | What it is |
|---|---|---|---|
| `SignedEdgeWeights(graph, inhibitory_init_scale, generator)` | `sign(pre) · abs(theta) / sqrt(in_degree)` | `theta[E]` | One random trainable weight per connection, `theta ~ N(0, 1)` (Cho et al. 2026, with signs added). |

| Member | Returns | What it does |
|---|---|---|
| `theta_init` (buffer) | `FloatTensor[E]` | Copy of `theta` at construction, after `inhibitory_init_scale`. In `state_dict()`, so it is saved in checkpoints (about 11 MB on the real fly). Never trained. |
| `drift_penalty()` | 0-dim tensor, with grad | `sum((theta − theta_init)²)`, the L2 pull toward the initial magnitudes. 0 at init. |

Notes:
- In `SignedEdgeWeights`, taking the absolute value of `theta` keeps every neuron excitatory or inhibitory during training. `inhibitory_init_scale` multiplies the inhibitory `theta` values once, at initialisation.
