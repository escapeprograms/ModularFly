# modularfly

The FlyWire v783 fly-brain connectome as a PyTorch recurrent network. The wiring is fixed; the weights come from a pluggable module.

| Folder / file | What it holds |
|---|---|
| [config.py](config.py) | Every tunable value (paths, thresholds, sign map, dynamics, scales, embeddings, interpretability). |
| [data/](data/README.md) | Download the raw files, build the graph, group neurons, rewire. |
| [model/](model/README.md) | `fly/`: the recurrent network, the sparse product and scale estimates; `weights/`: the weight modules; `io/`: the token embeddings; `end_to_end/`: complete tokens-to-logits models. |
| [interpretability/](interpretability/README.md) | Fourier analysis of a trained modular-addition model (Nanda et al. 2023): embedding spectra, per-neuron 2D spectra over `(a, b)`, and figures. |

Shapes used throughout: `N` neurons (139,248), `E` connections (2,700,429 at ≥5 synapses), `B` batch, `T` time steps.

## Putting it together

```python
import torch
from modularfly.config import DataConfig, DynamicsConfig
from modularfly.data.graph import graph_cache_path, load_graph
from modularfly.data.groups import group_neurons, pair_gain_index, select_neurons
from modularfly.model.fly.connectome import ConnectomeRNN
from modularfly.model.weights.gain_modulated import GainModulatedWeights, GroupGain
from modularfly.model.weights.signed_edge import SignedEdgeWeights
from modularfly.model.weights.synapse_count import SynapseCountWeights

data, dyn = DataConfig(), DynamicsConfig()
graph = load_graph(graph_cache_path(data.processed_dir, data.min_syn_count))
input_idx = select_neurons(graph, data.input_level, data.input_labels)     # 19,262 sensory + ascending
output_idx = select_neurons(graph, data.output_level, data.output_labels)  # 1,489 motor, endocrine, descending

w0 = 4.97e-4  # printed by scripts/build_graph.py

# 1. Frozen natural weights
frozen = SynapseCountWeights(graph, w0)

# 2. Natural weights with a learned gain per cell-type pair
edge_group, pair_keys = pair_gain_index(graph, group_neurons(graph, "cell_type"))
gained = GainModulatedWeights(SynapseCountWeights(graph, w0), GroupGain(edge_group, len(pair_keys)))

# 3. Random trainable weight per connection, signs kept
random = SignedEdgeWeights(graph, inhibitory_init_scale=1.0, generator=torch.Generator().manual_seed(0))

rnn = ConnectomeRNN(graph, frozen, input_idx, output_idx, dyn).cuda()
drive = torch.randn(32, input_idx.numel()).cuda()
readout = rnn(drive)  # drive [B, 19262] -> readout [B, 1489]
```

For modular addition, `SplitInputModel` wraps the network with a token embedding (each number drives a random half of the input neurons) and an output layer:

```python
from modularfly.config import EmbeddingConfig
from modularfly.model.end_to_end.split_input import SplitInputModel

model = SplitInputModel(rnn, EmbeddingConfig(), generator=torch.Generator().manual_seed(0)).cuda()
tokens = torch.randint(0, 113, (32, 2)).cuda()
logits = model(tokens)  # tokens [B, 2] -> logits [B, 113]
```

## config.py

All values are dataclass fields with defaults; construct with no arguments and override what you need.

**`RemoteFile(url, relative_path, md5)`**: one downloadable file. `CONNECTIONS_783` and `NEURON_ANNOTATIONS` are the two instances. The annotation file follows its repository's main branch, so its md5 can change upstream.

**`NT_SIGN`**: transmitter name → sign. Acetylcholine, dopamine, serotonin and octopamine are `+1`; glutamate and GABA are `-1` (Shiu et al. 2024).

**`DataConfig`**

| Field | Default | Meaning |
|---|---|---|
| `raw_dir`, `processed_dir` | `data/raw`, `data/processed` | Where downloads and the cached graph live. |
| `connections`, `annotations` | the two `RemoteFile`s | Source files. |
| `min_syn_count` | `5` | A connection needs at least this many synapses. |
| `annotation_levels` | `flow, super_class, cell_class, cell_sub_class, supertype, cell_type` | Annotation hierarchy, coarsest first. |
| `extra_annotation_columns` | `side, top_nt` | Extra labels stored on the graph. |
| `nt_column`, `nt_confidence_column` | `top_nt`, `top_nt_conf` | Columns giving each neuron's transmitter and its confidence. |
| `nt_sign` | copy of `NT_SIGN` | Transmitter → sign map. |
| `missing_nt_sign` | `1.0` | Sign for the 602 neurons with no prediction. |
| `input_level`, `input_labels` | `flow`, `("afferent",)` | Which neurons are inputs. |
| `output_level`, `output_labels` | `flow`, `("efferent",)` | Which neurons are outputs. |
| `download_chunk_bytes` | 1 MiB | Read size for download and md5. |

**`DynamicsConfig`**

| Field | Default | Meaning |
|---|---|---|
| `alpha` | `0.5` | Leak in `h ← (1 − alpha)·h + alpha·tanh(u + W·h + b)`. |
| `n_steps` | `64` | Time steps `T`. |
| `readout_steps` | `4` | The readout averages this many final steps. |
| `propagate_chunk_edges` | 262,144 | Connections per chunk when computing weight gradients (a memory knob). |

**`ScaleConfig`**

| Field | Default | Meaning |
|---|---|---|
| `target_spectral_radius` | `1.0` | `w0` is chosen so the natural-weight matrix has this spectral radius. At 1.0 the network is mostly quiet; 3–10 looked more usable. |
| `power_iterations` | `200` | Iterations for the spectral-radius estimate. |
| `inhibitory_init_scale` | `1.0` | Multiplier on inhibitory random weights at init. 1.57 balances excitation and inhibition. |
| `seed` | `0` | Seed for the estimate and the random weights. |

**`RewireConfig`**

| Field | Default | Meaning |
|---|---|---|
| `swaps_per_edge` | `5` | Accepted swaps per connection. |
| `max_rounds` | `200` | Rewiring raises if it needs more rounds than this. |
| `seed` | `0` | Seed for the random pairing. |

**`EmbeddingConfig`**

| Field | Default | Meaning |
|---|---|---|
| `modulus` | `113` | `p`: tokens are `0..p-1` and there are `p` answers (Nanda et al. 2023). |
| `init_std` | `1.0` | Embedding entries start `~ N(0, init_std²)`; each entry is one neuron's drive. |

**`InterpretabilityConfig`**

| Field | Default | Meaning |
|---|---|---|
| `pair_chunk` | `256` | `(a, b)` pairs per forward pass when collecting activation grids. At 256, one state is about 143 MB on the real fly. |
| `dominant_fraction` | `0.5` | A frequency is dominant if its norm is at least this fraction of the largest non-constant norm. |
| `fve_threshold` | `0.85` | A neuron counts as single-frequency if its best frequency explains at least this fraction of its variance (Nanda 2023). |
| `histogram_bins` | `20` | Bins in the FVE histogram. |
| `figure_size` | `(6.0, 4.0)` | Size in inches of bar charts and histograms. |
| `heatmap_size` | `(5.0, 4.5)` | Size in inches of `(a, b)` heatmaps. |
| `dpi` | `150` | Resolution of saved figures. |
| `figure_dir` | `figures` | Where figures are saved by default. |
