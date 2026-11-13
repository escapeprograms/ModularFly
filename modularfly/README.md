# modularfly

The FlyWire v783 fly-brain connectome as a PyTorch recurrent network. The wiring is fixed; the weights come from a pluggable module.

| Folder / file | What it holds |
|---|---|
| [config.py](config.py) | Every tunable value (paths, thresholds, sign map, dynamics, scales, embeddings, the channel-group interface, interpretability, task split, training, experiment outputs, W&B). |
| [data/](data/README.md) | Download the raw files, build the graph, group neurons, rewire. |
| [model/](model/README.md) | `fly/`: the recurrent network, the sparse product and scale estimates; `weights/`: the weight modules; `io/`: the token embedding and the channel-group interface functions; `end_to_end/`: complete tokens-to-logits models. |
| [interpretability/](interpretability/README.md) | Fourier analysis of a trained modular-addition model (Nanda et al. 2023): embedding spectra, per-neuron 2D spectra over `(a, b)`, and figures. |
| [training/](training/README.md) | Modular-addition data (all `p²` labeled pairs, seeded train/test split), the minibatch training and evaluation loop, and checkpoint save/load. |
| [experiments/](experiments/README.md) | Experiment composition: build the real-fly model for a weight condition, run train → per-epoch Gini → history/checkpoint → spectra report into one run directory. |

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

For modular addition, `GroupInterfaceModel` wraps the network with the channel-group interface of arXiv 2610.10014: a learned `[59, 22]` token embedding gives 44 channels, each copied onto its own group of 64 input neurons, and logit `k` is the mean readout of its 32-neuron output group divided by `temperature` (no learned readout weights). The groups are sampled with `sample_interface` from any connected neuron (out-degree > 0 for inputs, 129,343 candidates; in-degree > 0 for outputs, 125,184 candidates), all disjoint:

```python
from modularfly.config import EmbeddingConfig, InterfaceConfig
from modularfly.model.end_to_end.group_interface import GroupInterfaceModel
from modularfly.model.io.interface import sample_interface

emb, interface = EmbeddingConfig(), InterfaceConfig()
input_groups, output_groups = sample_interface(
    graph, 2 * emb.channels_per_token, interface.input_group_size, emb.modulus,
    interface.output_group_size, torch.Generator().manual_seed(interface.seed))
# input_groups [44, 64] (2,816 neurons), output_groups [59, 32] (1,888 neurons)
group_rnn = ConnectomeRNN(graph, random, input_groups.flatten(), output_groups.flatten(), dyn)
model = GroupInterfaceModel(group_rnn, emb, interface).cuda()
tokens = torch.randint(0, 59, (32, 2)).cuda()
logits = model(tokens)  # tokens [B, 2] -> logits [B, 59]
```

## Training and running an experiment

`experiments/` builds the model for a weight condition and runs one experiment into `runs/<name>/` (configs, per-epoch history with Gini, checkpoint, spectra report and figures); `training/` holds the pure loop it uses. From the command line: `python scripts/run_experiment1.py --epochs 50`. In Python:

```python
from modularfly.config import EmbeddingConfig, ExperimentConfig, InterfaceConfig, InterpretabilityConfig, ScaleConfig, TaskConfig, TrainConfig
from modularfly.experiments.fly_model import build_group_interface_model, random_trainable_weights
from modularfly.experiments.run import run_experiment

configs = {"task": TaskConfig(), "train": TrainConfig(), "embedding": EmbeddingConfig(),
           "interface": InterfaceConfig(),
           "interpretability": InterpretabilityConfig(), "experiment": ExperimentConfig()}
model = build_group_interface_model(graph, random_trainable_weights(graph, ScaleConfig()), dyn,
                                    configs["embedding"], configs["interface"],
                                    configs["train"].seed).cuda()
rows = run_experiment(model, configs, configs["experiment"].runs_dir / "demo", on_row=print)
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
| `input_level`, `input_labels` | `flow`, `("afferent",)` | Which neurons are inputs for `select_neurons` (used by `build_graph.py`, `smoke_gpu.py` and the tests). Experiment 1 does not use them: it samples its interface from any connected neuron (`InterfaceConfig`). |
| `output_level`, `output_labels` | `flow`, `("efferent",)` | Which neurons are outputs for `select_neurons`; likewise unused by experiment 1. |
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
| `modulus` | `59` | `p`: tokens are `0..p-1` and there are `p` answers. |
| `channels_per_token` | `22` | Embedding width `d` per operand; the two operands give `2d = 44` input channels (arXiv 2610.10014). |
| `init_std` | `1.0` | Embedding entries start `~ N(0, init_std²)`; each entry is one channel's drive. |

**`InterfaceConfig`** (how channels touch neurons and how neurons become logits; paper defaults)

| Field | Default | Meaning |
|---|---|---|
| `input_group_size` | `64` | Neurons driven by each input channel (44 × 64 = 2,816 input neurons on the real fly). |
| `output_group_size` | `32` | Neurons averaged into each class logit (59 × 32 = 1,888 output neurons). |
| `input_amplitude` | `1.0` | Each channel value is copied onto its group times this. |
| `temperature` | `0.1` | Logit = group-mean readout / temperature. |
| `seed` | `0` | Interface sampling seed, independent of `TrainConfig.seed`, so the interface stays fixed across training seeds. |

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

**`TaskConfig`**

| Field | Default | Meaning |
|---|---|---|
| `train_fraction` | `0.5` | Share of the `p²` pairs used for training; the rest is the test set. 0.3 is Nanda 2023's grokking regime; 0.5 comes first, to check the connectome can learn and generalize at all. |
| `seed` | `0` | Seed of the split permutation. |

`p` is not repeated here; it is `EmbeddingConfig.modulus`.

**`TrainConfig`**

| Field | Default | Meaning |
|---|---|---|
| `batch_size` | `32` | Training minibatch. Backprop keeps 64 `[N, B]` states; 2.41 GB peak on the real fly. |
| `eval_batch` | `512` | Chunk size for evaluation under `no_grad` (about 1 GB peak at 512). |
| `lr` | `1e-3` | AdamW learning rate of the token embedding. |
| `fly_lr` | `1e-3` | AdamW learning rate of the recurrent weights (`model.rnn`). Each `theta` moves at most about `fly_lr` per step, so 3e-3..1e-2 is worth trying. |
| `betas` | `(0.9, 0.98)` | AdamW betas (Nanda 2023). |
| `weight_decay` | `0.1` | Weight decay on the token embedding. AdamW shrinks it by `lr · wd` per step; 1.0 (Nanda 2023) would decay the `[59, 22]` table about 5x over a 50-epoch run. |
| `fly_weight_decay` | `0.0` | Weight decay on the recurrent weights. At 1.0 every `theta` would shrink by 0.999 per step and silence the dynamics. |
| `fly_penalty` | `0.0` | Coefficient on the drift penalty `sum((theta − theta_init)²)` added to the training loss: an L2 pull toward the initial fly weights instead of toward zero. The logged `train_loss` stays cross-entropy; the unweighted penalty is logged as `theta_drift`. Off (`0`) by default: 1e-3 pinned `theta` (measured drift plateau ~90, RMS 0.006 per edge) and the plateau scales ~1/lambda, so sweep 1e-6..1e-5. See [training/README.md](training/README.md#definitions). |
| `grad_clip_norm` | `1.0` | Gradient-norm clip, applied every step (arXiv 2610.10014 clips at 1). The pre-clip norm is always logged. |
| `n_epochs` | `50` | Epochs to train. At `p = 59` an epoch is 33 training steps of about 1.26 s each on the real fly, plus test evaluation. |
| `seed` | `0` | Seed for model init (`torch.manual_seed`, the embedding) and batch order. The interface has its own seed (`InterfaceConfig.seed`). |

**`ExperimentConfig`**

| Field | Default | Meaning |
|---|---|---|
| `runs_dir` | `runs` | Parent folder of the run directories (gitignored). |
| `device` | `"cuda"` | Device name, a plain string so `config.py` imports no torch. |
| `config_name` | `config.json` | File holding every config of a run. |
| `history_name` | `history.csv` | Per-epoch training history. |
| `checkpoint_name` | `checkpoint.pt` | Model and optimizer state. |
| `report_name` | `embedding_report.json` | Token-embedding spectrum summary. |
| `figure_dir_name` | `figures` | Subfolder of the run directory for figures. |

**`WandbConfig`**

| Field | Default | Meaning |
|---|---|---|
| `entity` | `"archimedesli"` | W&B user or team that owns the project. |
| `project` | `"ModularFly"` | W&B project the runs go to. |
| `mode` | `"online"` | Passed straight to `wandb`: `online`, `offline` (logs locally under `wandb/`) or `disabled` (no-op, no network). |
| `env_file` | `.env` (repo root, gitignored) | Dotenv file holding the API key. |
| `key_variable` | `"WANDB_KEY"` | Name of the API-key variable in `env_file`. |
| `artifact_name` | `"run-files"` | Name of the artifact holding `config.json`, `history.csv`, `embedding_report.json` and `figures/`. |
