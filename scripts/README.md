# scripts

Command-line entry points. Run them from the repository root with the `modular-fly` environment's Python, in this order:

```
python scripts/download_flywire.py
python scripts/build_graph.py
python scripts/smoke_gpu.py
python scripts/run_experiment1.py
```

Settings come from the defaults in [modularfly/config.py](../modularfly/config.py).

The scripts import the installed `modularfly` package. Install it once, editable, from the repository root: `python -m pip install -e .` (see [pyproject.toml](../pyproject.toml)).

## download_flywire.py

Downloads the two FlyWire files into `data/raw/` if they are missing and checks their md5. No arguments.

- **Output:** prints `ok <path>` for each file. Raises `ValueError` on an md5 mismatch.
- **`main()`**: calls `ensure_flywire_files(DataConfig())`.

## build_graph.py

Builds the graph from the raw files, saves it to `data/processed/flywire783_graph_min5.pt`, and prints a summary.

| Argument | Default | Meaning |
|---|---|---|
| `--device` | `cuda` if available, else `cpu` | Device for the spectral-radius estimate. |

**Output (printed):** build time; neuron and connection counts; connections dropped for unannotated endpoints; saved path; input and output neuron counts; neurons per `flow` and per `super_class`; excitatory/inhibitory neuron and connection counts; neurons with no transmitter prediction; the balancing inhibitory scale; the spectral radius and the derived `w0`.

| Function | What it does |
|---|---|
| `print_label_counts(graph, level)` | Prints neurons per label at `level`, largest first, plus the number missing. |
| `print_io_counts(graph, cfg)` | Prints how many neurons the configured input and output selections contain. |
| `print_ei_counts(graph)` | Prints excitatory/inhibitory counts, missing-transmitter count, and the balancing scale. |
| `print_scale(graph, scale, device)` | Estimates the spectral radius and prints it with the derived `w0`. |
| `main()` | Parses arguments, builds, saves, and calls the printers. |

## smoke_gpu.py

Loads the cached graph and runs one forward and backward pass for each of the three weight settings on the GPU. Needs CUDA and a graph already built.

| Argument | Default | Meaning |
|---|---|---|
| `--batch` | `32` | Batch size. |
| `--drive-scale` | `1.0` | Standard deviation of the random input drive. |
| `--level` | `cell_type` | Annotation level for the cell-type-pair gains. |

**Output (printed, per setting):** peak GPU memory (flagged above 6 GB), time for the first and second step, number of trainable parameters, which parameters received gradients, and final-step activity statistics (mean activity, fraction saturated, fraction near zero, output-neuron activity).

| Function | Returns | What it does |
|---|---|---|
| `condition_weights(graph, w0, scale, level)` | `dict[str, nn.Module]` | Builds the three weight modules, keyed by a display name. |
| `timed_step(rnn, drive)` | `float` | Seconds for one forward and backward pass. |
| `gradient_report(rnn, drive)` | `str` | Parameter count and which parameters got gradients. |
| `activity_report(rnn, drive)` | `str` | Final-step activity statistics. |
| `run_condition(name, weights, graph, io_idx, drive, dyn)` | `None` | Runs and prints one setting. |
| `main()` | `None` | Parses arguments, derives `w0`, and runs all three settings. |

## run_experiment1.py

Experiment 1 (`project_design.md`): trains the random-init fly weights together with the token embedding on `(a + b) mod p` (`p = 59`) through the channel-group interface of arXiv 2610.10014, then reports the embedding's Fourier spectrum. A learned `[59, 22]` embedding gives each operand 22 channels (44 in all); each channel drives its own randomly sampled group of 64 input neurons (2,816 in all). Each of the 59 classes reads a randomly sampled group of 32 output neurons (1,888 in all), and its logit is the group's mean readout divided by `temperature = 0.1`; there is no unembedding or other learned readout. The groups are sampled once with `InterfaceConfig.seed` and stay fixed across `--seed` values. Needs a graph already built; uses CUDA by default.

Every argument except `--run-name` defaults to `None`, meaning "keep the value in `config.py`".

| Argument | Overrides | Meaning |
|---|---|---|
| `--run-name` | (default: timestamp `YYYYMMDD-HHMMSS`) | Run directory name under `ExperimentConfig.runs_dir`. An existing name raises `FileExistsError`. |
| `--epochs` | `TrainConfig.n_epochs` | Epochs to train. |
| `--batch` | `TrainConfig.batch_size` | Training minibatch. |
| `--eval-batch` | `TrainConfig.eval_batch` | Evaluation chunk size. |
| `--lr` | `TrainConfig.lr` | AdamW learning rate of the token embedding. |
| `--fly-lr` | `TrainConfig.fly_lr` | AdamW learning rate of the recurrent weights. |
| `--weight-decay` | `TrainConfig.weight_decay` | Weight decay on the token embedding. |
| `--fly-weight-decay` | `TrainConfig.fly_weight_decay` | Weight decay on the recurrent weights. |
| `--fly-penalty` | `TrainConfig.fly_penalty` | Coefficient on the drift penalty `sum((theta − theta_init)²)` in the training loss (default `0`, off; sweep 1e-6..1e-5). |
| `--grad-clip-norm` | `TrainConfig.grad_clip_norm` | Gradient-norm clip applied every step (default 1.0). |
| `--train-fraction` | `TaskConfig.train_fraction` | Share of the `p²` pairs used for training. |
| `--seed` | `TrainConfig.seed` and `TaskConfig.seed` | Embedding init, batch order and the train/test split. Not the interface sampling (`InterfaceConfig.seed`). |
| `--device` | `ExperimentConfig.device` | Device name, e.g. `cuda` or `cpu`. |
| `--wandb-mode` | `WandbConfig.mode` | `online`, `offline` or `disabled`. |
| `--wandb-project` | `WandbConfig.project` | W&B project. |
| `--wandb-entity` | `WandbConfig.entity` | W&B user or team. |

**W&B:** the run is also logged to Weights & Biases under the run name: every history row per epoch, then the figures, the final Gini/dominant frequencies (run summary) and an artifact with `config.json`, `history.csv`, `embedding_report.json` and `figures/` (the checkpoint stays local). The API key is read from `WANDB_KEY` in the repo-root `.env`; the file must exist even with `--wandb-mode disabled`. See [modularfly/experiments/README.md](../modularfly/experiments/README.md#wb).

**Output (printed):** a header line

```
N <neurons>, E <connections>, n_in 2816, channels 44 x 64, n_out 1888, T 64, p 59, n_train 1740, batch 32, epochs 50, device cuda, run_dir <path>
```

then one line per epoch

```
epoch   0  train loss <..> acc <..>  test loss <..> acc <..>  grad <norm>  <seconds> s  gini emb <embedding Gini>  drift <theta_drift>
```

`drift` is the unweighted `sum((theta − theta_init)²)` at the end of the epoch (0 for frozen weights); the RMS drift per connection is `sqrt(drift / E)`.

and at the end `embedding: gini <..>, dominant frequencies [..]`. **Output (files):** the run directory described in [modularfly/experiments/README.md](../modularfly/experiments/README.md#run-directory) (`config.json`, `history.csv`, `checkpoint.pt`, `embedding_report.json`, `figures/*.png`).

| Function | Returns | What it does |
|---|---|---|
| `parse_args()` | `Namespace` | Defines the arguments above. |
| `override(cfg, **values)` | dataclass | A copy of `cfg` with every non-`None` value applied (`dataclasses.replace`). |
| `build_configs(args)` | `dict[str, object]` | Every config keyed by name (`task`, `train`, `embedding`, `interface`, `interpretability`, `experiment`, `data`, `dynamics`, `scale`, `wandb`), with the CLI overrides applied. `data` is kept for the graph path. |
| `print_row(row)` | `None` | Prints one history row (losses, accuracies, grad norm, seconds, embedding Gini, theta drift). |
| `print_and_log_row(run, row)` | `None` | The `on_row` callback (bound to the run with `partial`): `print_row`, then `log_row`. |
| `print_header(graph, model, configs, run_dir)` | `None` | Prints the header line, including the interface shape `channels C x G_in` from `model.input_groups`. |
| `print_summary(summary)` | `None` | Prints the Gini and dominant frequencies of each table of `spectra_summary`. |
| `main()` | `None` | Loads the graph, builds random trainable weights and the model with `build_group_interface_model` on the device, starts the W&B run, runs `run_experiment`, prints the summary, and finishes the W&B run (`finish_run`). |

Timing on the real graph (RTX 2000 Ada, 8 GB) with the defaults: `p = 59`, so the 50 % train split is 1,740 of the 3,481 pairs, which is 55 steps per epoch at batch 32. The per-step time is unchanged at about 1.26 s/step (the state `[N, B]` and the per-edge `theta` gradient dominate, not the interface). At the earlier 30 % split (33 steps) an epoch plus the test evaluation in chunks of 512 took about 50 s (measured); at 50 % that is about 55 steps × 1.26 s plus evaluation, 78 s (measured). Each checkpoint is 143 MB (measured; it includes the propagator's index buffers and the 11 MB `theta_init` buffer).

