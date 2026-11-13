# modularfly/experiments

Composes the other packages into one experiment: build the real-fly model for a weight condition, train it on modular addition, measure the token-embedding spectrum every epoch, and write everything into one run directory. `training/` stays pure; the file I/O of a run lives here. Experiments 1 and 2 share `report.py` and `run.py` unchanged and differ only in the weights passed to `build_group_interface_model`.

`p` = modulus, `K = p // 2` frequencies.

| File | Job |
|---|---|
| [fly_model.py](fly_model.py) | Weight conditions (Experiment 1: `random_trainable_weights`) and `build_group_interface_model`, which samples the channel-group interface and assembles the `GroupInterfaceModel` on it. |
| [report.py](report.py) | Pure builders for the spectrum summary and figures (token-embedding Fourier norms, training curves), plus `write_report`, which does every write of the report. |
| [wandb_logging.py](wandb_logging.py) | Weights & Biases monitoring: start a run, log each history row, and at the end the figures, the spectra summary and the run files as an artifact. The only module importing `wandb` or `dotenv`. |
| [run.py](run.py) | `run_experiment`: create the run directory, save configs, train with `iter_epochs`, log a history row per epoch (with Gini), overwrite the checkpoint each epoch, then write the report. Experiment-agnostic. |

Tunables are in [`ExperimentConfig`, `TrainConfig`, `TaskConfig`, `EmbeddingConfig`, `InterfaceConfig`, `WandbConfig`](../README.md) in `config.py`.

## fly_model.py

| Signature | Returns | What it does |
|---|---|---|
| `random_trainable_weights(graph, scale)` | `SignedEdgeWeights` | Experiment 1's condition: one random trainable `theta` per connection, presynaptic sign kept, generator seeded with `scale.seed`, inhibitory thetas scaled by `scale.inhibitory_init_scale`. |
| `build_group_interface_model(graph, weights, dyn, emb, interface, seed)` | `GroupInterfaceModel` | Calls `torch.manual_seed(seed)` first (the `nn.Embedding` init uses the global RNG), then samples the interface with `sample_interface(graph, N_OPERANDS * emb.channels_per_token, interface.input_group_size, emb.modulus, interface.output_group_size, generator)` on a CPU generator seeded with `interface.seed`, so the interface stays fixed across training seeds. Builds `ConnectomeRNN(graph, weights, input_groups.flatten(), output_groups.flatten(), dyn)` and wraps it in `GroupInterfaceModel(rnn, emb, interface)`. On the real fly: 44 × 64 = 2,816 input and 59 × 32 = 1,888 output neurons. On the CPU; the caller moves it. |

## report.py

`rows` are history rows: the `EpochRecord` fields plus `GINI_COLUMNS`, one dict per epoch. The model has one token matrix (`embedding.weight [p, d]`); its group-mean readout has no weight matrix, so there is no unembedding spectrum.

| Signature | Returns | What it does |
|---|---|---|
| `GINI_COLUMNS` | `tuple[str]` | `("embedding_gini",)`, the extra history column. A tuple so `run.open_history` and `history_figures` iterate it. |
| `TABLE_NAMES` | `tuple[str]` | `("embedding",)`, the spectrum tables in the summary; `wandb_logging.log_summary` iterates it. |
| `gini_row(model, cfg)` | `dict[str, float]` | `{"embedding_gini": ...}`: Gini of the current embedding's Fourier norms, via `model_embedding_spectrum`. |
| `spectra_summary(model, cfg)` | `dict` | `{"embedding": {"gini": float, "dominant": list[int], "freq_norms": list[float]}}` (`K+1` norms, index 0 = constant). JSON-able. |
| `plot_history(epochs, series, ylabel, cfg)` | `Figure` | One line per entry of `series` over `epochs`, with a legend; x label "epoch", size `cfg.figure_size`. Object-oriented matplotlib, no pyplot. |
| `history_figures(rows, cfg)` | `dict[str, Figure]` | `"loss"` (train/test), `"accuracy"` (train/test), `"gini"` (embedding), `"grad_norm"`, `"theta_drift"` (`sum (theta - theta_init)^2`, the unweighted drift penalty). |
| `spectra_figures(model, cfg)` | `dict[str, Figure]` | `"embedding_freq_norms"`: a bar chart via `plot_frequency_norms`. |
| `write_report(run_dir, model, rows, icfg, ecfg)` | `None` | Writes `spectra_summary` to `run_dir / ecfg.report_name` (indent 2) and all six figures (five history, one spectrum) to `run_dir / ecfg.figure_dir_name / <name>.png` with `save_figure`. |
| `_spectrum_summary`, `_columns` | (private) | One spectrum as a JSON-able dict; pull named columns out of the rows. |

## run.py

`configs` is a `dict[str, object]` that must contain `"task"`, `"train"`, `"embedding"`, `"interpretability"` and `"experiment"`. Other entries (the script adds `"interface"`, `"data"`, `"dynamics"`, `"scale"`, `"wandb"`) are only saved.

| Signature | Returns | What it does |
|---|---|---|
| `write_config(run_dir, configs, ecfg)` | `None` | One JSON file `{name: asdict(cfg)}`; `Path`s and other non-JSON values become strings. |
| `open_history(run_dir, ecfg)` | `(file, csv.DictWriter)` | Opens the history CSV and writes the header (`EpochRecord` fields + `GINI_COLUMNS`) at once. The caller closes the file. |
| `log_epoch(record, model, icfg, file, writer)` | `dict` | `asdict(record) \| gini_row(model, icfg)`, written and flushed, so a killed run keeps its history. |
| `train_and_log(model, data, configs, run_dir, on_row)` | `list[dict]` | `make_optimizer`, a CPU generator seeded with `train.seed`, then for each `iter_epochs` record: log the row, `on_row(row)`, overwrite the checkpoint with `record.epoch`. |
| `run_experiment(model, configs, run_dir, on_row)` | `list[dict]` | `run_dir.mkdir(parents=True, exist_ok=False)` (an existing directory raises `FileExistsError`, no silent overwrite), `write_config`, `make_split(emb.modulus, task)` moved to the model's device, `train_and_log`, `write_report`. Returns the rows. |

## wandb_logging.py

| Signature | Returns | What it does |
|---|---|---|
| `read_env_value(path, name)` | `str` | `dotenv_values(path)[name]`; raises `KeyError` if the name is missing or its value empty. |
| `wandb_settings(cfg)` | `wandb.Settings` | `Settings(api_key=<key from cfg.env_file>, mode=cfg.mode)`. Nothing is written to netrc and no environment variable is set. |
| `json_safe_configs(configs)` | `dict` | `{name: asdict(cfg)}` with `Path`s etc. as strings, same as `config.json`. |
| `start_run(cfg, run_name, configs)` | `wandb.Run` | `wandb.init` in `cfg.entity`/`cfg.project`, named `run_name`, with every config as the run config. |
| `log_row(run, row)` | `None` | `run.log(row, step=row["epoch"])`. |
| `log_figures(run, figure_dir)` | `None` | Every PNG as `figures/<stem>` in one `run.log` without a step, so it lands on the last logged epoch (steps never go backwards). |
| `log_summary(run, summary)` | `None` | `<table>/gini` and `<table>/dominant` for each of `TABLE_NAMES` (now `embedding/gini`, `embedding/dominant`) from `spectra_summary` into `run.summary`. |
| `log_run_files(run, run_dir, cfg, ecfg)` | `None` | Artifact `cfg.artifact_name` (type `run`) with `config.json`, `history.csv`, `embedding_report.json` and `figures/`. |
| `finish_run(run, run_dir, summary, cfg, ecfg)` | `None` | `log_figures`, `log_summary`, `log_run_files`, then `run.finish()`. |

### W&B

The hook is `run_experiment`'s `on_row` callback; `run.py` and `report.py` know nothing about W&B. `scripts/run_experiment1.py` calls `start_run` before training, passes an `on_row` that prints the row and calls `log_row`, and calls `finish_run` after `run_experiment` returns.

- **Per epoch** (step = epoch): `train_loss`, `train_acc`, `test_loss`, `test_acc`, `grad_norm`, `theta_drift`, `seconds`, `embedding_gini`, `epoch`.
- **At the end**: the six figures as images (`figures/<name>`), the final Gini and dominant frequencies in the run summary, and the artifact `run-files`. `checkpoint.pt` (143 MB on the real fly, including the 11 MB `theta_init` buffer) stays local.
- **Key**: read from `WANDB_KEY` in the repo-root `.env` (gitignored), set by `WandbConfig.env_file`/`key_variable`. A missing or empty key raises `KeyError`, even in `disabled` mode.
- **Turn off**: `--wandb-mode disabled` (no network, nothing logged) or `--wandb-mode offline` (logs to `wandb/`, uploadable later with `wandb sync`).

## Run directory

```
<ExperimentConfig.runs_dir>/<run name>/
    config.json              {name: asdict(config)} for every config passed
    history.csv              epoch, train_loss, train_acc, test_loss, test_acc, grad_norm, theta_drift,
                             seconds, embedding_gini   (one row per epoch, flushed each epoch)
    checkpoint.pt            {"model", "optimizer", "epoch"} after the latest epoch (load_checkpoint)
    embedding_report.json    spectra_summary of the final model
    figures/
        loss.png  accuracy.png  gini.png  grad_norm.png  theta_drift.png
        embedding_freq_norms.png
```

`runs/` and `*.pt` are gitignored. Runs made before the channel-group interface keep the old schema (an `unembedding_gini` column and figure, no `"interface"` entry in `config.json`). Runs made before the drift penalty have no `theta_drift` column or figure.

## Putting it together

```python
from modularfly.config import (
    DataConfig, DynamicsConfig, EmbeddingConfig, ExperimentConfig, InterfaceConfig,
    InterpretabilityConfig, ScaleConfig, TaskConfig, TrainConfig,
)
from modularfly.data.graph import graph_cache_path, load_graph
from modularfly.experiments.fly_model import build_group_interface_model, random_trainable_weights
from modularfly.experiments.run import run_experiment

configs = {"task": TaskConfig(), "train": TrainConfig(n_epochs=5), "embedding": EmbeddingConfig(),
           "interface": InterfaceConfig(),
           "interpretability": InterpretabilityConfig(), "experiment": ExperimentConfig()}
data, dyn = DataConfig(), DynamicsConfig()
graph = load_graph(graph_cache_path(data.processed_dir, data.min_syn_count))
weights = random_trainable_weights(graph, ScaleConfig())
model = build_group_interface_model(graph, weights, dyn, configs["embedding"],
                                    configs["interface"], configs["train"].seed).to("cuda")
rows = run_experiment(model, configs, configs["experiment"].runs_dir / "demo", on_row=print)
```

`scripts/run_experiment1.py` does exactly this with CLI overrides, plus W&B logging (see [W&B](#wb)).

## Next features

- **Experiment 2**: `frozen_natural_weights(graph, scale, device) -> SynapseCountWeights` beside `random_trainable_weights` (w0 via `graph_spectral_radius` + `w0_for_radius`, as `scripts/build_graph.py::print_scale` does), plus `scripts/run_experiment2.py`, passed to `build_group_interface_model` unchanged. `make_optimizer` then has only the embedding group (only `embedding.weight` trains); nothing in `report.py` or `run.py` changes.
- **Per-neuron analysis of a finished run**: `load_checkpoint` + `build_group_interface_model` (same configs from `config.json`, including `"interface"`, so the same groups are resampled) + `interpretability.neurons`, in a `scripts/analyze_run.py`. It must tolerate old runs without `"interface"`, with `unembedding_gini`, or without `theta_drift`.
- **Other input codes** (e.g. the paper's 47-channel one-hot digit code): a new encoder in `model/io/` and a sibling of `GroupInterfaceModel` that replaces `channels()`; the builder would pass its channel count to `sample_interface`.
- **Group-size / temperature sweeps**: `InterfaceConfig` fields plus CLI flags in the script.
- **Resume**: `iter_epochs` gains a `start_epoch`; `run_experiment` would load the checkpoint instead of requiring a fresh `run_dir`, and open the history in append mode.
