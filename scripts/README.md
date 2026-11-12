# scripts

Command-line entry points. Run them from the repository root with the `modular-fly` environment's Python, in this order:

```
python scripts/download_flywire.py
python scripts/build_graph.py
python scripts/smoke_gpu.py
```

Settings come from the defaults in [modularfly/config.py](../modularfly/config.py).

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

Each script adds the repository root to the import path by hand; this is recorded as debt-001 in [debt.md](../debt.md).
