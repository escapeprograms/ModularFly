# modularfly/training

Training on modular addition `(a + b) mod p`: the task data, the minibatch training and evaluation loop, and checkpoints. The package depends on torch and `config.py` only (plus `all_pairs` from `interpretability/neurons.py` for the pair order); `loop.py` imports nothing from `interpretability`. Experiments 1 and 2 share it unchanged.

`p` = modulus, `B` = batch, `n` = pairs in a split.

| File | Job |
|---|---|
| [task.py](task.py) | All `p²` labeled pairs and a seeded, disjoint train/test split. Pure tensors. |
| [loop.py](loop.py) | AdamW with two parameter groups (own learning rate and weight decay each), the regularized loss (cross-entropy plus drift penalty), one training step, one epoch, chunked evaluation, and `iter_epochs`, a generator of per-epoch records. Pure torch, no I/O. |
| [checkpoint.py](checkpoint.py) | Save/load the model and optimizer state dicts plus the epoch. The only file I/O in the package. |

Tunables are in [`TaskConfig` and `TrainConfig`](../README.md) in `config.py`.

## task.py

| Signature | Returns | What it does |
|---|---|---|
| `SplitData` | frozen dataclass | Fields: `train_tokens LongTensor[n_train, 2]`, `train_targets LongTensor[n_train]`, `test_tokens LongTensor[n_test, 2]`, `test_targets LongTensor[n_test]`. |
| `SplitData.to(device)` | `SplitData` | A new split with every tensor moved to `device`. |
| `labeled_pairs(p)` | `(LongTensor[p*p, 2], LongTensor[p*p])` | Every `(a, b)` in `all_pairs` order (row `a*p + b`) and its target `(a + b) % p`. |
| `make_split(p, cfg)` | `SplitData` | Permutes the `p²` pairs with `torch.randperm` seeded by `cfg.seed`; the first `round(cfg.train_fraction · p²)` are training pairs, the rest test pairs. On the CPU. |

## loop.py

The model is any callable `tokens [B, 2] -> logits [B, p]`. `make_optimizer` also needs its `.embedding` and `.rnn` submodules (as on `GroupInterfaceModel`).

| Signature | Returns | What it does |
|---|---|---|
| `EpochRecord` | frozen dataclass | Fields: `epoch` (0-based), `train_loss`, `train_acc` (running means over the epoch's minibatches, no second pass), `test_loss`, `test_acc`, `grad_norm` (mean pre-clip gradient norm over the epoch's steps), `theta_drift` (unweighted `drift_penalty()` of the fly weights at the end of the epoch; 0 for a frozen fly), `seconds` (wall time of train + eval, `time.perf_counter`). |
| `make_optimizer(model, cfg)` | `AdamW` | Group 1: `model.embedding` parameters (the token table) at `lr=cfg.lr`, `weight_decay=cfg.weight_decay`. Group 2: `model.rnn` parameters at `lr=cfg.fly_lr`, `weight_decay=cfg.fly_weight_decay`. A group with no parameters is left out, so frozen fly weights give a single group. `betas=cfg.betas`. |
| `param_group(module, lr, weight_decay)` | `dict` | One AdamW group: `{"params": list(module.parameters()), "lr": lr, "weight_decay": weight_decay}`. |
| `batch_loss(model, tokens, targets)` | `(Tensor, int)` | Mean cross-entropy (with grad) and the number of rows whose argmax equals the target. |
| `regularized_loss(model, tokens, targets, penalty_weight)` | `(Tensor, Tensor, int)` | `(ce + penalty_weight · model.rnn.edge_weights.drift_penalty(), ce, n_correct)`, using `batch_loss` for `ce`. |
| `epoch_drift(model)` | `float` | Under `no_grad`: `model.rnn.edge_weights.drift_penalty()` as a float (unweighted). |
| `optimizer_params(optimizer)` | `list[Tensor]` | Every parameter across the optimizer's groups. |
| `train_step(model, optimizer, tokens, targets, clip_norm, penalty_weight)` | `(float, int, float)` | `zero_grad(set_to_none=True)`, backward on the regularized loss, `clip_grad_norm_` over all optimizer parameters, step. Returns the cross-entropy (without the penalty), correct count and the pre-clip gradient norm (which includes the penalty's gradient). |
| `train_epoch(model, optimizer, tokens, targets, cfg, generator)` | `(float, float, float)` | One pass in `cfg.batch_size` minibatches, ordered by `torch.randperm(n, generator)` on the CPU (moved to the tokens' device). Passes `cfg.grad_clip_norm` and `cfg.fly_penalty` to `train_step`. Returns the batch-size-weighted mean cross-entropy, accuracy and mean grad norm. |
| `evaluate(model, tokens, targets, batch)` | `(float, float)` | Under `no_grad`, in chunks of `batch`: summed cross-entropy divided by `n` (exact with an uneven last chunk), and accuracy. |
| `iter_epochs(model, optimizer, data, cfg, generator)` | generator of `EpochRecord` | `cfg.n_epochs` times: `train_epoch` on the training split, `evaluate` on the test split with `cfg.eval_batch`, `epoch_drift`, then yields the record. The caller logs, measures and checkpoints between epochs. |

## checkpoint.py

| Signature | Returns | What it does |
|---|---|---|
| `save_checkpoint(path, model, optimizer, epoch)` | `Path` | Writes `{"model": state_dict, "optimizer": state_dict, "epoch": int}` with `torch.save`, creating parent directories and overwriting an existing file. |
| `load_checkpoint(path)` | `dict` | `torch.load(path, weights_only=True, map_location="cpu")`. The AdamW state dict loads under `weights_only`. |

Configs are not in the checkpoint (`Path` fields fail `weights_only`); they belong in a separate JSON file. The model state dict includes the graph index buffers and `theta_init`, 143 MB on the real fly (measured).

## Definitions

**Two parameter groups.** Each group has its own learning rate and weight decay: the token embedding uses `lr` and `weight_decay`, the fly weights `fly_lr` and `fly_weight_decay` (both learning rates default to 1e-3). Adam moves each `theta` by at most about `fly_lr` per step, so `fly_lr` sets how far the fly can travel in a run; 3e-3..1e-2 is worth trying. The token embedding gets weight decay `weight_decay = 0.1`. Nanda et al. 2023 use 1.0 to drive the generalizing (grokking) solution, but AdamW shrinks the table by `lr · wd` per step, and at 1.0 that decays the `[59, 22]` table about 5x over a 50-epoch run. `GroupInterfaceModel` has no learned readout (logits are fixed group means), so the embedding is the only token-side parameter. The recurrent fly weights get their own decay, `fly_weight_decay = 0.0` by default: AdamW's decoupled decay multiplies every parameter by `1 − lr · wd` each step, so at `wd = 1.0` each `theta` would shrink by 0.999 per step and the dynamics would go silent. Frozen weights (`SynapseCountWeights`) have no parameters, so their group is simply left out (AdamW is not given an empty group).

**Drift penalty.** Instead of decaying the fly weights toward zero, the training loss adds `fly_penalty · sum((theta − theta_init)²)`, an L2 pull toward the initial weights (each weights module's `drift_penalty()`; see [model/weights](../model/weights/README.md)). The sum form (not a mean over edges) makes the per-edge gradient `2 · fly_penalty · (theta − theta_init)`, which Adam compares with the per-edge task gradient, about `grad_norm / sqrt(E)` (roughly 1e-3 on the real fly early in training). The penalty is off by default (`fly_penalty = 0`). Measured on the real fly: at `fly_penalty = 1e-3` it pins `theta`, with `theta_drift` plateauing near 90 over 2.7M edges (RMS 0.006 per edge), and the model did not learn. The plateau scales about as 1/`fly_penalty`, so to allow RMS drift of order 0.1 per edge sweep 1e-6..1e-5. `train_loss` stays cross-entropy only, so it is comparable with `test_loss`. `theta_drift` logs the unweighted sum at the end of each epoch (the loss term is `fly_penalty · theta_drift`); the RMS drift per edge is `sqrt(theta_drift / E)`. Set `fly_penalty > 0` (CLI `--fly-penalty`) to turn it on.

**Gradient clipping.** `clip_grad_norm_` runs on every step with `cfg.grad_clip_norm` (1.0 by default, as in arXiv 2610.10014), and the pre-clip norm it returns is what `grad_norm` logs. There is no on/off flag.

**Reproducibility.** Batch order comes from the CPU generator passed to `iter_epochs`, so two runs with the same model init, split and generator seed give equal records on the CPU. `torch.use_deterministic_algorithms` is not enabled (CUDA `index_copy` would raise).

## Putting it together

Build `model` (a `GroupInterfaceModel`) with `build_group_interface_model` as in [modularfly/README.md](../README.md) (it calls `torch.manual_seed(train_cfg.seed)` itself), then:

```python
from pathlib import Path

import torch
from modularfly.config import EmbeddingConfig, TaskConfig, TrainConfig
from modularfly.training.checkpoint import load_checkpoint, save_checkpoint
from modularfly.training.loop import iter_epochs, make_optimizer
from modularfly.training.task import make_split

cfg = TrainConfig()
data = make_split(EmbeddingConfig().modulus, TaskConfig()).to("cuda")
optimizer = make_optimizer(model, cfg)
generator = torch.Generator().manual_seed(cfg.seed)
for record in iter_epochs(model, optimizer, data, cfg, generator):
    print(record)                                   # EpochRecord(epoch=0, train_loss=..., ...)
    save_checkpoint(Path("runs/demo/checkpoint.pt"), model, optimizer, record.epoch)

state = load_checkpoint(Path("runs/demo/checkpoint.pt"))
model.load_state_dict(state["model"]); optimizer.load_state_dict(state["optimizer"])
```

## Next features

- **Resume**: `iter_epochs` gains a `start_epoch`; the checkpoint already holds the optimizer state.
- **Experiment composition** (build → train → analyze → write) lives in `modularfly/experiments/`, which consumes `iter_epochs`.
