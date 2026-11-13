"""Minibatch training and evaluation of a modular-addition model.

The model is any callable tokens LongTensor[B, 2] -> logits [B, p]; make_optimizer also
needs its `.embedding` and `.rnn` submodules. Pure torch: no file I/O and
no interpretability imports. iter_epochs is a generator, so the caller owns logging,
extra per-epoch measurements and checkpointing.

The training loss is cross-entropy plus cfg.fly_penalty * the fly weights' drift_penalty()
(sum of squared deviation from the initial parameters; see regularized_loss). The logged
train_loss is cross-entropy only, so it stays comparable with test_loss; theta_drift logs
the unweighted penalty at the end of each epoch.

Shapes: B = batch, p = modulus, n = pairs in a split.
"""
import time
from collections.abc import Iterator
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from modularfly.config import TrainConfig
from modularfly.training.task import SplitData


@dataclass(frozen=True)
class EpochRecord:
    epoch: int  # 0-based
    train_loss: float  # running mean over the epoch's minibatches (no second pass)
    train_acc: float
    test_loss: float
    test_acc: float
    grad_norm: float  # mean pre-clip gradient norm over the epoch's steps
    theta_drift: float  # sum (theta - theta_init)^2 at the end of the epoch; 0 for a frozen fly
    seconds: float  # wall time of train + eval (time.perf_counter, no I/O)


def param_group(module: nn.Module, lr: float, weight_decay: float) -> dict:
    """One AdamW parameter group: the module's parameters at their own lr and weight decay."""
    return {"params": list(module.parameters()), "lr": lr, "weight_decay": weight_decay}


def make_optimizer(model: nn.Module, cfg: TrainConfig) -> torch.optim.AdamW:
    """AdamW with two parameter groups: the token embedding (model.embedding) at cfg.lr and
    cfg.weight_decay, and the recurrent weights (model.rnn) at cfg.fly_lr and
    cfg.fly_weight_decay. A group with no parameters (frozen fly weights) is left out."""
    groups = [param_group(model.embedding, cfg.lr, cfg.weight_decay),
              param_group(model.rnn, cfg.fly_lr, cfg.fly_weight_decay)]
    return torch.optim.AdamW([g for g in groups if g["params"]], betas=cfg.betas)


def batch_loss(model: nn.Module, tokens: torch.Tensor,
               targets: torch.Tensor) -> tuple[torch.Tensor, int]:
    """(mean cross-entropy, a scalar tensor with grad; number of argmax-correct rows)."""
    logits = model(tokens)
    n_correct = int((logits.argmax(dim=1) == targets).sum())
    return F.cross_entropy(logits, targets), n_correct


def regularized_loss(model: nn.Module, tokens: torch.Tensor, targets: torch.Tensor,
                     penalty_weight: float) -> tuple[torch.Tensor, torch.Tensor, int]:
    """(cross-entropy + penalty_weight * drift penalty, cross-entropy, n_correct)."""
    ce, n_correct = batch_loss(model, tokens, targets)
    return ce + penalty_weight * model.rnn.edge_weights.drift_penalty(), ce, n_correct


@torch.no_grad()
def epoch_drift(model: nn.Module) -> float:
    """The fly weights' unweighted drift_penalty() as a float."""
    return float(model.rnn.edge_weights.drift_penalty())


def optimizer_params(optimizer: torch.optim.Optimizer) -> list[torch.Tensor]:
    """Every parameter the optimizer updates, across its groups."""
    return [p for group in optimizer.param_groups for p in group["params"]]


def train_step(model: nn.Module, optimizer: torch.optim.Optimizer, tokens: torch.Tensor,
               targets: torch.Tensor, clip_norm: float,
               penalty_weight: float) -> tuple[float, int, float]:
    """One update on the regularized loss: (cross-entropy, n_correct, pre-clip gradient
    norm over all optimizer params, including the penalty's gradient)."""
    optimizer.zero_grad(set_to_none=True)
    total, ce, n_correct = regularized_loss(model, tokens, targets, penalty_weight)
    total.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(optimizer_params(optimizer), clip_norm)
    optimizer.step()
    return ce.item(), n_correct, grad_norm.item()


def train_epoch(model: nn.Module, optimizer: torch.optim.Optimizer, tokens: torch.Tensor,
                targets: torch.Tensor, cfg: TrainConfig,
                generator: torch.Generator) -> tuple[float, float, float]:
    """One shuffled pass in cfg.batch_size minibatches: (mean cross-entropy, accuracy, mean
    grad norm).
    The order comes from torch.randperm on the CPU generator, then moves to tokens.device."""
    n = tokens.shape[0]
    order = torch.randperm(n, generator=generator).to(tokens.device)
    loss_sum, correct, norm_sum, n_steps = 0.0, 0, 0.0, 0
    for start in range(0, n, cfg.batch_size):
        idx = order[start:start + cfg.batch_size]
        loss, n_correct, grad_norm = train_step(
            model, optimizer, tokens[idx], targets[idx], cfg.grad_clip_norm, cfg.fly_penalty)
        loss_sum += loss * idx.numel()
        correct += n_correct
        norm_sum += grad_norm
        n_steps += 1
    return loss_sum / n, correct / n, norm_sum / n_steps


@torch.no_grad()
def evaluate(model: nn.Module, tokens: torch.Tensor, targets: torch.Tensor,
             batch: int) -> tuple[float, float]:
    """(mean cross-entropy, accuracy) over all rows, in chunks of `batch`. Losses are
    summed per chunk and divided by n at the end, so an uneven last chunk is exact."""
    n = tokens.shape[0]
    loss_sum, correct = 0.0, 0
    for start in range(0, n, batch):
        logits = model(tokens[start:start + batch])
        chunk_targets = targets[start:start + batch]
        loss_sum += F.cross_entropy(logits, chunk_targets, reduction="sum").item()
        correct += int((logits.argmax(dim=1) == chunk_targets).sum())
    return loss_sum / n, correct / n


def iter_epochs(model: nn.Module, optimizer: torch.optim.Optimizer, data: SplitData,
                cfg: TrainConfig, generator: torch.Generator) -> Iterator[EpochRecord]:
    """Trains for cfg.n_epochs, yielding one EpochRecord after each epoch's train + test eval."""
    for epoch in range(cfg.n_epochs):
        start = time.perf_counter()
        train_loss, train_acc, grad_norm = train_epoch(
            model, optimizer, data.train_tokens, data.train_targets, cfg, generator)
        test_loss, test_acc = evaluate(model, data.test_tokens, data.test_targets, cfg.eval_batch)
        yield EpochRecord(epoch, train_loss, train_acc, test_loss, test_acc, grad_norm,
                          epoch_drift(model), time.perf_counter() - start)
