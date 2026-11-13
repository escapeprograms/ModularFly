"""Modular-addition task data: every (a, b) pair labeled (a + b) % p, and a seeded
train/test split of those pairs.

Shapes: p = modulus, n_train = round(train_fraction * p^2), n_test = p^2 - n_train.
Pure tensors on the CPU; move a split to the model's device with SplitData.to.
"""
from dataclasses import dataclass

import torch

from modularfly.config import TaskConfig
from modularfly.interpretability.neurons import all_pairs


@dataclass(frozen=True)
class SplitData:
    train_tokens: torch.Tensor  # LongTensor[n_train, 2]
    train_targets: torch.Tensor  # LongTensor[n_train]
    test_tokens: torch.Tensor  # LongTensor[n_test, 2]
    test_targets: torch.Tensor  # LongTensor[n_test]

    def to(self, device) -> "SplitData":
        """A new SplitData with every tensor on `device`."""
        return SplitData(
            self.train_tokens.to(device), self.train_targets.to(device),
            self.test_tokens.to(device), self.test_targets.to(device),
        )


def labeled_pairs(p: int) -> tuple[torch.Tensor, torch.Tensor]:
    """(tokens LongTensor[p*p, 2] in all_pairs order, targets LongTensor[p*p] = (a + b) % p)."""
    tokens = all_pairs(p)
    return tokens, tokens.sum(dim=1) % p


def make_split(p: int, cfg: TaskConfig) -> SplitData:
    """Disjoint train/test split of all p*p pairs by a permutation seeded with cfg.seed.
    The first round(cfg.train_fraction * p*p) permuted pairs are the training set."""
    tokens, targets = labeled_pairs(p)
    order = torch.randperm(p * p, generator=torch.Generator().manual_seed(cfg.seed))
    n_train = round(cfg.train_fraction * p * p)
    train, test = order[:n_train], order[n_train:]
    return SplitData(tokens[train], targets[train], tokens[test], targets[test])
