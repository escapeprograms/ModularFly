import torch

from modularfly.config import TaskConfig
from modularfly.interpretability.neurons import all_pairs
from modularfly.training.task import labeled_pairs, make_split

P = 7


def rows(tokens):
    return {tuple(r) for r in tokens.tolist()}


def test_split_sizes_disjoint_and_complete():
    data = make_split(P, TaskConfig(0.3, 0))
    assert data.train_tokens.shape == (15, 2) and data.train_targets.shape == (15,)
    assert data.test_tokens.shape == (34, 2) and data.test_targets.shape == (34,)
    train, test = rows(data.train_tokens), rows(data.test_tokens)
    assert len(train) == 15 and len(test) == 34 and not train & test
    assert train | test == rows(all_pairs(P))


def test_targets_are_sum_mod_p():
    tokens, targets = labeled_pairs(P)
    assert torch.equal(targets, (tokens[:, 0] + tokens[:, 1]) % P)
    data = make_split(P, TaskConfig(0.3, 0))
    for toks, tgts in ((data.train_tokens, data.train_targets), (data.test_tokens, data.test_targets)):
        assert torch.equal(tgts, toks.sum(dim=1) % P)


def test_split_is_seeded():
    same_a, same_b = make_split(P, TaskConfig(0.3, 0)), make_split(P, TaskConfig(0.3, 0))
    other = make_split(P, TaskConfig(0.3, 1))
    assert torch.equal(same_a.train_tokens, same_b.train_tokens)
    assert torch.equal(same_a.test_tokens, same_b.test_tokens)
    assert not torch.equal(same_a.train_tokens, other.train_tokens)


def test_to_returns_new_split():
    data = make_split(P, TaskConfig(0.3, 0))
    moved = data.to("cpu")
    assert moved is not data and torch.equal(moved.test_targets, data.test_targets)
