import math

import torch

from modularfly.config import TaskConfig, TrainConfig
from modularfly.model.weights.synapse_count import SynapseCountWeights
from modularfly.training.loop import (
    batch_loss, evaluate, iter_epochs, make_optimizer, train_step,
)
from modularfly.training.task import make_split
from tests.conftest import SMALL_P as P, make_model, trainable_fly

SMALL_TRAIN = TrainConfig(batch_size=8, n_epochs=2, eval_batch=16)


def seeded_model(graph):
    return make_model(graph, trainable_fly(graph))  # the builder seeds nn.Embedding init


def run_records(graph, cfg):
    model = seeded_model(graph)
    data = make_split(P, TaskConfig(0.3, 0))
    gen = torch.Generator().manual_seed(cfg.seed)
    return list(iter_epochs(model, make_optimizer(model, cfg), data, cfg, gen))


def test_fresh_model_logits_bounded_by_inverse_temperature(small_graph):
    model = seeded_model(small_graph)
    data = make_split(P, TaskConfig(0.3, 0))
    with torch.no_grad():
        logits = model(data.test_tokens)
    assert logits.abs().max() < 1 / model.cfg.temperature  # tanh states lie in (-1, 1)
    loss, acc = evaluate(model, data.test_tokens, data.test_targets, 16)
    assert math.isfinite(loss) and 0.0 <= acc <= 1.0


def test_train_steps_lower_fixed_batch_loss(small_graph):
    model = seeded_model(small_graph)
    optimizer = make_optimizer(model, TrainConfig(lr=0.1, fly_lr=0.1))
    tokens = torch.tensor([[0, 1], [2, 3], [4, 5], [6, 0], [3, 3]])
    targets = (tokens[:, 0] + tokens[:, 1]) % P
    before = batch_loss(model, tokens, targets)[0].item()
    for _ in range(5):
        train_step(model, optimizer, tokens, targets, clip_norm=1e3, penalty_weight=0.0)
    assert batch_loss(model, tokens, targets)[0].item() < before


def test_iter_epochs_yields_records(small_graph):
    cfg = TrainConfig(batch_size=8, n_epochs=3, eval_batch=16)
    records = run_records(small_graph, cfg)
    assert [r.epoch for r in records] == [0, 1, 2]
    for r in records:
        assert math.isfinite(r.grad_norm) and r.seconds > 0
        assert math.isfinite(r.train_loss) and math.isfinite(r.test_loss)


def test_same_seed_runs_give_equal_records(small_graph):
    first, second = run_records(small_graph, SMALL_TRAIN), run_records(small_graph, SMALL_TRAIN)
    strip = lambda rs: [(r.epoch, r.train_loss, r.train_acc, r.test_loss, r.test_acc, r.grad_norm,
                         r.theta_drift) for r in rs]
    assert strip(first) == strip(second)


def drift_after_step(graph, penalty_weight):
    """Seeded model with theta pushed 0.5 off theta_init (at init the penalty's gradient is 0),
    then one step at lr 0.1 on a fixed batch; returns the drift penalty after the step."""
    model = seeded_model(graph)
    with torch.no_grad():
        model.rnn.edge_weights.theta += 0.5
    optimizer = make_optimizer(model, TrainConfig(lr=0.1, fly_lr=0.1))
    tokens = torch.tensor([[0, 1], [2, 3], [4, 5], [6, 0], [3, 3]])
    train_step(model, optimizer, tokens, (tokens[:, 0] + tokens[:, 1]) % P,
               clip_norm=1e3, penalty_weight=penalty_weight)
    return model.rnn.edge_weights.drift_penalty().item()


def test_penalty_pulls_theta_toward_init(small_graph):
    assert drift_after_step(small_graph, 1e2) < drift_after_step(small_graph, 0.0)


def test_records_log_theta_drift(small_graph):
    trainable = run_records(small_graph, SMALL_TRAIN)
    assert trainable[0].theta_drift > 0 and math.isfinite(trainable[0].theta_drift)
    model = make_model(small_graph, SynapseCountWeights(small_graph, 0.01))
    data = make_split(P, TaskConfig(0.3, 0))
    gen = torch.Generator().manual_seed(SMALL_TRAIN.seed)
    frozen = list(iter_epochs(model, make_optimizer(model, SMALL_TRAIN), data, SMALL_TRAIN, gen))
    assert [r.theta_drift for r in frozen] == [0.0] * SMALL_TRAIN.n_epochs


def test_optimizer_groups(small_graph):
    model = make_model(small_graph, trainable_fly(small_graph))
    token_group, fly_group = make_optimizer(model, TrainConfig()).param_groups
    ids = lambda group: {id(p) for p in group["params"]}
    assert ids(token_group) == {id(model.embedding.weight)}
    assert ids(fly_group) == {id(model.rnn.edge_weights.theta)}
    assert token_group["weight_decay"] == 0.1 and fly_group["weight_decay"] == 0.0
    assert token_group["lr"] == 1e-3 and fly_group["lr"] == 1e-3


def test_fly_lr_sets_fly_group_only(small_graph):
    model = make_model(small_graph, trainable_fly(small_graph))
    token_group, fly_group = make_optimizer(model, TrainConfig(lr=1e-3, fly_lr=5e-3)).param_groups
    assert token_group["lr"] == 1e-3 and fly_group["lr"] == 5e-3


def test_frozen_fly_has_only_token_group(small_graph):
    model = make_model(small_graph, SynapseCountWeights(small_graph, 0.01))
    groups = make_optimizer(model, TrainConfig()).param_groups
    assert len(groups) == 1 and len(groups[0]["params"]) == 1


def test_evaluate_uneven_chunk_matches_single_chunk(small_graph):
    model = make_model(small_graph, trainable_fly(small_graph))
    data = make_split(P, TaskConfig(0.3, 0))  # 34 test rows: chunks of 16 leave 2
    chunked = evaluate(model, data.test_tokens, data.test_targets, 16)
    whole = evaluate(model, data.test_tokens, data.test_targets, 34)
    assert math.isclose(chunked[0], whole[0], rel_tol=1e-6) and chunked[1] == whole[1]
