import torch

from modularfly.config import TrainConfig
from modularfly.experiments.fly_model import build_group_interface_model
from modularfly.model.weights.signed_edge import SignedEdgeWeights
from modularfly.training.checkpoint import load_checkpoint, save_checkpoint
from modularfly.training.loop import make_optimizer, train_step
from tests.conftest import SMALL_DYN, SMALL_EMB, SMALL_INTERFACE, SMALL_P as P


def build(graph, seed):
    weights = SignedEdgeWeights(graph, 1.0, torch.Generator().manual_seed(seed))
    return build_group_interface_model(graph, weights, SMALL_DYN, SMALL_EMB, SMALL_INTERFACE, seed)


def test_roundtrip_restores_model_and_epoch(small_graph, tmp_path):
    model = build(small_graph, 0)
    optimizer = make_optimizer(model, TrainConfig())
    tokens = torch.tensor([[0, 1], [2, 3], [4, 5]])
    train_step(model, optimizer, tokens, tokens.sum(dim=1) % P, clip_norm=1e3,
               penalty_weight=TrainConfig().fly_penalty)  # fills AdamW state
    path = save_checkpoint(tmp_path / "sub" / "checkpoint.pt", model, optimizer, epoch=3)
    assert path.exists()

    state = load_checkpoint(path)  # weights_only=True, optimizer state included
    assert state["epoch"] == 3
    other = build(small_graph, 1)
    other.load_state_dict(state["model"])
    for name, tensor in model.state_dict().items():
        assert torch.equal(other.state_dict()[name], tensor), name
    with torch.no_grad():
        assert torch.equal(other(tokens), model(tokens))
    other_opt = make_optimizer(other, TrainConfig())
    other_opt.load_state_dict(state["optimizer"])
    assert other_opt.state_dict()["state"].keys() == optimizer.state_dict()["state"].keys()
