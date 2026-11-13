import torch

from modularfly.model.io.embedding import input_embedding


def test_input_embedding_shares_one_table():
    embedding = input_embedding(5, 3, 1.0)
    out = embedding(torch.tensor([[2, 2], [1, 4]]))
    assert out.shape == (2, 2, 3)
    assert torch.equal(out[0, 0], out[0, 1])
    assert torch.equal(out[1, 0], embedding.weight[1])


def test_input_embedding_init_std():
    std = input_embedding(1000, 64, 0.1).weight.std().item()
    assert abs(std - 0.1) < 0.01

