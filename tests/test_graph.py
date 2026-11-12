import pytest
import torch

from modularfly.data.graph import edge_keys, graph_cache_path, load_graph, save_graph


def test_edges_sorted_by_post_then_pre(small_graph):
    keys = edge_keys(small_graph.post, small_graph.pre, small_graph.n_neurons)
    assert bool((keys[1:] > keys[:-1]).all())


def test_duplicate_edges_rejected(small_graph):
    pre = torch.cat([small_graph.pre, small_graph.pre[:1]])
    post = torch.cat([small_graph.post, small_graph.post[:1]])
    count = torch.cat([small_graph.syn_count, small_graph.syn_count[:1]])
    with pytest.raises(ValueError):
        small_graph.with_edges(pre, post, count)


def test_in_degree(small_graph):
    expected = torch.zeros(small_graph.n_neurons, dtype=torch.long)
    for p in small_graph.post.tolist():
        expected[p] += 1
    assert torch.equal(small_graph.in_degree(), expected)


def test_save_load_roundtrip(small_graph, tmp_path):
    path = graph_cache_path(tmp_path, 5)
    assert "min5" in path.name
    save_graph(small_graph, path)
    loaded = load_graph(path)
    assert torch.equal(loaded.pre, small_graph.pre) and torch.equal(loaded.post, small_graph.post)
    assert loaded.vocab == small_graph.vocab
    assert torch.equal(loaded.annotations["cell_type"], small_graph.annotations["cell_type"])
