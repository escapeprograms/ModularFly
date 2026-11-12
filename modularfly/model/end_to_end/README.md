# modularfly/model/end_to_end

Complete tokens-to-logits models built from the recurrent network and the token layers, one per input mechanism. New input mechanisms go here.

| File | Job |
|---|---|
| [split_input.py](split_input.py) | `SplitInputModel`: each number's embedding drives a random half of the input neurons. |

## split_input.py

### `SplitInputModel(rnn, cfg, generator)`

Tokens `(a, b)` → logits over `p` answers. The input neurons are split once, at random, into two halves of `n_in // 2`; `a`'s embedding is the drive on the first half and `b`'s on the second. The embedding dimension is `n_in // 2`, so each entry drives one neuron. If `n_in` is odd, one input neuron stays undriven.

| Argument | Meaning |
|---|---|
| `rnn` | A `ConnectomeRNN` with at least 2 input neurons (otherwise `ValueError`). |
| `cfg` | An [`EmbeddingConfig`](../../README.md) (`modulus` = `p`, `init_std`). |
| `generator` | `torch.Generator` that fixes the random split. |

| Member | Returns | What it does |
|---|---|---|
| `forward(tokens)` | `[B, p]` | `unembedding(rnn(drive(tokens)))`. `tokens` is `LongTensor[B, 2]`. |
| `drive(tokens)` | `[B, n_in]` | Places the two embeddings on their halves; the leftover slot is zero. |
| `slots` (buffer) | `LongTensor[2, n_in // 2]` | Row 0 is `a`'s positions, row 1 is `b`'s. |
| `embedding`, `unembedding` | `nn.Embedding`, `nn.Linear` | The two layers from [io/embedding.py](../io/README.md). |

| Function | Returns | What it does |
|---|---|---|
| `random_halves(n, generator)` | `LongTensor[2, n // 2]` | Two disjoint random sets of positions in `0..n-1`. |

`slots` are positions in the drive vector, not neuron ids: the neurons `a` drives are `rnn.input_idx[model.slots[0]]`.

Experiment 2 (frozen natural weights) = pass [`SynapseCountWeights`](../weights/README.md) to the RNN; then only `embedding.weight` and `unembedding.weight` train.
