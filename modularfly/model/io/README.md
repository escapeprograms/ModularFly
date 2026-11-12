# modularfly/model/io

Factories for the layers that turn tokens into input drive and output activity into logits.

`B` = batch, `p` = modulus (number of tokens and of answers).

| File | Job |
|---|---|
| [embedding.py](embedding.py) | Factories for the token embedding and the output-to-logits layer. |

## embedding.py

| Signature | Returns | What it does |
|---|---|---|
| `input_embedding(n_tokens, dim, init_std)` | `nn.Embedding` | One table `[n_tokens, dim]` shared by both numbers, entries `~ N(0, init_std²)`. Maps `LongTensor[B, 2]` to `[B, 2, dim]`. |
| `unembedding(n_out, n_tokens)` | `nn.Linear` | Output-neuron readout `[B, n_out]` to logits `[B, n_tokens]`, no bias. |
