# modularfly/model/fly

The recurrent network, the sparse product it runs on, and the scale estimates for its weights.

`N` = neurons, `E` = connections, `B` = batch, `T` = time steps.

| File | Job |
|---|---|
| [connectome.py](connectome.py) | `ConnectomeRNN`: the weight-agnostic recurrent network. |
| [propagate.py](propagate.py) | The sparse product `W·h`, with memory-safe gradients. |
| [scaling.py](scaling.py) | Spectral-radius estimate and the scales derived from it. |

## connectome.py

### `ConnectomeRNN(graph, edge_weights, input_idx, output_idx, cfg)`

Runs `h ← (1 − alpha)·h + alpha·tanh(u + W·h + b)` for `T` steps from `h = 0`. The drive `u` is held constant on the input neurons and is zero elsewhere. `b` is a zero bias buffer of shape `[N, 1]`.

| Argument | Meaning |
|---|---|
| `graph` | A `ConnectomeGraph`. |
| `edge_weights` | A weight module (see the [weights/](../weights/README.md) files). |
| `input_idx`, `output_idx` | Neuron indices to drive and to read, usually from `select_neurons`. |
| `cfg` | A `DynamicsConfig` (`alpha`, `n_steps`, `readout_steps`). |

| Method | Returns | What it does |
|---|---|---|
| `forward(drive)` | `[B, n_out]` | Takes `drive` of shape `[B, n_in]`; returns output-neuron activity averaged over the last `readout_steps` steps. |
| `window_mean(drive, neuron_idx)` | `[n, B]` | Activity of the neurons in `neuron_idx` averaged over the last `readout_steps` steps (neurons first). `forward` is this with `output_idx`, transposed. |
| `iter_states(drive)` | generator of `[N, B]` | Yields the state after each of the `T` steps. Use it to inspect hidden activity. |
| `step(h, drive, weights)` | `[N, B]` | One update. `h` and `drive` are `[N, B]`; `weights` is `[E]`. |
| `expand_drive(drive)` | `[N, B]` | Places `[B, n_in]` drive onto the input neurons, zeros elsewhere. |

Internal states are `[N, B]` (neurons first); only `forward` uses batch-first shapes.

## propagate.py

PyTorch's built-in sparse product tries to build a dense `N × N` gradient (72 GB here) when the weights are trainable. This file computes the same product with gradients that fit in memory.

| Signature | Returns | What it does |
|---|---|---|
| `SparsePropagator(graph, chunk_edges)` | module | Stores the graph's sparse indices. `forward(weights, h)` returns `W·h` of shape `[N, B]`, where `W[post, pre] = weights`. |
| `SparsePropagate` | autograd function | The product with hand-written gradients, used by `SparsePropagator`. The weight gradient is skipped when weights are frozen. |
| `csr_row_pointers(rows, n_rows)` | `LongTensor[n_rows + 1]` | Row start offsets for a sorted row-index list. |
| `csr_matmul(crow, col, values, h)` | `[n_rows, B]` | Sparse matrix times dense matrix. |
| `edge_weight_grad(grad_out, h, pre, post, chunk_edges)` | `FloatTensor[E]` | Gradient for each connection weight, computed `chunk_edges` connections at a time. |

## scaling.py

| Signature | Returns | What it does |
|---|---|---|
| `spectral_radius(matvec, n, n_iter, generator, device=None, dtype=float32)` | `float` | Estimates the largest eigenvalue magnitude of a matrix given only a function that multiplies by it. Works when that eigenvalue is complex. |
| `graph_spectral_radius(graph, weights, n_iter, generator, device=None)` | `float` | The same estimate for the graph's weight matrix. |
| `w0_for_radius(radius, target_radius)` | `float` | `target_radius / radius`: the global scale that gives the natural weights the target radius. |
| `signed_count_weights(graph)` | `FloatTensor[E]` | `sign(pre) · syn_count`, the unscaled natural weights. |
| `ei_balance_scale(graph)` | `float` | Excitatory connections divided by inhibitory connections (1.57 here): the `inhibitory_init_scale` that balances the two. |

On the real graph the spectral radius of `sign · syn_count` is 2012, giving `w0 = 4.97e-4` for a target of 1.0.
