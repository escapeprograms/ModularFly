# modularfly/model

The recurrent network, the modules that supply its weights, and the token interface around it.

| Folder | Job |
|---|---|
| [fly/](fly/README.md) | The wiring: `ConnectomeRNN`, the sparse product `W·h`, and spectral-radius scale estimates. |
| [weights/](weights/README.md) | The pluggable weight modules. |
| [io/](io/README.md) | The token-embedding factory and the channel-group interface functions (sample input/output neuron groups, copy channels onto groups, group-mean logits). |
| [end_to_end/](end_to_end/README.md) | Complete tokens-to-logits models, one per input mechanism (now `GroupInterfaceModel`); new input mechanisms go there. |

`N` = neurons, `E` = connections, `B` = batch, `T` = time steps, `p` = modulus (number of tokens and of answers). A weight module is any `nn.Module` whose `forward()` takes no arguments and returns `FloatTensor[E]`, one weight per connection in the graph's order, and whose `drift_penalty()` returns the 0-dim sum of squared deviation of its trainable parameters from their initial values (zero when frozen).
