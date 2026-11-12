# modularfly/model

The recurrent network, the modules that supply its weights, and the token embeddings around it.

| Folder | Job |
|---|---|
| [fly/](fly/README.md) | The wiring: `ConnectomeRNN`, the sparse product `W·h`, and spectral-radius scale estimates. |
| [weights/](weights/README.md) | The pluggable weight modules. |
| [io/](io/README.md) | Factories for the token embedding and the output-to-logits layer. |
| [end_to_end/](end_to_end/README.md) | Complete tokens-to-logits models, one per input mechanism; new input mechanisms go there. |

`N` = neurons, `E` = connections, `B` = batch, `T` = time steps, `p` = modulus (number of tokens and of answers). A weight module is any `nn.Module` whose `forward()` takes no arguments and returns `FloatTensor[E]`, one weight per connection in the graph's order.
