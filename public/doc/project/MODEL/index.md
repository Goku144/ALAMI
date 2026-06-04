# MODEL Namespace

> **Reading Path**  
> Home: [Project Manual](../index.md) | Previous: [OPERATOR](../OPERATOR/index.md) | Next: [Usage](../usage.md)

`MODEL` is the model orchestration layer above the low-level runtime
primitives. It is where tensors and operators become a trainable graph.

The current implementation contains one concrete class:

| Class | File | Purpose |
|---|---|---|
| `MODEL::DL` | [dl.md](dl.md) | Concrete CNN-style digit classifier/trainer built from handlers and operators. |
| Layer math | [layers.md](layers.md) | Mathematical proof of the current CNN forward pass, backward pass, optimization, and result interpretation. |

Future generic abstractions such as reusable layers, sequential containers, and
optimizer wrappers can still be added later. For now, `MODEL::DL` wires the
runtime manually and documents the real model behavior.

## Current Responsibilities

- own runtime handlers
- load dataset metadata and images
- manage trainable parameters
- coordinate forward propagation
- coordinate backward propagation
- call optimizer/update operations through `OPERATOR::SGD`
- save and restore raw checkpoints
- run single-image inference
- run the final held-out test benchmark used by comparison tooling

## Relationship To APP

`MODEL::DL` owns the CUDA model behavior. The app entry point `app/src/dl.cu`
chooses the training schedule:

```cpp
MODEL::DL dl(64);
dl.train(60000, 0.0005f, 3000);
```

The comparison script does not inspect private model state. It reads the
printed confusion matrix from the DL log and compares that evidence with the ML
baseline.

---

> **Continue Reading**  
> [MODEL::DL](dl.md) | Previous: [OPERATOR](../OPERATOR/index.md) | Next: [APP](../APP/index.md)
