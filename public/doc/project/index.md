# DL Project Manual

> **Reading Path**  
> Home: **Project Manual** | Previous: None | Next: [Architecture](architecture.md)

This manual explains the DL runtime from the top of the hierarchy down to each public class and function.
For the final `remise`, start with the [Explanatory Submission Plan](submission_plan.md);
it maps the same project to the Machine Learning and Practical IA / Deep
Learning grading criteria.

DL is organized as a small CUDA/C++ neural-network runtime plus an application
layer that proves the runtime against a Python machine-learning baseline. It is
not a general deep-learning framework. It is a controlled system: explicit
tensor metadata, explicit memory handlers, explicit GPU execution, a concrete
CNN model, and a comparison script that stores evidence.

## Source Hierarchy

```text
CORE/
  Shared constants, alignment, errors, and logging.

VIEW/
  Tensor shape and tensor view objects.

HANDLER/
  CPU/GPU memory arenas, tensor binding, file IO, and execution workspace.

OPERATOR/
  GPU operators that consume and produce VIEW::Math tensors.

MODEL/
  Current model orchestration, including MODEL::DL.

APP/
  Runnable experiments: CUDA DL, Python ML baseline, and comparison.
```

## Documentation Map

| Order | Document | What It Explains |
|---:|---|---|
| 1 | [Architecture](architecture.md) | How all classes are orchestrated together. |
| 2 | [CORE](CORE/index.md) | Shared project definitions and error vocabulary. |
| 3 | [VIEW](VIEW/index.md) | Shape and tensor view objects. |
| 4 | [HANDLER](HANDLER/index.md) | Memory, data movement, files, and workspace classes. |
| 5 | [OPERATOR](OPERATOR/index.md) | Every operator class, operand contract, and function. |
| 6 | [MODEL](MODEL/index.md) | Current model orchestration and future model/layer direction. |
| 7 | [APP](APP/index.md) | DL app, ML baseline, and comparison pipeline. |
| 8 | [Dependencies](dependencies.md) | Required CUDA, Python, and dataset dependencies. |
| 9 | [Proof And Video](proof_video.md) | How to present and verify the project result. |
| 10 | [Submission Plan](submission_plan.md) | Grading-oriented guide for the final report, video, and defense. |
| 11 | [Usage](usage.md) | Minimal practical usage recipe. |
| 12 | [Full Overview](overview.md) | Long-form narrative overview. |
| 13 | [Conclusion](conclusion.md) | Mathematical and conceptual interpretation of the final result. |

## Most Important Rule

Every operator follows the same life cycle:

```text
create handlers
create VIEW::Math tensors
set tensor shapes
bind memory
fill input CPU memory
copy inputs to GPU
attach operands to operator
run operation
copy outputs back if needed
```

Outputs must already exist, have a shape, and be bound before an operator writes to them.

## Proof Workflow

The current end-to-end proof command is:

```bash
make compaire
```

It runs the Python baseline and CUDA DL app, stores raw logs, extracts confusion
matrices, writes a comparison report, and saves a chart under
`public/checkpoints`.

---

> **Continue Reading**  
> Next: [Architecture](architecture.md)
