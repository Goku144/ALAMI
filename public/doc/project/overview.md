# Full Project Overview

> **Reading Path**  
> Home: [Project Manual](index.md) | Previous: [Usage](usage.md) | Next: Finished

DL is a compact CUDA/C++ runtime for building deep-learning pieces manually,
plus a small application layer that proves the CUDA model against a Python
baseline.

For final submission, the project should be presented through the
[Explanatory Submission Plan](submission_plan.md). That guide separates the
same work into the Machine Learning grade and the Practical IA / Deep Learning
grade.

The code is split by responsibility. That is the central design choice. The
project does not hide memory, shape, data movement, or model wiring behind a
large framework. It makes those choices visible.

## The Hierarchy

```text
CORE
  vocabulary and rules

VIEW
  tensor descriptions and tensor handles

HANDLER
  memory ownership, file IO, and execution resources

OPERATOR
  GPU math operations

MODEL
  concrete CUDA CNN orchestration plus future abstraction layer

APP
  runnable experiments, baseline, comparison, and proof artifacts
```

## Why This Shape

The project keeps memory explicit. That makes it easier to see what is on CPU,
what is on GPU, and when copies happen.

The central object is:

```cpp
VIEW::Math
```

Every operator reads or writes `VIEW::Math`.

`VIEW::Math` does not allocate. It is connected to memory through:

```cpp
HANDLER::IO
```

Operators share execution state through:

```cpp
HANDLER::Workspace
```

The application layer then turns the runtime into evidence:

```text
MODEL::DL
  -> CUDA CNN training and test benchmark

ml.py
  -> Random Forest baseline

comparaison.py
  -> saved logs, parsed matrices, report, and chart
```

## CORE In One Sentence

`CORE` defines constants, memory sizes, error enums, alignment helpers, and logging.

It should not depend on the rest of the project.

## VIEW In One Sentence

`VIEW` defines tensor metadata and tensor references.

`Shape` says what the tensor is.

`Math` says where the tensor data lives.

## HANDLER In One Sentence

`HANDLER` owns resources.

It owns the memory arenas, the copy logic, file/data loading, and GPU library handles.

## OPERATOR In One Sentence

`OPERATOR` performs math on tensors already prepared by `VIEW` and `HANDLER`.

It assumes:

- inputs contain valid GPU data
- outputs are already shaped and bound
- workspace has valid CUDA/cuDNN/cuBLASLt resources

## MODEL In One Sentence

`MODEL` currently contains `MODEL::DL`, which wraps the lower-level pieces into
a concrete trainable digit-classifier pipeline with dataset loading,
forward/backward propagation, SGD updates, checkpointing, and inference.

## APP In One Sentence

`APP` contains the runnable proof layer: `dl.cu` trains and benchmarks the CUDA
CNN, `ml.py` trains or loads the Random Forest baseline, and `comparaison.py`
compares both systems from saved confusion matrices.

## Function Declaration vs Implementation

Headers:

```text
public/inc
```

Implementations:

```text
lib/src
```

Example:

```text
public/inc/OPERATOR/Relu.hpp
lib/src/OPERATOR/Relu.cu
```

The header tells you what the class exposes. The source file tells you how it works.

Application entry points live in:

```text
app/src/dl.cu
app/src/ml.py
app/src/comparaison.py
```

## The Most Common Mistake

Do not bind before shape:

```cpp
// wrong
io.bind(x);
x.getLayout().setShape(dims, 1, VIEW::F16);
```

Correct:

```cpp
x.getLayout().setShape(dims, 1, VIEW::F16);
io.bind(x);
```

Binding calculates memory size from the current shape.

## Current Operators

| Operator | Main Use |
|---|---|
| `Normalize` | scale input values |
| `Relu` | activation |
| `SGD` | in-place parameter update |
| `Softmax` | logits to probabilities |
| `CrossEntropy` | loss and probability gradient |
| `Pool` | max-pooling |
| `Conv2DRelu` | currently convolution plus bias |
| `MatrixMulBias` | linear layer math |

## DL Model Concept

The CUDA model follows a compact CNN structure:

```text
image
  -> Normalize
  -> Conv2D + bias
  -> ReLU
  -> MaxPool
  -> Dense
  -> ReLU
  -> Dense
  -> Softmax
  -> CrossEntropy
  -> SGD
```

Each stage has a role:

- normalization turns raw pixel values into a stable numeric range
- convolution learns local visual patterns such as strokes and corners
- ReLU keeps positive evidence and removes negative activation noise
- max pooling reduces spatial size while keeping strong local signals
- dense layers combine learned features into class evidence
- softmax converts logits into class probabilities
- cross entropy measures the probability assigned to the true label
- SGD moves trainable weights in the direction that reduces loss

The CUDA path uses custom kernels where the operation is compact and direct, and
uses cuDNN/cuBLASLt where production GPU libraries already provide excellent
optimized primitives.

## ML Baseline Concept

The Python baseline flattens each image into 784 numeric features and trains a
Random Forest. It is CPU-based, classical, and reliable. It does not understand
image locality the way a CNN does, but it is a strong baseline for MNIST.

The project compares against it because a custom CUDA model should be measured
against something real, not only against itself.

## Proof Artifacts

Comparison output is written to:

```text
public/checkpoints/doc
public/checkpoints/img
```

The important files are:

```text
comparison_report.txt
dl_confusion_matrix.txt
ml_confusion_matrix.txt
comparison.png
```

These files make the result reviewable and reusable in reports or video.

## What To Read For Each Question

| Question | Read |
|---|---|
| How do classes fit together? | [Architecture](architecture.md) |
| What are the error codes? | [CORE](CORE/index.md) |
| How do tensors work? | [VIEW](VIEW/index.md) |
| How does memory/copying work? | [HANDLER](HANDLER/index.md) |
| What does each operator do? | [OPERATOR](OPERATOR/index.md) |
| How does the current model work? | [MODEL](MODEL/index.md) |
| How do app scripts fit together? | [APP](APP/index.md) |
| What must be installed? | [Dependencies](dependencies.md) |
| How do I map the project to the professor's grading criteria? | [Submission Plan](submission_plan.md) |
| How do I present the result? | [Proof And Video](proof_video.md) |
| What does the final result mean? | [Conclusion](conclusion.md) |
| How do I write a tiny program? | [Usage](usage.md) |

---

> **End Of Guide**  
> Previous: [Usage](usage.md) | Back to: [Project Manual](index.md)
