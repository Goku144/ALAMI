# DL App

> **Reading Path**  
> Home: [Project Manual](../index.md) | Section: [APP](index.md) | Next: [ML Baseline](ml.md)

`app/src/dl.cu` is the CUDA deep-learning entry point. It is intentionally
small:

```cpp
MODEL::DL dl(64);
dl.train(60000, 0.0005f, 3000);
```

That compact call hides a full path:

```text
load training metadata
shuffle training rows
load images
bind tensors
initialize or restore weights
train CNN mini-batches
save checkpoints
load test metadata
run final benchmark
print confusion matrix
```

The app does not implement kernels. It delegates the deep-learning work to
`MODEL::DL`, which in turn delegates math to `OPERATOR` classes and CUDA
libraries.

## Current Training Schedule

The current app asks for:

| Value | Meaning |
|---|---|
| `64` | Mini-batch size. |
| `60000` | Target training iteration. |
| `0.0005f` | Refinement learning rate. |
| `3000` | Save checkpoint every 3000 iterations. |

One iteration is one mini-batch SGD update. With batch size 64 and 60,000 train
images, one epoch is roughly:

```text
60000 / 64 = 938 updates
```

So 60,000 iterations is a long refinement run. The checkpoint-resume logic means
the model does not start from zero every time. It finds the latest
`dl_iter_*.bin` checkpoint, loads it, and continues only if the requested target
iteration is higher.

## Why Checkpoint Resume Exists

CUDA training can be expensive. A run should be able to stop and continue
without wasting previous work.

`MODEL::DL::train()` searches for checkpoints matching the configured prefix:

```text
public/checkpoints/dl_iter_*.bin
```

It chooses the highest iteration number. If that checkpoint already satisfies
the requested training target, the program uses it directly for the benchmark.
If not, it resumes training from that point.

Conceptually:

```text
checkpoint exists?
  no  -> initialize weights and train from zero
  yes -> load latest weights
          if latest < target: continue training
          else: benchmark immediately
```

This makes comparison fast and protects long CUDA runs from being disposable.

## Why CUDA

The DL app exists to prove the custom runtime. CUDA is used because the model is
not just calling a Python framework. The project owns the tensor shapes, memory
binding, GPU copies, and operator calls explicitly.

CUDA gives:

- F16 tensor storage for compact GPU memory use
- custom kernels for normalization, ReLU, SGD, and cross entropy
- cuDNN for convolution, pooling, and softmax
- cuBLASLt for linear layers

The point is control. Every model stage has an explicit place in the codebase.

## Training And Evaluation Split

The DL constructor defaults to:

```text
public/target/meta/train.csv
```

The final benchmark uses:

```text
public/target/meta/test.csv
```

This matters. Accuracy is only meaningful when the model is measured on images
it did not use for training.

## Concept Flow

```text
image
  -> Normalize
  -> Conv2D + bias
  -> ReLU
  -> MaxPool
  -> Flatten
  -> MatrixMulBias
  -> ReLU
  -> MatrixMulBias
  -> Softmax
  -> CrossEntropy gradient
  -> Backward pass
  -> SGD update
```

Each concept is represented by a concrete operator class:

| Concept | Runtime Class |
|---|---|
| Normalize pixels | `OPERATOR::Normalize` |
| Convolution | `OPERATOR::Conv2DRelu` |
| Activation | `OPERATOR::Relu` |
| Downsampling | `OPERATOR::Pool` |
| Dense layer | `OPERATOR::MatrixMulBias` |
| Probability distribution | `OPERATOR::Softmax` |
| Loss and output gradient | `OPERATOR::CrossEntropy` |
| Parameter update | `OPERATOR::SGD` |

## How To Run

```bash
make dl
```

The output contains training progress and ends with a `CONFUSION MATRIX` block.

---

> **Continue Reading**  
> Previous: [APP](index.md) | Next: [ML Baseline](ml.md)
