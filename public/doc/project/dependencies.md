# Dependencies

> **Reading Path**  
> Home: [Project Manual](index.md) | Previous: [APP](APP/index.md) | Next: [Proof And Video](proof_video.md)

This project has two execution paths:

```text
CUDA/C++ deep-learning runtime
Python Random Forest baseline and comparison tooling
```

Both paths use the same MNIST PNG dataset, but they require different software.

## Required For CUDA DL

| Dependency | Why It Is Needed |
|---|---|
| NVIDIA GPU | Executes CUDA kernels and library calls. |
| NVIDIA driver | Allows CUDA runtime to communicate with the GPU. |
| CUDA Toolkit | Provides `nvcc`, CUDA runtime headers, and device libraries. |
| cuDNN | Used for convolution, pooling, and softmax. |
| cuBLASLt | Used for F16 matrix multiplication in dense layers. |
| C++ compiler | Host compiler used by `nvcc`. |
| GNU Make | Runs project build targets. |

The Makefile includes CUDA 13.2 include paths:

```text
/usr/local/cuda-13.2/include
/usr/local/cuda-13.2/include/cccl
/usr/local/cuda-13.2/targets/x86_64-linux/include
```

If `nvcc` on `PATH` is not from the same CUDA version as those headers, the
build can fail with an incompatible CUDA compiler/header error. In that case run
with an explicit compiler:

```bash
make dl NVCC=/usr/local/cuda-13.2/bin/nvcc
```

## Required For Python ML And Comparison

| Dependency | Why It Is Needed |
|---|---|
| Python 3 | Runs `ml.py`, `comparaison.py`, and dataset utilities. |
| NumPy | Stores image features and computes comparison metrics. |
| Pillow | Loads PNG images for the Random Forest baseline. |
| scikit-learn | Provides `RandomForestClassifier` and metrics. |
| joblib | Saves and loads the trained Random Forest checkpoint. |
| matplotlib | Draws the comparison chart. |
| GitPython | Used by the dataset downloader script. |

The dataset utility requirements are listed at:

```text
public/target/src/requirements.txt
```

The comparison script imports `matplotlib` only when plotting. If comparison
parsing works but chart creation fails, install matplotlib in the Python
environment.

## Dataset

The project expects MNIST PNG data under:

```text
public/target/meta
```

Expected files:

```text
public/target/meta/train.csv
public/target/meta/test.csv
public/target/meta/train/
public/target/meta/test/
```

The train split contains 60,000 images. The test split contains 10,000 images.

## Generated Artifacts

The project writes runtime outputs to:

```text
public/checkpoints
```

Important outputs:

```text
public/checkpoints/dl_iter_*.bin
public/checkpoints/random_forest_mnist.joblib
public/checkpoints/doc/*.txt
public/checkpoints/img/comparison.png
```

Checkpoint files can accumulate quickly. Use a reasonable checkpoint interval
for long DL runs.

## Minimal Run Commands

Build and run DL:

```bash
make dl NVCC=/usr/local/cuda-13.2/bin/nvcc
```

Run ML:

```bash
make ml
```

Run comparison:

```bash
make compaire
```

Regenerate comparison from existing logs:

```bash
python3 app/src/comparaison.py --no-run
```

---

> **Continue Reading**  
> Previous: [APP](APP/index.md) | Next: [Proof And Video](proof_video.md)
