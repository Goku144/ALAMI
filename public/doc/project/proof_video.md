# Proof And Video Guide

> **Reading Path**  
> Home: [Project Manual](index.md) | Previous: [Dependencies](dependencies.md) | Next: [Conclusion](conclusion.md)

This project produces measurable evidence. The best way to present it is to
show the pipeline, not only the final score.

For the full final `remise` structure, use the
[Explanatory Submission Plan](submission_plan.md). This video guide is the
recording companion to that grading-oriented document.

## Proof Artifacts

After running:

```bash
make compaire
```

the project writes:

```text
public/checkpoints/doc/dl.txt
public/checkpoints/doc/ml.txt
public/checkpoints/doc/dl_confusion_matrix.txt
public/checkpoints/doc/ml_confusion_matrix.txt
public/checkpoints/doc/comparison_report.txt
public/checkpoints/img/comparison.png
```

These files are the proof package:

- raw logs show what was actually executed
- confusion matrices show class-level behavior
- the report summarizes accuracy, macro F1, and recall
- the chart makes the comparison readable in a presentation

## Suggested Video Structure

Use this structure for a short technical video:

```text
1. Project explanation
2. Realized pipeline
3. Models used
4. Demonstration of the work
5. Results obtained
6. Final interpretation
```

## Scene Notes

### 1. Project Purpose

Show the source tree:

```text
CORE -> VIEW -> HANDLER -> OPERATOR -> MODEL -> APP
```

Narration:

```text
This project builds a CUDA/C++ neural-network runtime from explicit tensor
views, memory handlers, GPU operators, and one concrete digit classifier. A
Python Random Forest baseline is used to verify whether the custom DL path is
competitive.
```

### 2. Runtime Architecture

Show [Architecture](architecture.md). Emphasize:

```text
VIEW::Math does not own memory.
HANDLER owns memory and IO.
OPERATOR performs GPU math.
MODEL wires operators into a trainable graph.
APP runs experiments and proof scripts.
```

### 3. DL Model Path

Show [DL App](APP/dl.md) and [MODEL::DL](MODEL/dl.md).

Narration:

```text
The CUDA model keeps images as tensors. It normalizes pixels, applies
convolution, ReLU, max pooling, dense layers, softmax, cross entropy, and SGD.
The code uses custom kernels where control matters and cuDNN/cuBLASLt where
optimized library math is the right tool.
```

### 4. ML Baseline Path

Show [ML Baseline](APP/ml.md).

Narration:

```text
The Random Forest baseline flattens each image into 784 features. It gives a
strong CPU baseline and helps prove that the custom CUDA model is not only
running, but competitive.
```

### 5. Comparison Pipeline

Show [Comparison](APP/comparaison.md).

Run:

```bash
make compaire
```

or, if logs already exist:

```bash
python3 app/src/comparaison.py --no-run
```

### 6. Results

Show:

```text
public/checkpoints/doc/comparison_report.txt
public/checkpoints/img/comparison.png
```

Interpret the result by class, not only by global accuracy. A good comparison
explains where DL wins, where ML wins, and which digits remain difficult.

## Optional Recording Command

If `ffmpeg` is installed, one simple Linux screen-recording command is:

```bash
ffmpeg -video_size 1920x1080 -framerate 30 -f x11grab -i :0.0 -f pulse -i default public/checkpoints/img/project_demo.mp4
```

This is optional. The project does not require video recording to run.

## What Makes The Proof Honest

The comparison is honest when:

- DL trains on `train.csv`
- ML trains on `train.csv`
- DL evaluates on `test.csv`
- ML evaluates on `test.csv`
- the confusion matrices come from saved logs
- the report is computed from those matrices

That is the current intended workflow.

---

> **Continue Reading**  
> Previous: [Dependencies](dependencies.md) | Next: [Conclusion](conclusion.md)
