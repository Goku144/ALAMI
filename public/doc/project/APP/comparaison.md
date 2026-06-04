# Comparison Pipeline

> **Reading Path**  
> Home: [Project Manual](../index.md) | Section: [APP](index.md) | Previous: [ML Baseline](ml.md)

`app/src/comparaison.py` is the evidence script. It does not define a model. It
runs both model paths, saves their outputs, parses their confusion matrices, and
turns the result into text and an image.

The filename keeps the project spelling: `comparaison.py`.

## Why This Script Exists

Accuracy claims should be reproducible. A terminal screenshot is useful, but it
is not enough. The comparison script creates a durable trail:

```text
raw DL log
raw ML log
parsed DL confusion matrix
parsed ML confusion matrix
summary report
comparison chart
```

That trail makes the result inspectable after the run finishes.

## Default Workflow

```bash
make compaire
```

The Makefile target runs:

```text
python3 app/src/comparaison.py
```

The script then runs:

```text
make ml
make dl
```

and stores the outputs.

## Stored Evidence

```text
public/checkpoints/doc/ml.txt
public/checkpoints/doc/dl.txt
public/checkpoints/doc/ml_confusion_matrix.txt
public/checkpoints/doc/dl_confusion_matrix.txt
public/checkpoints/doc/comparison_report.txt
public/checkpoints/img/comparison.png
```

The raw logs remain untouched. The parsed matrices are written separately so a
reader can verify the numbers without reading all training output.

## Metrics

The script computes metrics from the confusion matrices themselves:

| Metric | Meaning |
|---|---|
| Accuracy | Correct predictions divided by total predictions. |
| Per-class recall | How often each true class was recovered. |
| Macro F1 | Average class-balanced F1 score. |

This is deliberately conservative. The comparison does not use hidden model
state. It uses only the printed evidence from each run.

## Chart

The chart at `public/checkpoints/img/comparison.png` contains:

- DL confusion matrix
- ML confusion matrix
- per-class recall comparison
- DL training progress when training log lines are available

This chart is designed for reports, demos, and video narration.

## Reusing Existing Logs

When you do not want to rerun training:

```bash
python3 app/src/comparaison.py --no-run
```

This reads the existing logs in `public/checkpoints/doc` and regenerates the
report and chart.

## Conceptual Role

The comparison script is the bridge between engineering and proof. The runtime
answers, "can we build and train the model?" The comparison script answers,
"what did it achieve, compared to a known baseline?"

---

> **Continue Reading**  
> Back to: [APP](index.md)
