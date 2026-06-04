# APP Layer

> **Reading Path**  
> Home: [Project Manual](../index.md) | Previous: [MODEL](../MODEL/index.md) | Next: [DL App](dl.md)

The `app/src` folder is the experiment and reporting layer of the project. It is
where the low-level CUDA runtime is turned into runnable programs and where the
classical machine-learning baseline is evaluated beside it.

`APP` is intentionally separate from the runtime. The runtime lives in
`public/inc` and `lib/src`; the app folder answers a different question:

```text
Given the runtime, what do we run, measure, save, and compare?
```

## Files

| File | Document | Responsibility |
|---|---|---|
| `app/src/dl.cu` | [DL App](dl.md) | Runs the CUDA CNN training and benchmark path. |
| `app/src/ml.py` | [ML Baseline](ml.md) | Trains or loads a Random Forest baseline and evaluates it. |
| `app/src/comparaison.py` | [Comparison](comparaison.md) | Runs both systems, stores logs, parses matrices, and draws charts. |

## Why This Layer Matters

The CUDA runtime by itself only gives primitives: tensors, handlers, operators,
and model orchestration. The app layer gives proof. It defines repeatable
workflows:

```text
make ml
  -> Python baseline result

make dl
  -> CUDA CNN result

make compaire
  -> saved logs, matrices, report, and chart
```

The app layer also keeps the comparison honest. Both systems train from
`train.csv` and evaluate on `test.csv`. The result is not a training-set
memorization score; it is a held-out test comparison.

## Output Contract

Generated evidence is written under `public/checkpoints`:

```text
public/checkpoints/
  dl_iter_*.bin
  random_forest_mnist.joblib
  doc/
    dl.txt
    ml.txt
    dl_confusion_matrix.txt
    ml_confusion_matrix.txt
    comparison_report.txt
  img/
    comparison.png
```

The project therefore has both executable artifacts and human-readable proof.

---

> **Continue Reading**  
> [DL App](dl.md) | [ML Baseline](ml.md) | [Comparison](comparaison.md)
