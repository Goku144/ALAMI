# ML Baseline

> **Reading Path**  
> Home: [Project Manual](../index.md) | Section: [APP](index.md) | Previous: [DL App](dl.md) | Next: [Comparison](comparaison.md)

`app/src/ml.py` is the classical machine-learning baseline. It is not part of
the CUDA runtime. It exists so the custom DL model has a practical reference
point.

The baseline uses:

```text
RandomForestClassifier
```

from scikit-learn.

## Why A Random Forest

A Random Forest is a strong non-neural baseline for small images such as MNIST.
It does not need CUDA and it does not learn through gradient descent. It builds
many decision trees and combines their votes.

This makes it useful as a comparison:

- it is simple to run
- it is reliable on tabularized pixel data
- it provides a good accuracy target
- it gives a different learning philosophy from the CNN

The comparison is therefore meaningful: a custom CUDA CNN is measured against a
mature CPU-based classical model.

## Data Flow

```text
train.csv
  -> load image paths
  -> decode PNG images with PIL
  -> convert to grayscale
  -> resize to 28x28 when needed
  -> normalize pixels to [0, 1]
  -> flatten each image to 784 features
  -> train Random Forest

test.csv
  -> same image preparation
  -> model.predict
  -> accuracy and confusion matrix
```

The training and test files are separate by default:

| Argument | Default |
|---|---|
| `--train-csv` | `public/target/meta/train.csv` |
| `--test-csv` | `public/target/meta/test.csv` |

## Checkpoint Reuse

The baseline saves the trained forest to:

```text
public/checkpoints/random_forest_mnist.joblib
```

If that file already exists, `ml.py` loads it instead of training again. It
still loads and evaluates the test set, so the confusion matrix remains fresh
while avoiding expensive repeated forest training.

Conceptually:

```text
model checkpoint exists?
  yes -> load model, load test images, evaluate
  no  -> load train images, train model, save model, load test images, evaluate
```

## Output

`ml.py` prints:

```text
Accuracy : ...
Confusion matrix
[[...]]
```

That printed matrix is consumed by `comparaison.py`.

## Difference From DL

The Random Forest sees every image as a flat vector of 784 numbers. It does not
know about spatial structure directly.

The CUDA CNN keeps the image as a tensor:

```text
batch x channels x height x width
```

The CNN can learn local patterns through convolution. The Random Forest learns
decision rules over individual pixel values and combinations of them.

## How To Run

```bash
make ml
```

---

> **Continue Reading**  
> Previous: [DL App](dl.md) | Next: [Comparison](comparaison.md)
