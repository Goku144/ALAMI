# Explanatory Submission Plan

> **Reading Path**  
> Home: [Project Manual](index.md) | Previous: [Proof And Video](proof_video.md) | Next: [Conclusion](conclusion.md)

This document is the final `remise` guide. It explains how to present the same
project for two evaluations:

```text
Machine Learning note
Practical IA / Deep Learning note
```

The project is common, but it must be read in two different ways. For the
Machine Learning grade, the evaluator looks for data preparation, feature
engineering, a classical model, evaluation metrics, and result analysis. For
the Practical IA / Deep Learning grade, the evaluator looks for a deep-learning
model, its architecture, its training logic, the optimization process, the
comparison with ML, and the practical intelligent system that was built.

The goal of the final submission is therefore not only to show that the code
runs. The goal is to prove, with source code, report, results, notebooks or
Colab work, and video demonstration, that the project satisfies both modules.

## Chapter Index

| Chapter | Topic |
|---:|---|
| 1 | What The Final Submission Must Prove |
| 2 | Machine Learning Evaluation |
| 3 | Practical IA / Deep Learning Evaluation |
| 4 | ML vs DL Comparison |
| 5 | Final Deliverables |
| 6 | Video Presentation Script |
| 7 | Defense Checklist |

---

## Chap 1: What The Final Submission Must Prove

The professor's message asks for one project, but the grading is separated into
two marks. This means the final document, `rapport`, and video should make the
separation explicit:

```text
Same dataset
Same digit-classification problem
Same final comparison package

but:

Machine Learning evaluates the classical ML pipeline.
Practical IA / Deep Learning evaluates the custom CUDA CNN pipeline.
```

The common task is digit classification. The system receives MNIST-style
grayscale images and predicts one class:

```text
0, 1, 2, 3, 4, 5, 6, 7, 8, 9
```

The common data split is:

```text
training data   -> public/target/meta/train.csv
evaluation data -> public/target/meta/test.csv
```

This split is important for both grades. `train.csv` is used to learn model
parameters. `test.csv` is held out for final measurement. The final reported
accuracy is therefore a generalization result, not only a memorization result.

The project has two model paths:

| Path | Module Meaning | Main Artifact |
|---|---|---|
| Random Forest | Machine Learning | [APP/ml.md](APP/ml.md) |
| CUDA CNN | Practical IA / Deep Learning | [MODEL/layers.md](MODEL/layers.md) |

The project also has one comparison path:

| Path | Purpose |
|---|---|
| [APP/comparaison.md](APP/comparaison.md) | Runs or reuses both outputs, parses confusion matrices, writes the report, and saves the chart. |

The final proof is not a single screenshot. It is a chain:

```text
source code
  -> data loading
  -> preprocessing
  -> ML model
  -> DL model
  -> training
  -> evaluation
  -> confusion matrices
  -> comparison report
  -> chart
  -> video demonstration
```

When presenting, say clearly:

```text
This project supports two grades. The ML part is the Random Forest baseline and
its preprocessing/evaluation. The Practical IA / DL part is the custom CUDA CNN
runtime, its mathematical architecture, training loop, GPU operators, and its
comparison against the ML baseline.
```

---

## Chap 2: Machine Learning Evaluation

The Machine Learning grade should be presented as a complete classical ML
pipeline. The evaluator should be able to see:

```text
preprocessing
feature engineering
model choice
performance evaluation
result analysis
```

The relevant source is:

```text
app/src/ml.py
```

The relevant documentation is:

```text
public/doc/project/APP/ml.md
public/doc/project/APP/comparaison.md
public/doc/project/conclusion.md
```

### 2.1 Preprocessing

The ML path starts from CSV metadata:

```text
public/target/meta/train.csv
public/target/meta/test.csv
```

Each CSV row points to a PNG image and its label. The preprocessing pipeline is:

```text
CSV loading
  -> PNG reading
  -> grayscale conversion
  -> resize to 28x28 if needed
  -> normalization to [0, 1]
```

Each step has a reason.

CSV loading gives a reproducible dataset split. The model does not randomly mix
training and test data during final evaluation.

PNG reading converts the stored image file into numeric pixel values.

Grayscale conversion ensures the model receives one channel per image. MNIST is
a grayscale task, so color channels would add unnecessary dimensions.

Resize to `28x28` standardizes every image. A classical ML model needs a fixed
number of features per sample.

Normalization converts pixel intensity from:

```text
0..255
```

to:

```text
0..1
```

For a pixel value `p`, the normalized value is:

```text
p_norm = p / 255
```

For the ML model, normalization makes feature values comparable and avoids
feeding raw integer magnitudes into the forest.

### 2.2 Feature Engineering

The Random Forest does not receive a two-dimensional image tensor. It receives
a feature vector.

The image shape is:

```text
28 x 28
```

The flattened feature count is:

```text
28 * 28 = 784
```

So one image becomes:

```text
image[28][28] -> vector[784]
```

Conceptually:

```text
x = [
  pixel_0,
  pixel_1,
  pixel_2,
  ...
  pixel_783
]
```

This is the central ML feature-engineering step. The model no longer sees
rows, columns, neighborhoods, or local stroke structure directly. It sees 784
numeric variables.

That design is acceptable for a classical baseline because Random Forests work
well with tabular feature vectors. It also creates a useful contrast with the
CNN, which keeps spatial structure.

### 2.3 Machine Learning Model

The selected ML model is:

```text
RandomForestClassifier
```

The Random Forest is a strong baseline because it combines many decision trees.
Each tree learns rules over pixel features, then the forest combines their
votes.

In simplified form:

```text
tree_1 predicts digit
tree_2 predicts digit
tree_3 predicts digit
...
final prediction = majority vote
```

The Random Forest is not a deep-learning model. It does not use convolution,
backpropagation, GPU tensors, or learned filters. Its learning process is based
on decision trees and voting.

This makes it valuable for grading because the ML part is clearly distinct
from the DL part:

```text
ML:
  flattened pixels
  Random Forest
  CPU/scikit-learn
  decision trees and voting

DL:
  spatial tensors
  CUDA CNN
  GPU operators
  backpropagation and SGD
```

The ML model also uses a checkpoint:

```text
public/checkpoints/random_forest_mnist.joblib
```

If the checkpoint exists, the script loads it instead of training again. It
still evaluates on `test.csv`, so the final ML result remains tied to the held
out test set.

### 2.4 Performance Evaluation

The ML evaluation should not be reduced to one number. The project reports:

```text
accuracy
confusion matrix
macro F1
per-class recall
```

Accuracy measures global correctness:

```text
accuracy = correct predictions / total predictions
```

The final ML accuracy is:

```text
ML accuracy = 0.9702
```

On a 10,000-image test set, this means approximately:

```text
0.9702 * 10000 = 9702 correct predictions
```

The confusion matrix shows which classes were predicted correctly and which
classes were confused. A row represents the true class. A column represents the
predicted class.

The diagonal entries are correct predictions:

```text
true class = predicted class
```

Off-diagonal entries are mistakes:

```text
true class != predicted class
```

Macro F1 gives a class-balanced summary. It is useful because it does not let
large or easy classes hide weaker class behavior.

The final ML macro F1 is:

```text
ML macro F1 = 0.9700
```

Per-class recall asks:

```text
For all real examples of digit k, how many did the model recover?
```

Mathematically:

```text
recall(k) = true_positives(k) / total_true_examples(k)
```

The final ML recalls are:

| Class | ML Recall |
|---:|---:|
| 0 | 0.9898 |
| 1 | 0.9885 |
| 2 | 0.9719 |
| 3 | 0.9624 |
| 4 | 0.9766 |
| 5 | 0.9619 |
| 6 | 0.9791 |
| 7 | 0.9611 |
| 8 | 0.9548 |
| 9 | 0.9534 |

### 2.5 Result Analysis For ML

The Random Forest performs well overall:

```text
accuracy  = 0.9702
macro F1  = 0.9700
```

This proves the preprocessing and feature engineering are valid. A weak data
pipeline would usually produce unstable or very low results. Here, the model
learns a strong classifier from flattened pixels.

Its weaker classes are:

```text
8, 9, 7, 5, 3
```

Those digits can require shape reasoning. For example, distinguishing a `3`
from an `8`, or a `9` from a `4`, often depends on curves, gaps, and stroke
placement. A Random Forest can learn useful pixel rules, but it does not have
the built-in spatial bias of convolution.

For the Machine Learning grade, the conclusion should be:

```text
The ML pipeline is complete: it loads the data, preprocesses images, converts
them to 784-dimensional feature vectors, trains or loads a Random Forest,
evaluates on the held-out test set, and produces accuracy plus confusion-matrix
based analysis.
```

---

## Chap 3: Practical IA / Deep Learning Evaluation

The Practical IA / Deep Learning grade should be presented as the custom
intelligent runtime and CNN model. The evaluator should be able to see:

```text
deep-learning model
network architecture
training and optimization
practical implementation
ML vs DL comparison
```

The relevant source files are:

```text
app/src/dl.cu
lib/src/MODEL/DL.cu
public/inc/MODEL/DL.hpp
```

The relevant mathematical explanation is:

```text
public/doc/project/MODEL/layers.md
```

The relevant operator documentation is:

```text
public/doc/project/OPERATOR/index.md
```

### 3.1 Deep-Learning Model

The DL model is a compact convolutional neural network implemented through the
custom CUDA/C++ runtime.

The high-level architecture is:

```text
image
  -> Normalize
  -> Conv2D + bias
  -> ReLU
  -> MaxPool
  -> Flatten
  -> Dense + bias
  -> ReLU
  -> Dense + bias
  -> Softmax
```

The training architecture adds:

```text
CrossEntropy
  -> backward propagation
  -> SGD update
```

The model learns trainable parameters:

```text
kerW0, b0, W1, b1, W2, b2
```

Their roles are:

| Parameter | Meaning |
|---|---|
| `kerW0` | Convolution filters that learn local stroke patterns. |
| `b0` | Bias for convolution outputs. |
| `W1` | Dense projection from pooled features to hidden representation. |
| `b1` | Hidden dense bias. |
| `W2` | Final projection from hidden features to class logits. |
| `b2` | Final class-logit bias. |

The model input remains spatial:

```text
batch x channel x height x width
```

For this project:

```text
B = 64
C = 1
H = 28
W = 28
```

This is the main conceptual difference from the ML baseline. The CNN does not
begin by destroying the image shape. It first learns local visual features.

### 3.2 Network Architecture Logic

The full forward graph is:

```text
layer 0
  image(64, 1, H, W)
  x = Normalize(image)

layer 1
  z0 = conv(x, kerW0) + b0
  A0 = ReLU(z0)

layer 1.5
  p = pool(A0)

layer 2
  p_flat = flatten(p)
  z1 = p_flat @ W1 + b1
  A1 = ReLU(z1)

layer 3
  z2 = A1 @ W2 + b2
  out = softmax(z2)
```

Layer 0 normalizes pixels:

```text
x = image / 255
```

This stabilizes the numeric scale.

Layer 1 applies convolution. For one output position, the idea is:

```text
z0[b, f, i, j] =
  b0[f] + sum over c, u, v of x[b, c, i+u, j+v] * kerW0[f, c, u, v]
```

The filter slides over the image and produces evidence for local patterns.
Those patterns can be strokes, curves, corners, or gaps.

ReLU keeps positive evidence:

```text
A0 = max(0, z0)
```

Max pooling reduces spatial size while preserving strong local activations:

```text
p[b, f, i, j] = max value in the 2x2 local window of A0
```

Flattening converts the pooled feature maps into a vector only after the model
has already extracted local spatial features:

```text
p_flat = flatten(p)
```

The first dense layer combines local features into a hidden representation:

```text
z1 = p_flat @ W1 + b1
A1 = ReLU(z1)
```

The final dense layer creates class evidence:

```text
z2 = A1 @ W2 + b2
```

Softmax converts evidence into probabilities:

```text
out[k] = exp(z2[k]) / sum_j exp(z2[j])
```

The predicted class is:

```text
prediction = argmax_k out[k]
```

The detailed mathematical proof is in [Layer Mathematics And Model Process](MODEL/layers.md).

### 3.3 Training And Optimization

Training uses mini-batches:

```text
MODEL::DL dl(64)
```

Each update processes 64 images. The current app training request is:

```cpp
dl.train(60000, 0.0005f, 3000);
```

This means:

| Value | Meaning |
|---|---|
| `60000` | Target training iteration. |
| `0.0005f` | Learning rate. |
| `3000` | Checkpoint interval. |

One training step is:

```text
load mini-batch
forward pass
compute softmax probabilities
compute cross-entropy loss
compute output gradient
backpropagate gradients
update parameters with SGD
```

The loss is cross entropy. For a true class `y`, it penalizes the model when
the probability assigned to `y` is low:

```text
loss = -log(out[y])
```

The useful softmax-cross-entropy gradient is:

```text
dz2 = out - Y
```

where `Y` is the one-hot target distribution.

The backward chain is:

```text
layer 3
  dz2 = out - Y
  dW2 = A1.T @ dz2
  db2 = sum_rows(dz2)
  dA1 = dz2 @ W2.T

layer 2
  dz1 = dA1 .* ReLU'(z1)
  dW1 = p_flat.T @ dz1
  db1 = sum_rows(dz1)
  dp_flat = dz1 @ W1.T
  dp = reshape(dp_flat, shape_of_p)

layer 1.5
  dA0 = poolMax_backward(dp, A0)

layer 1
  dz0 = dA0 .* ReLU'(z0)
  dkerW0 = conv_backward_filter(x, dz0)
  db0 = sum_over_batch_height_width(dz0)
  dx = conv_backward_input(kerW0, dz0)
```

The SGD update is:

```text
parameter = parameter - learning_rate * gradient
```

For this model:

```text
kerW0 = kerW0 - lr * dkerW0
b0    = b0    - lr * db0
W1    = W1    - lr * dW1
b1    = b1    - lr * db1
W2    = W2    - lr * dW2
b2    = b2    - lr * db2
```

The checkpoint system saves weights under:

```text
public/checkpoints/dl_iter_*.bin
```

This is a practical part of the project. Training can resume from the latest
checkpoint instead of starting from zero each time.

### 3.4 Practical IA Implementation

The project is practical because it is not only a notebook model call. It builds
a runtime with explicit responsibilities:

```text
CORE
  constants, errors, alignment, logging

VIEW
  tensor shapes and tensor handles

HANDLER
  CPU memory, GPU memory, file IO, tensor binding, data movement

OPERATOR
  CUDA/cuDNN/cuBLASLt mathematical operations

MODEL
  trainable CNN orchestration

APP
  runnable experiments and evidence scripts
```

The Practical IA proof is that all these layers work together:

```text
load data
bind tensors
copy to GPU
run operators
compute loss
backpropagate
update weights
save checkpoints
benchmark on test data
print confusion matrix
compare to ML
```

The project uses GPU logic directly:

| Component | Practical Role |
|---|---|
| Custom CUDA kernels | Compact operations such as normalization, ReLU, SGD, and loss support. |
| cuDNN | Optimized convolution, pooling, and softmax operations. |
| cuBLASLt | Optimized dense matrix multiplication. |
| Explicit memory handlers | Show CPU/GPU ownership and data transfer. |
| Checkpoints | Preserve trained model states. |
| Comparison script | Converts runtime output into evidence. |

### 3.5 Intelligent Part

The intelligent part is not only that the program uses CUDA. The intelligent
part is that the learned parameters change the behavior of the classifier.

Before training, the filters and dense weights do not encode useful digit
knowledge. During training, the loss gradient modifies them. After training,
the convolution filters respond to useful visual structures, and the dense
layers combine those structures into class decisions.

The CNN has a useful inductive bias for images:

```text
nearby pixels are related
local strokes form shapes
shapes help identify digits
```

Convolution uses that bias by applying the same learned filter across many
spatial positions. This gives the model a better way to recognize patterns
than treating every pixel coordinate as isolated tabular data.

For the Practical IA / DL grade, the conclusion should be:

```text
The DL pipeline is a custom CUDA CNN that keeps image structure, learns filters,
trains with backpropagation and SGD, saves checkpoints, evaluates on the test
set, and produces stronger results than the classical ML baseline.
```

---

## Chap 4: ML vs DL Comparison

The comparison is the bridge between the two grades.

Both models use the same final evaluation protocol:

```text
train on train.csv
evaluate on test.csv
```

This matters because the comparison is fair only if both systems are judged on
the same held-out images.

The comparison command is:

```bash
make compaire
```

If existing logs should be reused:

```bash
python3 app/src/comparaison.py --no-run
```

The comparison artifacts are:

```text
public/checkpoints/doc/dl.txt
public/checkpoints/doc/ml.txt
public/checkpoints/doc/dl_confusion_matrix.txt
public/checkpoints/doc/ml_confusion_matrix.txt
public/checkpoints/doc/comparison_report.txt
public/checkpoints/img/comparison.png
```

### 4.1 Final Numerical Result

The final result is:

```text
DL accuracy : 0.9782
ML accuracy : 0.9702

DL macro F1 : 0.9781
ML macro F1 : 0.9700
```

The accuracy gain is:

```text
0.9782 - 0.9702 = 0.0080
```

That is:

```text
0.80 percentage points
```

On 10,000 test samples:

```text
0.0080 * 10000 = 80
```

So the DL model makes about 80 additional correct predictions compared with the
ML baseline.

This is a good result because the DL model is not imported from a Python deep
learning framework. It is produced by the custom CUDA/C++ runtime.

### 4.2 Per-Class Recall

The recall comparison is:

| Class | DL Recall | ML Recall | DL - ML |
|---:|---:|---:|---:|
| 0 | 0.9888 | 0.9898 | -0.0010 |
| 1 | 0.9912 | 0.9885 | +0.0026 |
| 2 | 0.9748 | 0.9719 | +0.0029 |
| 3 | 0.9851 | 0.9624 | +0.0228 |
| 4 | 0.9756 | 0.9766 | -0.0010 |
| 5 | 0.9697 | 0.9619 | +0.0078 |
| 6 | 0.9854 | 0.9791 | +0.0063 |
| 7 | 0.9728 | 0.9611 | +0.0117 |
| 8 | 0.9713 | 0.9548 | +0.0164 |
| 9 | 0.9653 | 0.9534 | +0.0119 |

The DL model wins on most classes. The largest gains are:

```text
class 3: +0.0228
class 8: +0.0164
class 9: +0.0119
class 7: +0.0117
```

This supports the conceptual explanation: digits such as `3`, `8`, `9`, and
`7` benefit from spatial reasoning about curves, gaps, loops, and stroke
position.

### 4.3 Why Confusion Matrices Are More Honest Than Accuracy Alone

Accuracy can hide where mistakes happen. Two models can have similar accuracy
but fail on different classes.

A confusion matrix shows:

```text
which true classes are easy
which true classes are difficult
which wrong classes are confused
whether a model has class-specific weakness
```

For example, if many real `9` images are predicted as `4`, the accuracy number
only counts those as wrong. The confusion matrix reveals the exact confusion
pattern.

This is why the report and video should show:

```text
accuracy
macro F1
confusion matrices
per-class recall
```

The final interpretation should be:

```text
The CNN is slightly but clearly better than the Random Forest. The gain is
small in percentage, but meaningful on 10,000 test images. The per-class recall
shows that the CNN advantage appears especially on digits where local shape and
spatial structure matter.
```

---

## Chap 5: Final Deliverables

The final `remise` should contain or point to the following deliverables.

### 5.1 Source Code

Show the project source tree:

```text
app/src
lib/src
public/inc
public/doc/project
```

Important files to mention:

| File | Why It Matters |
|---|---|
| `app/src/ml.py` | Machine Learning baseline. |
| `app/src/dl.cu` | Deep-learning app entry. |
| `app/src/comparaison.py` | Final comparison and report generator. |
| `lib/src/MODEL/DL.cu` | CNN training, checkpoint, benchmark behavior. |
| `public/inc/MODEL/DL.hpp` | DL model public interface. |

### 5.2 Notebooks / Colab Work

If a notebook or Colab is submitted, it should not replace the project code.
It should explain or reproduce the important evidence:

```text
dataset inspection
preprocessing examples
ML baseline explanation
DL architecture explanation
metric calculation
result tables
comparison chart
```

The notebook should make the grading story easy to follow, but the source code
remains the main implementation artifact.

### 5.3 Report

The report should be organized in the same order as the grading criteria:

```text
1. Project objective
2. Dataset and preprocessing
3. Machine Learning model
4. Deep Learning model
5. Training and optimization
6. Results
7. ML vs DL comparison
8. Member contribution
9. Conclusion
```

For the ML section, include:

```text
CSV loading
PNG reading
grayscale conversion
resize to 28x28
normalization
flattening to 784 features
Random Forest
accuracy and confusion matrix
```

For the DL section, include:

```text
CUDA CNN architecture
normalization
convolution
ReLU
max pooling
flatten
dense layers
softmax
cross entropy
backpropagation
SGD
checkpointing
```

For the results section, include:

```text
DL accuracy 0.9782
ML accuracy 0.9702
DL macro F1 0.9781
ML macro F1 0.9700
about 80 extra correct DL predictions
```

### 5.4 Results

Submit or show these files:

```text
public/checkpoints/doc/comparison_report.txt
public/checkpoints/doc/dl_confusion_matrix.txt
public/checkpoints/doc/ml_confusion_matrix.txt
public/checkpoints/img/comparison.png
```

The result files prove that the numbers are computed from model outputs.

### 5.5 Video Recording

The video should show:

```text
project explanation
realized pipeline
models used
results obtained
demonstration of the work
```

This is the professor's required structure. The video should be technical but
not too slow. It should show code, documentation, commands, and final results.

### 5.6 Member Contributions

The report should include a contribution table. A simple structure is:

| Member | Contribution |
|---|---|
| Member 1 | Dataset preparation, ML baseline, metrics. |
| Member 2 | CUDA operators, memory handlers, DL model. |
| Member 3 | Report, documentation, video, comparison analysis. |

Adjust the names and responsibilities to match the real team. The important
thing is to connect each member to concrete project work.

---

## Chap 6: Video Presentation Script

The video should follow this scene order.

### Scene 1: Project Explanation

Show:

```text
public/doc/project/index.md
```

Say:

```text
This project is a digit-classification system built in two parts. The first
part is a classical Machine Learning baseline using Random Forest. The second
part is a Practical IA / Deep Learning implementation using a custom CUDA/C++
CNN runtime. The same project supports two grades, but each grade evaluates a
different part of the work.
```

Show the hierarchy:

```text
CORE -> VIEW -> HANDLER -> OPERATOR -> MODEL -> APP
```

Explain:

```text
CORE defines shared rules.
VIEW describes tensors.
HANDLER owns memory and data movement.
OPERATOR performs GPU math.
MODEL wires a trainable CNN.
APP runs experiments and comparison.
```

### Scene 2: Realized Pipeline

Show:

```text
public/doc/project/APP/index.md
```

Say:

```text
The realized pipeline starts from train.csv and test.csv. Images are loaded
from PNG files. The ML path converts them to flat 784-feature vectors. The DL
path keeps them as image tensors. Both models are evaluated on the same test
split, and the comparison script saves logs, matrices, a report, and a chart.
```

Show:

```text
public/checkpoints/doc
public/checkpoints/img
```

### Scene 3: Models Used

Show the ML baseline:

```text
public/doc/project/APP/ml.md
app/src/ml.py
```

Say:

```text
For Machine Learning, the model is RandomForestClassifier. Each 28x28 image is
normalized and flattened into 784 features. The forest learns decision trees
over those pixel features and combines their votes.
```

Then show the DL model:

```text
public/doc/project/MODEL/layers.md
app/src/dl.cu
lib/src/MODEL/DL.cu
```

Say:

```text
For Practical IA / Deep Learning, the model is a CUDA CNN. It applies
normalization, convolution, ReLU, max pooling, dense layers, softmax, cross
entropy, backpropagation, and SGD. The model saves checkpoints and benchmarks
on the test set.
```

### Scene 4: Demonstration Of The Work

Run the full comparison if time and hardware allow:

```bash
make compaire
```

If trained checkpoints and logs already exist, run:

```bash
python3 app/src/comparaison.py --no-run
```

Say:

```text
The comparison script is the proof layer. It runs or reuses both model outputs,
extracts confusion matrices, computes metrics, and writes the final comparison
chart.
```

### Scene 5: Results Obtained

Show:

```text
public/checkpoints/doc/comparison_report.txt
public/checkpoints/img/comparison.png
```

Say:

```text
The final DL accuracy is 0.9782 and the final ML accuracy is 0.9702. The DL
macro F1 is 0.9781 and the ML macro F1 is 0.9700. The absolute accuracy gain is
0.0080, which corresponds to about 80 additional correct predictions on 10,000
test images.
```

Then explain:

```text
The confusion matrices are more informative than accuracy alone because they
show class-level mistakes. The CNN wins on most classes, especially digits
where local shape matters, such as 3, 8, 9, and 7.
```

### Scene 6: Final Conclusion

Say:

```text
For the Machine Learning grade, the project proves a complete preprocessing,
feature engineering, Random Forest, and evaluation pipeline. For the Practical
IA / Deep Learning grade, it proves a custom CUDA CNN runtime with architecture,
training, optimization, checkpointing, and comparison. The same dataset and
test split are used for both, making the final comparison fair and reproducible.
```

---

## Chap 7: Defense Checklist

Use this checklist before the final defense or upload.

### 7.1 What To Open

Open these documentation files:

```text
public/doc/project/submission_plan.md
public/doc/project/index.md
public/doc/project/APP/ml.md
public/doc/project/APP/dl.md
public/doc/project/APP/comparaison.md
public/doc/project/MODEL/layers.md
public/doc/project/conclusion.md
public/doc/project/proof_video.md
```

Open these source files:

```text
app/src/ml.py
app/src/dl.cu
app/src/comparaison.py
lib/src/MODEL/DL.cu
```

Open these result files:

```text
public/checkpoints/doc/comparison_report.txt
public/checkpoints/doc/dl_confusion_matrix.txt
public/checkpoints/doc/ml_confusion_matrix.txt
public/checkpoints/img/comparison.png
```

### 7.2 What Command To Run

For the complete proof:

```bash
make compaire
```

For a faster proof from existing logs:

```bash
python3 app/src/comparaison.py --no-run
```

For individual paths:

```bash
make ml
make dl
```

### 7.3 What Numbers To Mention

Memorize these final values:

```text
DL accuracy : 0.9782
ML accuracy : 0.9702
DL macro F1 : 0.9781
ML macro F1 : 0.9700
DL gain     : 0.0080 accuracy = about 80 extra correct predictions
```

Also mention:

```text
DL wins on most classes.
ML is still strong.
The comparison is fair because both use train.csv and test.csv.
Confusion matrices explain class-level behavior.
```

### 7.4 What Each Member Should Be Ready To Explain

At least one member should explain the ML pipeline:

```text
CSV loading
PNG reading
grayscale conversion
resize
normalization
flattening to 784 features
Random Forest
accuracy and confusion matrix
```

At least one member should explain the DL architecture:

```text
Normalize
Conv2D
ReLU
MaxPool
Flatten
Dense
Softmax
CrossEntropy
Backpropagation
SGD
```

At least one member should explain the practical runtime:

```text
CORE
VIEW
HANDLER
OPERATOR
MODEL
APP
CUDA
cuDNN
cuBLASLt
checkpoints
```

At least one member should explain the comparison:

```text
same train/test split
confusion matrices
accuracy
macro F1
per-class recall
about 80 extra correct DL predictions
```

### 7.5 Final Submission Sentence

Use this as the final defense sentence:

```text
Our project is one digit-classification system evaluated in two ways: the
Machine Learning part proves a complete Random Forest baseline with
preprocessing, feature engineering, and metrics; the Practical IA / Deep
Learning part proves a custom CUDA CNN with mathematical architecture, training,
optimization, checkpointing, and a fair comparison against the ML baseline.
```

---

> **Continue Reading**  
> Previous: [Proof And Video](proof_video.md) | Next: [Conclusion](conclusion.md)
