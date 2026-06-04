# Layer Mathematics And Model Process

> **Reading Path**  
> Home: [Project Manual](../index.md) | Section: [MODEL](index.md) | Previous: [MODEL::DL](dl.md) | Next: [Conclusion](../conclusion.md)

This document explains the model from `layers.txt` as a mathematical and
conceptual system. It is not a code walkthrough. It is the layer-by-layer story
of what the model computes, why each stage exists, how tensors flow through the
network, how gradients flow backward, and how the final comparison chart should
be interpreted.

The model is a compact convolutional neural network for digit classification.
It learns from MNIST-style images and compares its result against a Random
Forest baseline.

## Chapter Index

| Chapter | Topic |
|---:|---|
| 1 | Model Purpose |
| 2 | Symbols And Tensor Shapes |
| 3 | Batch Size And Dataset Split |
| 4 | Full Forward Graph |
| 5 | Layer 0: Image Input And Normalization |
| 6 | Layer 1: Convolution |
| 7 | Layer 1 Activation: ReLU |
| 8 | Layer 1.5: Max Pooling |
| 9 | Layer 2: Flattening |
| 10 | Layer 2 Dense Projection |
| 11 | Layer 2 Activation: ReLU |
| 12 | Layer 3 Dense Projection |
| 13 | Layer 3 Softmax |
| 14 | Loss Function: Cross Entropy |
| 15 | Backward Pass Overview |
| 16 | Backward Layer 3 |
| 17 | Backward Layer 2 |
| 18 | Backward Pooling |
| 19 | Backward Convolution |
| 20 | SGD Update |
| 21 | CUDA And Numerical Design |
| 22 | Checkpoints And Training Continuation |
| 23 | Confusion Matrix Mathematics |
| 24 | Comparison Chart Explanation |
| 25 | Final Interpretation |

---

## Chapter 1: Model Purpose

The model receives a batch of grayscale digit images and predicts one digit
class for each image.

The conceptual task is:

```text
image -> digit class
```

For MNIST, the real classes are:

```text
0, 1, 2, 3, 4, 5, 6, 7, 8, 9
```

The model computes a probability distribution over classes. The predicted class
is the class with the highest probability.

Mathematically:

```text
prediction = argmax_k probability[k]
```

The network does not memorize a text label. It learns numerical parameters:

```text
kerW0, b0, W1, b1, W2, b2
```

Those parameters are adjusted through backpropagation and SGD.

The high-level model is:

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

The high-level training rule is:

```text
forward pass
  -> compute loss
  -> backward pass
  -> update weights
```

The model is intentionally small. Its importance is not size. Its importance is
that the full system is custom: tensor views, memory handlers, CUDA kernels,
cuDNN/cuBLASLt operators, checkpoints, and comparison reports all work
together.

---

## Chapter 2: Symbols And Tensor Shapes

The network is easiest to understand if the symbols are fixed.

| Symbol | Meaning |
|---|---|
| `B` | Batch size. Current app uses `B = 64`. |
| `C` | Input channels. MNIST grayscale uses `C = 1`. |
| `H` | Input image height. MNIST uses `H = 28`. |
| `W` | Input image width. MNIST uses `W = 28`. |
| `F` | Number of convolution filters. Current model uses `F = 16`. |
| `K` | Convolution kernel size. Current model uses `K = 3`. |
| `P` | Pool window size. Current model uses `P = 2`. |
| `M` | Hidden dense size. Current model uses `M = 128`. |
| `R` | Real digit classes. Current task uses `R = 10`. |
| `Q` | Physical softmax columns. Current model uses `Q = 16`. |

The model uses 16 output columns even though only 10 classes are real. The
extra columns are not real digit labels. They support vectorized memory and
loss logic.

For MNIST:

```text
B = 64
C = 1
H = 28
W = 28
F = 16
K = 3
P = 2
M = 128
R = 10
Q = 16
```

The main forward tensor shapes are:

| Tensor | Shape | Meaning |
|---|---|---|
| `image` | `(B, 1, H, W)` | Raw grayscale image batch. |
| `x` | `(B, 1, H, W)` | Normalized image batch. |
| `kerW0` | `(16, 1, 3, 3)` | Convolution filters. |
| `b0` | `(16)` | Convolution bias. |
| `z0` | `(B, 16, H, W)` | Convolution output before ReLU. |
| `A0` | `(B, 16, H, W)` | Convolution activation after ReLU. |
| `p` | `(B, 16, H/2, W/2)` | Max-pooled activation. |
| `p_flat` | `(B, 16 * H/2 * W/2)` | Flattened pooled tensor. |
| `W1` | `(16 * H/2 * W/2, 128)` | First dense weight. |
| `b1` | `(128)` | First dense bias. |
| `z1` | `(B, 128)` | First dense output before ReLU. |
| `A1` | `(B, 128)` | First dense activation after ReLU. |
| `W2` | `(128, 16)` | Final dense weight. |
| `b2` | `(16)` | Final dense bias. |
| `z2` | `(B, 16)` | Final logits. |
| `out` | `(B, 16)` | Softmax probabilities. |
| `Y` | `(B)` conceptually or `(B, 16)` one-hot conceptually | True labels. |

For MNIST, after pooling:

```text
H/2 = 14
W/2 = 14
flat = 16 * 14 * 14 = 3136
```

So:

```text
p_flat shape = (64, 3136)
W1 shape     = (3136, 128)
W2 shape     = (128, 16)
```

---

## Chapter 3: Batch Size And Dataset Split

The app creates:

```text
MODEL::DL dl(64)
```

So the normal training batch size is:

```text
B = 64
```

A batch is a group of images processed together. Instead of updating weights
after one image, the model computes an average gradient over 64 images.

One update step is:

```text
one mini-batch forward pass
one mini-batch loss
one mini-batch backward pass
one SGD update
```

The train dataset has 60,000 images. With batch size 64:

```text
steps per epoch = 60000 / 64 ~= 938
```

An epoch means one approximate pass over the training set.

The project uses an honest split:

```text
train.csv -> used for learning
test.csv  -> used for final benchmark
```

That distinction matters. Training accuracy can be inflated by memorization.
Test accuracy is the real proof that the model generalizes.

The CSV rows are shuffled before image loading. This is mathematically
important because mini-batch SGD assumes each batch gives a reasonable sample
of the full distribution. If the data is sorted by class, a batch can contain
mostly one digit, and the gradient becomes biased toward that temporary class
block.

---

## Chapter 4: Full Forward Graph

From `layers.txt`, the forward graph is:

```text
layer 0
  image(B, 1, H, W)
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

loss
  LossFunc(out, Y)
```

The model gradually transforms raw pixels into class evidence.

The early layers preserve spatial structure:

```text
image -> convolution -> pooling
```

The later layers combine learned features:

```text
flatten -> dense -> dense -> softmax
```

This is a classic CNN idea:

```text
local patterns first
global class decision later
```

---

## Chapter 5: Layer 0 - Image Input And Normalization

### Purpose

The input image is a grayscale digit. Each pixel is an integer-like value:

```text
0..255
```

The model normalizes it:

```text
x = image / 255
```

So the pixel range becomes:

```text
0..1
```

### Shape

```text
image shape = (B, 1, H, W)
x shape     = (B, 1, H, W)
```

For the current app:

```text
image shape = (64, 1, 28, 28)
x shape     = (64, 1, 28, 28)
```

### Mathematical Formula

For every batch item `n`, channel `c`, row `h`, and column `w`:

```text
x[n, c, h, w] = image[n, c, h, w] / 255
```

### Why It Matters

Normalization makes the optimization problem smoother. If pixels remain in
`0..255`, early activations can become too large, and gradients can become less
stable. Normalization does not create new information; it gives the same
information a better numerical scale.

### Backward Concept

If the model needed a gradient with respect to the raw image:

```text
dimage = dx / 255
```

In classification training, the image itself is not a trainable parameter, so
this gradient is conceptually useful but not used for parameter updates.

---

## Chapter 6: Layer 1 - Convolution

### Purpose

Convolution learns local visual detectors. A 3x3 filter can detect small
patterns such as:

```text
short vertical strokes
horizontal strokes
corners
curves
small gaps
```

Digits are made from these local visual structures. The model learns filters
that become useful for recognizing them.

### Shape

```text
x shape     = (B, 1, H, W)
kerW0 shape = (16, 1, 3, 3)
b0 shape    = (16)
z0 shape    = (B, 16, H, W)
```

The output height and width remain `H` and `W` because the convolution uses
padding of 1 with a 3x3 kernel.

For MNIST:

```text
x shape  = (64, 1, 28, 28)
z0 shape = (64, 16, 28, 28)
```

### Mathematical Formula

Let:

```text
n  = batch index
f  = filter index
c  = channel index
i  = output row
j  = output column
u  = kernel row offset
v  = kernel column offset
pad = 1
```

Then:

```text
z0[n, f, i, j] =
  b0[f]
  + sum over c, u, v of
      x[n, c, i + u - pad, j + v - pad] * kerW0[f, c, u, v]
```

Out-of-bound positions created by padding are treated as zero.

### Conceptual Meaning

Each filter is a learned question:

```text
Does this local 3x3 region look like the pattern I learned?
```

The filter is applied everywhere. This is called weight sharing. It means the
same pattern can be detected whether it appears near the top, bottom, left, or
right of the digit.

### Why 16 Filters

The model uses 16 filters. That gives it 16 different local detectors at the
first convolution stage.

This is small enough to keep the model compact and fast, but large enough to
learn multiple stroke types.

---

## Chapter 7: Layer 1 Activation - ReLU

### Purpose

After convolution, the model applies:

```text
A0 = ReLU(z0)
```

ReLU means:

```text
ReLU(t) = max(0, t)
```

### Shape

```text
z0 shape = (B, 16, H, W)
A0 shape = (B, 16, H, W)
```

### Mathematical Formula

For every element:

```text
A0[n, f, i, j] = max(0, z0[n, f, i, j])
```

### Conceptual Meaning

The convolution creates positive and negative evidence. ReLU keeps the positive
evidence and suppresses negative values.

Without nonlinear activation, a stack of linear layers would behave like one
larger linear layer. ReLU lets the network create nonlinear decision surfaces.

### Backward Formula

The derivative of ReLU is:

```text
ReLU'(t) = 1 if t > 0
ReLU'(t) = 0 if t <= 0
```

So:

```text
dz0 = dA0 .* ReLU'(z0)
```

The symbol `.*` means elementwise multiplication.

---

## Chapter 8: Layer 1.5 - Max Pooling

### Purpose

Max pooling reduces spatial size while keeping strong local evidence.

The model uses:

```text
2x2 max pooling
stride 2
```

This converts a 28x28 map into a 14x14 map.

### Shape

```text
A0 shape = (B, 16, H, W)
p shape  = (B, 16, H/2, W/2)
```

For MNIST:

```text
A0 shape = (64, 16, 28, 28)
p shape  = (64, 16, 14, 14)
```

### Mathematical Formula

For pool row `i` and pool column `j`:

```text
p[n, f, i, j] =
  max over a in {0, 1}, b in {0, 1} of
    A0[n, f, 2*i + a, 2*j + b]
```

### Conceptual Meaning

Pooling says:

```text
If a feature appears strongly somewhere in this small region, keep it.
```

It gives the model tolerance to small shifts. A handwritten stroke does not
need to land in exactly the same pixel location every time.

### Backward Concept

The gradient goes only to the element that won the max operation:

```text
dA0 receives dp at the max location
all other locations in the 2x2 region receive 0
```

This is not matrix multiplication. It is routing. The forward max selected a
winner; the backward pass sends gradient back to that winner.

---

## Chapter 9: Layer 2 - Flattening

### Purpose

After pooling, the tensor is still spatial:

```text
(B, 16, 14, 14)
```

A dense layer expects a vector per image. Flattening reinterprets the pooled
features as:

```text
(B, 3136)
```

### Shape

General:

```text
p shape      = (B, 16, H/2, W/2)
p_flat shape = (B, 16 * H/2 * W/2)
```

For MNIST:

```text
p_flat shape = (64, 3136)
```

### Mathematical Meaning

Flattening does not change the data values. It changes the indexing.

If:

```text
q = f * (H/2 * W/2) + i * (W/2) + j
```

then:

```text
p_flat[n, q] = p[n, f, i, j]
```

Flattening is a bridge between convolutional feature maps and dense feature
combination.

---

## Chapter 10: Layer 2 Dense Projection

### Purpose

The first dense layer combines all pooled local features into a hidden feature
representation.

It computes:

```text
z1 = p_flat @ W1 + b1
```

### Shape

```text
p_flat shape = (B, 3136)
W1 shape     = (3136, 128)
b1 shape     = (128)
z1 shape     = (B, 128)
```

### Mathematical Formula

For image `n` and hidden unit `m`:

```text
z1[n, m] =
  b1[m]
  + sum over q of p_flat[n, q] * W1[q, m]
```

### Conceptual Meaning

Each hidden unit is a learned combination of pooled convolution features. It can
learn ideas like:

```text
loop-like evidence
top stroke plus right curve
vertical stroke and lower hook
absence of a center gap
```

These are not hand-coded. They emerge from training.

---

## Chapter 11: Layer 2 Activation - ReLU

### Purpose

The model applies:

```text
A1 = ReLU(z1)
```

This introduces nonlinearity after the first dense projection.

### Shape

```text
z1 shape = (B, 128)
A1 shape = (B, 128)
```

### Mathematical Formula

```text
A1[n, m] = max(0, z1[n, m])
```

### Conceptual Meaning

The hidden layer creates many possible pieces of digit evidence. ReLU selects
which pieces are active for each image.

The same hidden unit can be active for one digit and inactive for another.

### Backward Formula

```text
dz1 = dA1 .* ReLU'(z1)
```

Again:

```text
ReLU'(z1) = 1 where z1 > 0
ReLU'(z1) = 0 where z1 <= 0
```

---

## Chapter 12: Layer 3 Dense Projection

### Purpose

The final dense layer maps hidden features to class logits.

It computes:

```text
z2 = A1 @ W2 + b2
```

### Shape

```text
A1 shape = (B, 128)
W2 shape = (128, 16)
b2 shape = (16)
z2 shape = (B, 16)
```

### Mathematical Formula

For image `n` and output column `k`:

```text
z2[n, k] =
  b2[k]
  + sum over m of A1[n, m] * W2[m, k]
```

### Conceptual Meaning

The logits are raw class evidence. They are not probabilities yet. A high logit
for class `k` means the model has strong evidence for class `k`.

Only classes `0..9` are real digits. Columns `10..15` exist for vectorized
layout and loss logic, not because the dataset has those classes.

---

## Chapter 13: Layer 3 Softmax

### Purpose

Softmax converts logits into probabilities.

```text
out = softmax(z2)
```

### Shape

```text
z2 shape  = (B, 16)
out shape = (B, 16)
```

### Mathematical Formula

For image `n` and output column `k`:

```text
out[n, k] = exp(z2[n, k]) / sum over j of exp(z2[n, j])
```

This guarantees:

```text
sum over k of out[n, k] = 1
```

### Prediction

The predicted digit is:

```text
pred[n] = argmax over k in {0..9} of out[n, k]
```

The comparison ignores physical columns `10..15` when choosing the real digit
answer.

### Conceptual Meaning

Softmax turns raw evidence into a distribution. It answers:

```text
How much probability does the model assign to each class?
```

---

## Chapter 14: Loss Function - Cross Entropy

### Purpose

The loss tells the model how wrong it is.

For a true class `y[n]`, cross entropy is:

```text
loss[n] = -log(out[n, y[n]])
```

For a batch:

```text
Loss = (1 / B) * sum over n of loss[n]
```

### One-Hot Target Concept

The label can be represented conceptually as a one-hot vector:

```text
Y[n, k] = 1 if k == y[n]
Y[n, k] = 0 otherwise
```

Then the loss can be written:

```text
Loss = -(1 / B) * sum over n, k of Y[n, k] * log(out[n, k])
```

### Gradient Concept

For softmax plus cross entropy, the output gradient simplifies:

```text
dz2 = (out - Y) / B
```

This is one of the most useful results in neural-network math. It means the
model does not need to separately compute a complicated softmax derivative for
the final layer. The probability error becomes the gradient.

---

## Chapter 15: Backward Pass Overview

The backward pass starts at the loss and moves in reverse order.

From `layers.txt`:

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

The backward pass is not a separate model. It is the same model read in reverse
through the chain rule.

The chain rule says:

```text
if loss depends on b, and b depends on a,
then loss depends on a through b
```

Symbolically:

```text
dLoss/da = dLoss/db * db/da
```

---

## Chapter 16: Backward Layer 3

### Output Error

The final layer receives:

```text
dz2 = (out - Y) / B
```

`dz2` has shape:

```text
(B, 16)
```

### Gradient For W2

Forward:

```text
z2 = A1 @ W2 + b2
```

Backward:

```text
dW2 = A1.T @ dz2
```

Shape check:

```text
A1.T shape = (128, B)
dz2 shape  = (B, 16)
dW2 shape  = (128, 16)
```

### Gradient For b2

Bias is added to every row, so the bias gradient sums over the batch:

```text
db2 = sum_rows(dz2)
```

Shape:

```text
db2 shape = (16)
```

### Gradient For A1

To pass error to the previous layer:

```text
dA1 = dz2 @ W2.T
```

Shape check:

```text
dz2 shape  = (B, 16)
W2.T shape = (16, 128)
dA1 shape  = (B, 128)
```

---

## Chapter 17: Backward Layer 2

### ReLU Backward

The dense hidden activation was:

```text
A1 = ReLU(z1)
```

So:

```text
dz1 = dA1 .* ReLU'(z1)
```

Shape:

```text
dz1 shape = (B, 128)
```

### Gradient For W1

Forward:

```text
z1 = p_flat @ W1 + b1
```

Backward:

```text
dW1 = p_flat.T @ dz1
```

Shape check:

```text
p_flat.T shape = (3136, B)
dz1 shape      = (B, 128)
dW1 shape      = (3136, 128)
```

### Gradient For b1

```text
db1 = sum_rows(dz1)
```

Shape:

```text
db1 shape = (128)
```

### Gradient For p_flat

```text
dp_flat = dz1 @ W1.T
```

Shape check:

```text
dz1 shape  = (B, 128)
W1.T shape = (128, 3136)
dp_flat    = (B, 3136)
```

### Reshape Back To Pool Shape

```text
dp = reshape(dp_flat, shape_of_p)
```

For MNIST:

```text
dp shape = (B, 16, 14, 14)
```

---

## Chapter 18: Backward Pooling

Forward max pooling was:

```text
p[n, f, i, j] =
  max over a, b of A0[n, f, 2*i + a, 2*j + b]
```

Backward max pooling sends the gradient only to the location that won the max.

If:

```text
(a_star, b_star) = argmax in the 2x2 region
```

then:

```text
dA0[n, f, 2*i + a_star, 2*j + b_star] += dp[n, f, i, j]
```

All other elements in that 2x2 region receive zero from that output.

This is why max pooling backward is not a normal matrix formula. It is a
selection-mask formula.

Conceptually:

```text
the strongest activation got credit in forward
the strongest activation receives responsibility in backward
```

---

## Chapter 19: Backward Convolution

### ReLU Backward

The first activation was:

```text
A0 = ReLU(z0)
```

So:

```text
dz0 = dA0 .* ReLU'(z0)
```

Shape:

```text
dz0 shape = (B, 16, H, W)
```

### Gradient For Convolution Filters

Forward convolution used:

```text
z0[n, f, i, j] =
  b0[f] + sum x[...] * kerW0[f, c, u, v]
```

The filter gradient asks:

```text
How much did this filter weight contribute to the loss?
```

For every filter weight:

```text
dkerW0[f, c, u, v] =
  sum over n, i, j of
    x[n, c, i + u - pad, j + v - pad] * dz0[n, f, i, j]
```

Shape:

```text
dkerW0 shape = (16, 1, 3, 3)
```

### Gradient For b0

The bias is shared over batch and spatial positions:

```text
db0[f] = sum over n, i, j of dz0[n, f, i, j]
```

Shape:

```text
db0 shape = (16)
```

### Gradient For x

The input gradient asks:

```text
How did each input pixel affect the loss through convolution?
```

Conceptually:

```text
dx = conv_backward_input(kerW0, dz0)
```

This is useful for continuing the chain rule. Since the raw image is not a
trainable parameter, the model does not update the image. It updates weights.

---

## Chapter 20: SGD Update

After gradients are computed, parameters are updated:

```text
kerW0 = kerW0 - lr * dkerW0
b0    = b0    - lr * db0
W1    = W1    - lr * dW1
b1    = b1    - lr * db1
W2    = W2    - lr * dW2
b2    = b2    - lr * db2
```

The learning rate `lr` controls how far the model moves.

Large `lr`:

```text
faster movement
higher risk of overshooting
```

Small `lr`:

```text
slower movement
better for refinement
```

The successful refinement schedule uses:

```text
lr = 0.0005
```

This makes sense after a strong checkpoint already exists. The model no longer
needs dramatic changes; it needs careful local improvement.

---

## Chapter 21: CUDA And Numerical Design

The project uses CUDA because the model is designed around explicit GPU work.

The operations are naturally parallel:

| Operation | Parallel Structure |
|---|---|
| Normalize | One operation per pixel. |
| ReLU | One operation per activation. |
| SGD | One operation per parameter. |
| Convolution | Many filter applications across batch and spatial positions. |
| Matrix multiply | Many dot products. |
| Softmax | One distribution per image. |

The project uses:

```text
custom CUDA kernels
cuDNN
cuBLASLt
```

The split is conceptual:

```text
small direct elementwise math -> custom kernels
large optimized primitives    -> cuDNN/cuBLASLt
```

The model uses F16 tensors for many values. F16 reduces memory bandwidth and
storage. Some operations accumulate in float where library calls support it,
which helps preserve numerical quality.

Vectorized kernels often process multiple F16 values together. This is why the
model uses 16 output columns even though the real digit classes are 10.

---

## Chapter 22: Checkpoints And Training Continuation

Checkpoints store trainable parameters:

```text
kerW0, b0, W1, b1, W2, b2
```

They are saved as:

```text
public/checkpoints/dl_iter_<iteration>.bin
```

The model can load the latest checkpoint and continue training.

Important concept:

```text
train(target_iteration, learning_rate, checkpoint_every)
```

The first value is a target iteration, not always the number of new iterations.

If:

```text
latest checkpoint = 30000
target iteration  = 60000
```

then:

```text
new updates = 60000 - 30000 = 30000
```

If:

```text
latest checkpoint = 60000
target iteration  = 60000
```

then:

```text
new updates = 0
```

The program benchmarks the checkpoint.

This is why results may not change if the target iteration is not increased
past the latest checkpoint.

---

## Chapter 23: Confusion Matrix Mathematics

A confusion matrix counts predictions.

Rows are true classes:

```text
row = actual digit
```

Columns are predicted classes:

```text
column = predicted digit
```

A diagonal element means correct classification:

```text
matrix[k, k]
```

An off-diagonal element means confusion:

```text
matrix[true_class, predicted_class]
```

Accuracy is:

```text
accuracy = sum diagonal / sum all entries
```

Recall for class `k` is:

```text
recall(k) = matrix[k, k] / sum over j of matrix[k, j]
```

Precision for class `k` is:

```text
precision(k) = matrix[k, k] / sum over i of matrix[i, k]
```

F1 for class `k` is:

```text
F1(k) = 2 * precision(k) * recall(k) / (precision(k) + recall(k))
```

Macro F1 is:

```text
macro_F1 = average over classes of F1(k)
```

Macro F1 matters because it gives each class equal importance.

---

## Chapter 24: Comparison Chart Explanation

The comparison chart has four panels.

### Panel 1: DL Confusion Matrix

This shows the CUDA CNN predictions on the test set.

Strong diagonal:

```text
model is correctly classifying most digits
```

Light off-diagonal cells:

```text
few class confusions
```

The title includes DL accuracy.

### Panel 2: ML Confusion Matrix

This shows the Random Forest baseline predictions on the same test set.

It is the same kind of evidence as the DL matrix, so the comparison is fair.

### Panel 3: Per-Class Recall

This bar chart compares recall for each digit.

If DL is higher for a class:

```text
DL recovered more real examples of that digit
```

If ML is higher:

```text
Random Forest recovered more real examples of that digit
```

This chart is more informative than accuracy alone because it shows where each
model is strong.

### Panel 4: DL Training Progress

This panel shows DL batch loss and DL batch accuracy from training logs.

Important: batch accuracy is not final test accuracy. It is the accuracy on the
mini-batch that was logged at that iteration.

Batch curves can be jagged because each batch is a different sample of the
training set.

The final proof is the test confusion matrix, not one training batch point.

---

## Chapter 25: Final Interpretation

The model process is coherent:

```text
normalize pixels
learn local filters
activate useful evidence
pool spatial evidence
combine features
produce probabilities
measure cross entropy
backpropagate responsibility
update parameters
benchmark on held-out data
```

The DL model has the right mathematical structure for images. Convolution gives
it local pattern recognition. Pooling gives it spatial tolerance. Dense layers
combine features into class evidence. Cross entropy and SGD turn errors into
parameter updates.

The Random Forest baseline is strong, but it sees the image as a flat vector.
The CNN sees spatial structure. That is the architectural reason the DL model
can exceed the baseline.

The final result is therefore more than a number. It is proof that the full
pipeline works:

```text
dataset
  -> tensors
  -> CUDA operators
  -> trainable CNN
  -> checkpoint continuation
  -> held-out benchmark
  -> comparison report
```

That is the process captured by `layers.txt`, implemented by `MODEL::DL`, and
measured by `comparaison.py`.

---

> **End Of Layer Guide**  
> Previous: [MODEL::DL](dl.md) | Next: [Conclusion](../conclusion.md)
