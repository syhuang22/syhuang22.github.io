---
title: ML Foundamentals - Part 3
date: 2025-10-26 10:00
categories: [ML]
tags: [backpropagation]
author: James Huang
---


<!-- ✅ MathJax setup -->
<script>
window.MathJax = {
 tex: {
   inlineMath: [['$', '$'], ['\\(', '\\)']],
   displayMath: [['$$', '$$'], ['\\[', '\\]']],
   processEscapes: true,
 },
 options: {
   skipHtmlTags: ['script', 'noscript', 'style', 'textarea', 'pre'],
 }
};
</script>
<script type="text/javascript" async
 src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js">
</script>


## Backpropagation and Optimizers

Today’s topic focuses on **how gradients flow backward** through a neural network — the heart of how learning happens.


## 1. Setup

We start from a fully connected (dense) layer:

$$
z = xW + b
$$

### Variable Definitions

| Symbol | Description | Shape |
|---------|--------------|--------|
| $x$ | Input vector (one sample) | $[1, n_{in}]$ |
| $W$ | Weight matrix connecting inputs to outputs | $[n_{in}, n_{out}]$ |
| $b$ | Bias vector added to each output neuron | $[1, n_{out}]$ |
| $z$ | Linear (pre-activation) output | $[1, n_{out}]$ |
| $y = f(z)$ | Activated output after applying the nonlinearity | $[1, n_{out}]$ |
| $\mathcal{L}(y)$ | Scalar loss computed from model predictions | scalar |

If we process a batch of samples with batch size $m$:

| Symbol | Description | Shape |
|---------|--------------|--------|
| $X$ | Input matrix (batch of samples) | $[m, n_{in}]$ |
| $Z$ | Pre-activation outputs for all samples | $[m, n_{out}]$ |
| $dZ = \frac{\partial \mathcal{L}}{\partial Z}$ | Upstream gradient received from the next layer | $[m, n_{out}]$ |

---

## 2. Upstream Gradient

# Understanding $dZ$ (Upstream Gradient)

## 1. What $dZ$ Represents

In backpropagation, each layer receives a gradient signal from the layer **after it** —  
this is the **upstream gradient**, denoted as $dZ$ (or sometimes $δ$).

Mathematically, it’s:

$$
dZ = \frac{\partial \mathcal{L}}{\partial Z}
$$

It tells the current layer:
> “How much does the loss change if the layer’s pre-activation output $Z$ changes?”

So $dZ$ measures **how sensitive the loss is** to each element of the pre-activation output.

---

## 2. Where $dZ$ Comes From

Let’s consider the flow of computation in a neural network:

$$
X \xrightarrow[]{W,b} Z = XW + b \xrightarrow[]{f(\cdot)} A = f(Z) \xrightarrow[]{\text{next layers}} \mathcal{L}
$$

During backprop:

1. The **next layer** computes its own gradient w.r.t. its input (say $dA$).
2. The current layer receives that gradient and transforms it into $dZ$:

   $$
   dZ = dA \odot f'(Z)
   $$

   where $\odot$ is element-wise multiplication, and $f'(Z)$ is the derivative of the activation.

   - Example: For **ReLU**, $f'(Z)$ is `1` when $Z > 0` and `0` otherwise.
   - So $dZ$ is simply $dA$ “masked” by where the neuron was active.


## 3. Gradient with Respect to Weights

We know that

$$
z = xW + b
$$

To find how the loss changes with each weight, apply the chain rule:

$$
\frac{\partial \mathcal{L}}{\partial W}
= 
\frac{\partial \mathcal{L}}{\partial z}
\cdot
\frac{\partial z}{\partial W}
$$

Since each element of $z$ depends linearly on the corresponding elements of $W$ through $x$,  
the derivative of $z$ with respect to $W$ is the input itself: $\frac{\partial z}{\partial W} = x^\top$

$$
dW = x^\top dZ
$$

**Interpretation:**  
Each weight connects one input neuron to one output neuron.  
The gradient tells us how much changing that connection would affect the loss.

**Shape check:**
- $x$: $[1, n_{in}]$  
- $dZ$: $[1, n_{out}]$  
- $x^\top dZ$: $[n_{in}, n_{out}]$ → same shape as $W$

**Batch version:**

If we have multiple samples stacked in `X`:

$$
dW = X^\top dZ
$$

This sums contributions from all samples.

---

## 4. Gradient with Respect to Input

We also need to send gradients backward to the previous layer:

$$
\frac{\partial \mathcal{L}}{\partial x} =
\frac{\partial \mathcal{L}}{\partial z}
\cdot
\frac{\partial z}{\partial x}
$$

Since each element of $z$ is a weighted sum of the inputs,  
the derivative of $z$ with respect to $x$ is the weight matrix itself:
$$
dX = dZ W^\top
$$

**Interpretation:**  
Each input receives a gradient equal to the sum of all output gradients weighted by their corresponding connection strengths.  
This is what we pass back to the previous layer.

**Shape check:**
- $dZ$: $[1, n_{out}]$  
- $W^\top$: $[n_{out}, n_{in}]$  
- $dX$: $[1, n_{in}]$ → same shape as the input

---

## 5. Gradient with Respect to Bias

Bias contributes to each neuron linearly:

$$
z_j = \sum_i x_i W_{ij} + b_j
$$

and

$$
\frac{\partial z_j}{\partial b_j} = 1
$$

Thus,

$$
\frac{\partial \mathcal{L}}{\partial b} = \frac{\partial \mathcal{L}}{\partial z} = dZ
$$

**Interpretation:**  
Each bias term gets the same gradient as its neuron’s upstream gradient.

**Batch version:**

Since biases are shared across samples:

$$
dB = \sum_{\text{batch}} dZ
$$

**Shape check:**  
$b$: $[1, n_{out}]$  
$dB$: $[1, n_{out}]$

### Intuitive Meaning

- The upstream gradient tells each layer *“how much the output affected the loss.”*  
- Multiplying by inputs or weights redistributes that influence backward through the network.
- Gradients flow **in reverse** but follow the **same connections** used in the forward pass.

### Code Example of full connected layer with bactch inputs 
```python
import numpy as np
# Batch of 4 samples, each with 3 features
X = np.random.randn(4, 3)
W = np.random.randn(3, 2)
b = np.zeros((1, 2))

# Forward
Z = np.dot(X, W) + b

# Upstream gradient from next layer
dZ = np.random.randn(4, 2)

# Backward
dW = np.dot(X.T, dZ)
dB = np.sum(dZ, axis=0, keepdims=True)
dX = np.dot(dZ, W.T)

print("dW shape:", dW.shape)  # (3, 2)
print("dB shape:", dB.shape)  # (1, 2)
print("dX shape:", dX.shape)  # (4, 3)
```

## Optimizers in Deep Learning

After computing gradients through **backpropagation**,  
the next question is: *how do we actually update the weights efficiently and reliably?*  
That’s the job of **optimizers**.

### The Goal

Training a neural network means minimizing a loss function:
$$
\mathcal{L}(\mathbf{W})
$$
We use **gradient descent** to find parameters that make the loss as small as possible:
$$
\mathbf{W} \leftarrow \mathbf{W} - \eta \nabla_\mathbf{W}\mathcal{L}
$$

where:
- $$\eta$$ = learning rate (step size)  
- $$\nabla_\mathbf{W}\mathcal{L}$$ = gradient from backpropagation

This is the simplest optimizer — **Stochastic Gradient Descent (SGD)**.


### Problem with Basic SGD

While conceptually simple, SGD can be:
- **Slow** to converge (especially in deep networks)
- **Noisy**, since each batch may point in a slightly different direction
- **Unstable**, if gradients vary wildly in magnitude

To fix this, modern optimizers introduce *momentum, adaptive learning rates,* or both.

### Momentum

**Momentum** is one of the earliest and most effective improvements to basic gradient descent.  
The idea comes directly from physics — imagine a ball rolling down a hill.  
Instead of moving purely based on the current slope, the ball **builds up velocity** from past gradients, allowing it to roll faster through shallow regions and smooth out oscillations.

---

#### 1. Intuition

In standard gradient descent, each parameter update depends only on the **current gradient**:

$$
w_{t+1} = w_t - \eta \nabla_w \mathcal{L}_t
$$

where:
- $w_t$ = current parameter,
- $\eta$ = learning rate,
- $\nabla_w \mathcal{L}_t$ = gradient at step $t$.

This can lead to zig-zagging, especially in regions where the gradient changes direction rapidly (common in deep networks).

Momentum solves this by introducing a **velocity** term $v_t$ that accumulates past gradients:

$$
v_t = \beta v_{t-1} + (1 - \beta)\nabla_w \mathcal{L}_t
$$

$$
w_{t+1} = w_t - \eta v_t
$$

Here:
- $v_t$ is an **exponentially weighted moving average** of past gradients.  
- $\beta$ (typically around 0.9) controls how much past momentum to keep.

This lets the optimizer “remember” previous directions and continue moving smoothly toward a minimum.

---

#### 2. Intuitive Analogy

Think of a marble rolling down a valley:
- Without momentum, it takes tiny steps and wobbles side to side.
- With momentum, it gains **inertia**, moving faster in the consistent downhill direction and ignoring small oscillations.

This makes convergence faster and more stable.

---

#### 3. Code Example - Momentum

Below is a simple implementation of gradient descent **with** and **without** momentum.

```python
import numpy as np

# Simulated gradient sequence (toy example)
grads = np.array([0.9, 0.5, 0.2, -0.1, -0.3, -0.4, -0.2])

# Hyperparameters
eta = 0.1       # learning rate
beta = 0.9      # momentum coefficient

# Without momentum
w_gd = 0.0
for g in grads:
    w_gd -= eta * g
print("Final parameter (no momentum):", w_gd)

# With momentum
w_mom = 0.0
v = 0.0
for g in grads:
    v = beta * v + (1 - beta) * g
    w_mom -= eta * v
print("Final parameter (with momentum):", w_mom)
```



### Common Optimizers

| Optimizer | Formula | Key Idea | Pros | Cons |
|------------|----------|----------|------|------|
| **SGD (Vanilla)** | $$\mathbf{W} \leftarrow \mathbf{W} - \eta \nabla_\mathbf{W}\mathcal{L}$$ | Simple gradient step | Easy to implement | Can oscillate, slow convergence |
| **SGD + Momentum** | $$\begin{aligned} v_t &= \beta v_{t-1} + (1-\beta)\nabla_\mathbf{W}\mathcal{L}_t \\ \mathbf{W} &\leftarrow \mathbf{W} - \eta v_t \end{aligned}$$ | Adds velocity term to smooth gradients | Faster convergence, less noise | May overshoot minima |
| **Nesterov Momentum** | $$v_t = \beta v_{t-1} + (1-\beta)\nabla_\mathbf{W}\mathcal{L}(\mathbf{W}-\eta\beta v_{t-1})$$ | “Look ahead” before applying momentum | More accurate correction | Slightly more computation |
| **AdaGrad** | $$\mathbf{W} \leftarrow \mathbf{W} - \frac{\eta}{\sqrt{G_t+\epsilon}} \nabla_\mathbf{W}\mathcal{L}$$ | Scales learning rate by past gradient magnitude | Great for sparse data | Learning rate decays too fast |
| **RMSProp** | $$\begin{aligned} E[g^2]_t &= \beta E[g^2]_{t-1} + (1-\beta)(\nabla_\mathbf{W}\mathcal{L})^2 \\ \mathbf{W} &\leftarrow \mathbf{W} - \frac{\eta}{\sqrt{E[g^2]_t + \epsilon}}\nabla_\mathbf{W}\mathcal{L} \end{aligned}$$ | Keeps moving average of squared gradients | Works well for non-stationary loss | Sensitive to hyperparameters |
| **Adam** | $$\begin{aligned} m_t &= \beta_1 m_{t-1} + (1-\beta_1)\nabla_\mathbf{W}\mathcal{L}_t \\ v_t &= \beta_2 v_{t-1} + (1-\beta_2)(\nabla_\mathbf{W}\mathcal{L}_t)^2 \\ \hat{m}_t &= \frac{m_t}{1-\beta_1^t}, \ \hat{v}_t = \frac{v_t}{1-\beta_2^t} \\ \mathbf{W} &\leftarrow \mathbf{W} - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t}+\epsilon} \end{aligned}$$ | Combines momentum + adaptive scaling | Fast, robust, widely used | Can overfit or “stall” near minima |

---

## All Together: Building a Complete Neural Network from Scratch

In this section, we combine every concept we’ve learned so far into a full end-to-end neural network implementation using only NumPy.  

This network includes:

- **Multiple fully connected (dense) layers** for transforming inputs linearly  
- **ReLU activation** for introducing nonlinearity and enabling deep learning  
- **Softmax activation** for converting raw scores into probabilities  
- **Categorical Cross-Entropy loss** for measuring how well predictions match labels  
- **Backpropagation** for computing gradients across all layers  
- **Batch input support** for efficient parallel computation  
- **Momentum-based updates** for faster, smoother convergence

By connecting all these components, we can forward-propagate data, compute a loss, backpropagate errors, and iteratively update weights to train a small neural network — the foundation of modern deep learning systems.


```python
import numpy as np

# -----------------------------
# Layer: Fully Connected (Dense)
# -----------------------------
class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        # He initialization for weights
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
        # For momentum
        self.weight_momentums = np.zeros_like(self.weights)
        self.bias_momentums = np.zeros_like(self.biases)

    def forward(self, inputs):
        # Store inputs for backprop
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward(self, dvalues):
        # Gradients with respect to weights, biases, and inputs
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        self.dinputs = np.dot(dvalues, self.weights.T)


# -----------------------------
# Activation: ReLU
# -----------------------------
class Activation_ReLU:
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0


# -----------------------------
# Activation: Softmax
# -----------------------------
class Activation_Softmax:
    def forward(self, inputs):
        # Numerically stable softmax
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities

    def backward(self, dvalues):
        # Placeholder (we’ll handle it combined with loss)
        self.dinputs = dvalues


# -----------------------------
# Loss: Categorical Cross-Entropy
# -----------------------------
class Loss_CategoricalCrossentropy:
    def forward(self, y_pred, y_true):
        # Clip predictions to avoid log(0)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
        # Probabilities for target values
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[
                range(len(y_pred_clipped)), y_true
            ]
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods

    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        labels = dvalues.shape[1]

        # If labels are sparse, convert to one-hot
        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]

        self.dinputs = -y_true / dvalues
        self.dinputs = self.dinputs / samples


# -----------------------------
# Combined Softmax + Cross-Entropy
# (for more efficient backward pass)
# -----------------------------
class Activation_Softmax_Loss_CategoricalCrossentropy:
    def __init__(self):
        self.activation = Activation_Softmax()
        self.loss = Loss_CategoricalCrossentropy()

    def forward(self, inputs, y_true):
        self.activation.forward(inputs)
        self.output = self.activation.output
        return np.mean(self.loss.forward(self.output, y_true))

    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)
        self.dinputs = dvalues.copy()
        self.dinputs[range(samples), y_true] -= 1
        self.dinputs = self.dinputs / samples


# -----------------------------
# Optimizer: SGD with Momentum
# -----------------------------
class Optimizer_SGD:
    def __init__(self, learning_rate=1.0, momentum=0.9):
        self.learning_rate = learning_rate
        self.momentum = momentum

    def update_params(self, layer):
        # Momentum update
        layer.weight_momentums = self.momentum * layer.weight_momentums - \
            self.learning_rate * layer.dweights
        layer.bias_momentums = self.momentum * layer.bias_momentums - \
            self.learning_rate * layer.dbiases

        # Apply updates
        layer.weights += layer.weight_momentums
        layer.biases += layer.bias_momentums


# -----------------------------
# Example: Network Training
# -----------------------------
if __name__ == "__main__":
    np.random.seed(42)

    # Synthetic data (batch of 5, 2 features, 3 classes)
    X = np.random.randn(5, 2)
    y = np.array([0, 1, 2, 1, 0])

    # Define network
    dense1 = Layer_Dense(2, 64)
    activation1 = Activation_ReLU()
    dense2 = Layer_Dense(64, 3)
    loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()
    optimizer = Optimizer_SGD(learning_rate=0.1, momentum=0.9)

    # Training loop
    for epoch in range(1001):
        # Forward pass
        dense1.forward(X)
        activation1.forward(dense1.output)
        dense2.forward(activation1.output)

        # Compute loss
        loss = loss_activation.forward(dense2.output, y)

        # Accuracy
        predictions = np.argmax(loss_activation.output, axis=1)
        accuracy = np.mean(predictions == y)

        # Backward pass
        loss_activation.backward(loss_activation.output, y)
        dense2.backward(loss_activation.dinputs)
        activation1.backward(dense2.dinputs)
        dense1.backward(activation1.dinputs)

        # Update parameters
        optimizer.update_params(dense1)
        optimizer.update_params(dense2)

        # Print progress
        if epoch % 100 == 0:
            print(f"epoch {epoch:4d} | loss: {loss:.3f} | acc: {accuracy:.3f}")
```
## What's Coming
Now that we’ve built and understood a complete neural network from the ground up.
We’ll move beyond these fundamentals and step into the architectures that power today’s large language models (LLMs):  
the **Transformer** — a model built not on dense layers alone, but on the powerful concepts of **attention**, **context**, and **parallelism**.

From here, we’ll start exploring how Transformers process sequences, learn long-range dependencies, and ultimately become the foundation of GPT, BERT, and other state-of-the-art models.



