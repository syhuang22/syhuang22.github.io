---
title: ML Foundamentals - Part 1
date: 2025-10-11 10:50
categories: [ML]
tags: [Neuron, Dot Product, Activation Functions, Loss]
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


In this post, we’ll peel back the curtain on the core building blocks of neural networks.


- **Neuron**: how a single neuron (or “node”) aggregates inputs and produces an output 
- **Dot Product**: the mathematical engine that lets inputs and weights interact 
- **Activation Functions**: how we inject non‑linearity so our networks can learn complex patterns 
- **Loss**: how we measure how well (or poorly) our model is doing, and how that drives learning 


## Neurons


A **neuron** (also called a “node” or “unit”) is the fundamental building block of a neural network. Here’s how it works, in simple terms:


### Inputs and Weights


A **neuron** takes in inputs, applies weights, adds a bias, and produces an output. Let’s break this down:


- A neuron receives **inputs** — these could be features from your data, like (x₁, x₂, x₃, …).
- Each input has a **weight** — (w₁, w₂, w₃, …) — telling the neuron how important that input is.
- The neuron multiplies each input by its weight and adds them all together.
- We usually also add a **bias** (b), which helps shift the result.


### Dot Products: Why They're Perfect for Neurons


Why use dot products?


- It’s the **simplest way** to combine multiple inputs with learned importance (weights)
- It’s **efficient** — dot products are fast and optimized in hardware (especially GPUs)
- It’s **differentiable** — so it fits perfectly with gradient descent and backpropagation


But the real reason is: they let each neuron carve out a **linear decision boundary** in the input space — the basic building block of pattern recognition.


### Code Example: Single Neuron


```python
inputs = [1.2, 2.5, -1.0]
weights = [0.8, -0.5, 1.0]
bias = 2.0


output = sum(x * w for x, w in zip(inputs, weights)) + bias
print("Output:", output) # Output: 0.71
```
### Why Neurons Are Useful


- Neurons let us model **complex relationships** by combining input features in flexible ways. 
- By stacking many neurons into **layers**, a neural network can learn hierarchical patterns (e.g. edges → shapes → objects). 


## Layers


So far, we’ve seen how a **single neuron** takes in inputs, applies weights and bias, and produces a value.
That’s powerful — but it can only learn one pattern at a time. Imagine you’re trying to classify an image: is it a cat or a dog?
You give a neuron some inputs — say pixel values or features. It can learn a **single weighted combination** of those features — basically drawing a straight line (or hyperplane) to separate "cat" from "dog", but real-world problems are rarely so simple.


What if the data isn’t linearly separable?
What if multiple patterns are needed to make a good decision?


### The Intuition: What If We Used Multiple Neurons?


Instead of relying on a single neuron, what if we had **multiple neurons**, each learning a different pattern?


- Neuron 1 learns pattern A (e.g. edge in top-left)
- Neuron 2 learns pattern B (e.g. circle in center)
- Neuron 3 learns pattern C (e.g. texture difference)


This idea — multiple neurons working together — leads us naturally to:


### A Layer


A **layer** is simply a group of neurons that:


- All take the same input
- Each have their own weights and bias
- Each produce their own output


We use layers to extract **multiple features in parallel**.


This unlocks **representation power**: instead of learning one decision boundary, we learn many, and combine them to make smarter choices.


### Scaling Up: Layers in Matrix Form


When we compute multiple neurons (a layer) at once, we can use matrix operations:


$$
\mathbf{z} = W \mathbf{x} + \mathbf{b}
$$


Where:
- $ W $ is the **weight matrix** (each row is a neuron’s weights) 
- $ \mathbf{x} $ is the input vector 
- $ \mathbf{b} $ is the bias vector 
- $ \mathbf{z} $ is the output vector — one value per neuron


This is just **many dot products computed in parallel**.


### Code Example
```python
import numpy as np


# Input vector (shape: [3])
inputs = np.array([1.2, 2.5, -1.0])


# Weight matrix (shape: [3 neurons, 3 inputs])
weights = np.array([
   [0.8, -0.5, 1.0],     # Neuron 1
   [0.1, 0.7, -1.2],     # Neuron 2
   [0.3, -0.3, 0.5]      # Neuron 3
])


# Bias vector (shape: [3])
biases = np.array([2.0, 0.5, -1.0])


# Layer output: vector of 3 outputs
layer_output = np.dot(weights, inputs) + biases
print("Layer output:", layer_output) # Layer output: [ 0.71  3.57 -1.89]
```
## Part 3: Activation Functions ⚙️


We’ve now seen how each neuron computes a weighted sum (dot product + bias). 
But if all we ever do is sum up and pass forward linearly, stacking layers would be pointless. 


### ❗ Why We Need Activation Functions


- Without activation, every layer is just a linear function. 
 Stacking linear functions = another linear function → no added expressive power.
- We need **non‑linearity** so the network can learn complex, non‑linear relationships (like shapes, boundaries, features). 
- Activation functions inject that non‑linearity while still being differentiable (so we can train via gradients).


### Common Activation Functions


Here are a few widely used ones:


| Activation | Formula | Range | Pros / Use Cases |
|------------|---------|-------|------------------|
| **Sigmoid** | $ \sigma(z) = \frac{1}{1 + e^{-z}} $ | (0, 1) | Good for binary probabilities; saturates at extremes |
| **Tanh** | $ \tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}} $ | (−1, 1) | Zero‑centered, often better convergence than sigmoid |
| **ReLU (Rectified Linear Unit)** | $ \mathrm{ReLU}(z) = \max(0, z) $ | [0, ∞) | Simpler, less vanishing gradient, commonly used in hidden layers |


### Example Code


```python
import numpy as np


def sigmoid(z):
   return 1 / (1 + np.exp(-z))


def tanh(z):
   return np.tanh(z)


def relu(z):
   return np.maximum(0, z)


# Example vector of pre-activation values
z = np.array([-2.0, -0.5, 0.0, 1.0, 3.0])


print("Sigmoid:", sigmoid(z))
print("Tanh:", tanh(z))
print("ReLU:", relu(z))
```
## Softmax: Turning Scores into Probabilities


When you're doing **multi‑class classification** (more than two classes), you want your neural network’s final layer to output probabilities for each class. That’s where **Softmax** comes in.


### Why Softmax? What Problem Does It Solve?


- After all the dot products and activations in hidden layers, the network outputs a vector of raw scores (often called **logits**), e.g. `z = [2.5, 0.3, -1.2]`. 
 These logits are arbitrary real numbers — there’s no direct interpretation. 
- You want to interpret them as **“this is the probability the input belongs to class 0, class 1, class 2, etc.”** 
- Softmax **normalizes** the raw scores into a **probability distribution**: 
 1. All outputs are between 0 and 1 
 2. They **sum to 1** 
 3. Higher logits lead to higher probabilities, but in a smooth, differentiable way 
- Unlike using `argmax` (which picks one class), `argmax` is not differentiable, so you cannot backpropagate through it. Softmax preserves differentiability. 
 :contentReference[oaicite:0]{index=0}


Thus Softmax is typically used as the **final activation** in multi-class neural networks. :contentReference[oaicite:1]{index=1}


### Math Formula


Given a vector of logits $ \mathbf{z} = [z_1, z_2, \dots, z_K] $, the softmax output for class $ i $ is:


$$
\text{softmax}(\mathbf{z})_i = \frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}}
$$


**Key properties:**


- Each $ \text{softmax}(\mathbf{z})_i \in (0, 1) $.
- The sum across all classes is 1:


 $$
 \sum_{i=1}^{K} \text{softmax}(\mathbf{z})_i = 1
 $$


- Softmax preserves the **order** of scores: 
 If $ z_i > z_j $, then $ \text{softmax}(\mathbf{z})_i > \text{softmax}(\mathbf{z})_j $.
- Softmax is **invariant to constant shifts**: 
 Adding the same constant $ c $ to all logits doesn’t change the output:


 $$
 \text{softmax}(\mathbf{z} + c) = \text{softmax}(\mathbf{z})
 $$


 This is useful for numerical stability.


**One common trick for numerical stability is:**


$$
\text{softmax}(\mathbf{z})_i = \frac{e^{z_i - \max(\mathbf{z})}}{\sum_{j} e^{z_j - \max(\mathbf{z})}}
$$


This avoids overflow from large exponentials.




### Code Example (Python / NumPy)


```python
import numpy as np


def softmax(logits):
   # logits: 1D numpy array of shape (K,)
   # For numerical stability, subtract max
   z = logits - np.max(logits)
   exp_z = np.exp(z)
   sum_exp = np.sum(exp_z)
   return exp_z / sum_exp


# Example usage
logits = np.array([2.5, 0.3, -1.2])
probs = softmax(logits)
print("Probabilities:", probs)
print("Sum of probs:", np.sum(probs))  # should be ~1.0
```
## All Together
let’s tie everything together into a mini neural network forward pass using NumPy only
### Full Forward Pass Example (ReLU + Softmax)
This example shows how data flows through a small feed-forward neural network:


1. Inputs → 
2. Layer 1 (Dense) → ReLU Activation → 
3. Layer 2 (Dense) → Softmax Activation → Output probabilities


```python
import numpy as np


# ------------------------------
# Activation Functions
# ------------------------------


def relu(x):
   """ReLU activation: sets negative values to 0"""
   return np.maximum(0, x)


def softmax(x):
   """Softmax activation for output layer"""
   exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))  # stability trick
   return exp_x / np.sum(exp_x, axis=1, keepdims=True)


# ------------------------------
# Input Data (batch of 3 samples, each with 4 features)
# ------------------------------
X = np.array([
   [1.0, 2.0, 3.0, 2.5],
   [2.0, 5.0, -1.0, 2.0],
   [-1.5, 2.7, 3.3, -0.8]
])


# ------------------------------
# Layer 1: 4 inputs → 5 neurons
# ------------------------------
weights1 = np.array([
   [0.2, 0.8, -0.5, 1.0],
   [0.5, -0.91, 0.26, -0.5],
   [-0.26, -0.27, 0.17, 0.87],
   [0.8, 0.45, -0.15, -0.33],
   [0.4, -0.2, 0.9, -0.5]
])


biases1 = np.array([[2.0, 3.0, 0.5, -1.0, 2.5]])


# Forward pass for Layer 1
layer1_output = np.dot(X, weights1.T) + biases1  # z = XWᵀ + b
relu_output = relu(layer1_output)


# ------------------------------
# Layer 2: 5 inputs → 3 neurons (output layer)
# ------------------------------
weights2 = np.array([
   [0.1, -0.14, 0.5, 0.3, -0.2],
   [-0.5, 0.12, -0.33, 0.5, 0.8],
   [0.3, 0.5, -0.3, 0.2, 0.1]
])


biases2 = np.array([[0.5, -0.3, 0.8]])


# Forward pass for Layer 2
layer2_output = np.dot(relu_output, weights2.T) + biases2
softmax_output = softmax(layer2_output)


# ------------------------------
# Final Results
# ------------------------------
print("Layer 1 Output (ReLU):")
print(relu_output)
print("\nLayer 2 Output (Softmax Probabilities):")
print(softmax_output)
print("\nSum per sample (should be 1):", np.sum(softmax_output, axis=1))


#Layer 1 Output (ReLU):
#[[4.8   1.21  2.385 0.    3.95 ]
# [8.9   0.    0.2   2.34  0.4  ]
# [1.41  1.051 0.026 0.    4.73 ]]


#Layer 2 Output (Softmax Probabilities):
#[[2.01644962e-01 4.99767821e-02 7.48378255e-01]
# [1.41035714e-01 6.13247392e-04 8.58351038e-01]
# [2.30990753e-02 6.49025876e-01 3.27875049e-01]]


#Sum per sample (should be 1): [1. 1. 1.]
```
So far, we’ve done a good job answering **What does the model predict?**
Now comes the next critical questions:
1. Was the prediction right? How wrong was it?
3. How can the model get better next time?


To train a neural network, we need a way to:


1. **Measure error** — using a **loss function**
2. **Propagate that error backward** — via **backpropagation**
3. **Adjust the weights** — using an **optimizer**


These are the building blocks of learning.


In the next chapter, we’ll dive into how neural networks *learn from their mistakes*, step by step — transforming raw predictions into better performance.


