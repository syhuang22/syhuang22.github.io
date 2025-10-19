---
title: ML Foundamentals - Part 2
date: 2025-10-19 10:00
categories: [ML]
tags: [Loss Function]
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


Now that we’ve walked through the forward‑pass of a neural network (inputs → neurons → layers → activations → softmax), we reach a pivotal moment: 
> **How wrong is the network’s prediction?** 
> And more importantly: **How can the network use that information to improve?**


This chapter introduces the concept of
- **loss functions** — the key mechanism for measuring error in neural networks.
- **Gradients, Partial Derivatives, and the Chain Rule**


---


### Why Loss Matters


- Up to now, our network produces probabilities (via softmax) or outputs from hidden layers. But a probability by itself doesn’t tell us *how far off* we are from the correct answer. 
- A **loss function** (also called a cost function) quantifies the difference between the network’s prediction and the true label(s). 
- Without a metric for error, we **cannot train**: no feedback means no adjustment to weights, no learning. 
- Loss functions provide both: 
 1. A **scalar value** representing how wrong the network is 
 2. A **signal** (via gradients) that tells us *which weights to adjust* and *how much*


### Common Loss Functions


Depending on the task type, we choose different loss functions:


#### Classification (e.g., multi‑class with softmax)


- **Categorical Cross‑Entropy** 
 Let **y** be the one-hot true label vector: 
 $$ \mathbf{y} = [y_1, y_2, \dots, y_K] $$ 
 and **p** be the predicted probability vector from softmax: 
 $$ \mathbf{p} = [p_1, p_2, \dots, p_K] $$ 


 The cross-entropy loss is:


 $$
 \text{Loss} = - \sum_{i=1}^K y_i \log(p_i)
 $$


 This punishes confident wrong predictions more than uncertain ones. It’s ideal when using softmax in the output layer.




#### Regression (continuous output)


- **Mean Squared Error (MSE)** 
 Let \( y \) be the true value and \( \hat{y} \) the predicted value. Then the loss is:


 $$
 \text{MSE} = \frac{1}{N} \sum_{i=1}^N (y_i - \hat{y}_i)^2
 $$


 MSE punishes larger errors more heavily and is widely used in tasks like forecasting, curve fitting, and any continuous output prediction.




### Code Example: Cross‑Entropy Loss (NumPy)


```python
import numpy as np


# Example: predictions (softmax output) for two samples
predictions = np.array([
   [0.1, 0.7, 0.2],
   [0.3, 0.4, 0.3]
])


# True one‑hot encoded labels
targets = np.array([
   [0,   1,   0],
   [1,   0,   0]
])


def categorical_cross_entropy(p, y):
   # p: predicted probabilities, y: one‑hot true labels
   # Add small value to log for numerical stability
   return -np.sum(y * np.log(p + 1e‑8)) / p.shape[0]


loss_value = categorical_cross_entropy(predictions, targets)
print("Cross‑Entropy Loss:", loss_value)
```


## Gradients, Partial Derivatives, and the Chain Rule


In the last chapter, we learned how to calculate the loss — a single number that tells us how far off the network's prediction was.


Now we ask the next question:


**How do we use that loss to improve the network?**


To do that, we need to understand how each parameter (like weights and biases) contributed to that loss. This leads us to: **gradients**, **partial derivatives**, and the **chain rule**.




### What Is a Gradient?


A **gradient** tells us how much the output of a function changes if we change its input slightly.


In the context of neural networks, the gradient tells us:


- Which direction the loss increases or decreases
- How quickly it changes


We use this to guide how we update each weight in the network to reduce the loss.


### Partial Derivatives: One Variable at a Time


A neural network has many weights. For each one, we ask:


**"If I nudge this weight a little, what happens to the loss?"**


This is what a **partial derivative** tells us:


$$
\frac{\partial \mathcal{L}}{\partial w_{ij}}
$$


It measures how the loss \($\mathcal{L} $\) changes with respect to just one weight \($ w_{ij} $\), holding all others constant.


We do this for every weight in the model.


### The Chain Rule: How Gradients Flow Back


Neural networks are built from **functions inside functions**. For example:


- A neuron computes a dot product: $\( z = w \cdot x + b \)$
- Applies an activation function: $\( a = \sigma(z) \)$
- Then the network computes a loss: \( $\mathcal{L} = \text{Loss}(a, y) \$)


To find how the loss changes with respect to a weight, we apply the **chain rule**:


$$
\frac{\partial \mathcal{L}}{\partial w} =
\frac{\partial \mathcal{L}}{\partial a} \cdot
\frac{\partial a}{\partial z} \cdot
\frac{\partial z}{\partial w}
$$


This breaks the computation into small, manageable parts — one per function — and multiplies the gradients step by step.




### Why Gradients and the Chain Rule Matter


- Gradients are the key to learning — they tell us how to adjust weights to reduce loss.
- Partial derivatives let us isolate the effect of individual weights.
- The chain rule lets us connect the loss all the way back to any parameter in the network.


This is the foundation of **backpropagation** — the algorithm that computes gradients efficiently in multi-layer neural networks.


## Backpropagation, Loss + Activation Derivatives


### Backpropagation Overview 
Once we know the loss (how wrong the model is), we need a way to **propagate** that error back through the network to update weights. 
This process is called **backpropagation**. 


We apply the **chain rule** repeatedly so that the loss’s effect on each parameter is calculated:


$$
\frac{dL}{dw} = \frac{dL}{dz} \cdot \frac{dz}{dw}
$$


This allows us to trace how changes in the loss relate to changes in weights, biases, and activations throughout the network.




### Derivative of Categorical Cross‑Entropy Loss 
For a multi‑class classification with one‑hot true label vector 
$$ \mathbf{y} = [y_1, y_2, \dots, y_K] $$ 
and predicted probability vector 
$$ \mathbf{p} = [p_1, p_2, \dots, p_K] $$ 
(using softmax in the final layer) the loss is: 
$$
\mathcal{L} = - \sum_{i=1}^{K} y_i \log(p_i)
$$ 
When we compute the derivative of this loss with respect to the inputs (logits) of the softmax layer, we get the form: 
$$
\frac{\partial \mathcal{L}}{\partial z_j} = p_j - y_j
$$ 
The softmax output is defined as
$$
p_j = \frac{e^{z_j}}{\sum_k e^{z_k}}
$$
for each class index \( $j$ \).



### Derivative of Softmax Activation 


Suppose we have a logits vector: 


$$
\mathbf{z} = [z_1, z_2, \dots, z_K]
$$ 


and corresponding softmax outputs: 


$$
p_j = \frac{e^{z_j}}{\sum_{k=1}^{K} e^{z_k}}
$$ 


The derivative of the softmax function can be expressed using the **Jacobian matrix**: 


$$
\frac{\partial p_j}{\partial z_k} =
\begin{cases}
p_j (1 - p_j), & \text{if } j = k \\
- p_j p_k, & \text{if } j \neq k
\end{cases}
$$ 


This describes how a change in the input \($ z_k $\) affects each output \($ p_j $\). 
It shows that each output depends on all inputs — making the Jacobian dense and computationally expensive to calculate directly.






### Combined Derivative: Softmax + Cross‑Entropy 
A useful simplification arises when we use the cross‑entropy loss together with a softmax final layer. Because of the form of softmax and the loss, the derivatives combine cleanly. In fact: 
$$
\frac{\partial \mathcal{L}}{\partial z_j} = p_j - y_j
$$ 
This relationship holds directly, meaning we **don’t have to separately compute** the full Jacobian of softmax when paired with cross-entropy.


Instead of computing:


$$
\frac{\partial \mathcal{L}}{\partial \mathbf{z}} = \mathbf{J}_{\text{softmax}}^\top \cdot \frac{\partial \mathcal{L}}{\partial \mathbf{p}}
$$


it simplifies to:


$$
\frac{\partial \mathcal{L}}{\partial \mathbf{z}} = \mathbf{p} - \mathbf{y}
$$


This simplification is what makes training multi-class networks efficient.


### Code Example:
```python
import numpy as np


# Combined Softmax activation and Cross-Entropy loss
class Activation_Softmax_Loss_CategoricalCrossentropy:
   def __init__(self):
       self.activation = Activation_Softmax()
       self.loss = Loss_CategoricalCrossentropy()


   # Forward pass
   def forward(self, inputs, y_true):
       self.activation.forward(inputs)
       self.output = self.activation.output
       return self.loss.calculate(self.output, y_true)


   # Backward pass
   def backward(self, dvalues, y_true):
       # Number of samples
       samples = len(dvalues)


       # If labels are one-hot encoded, convert them to class indices
       if len(y_true.shape) == 2:
           y_true = np.argmax(y_true, axis=1)


       # Copy the predictions
       self.dinputs = dvalues.copy()


       # Subtract 1 from the correct class index
       self.dinputs[range(samples), y_true] -= 1


       # Normalize gradient
       self.dinputs = self.dinputs / samples
```
