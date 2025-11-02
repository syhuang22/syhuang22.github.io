---
title: Note About Attention Is All You Need 
date: 2025-11-01 10:00
categories: [ML]
tags: [Attention]
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

## Attention Is All You Need — Foundations and Motivation

The 2017 paper *"Attention Is All You Need"* by Vaswani et al. marked a turning point in deep learning.  
It replaced recurrence and convolution with a purely **attention-based architecture** — the **Transformer**.  
To understand why this was revolutionary, we first revisit what came before.


### Background: RNNs and LSTMs

Before Transformers, **Recurrent Neural Networks (RNNs)** were the standard for sequence modeling —  
tasks like translation, speech recognition, or time series.

#### RNN Overview

RNNs process sequences *step by step*:
$$
h_t = f(Wx_t + Uh_{t-1})
$$

- $$x_t$$ = input at time *t*  
- $$h_t$$ = hidden state carrying information from previous steps  
- $$f$$ = nonlinear activation (e.g., tanh)

They share weights across time and model temporal dependencies.

#### The Problem: Vanishing and Exploding Gradients

As sequences grow longer, gradients must flow through many timesteps:
$$
\frac{\partial L}{\partial h_t} \propto \prod_i \frac{\partial h_i}{\partial h_{i-1}}
$$
This product can:
- shrink to near zero → **vanishing gradients** (can’t learn long dependencies)  
- blow up exponentially → **exploding gradients**  

As a result, RNNs struggle to “remember” information beyond ~20–50 timesteps.

### LSTMs and GRUs — Temporary Fixes

**LSTM (Long Short-Term Memory)** and **GRU (Gated Recurrent Unit)** architectures introduced *gates* to better control what information gets stored or forgotten.

#### Example (simplified):
$$
h_t = f_t \odot h_{t-1} + i_t \odot \tilde{h_t}
$$

where gates $$f_t, i_t$$ decide what to keep or replace.

**Pros:**
- Helped with vanishing gradients  
- Captured longer-range dependencies better than vanilla RNN

### The Bottleneck of Sequential Processing

All RNN-based models process tokens *in order*, one by one.

| Limitation | Description |
|-------------|--------------|
| **No parallelism** | Each token waits for the previous step — slow training |
| **Fixed-length context** | Hidden state compresses all past info into one vector |
| **Long-distance decay** | Distant tokens influence output weakly |
| **Difficult gradient flow** | Deep recurrence chains → unstable training |

These constraints made scaling to long sequences (like full sentences or documents) extremely difficult.


### Enter Attention: The Key Idea

The **attention mechanism** broke this bottleneck.  
Instead of encoding everything into a single hidden vector,  
it allows the model to **directly “look at” all past tokens** when processing the current one.

#### Core Concept
For a given query token, compute how much it should “attend to” each other token:

$$
\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
$$

- **Q (Query):** representation of the current token  
- **K (Key):** representation of all tokens (used for matching)  
- **V (Value):** information to be aggregated  
- The dot product $$QK^\top$$ measures similarity → tells the model *where to focus*  

This allows each token to access *context from anywhere in the sequence*,  
not just nearby positions.

### The Transformer: “Attention Is All You Need”

Vaswani et al. realized that **if attention can capture sequence dependencies**,  
we can **remove recurrence entirely**.

#### Transformer Highlights:
- Fully parallelizable — all tokens processed at once  
- Scales to long sequences efficiently  
- Learns global dependencies directly via self-attention  
- Uses **positional encoding** to retain order information

#### Main Components

| Component | Purpose |
|------------|----------|
| **Self-Attention** | Enables each token to attend to every other token in the same sequence, capturing contextual relationships. |
| **Multi-Head Attention** | Runs multiple attention operations in parallel so the model can learn different types of relationships simultaneously (e.g., syntax, semantics). |
| **Feedforward Layers** | Apply nonlinear transformations independently to each token, expanding the model’s representational power. |
| **Residual Connections + Layer Normalization** | Help stabilize training, maintain gradient flow, and allow deeper stacking of layers. |
| **Positional Encoding** | Injects information about the order of tokens (since attention alone has no notion of sequence). |


### Residual Connections

Residual (skip) connections were first popularized by **ResNet** in computer vision and later became critical for Transformers.

#### Intuition

Deep models can suffer from **vanishing gradients** —  
as layers stack up, early layers stop receiving clear training signals.

Residuals solve this by letting information and gradients **flow directly** around layers:

$$
\text{output} = \text{Layer}(x) + x
$$

The layer learns a *correction* to the input rather than a full transformation.

### Layer Normalization

Layer Normalization (LayerNorm) ensures that the activations within a layer have a **stable mean and variance**, improving convergence and training stability.

#### Formula

For input vector $$x = [x_1, x_2, ..., x_d]$$:

$$
\text{LayerNorm}(x) = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} \cdot \gamma + \beta
$$

where:
- $$\mu$$ = mean of elements in $$x$$  
- $$\sigma^2$$ = variance  
- $$\gamma, \beta$$ = learnable scale and shift parameters  

#### Intuition

Normalization keeps the network from having wildly varying activations,  
which can cause exploding gradients or slow learning.

Unlike BatchNorm (which normalizes across a batch),  
**LayerNorm** normalizes *across features within a single token’s representation* —  
making it ideal for sequence models where batch statistics vary.

### Self-Attention

Self-Attention is the heart of the Transformer.  
It allows the model to *dynamically focus on relevant parts of the input sequence* when processing each token.

#### Intuition

Instead of compressing all context into a single hidden state (as RNNs do),  
attention explicitly computes *how much each token should care about every other token*.

> Each token “looks at” others and gathers information weighted by importance.

#### Mathematical Formulation

Given input vectors (after linear projection):
- **Query (Q)** — represents the current token we are processing  
- **Key (K)** — represents how “relevant” each token is  
- **Value (V)** — carries the actual content to be aggregated  

The **Scaled Dot-Product Attention** is defined as:

$$
\text{Attention}(Q, K, V) =
\text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
$$

- The term $$QK^\top$$ computes pairwise similarity between tokens.  
- Dividing by $$\sqrt{d_k}$$ prevents large dot products from dominating.  
- The softmax converts scores into normalized attention weights.  
- Multiplying by $$V$$ aggregates the contextual information.

#### Example

If a token is “dog” in the sentence “The dog chased the ball,”  
it will likely attend strongly to “chased” and “ball,”  
but weakly to “The.”

Thus, attention dynamically distributes context where it matters most.

---
