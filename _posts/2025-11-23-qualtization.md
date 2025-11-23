---
title: Introduction of Quantization
date: 2025-11-23 09:10
categories: [Quantization]
tags: [Quantization]
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

This week, we introduce **Quantization**, one of the most important techniques for improving LLM inference efficiency.  
As models become larger (Llama 3 70B, GPT-4-class models, Gemini, Claude), inference bottlenecks increasingly come from:

- **Memory capacity (HBM)**
- **Memory bandwidth (HBM throughput)**
- **KV-cache size**
- **Compute–memory imbalance**

Quantization directly addresses these constraints.

---

# 1. What Is Quantization?

Quantization means representing model weights or activations using **lower-precision numeric formats**.

In general:

$$
w_{\text{quant}} = \text{round}\left(\frac{w}{s}\right),
\qquad
w \approx s \cdot w_{\text{quant}}
$$

Where:

- \( w \) = original FP32/FP16 weight  
- \( s \) = scaling factor  
- $\( w_{\text{quant}} \)$ = integer representation (INT8, INT4, etc.)

Quantization preserves the model's behavior while reducing memory and bandwidth requirements.

---

# 2. Why Quantization Matters for LLMs

LLMs are memory-bound.  
A 70B model in FP16 requires >140 GB just for weights, not counting KV cache.

Quantization reduces:

| Precision | Bytes per parameter | Memory reduction |
|----------|---------------------|------------------|
| FP32     | 4 bytes             | baseline         |
| FP16/BF16| 2 bytes             | 2× smaller       |
| INT8     | 1 byte              | 4× smaller       |
| INT4     | 0.5 bytes           | 8× smaller       |

For inference, reduced memory →  
**lower latency**, **higher throughput**, **larger batch sizes**, and **longer context**.

# 3. Mathematical View of Quantization

Given weight matrix $\( W \in \mathbb{R}^{m \times n} \)$:

Quantize per-channel or per-group:

$$
W_{q} = \text{round}\left(\frac{W}{s}\right)
$$

Dequantization during inference:

$$
W \approx s \cdot W_{q}
$$

Where \( s \) may be:
- A scalar per layer  
- A vector per channel  
- A matrix per group/block  

More granular scaling → better accuracy, slightly more overhead.

# 4. PTQ: Post-Training Quantization

Post-Training Quantization (PTQ) applies quantization **after** the model has been trained.  
It does not modify the training procedure and requires only a small calibration dataset.

The core idea:

1. Train a full-precision (FP32/FP16) model as usual.
2. Analyze the value ranges of **weights** and optionally **activations**.
3. Compute scale and zero-point parameters.
4. Convert the model to a lower-precision format (INT8, INT4) for inference.

PTQ answers two questions:

- How do we map continuous FP values to discrete integers?
- How do we choose the scale for this mapping?

---

## 4.1 Linear Quantization: Scale and Zero-Point

For a real-valued tensor \( F \) (weights or activations), we define a linear mapping:

- Quantization (float → int):
  $$
  q = \text{round}\left(\frac{F}{s}\right) + z
  $$

- Dequantization (int → float):
  $$
  \hat{F} = s \cdot (q - z)
  $$

Where:

- \( s \) = scale factor  
- \( z \) = zero point (integer offset)  
- \( q \) = quantized integer (e.g., INT8)  
- ( $\hat{F} $) ≈ reconstructed float

Two common cases:

### Symmetric Quantization

- Target integer range: $\([-Q_{\max}, Q_{\max}]\)$, e.g. $\([-127, 127]\)$ for signed INT8.
- We use only a scale, zero point is zero.

Given:
$$
F_{\max}^{\text{abs}} = \max_{i} |F_i|
$$

Then:
$$
s = \frac{F_{\max}^{\text{abs}}}{Q_{\max}}, \quad z = 0
$$

Quantization:
$$
q_i = \text{round}\left(\frac{F_i}{s}\right)
$$

Dequantization:
$$
\hat{F}_i = s \cdot q_i
$$

### Asymmetric Quantization

- Target integer range: e.g. \([0, 255]\) for unsigned INT8.

Given:
$$
F_{\min} = \min_i F_i,\quad F_{\max} = \max_i F_i
$$

Scale and zero point:
$$
s = \frac{F_{\max} - F_{\min}}{Q_{\max} - Q_{\min}}
$$

$$
z = \text{round}\left(Q_{\min} - \frac{F_{\min}}{s}\right)
$$

Where typically \(Q_{\min} = 0\), \(Q_{\max} = 255\).

Quantization:
$$
q_i = \text{round}\left(\frac{F_i}{s}\right) + z
$$

Dequantization:
$$
\hat{F}_i = s \cdot (q_i - z)
$$

---

## 4.2 PTQ Workflow for Weights

For a given weight tensor $W$:

1. **Collect statistics**  
   - Read all values of $W$.
   - Compute the range using one of the following methods:
    - **Symmetric quantization:**  
        $F_{\max}^{\text{abs}} = \max_i |W_i|$
    - **Asymmetric quantization:**  
        $W_{\min},\; W_{\max}$


2. **Compute $s, z$**  
   - Use the formulas above (symmetric or asymmetric).
   - Optionally per-tensor, per-channel, or per-group.

3. **Quantize weights**  
   $$
   q_{ij} = \text{round}\left(\frac{W_{ij}}{s}\right) + z
   $$

4. **Store only integers and scales**  
   - At inference time, kernels use $q$ and $s, z$ to reconstruct approximate weights on the fly or incorporate the scale into the computation.

---

## 4.3 PTQ Workflow for Activations (Calibration)

Weights are static; activations depend on input.

To quantize activations:

1. Prepare a **small representative dataset**.
2. Run the full-precision model and collect activation values for each layer:
   - Track min/max or a more robust statistic (e.g. percentile, KL-based histograms).
3. For each activation tensor \( A \), compute its \( s_A, z_A \).
4. Use these scales during inference to quantize/dequantize activations.

This calibration step is what makes PTQ work without retraining.

---

## 4.4 Example: PTQ for a Single Weight Tensor

Suppose we have a weight tensor with:
- \( $W_{\min} = -1.0 \$)
- \( $W_{\max} = 3.0 \$)
- Use unsigned INT8 in \([0, 255]\).

Then:

1. Scale:
   $$
   s = \frac{W_{\max} - W_{\min}}{255 - 0}
     = \frac{3.0 - (-1.0)}{255}
     \approx 0.01568
   $$

2. Zero point:
   $$
   z = \text{round}\left(0 - \frac{-1.0}{0.01568}\right)
     \approx \text{round}(63.76)
     = 64
   $$

Now, quantize \(W_{ij} = 1.5\):

$$
q_{ij} = \text{round}\left(\frac{1.5}{0.01568}\right) + 64
       \approx \text{round}(95.64) + 64
       = 96 + 64
       = 160
$$

During inference:

$$
\hat{W}_{ij} = s \cdot (q_{ij} - z)
             \approx 0.01568 \cdot (160 - 64)
             = 0.01568 \cdot 96
             \approx 1.505
$$

So the quantization error here is small.

---

## 4.5 Strengths and Weaknesses of PTQ

**Strengths:**

- No retraining required.
- Fast and easy to integrate into existing pipelines.
- Works well with INT8 on many LLMs.

**Weaknesses:**

- Quantization error is introduced *after* training.
- Model has no chance to adapt to that error.
- For aggressive schemes (e.g. INT4), accuracy can drop significantly unless advanced methods (GPTQ, AWQ) are used.

PTQ is ideal when you:

- Need fast deployment.
- Can tolerate some accuracy loss.
- Do not have resources to retrain or fine-tune the model.

---

# 5. QAT: Quantization-Aware Training

Quantization-Aware Training (QAT) integrates quantization into the **training process itself**.  
Instead of quantizing once at the end, QAT simulates quantization on every forward pass, so the model learns to **adapt** to quantization noise.

Key idea:

- Forward propagation: simulate quantization.
- Backward propagation: update full-precision weights with gradients that “see” quantization effects.

---

## 5.1 Fake Quantization in the Forward Pass

In QAT, we keep a full-precision weight \( W \), but during the forward pass we use a "fake-quantized" version:

1. Compute \( s, z \) (fixed or updated slowly during training).
2. Quantize to integer:
   $$
   q = \text{round}\left(\frac{W}{s}\right) + z
   $$
3. Dequantize back to float:
   $$
   \hat{W} = s \cdot (q - z)
   $$

4. Use \( $\hat{W} \$) in the forward computation:
   $$
   y = \hat{W} x
   $$

The forward pass therefore includes quantization error, and the loss function sees it.

---

## 5.2 Straight-Through Estimator (STE) for Backpropagation

The challenge:  
The `round()` function is non-differentiable, and its derivative is effectively zero almost everywhere.

To train through this, QAT uses the **Straight-Through Estimator (STE)**:

- In the **forward pass**, we use:
  $$
  q = \text{round}\left(\frac{W}{s}\right) + z
  $$
- In the **backward pass**, we *pretend* that this operation is the identity with respect to \( W \):

  Conceptually:
  $$
  \frac{\partial q}{\partial W} \approx 1
  $$

This means:

- Gradients are computed as if:
  $$
  \hat{W} \approx W
  $$
- But the forward pass still experiences quantization noise.

Effect:

- The model is trained under quantization noise.
- Parameters are updated to become robust to that noise.
- The final quantized model (INT8/INT4) usually matches FP32 accuracy closely.

---

## 5.3 QAT Workflow

1. Start from a pretrained FP32/FP16 model (or train from scratch).
2. Insert "fake quantization" modules around weights and/or activations.
3. During training:
   - Forward:
     - Apply quantization → dequantization (fake quant).
     - Compute loss with quantized values.
   - Backward:
     - Use STE to propagate gradients back to full-precision parameters.
4. After training:
   - Export true quantized weights (e.g., INT8 tensors + scale/zero-point metadata).
   - Use real integer kernels for inference.

---

## 5.4 When to Use QAT

**Use QAT when:**

- You need **maximum accuracy** under low-bit quantization.
- INT4 / INT2 quantization is required.
- You control the training pipeline and can afford additional training cost.

**Avoid QAT when:**

- You only have access to a frozen model.
- You cannot afford the computational cost of retraining / fine-tuning.
- PTQ already gives acceptable accuracy.

---

# 6. PTQ vs QAT: Summary

| Aspect        | PTQ (Post-Training Quantization)               | QAT (Quantization-Aware Training)                         |
|---------------|-------------------------------------------------|-----------------------------------------------------------|
| When applied  | After training                                  | During training                                           |
| Training cost | None (no gradient steps)                        | High (requires fine-tuning or full training)             |
| Accuracy      | Good with INT8, weaker with INT4                | Best, even for INT4 / aggressive schemes                 |
| Data needs    | Small calibration set                           | Full or partial training data                            |
| Use cases     | Fast deployment, limited resources              | High-accuracy production models, custom training         |
| Main idea     | Calibrate ranges, then quantize weights/acts    | Inject quantization in forward pass, learn to adapt via STE |

In practice:

- **PTQ** is the default for quickly deploying LLMs with INT8 / moderate INT4 quantization.
- **QAT** is used when you care about every bit of accuracy and can afford the cost of retraining.


