---
title: Page Attention
date: 2025-11-16 10:10
categories: [Attention]
tags: [Page Attention]
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

## PagedAttention — Solving Memory Fragmentation in LLM Serving

FlashAttention made **training and prefill** efficient,  
but **serving (inference)** introduces a different challenge —  
managing **KV cache memory** during **token-by-token decoding**.

---

### The Problem: Inefficient Memory Usage in LLM Serving

During inference, each request (user prompt) creates its own **Key–Value (KV) cache**:
- For each layer, store all past keys and values (size grows linearly with context length).
- Needed for the model to compute attention for new tokens efficiently.

When serving many users in parallel:
- Requests have **different sequence lengths**.
- Each user’s KV cache grows at a different rate.
- When sequences finish, their KV memory becomes **partially free**, leaving small unusable gaps.

This leads to **memory fragmentation**:
> GPU memory is full of small holes — total free memory is high,  
> but none of the holes are large enough for a new long sequence.

## ⚠️ Example: How KV Cache Blows Up in Llama 3 70B (and Causes Fragmentation)

To understand why memory fragmentation is a serious issue in LLM serving,  
let’s look at a concrete example using **Llama 3 70B**.

### KV Cache Memory Formula

For each token, each layer stores:

- Key vector: size = `d_k`
- Value vector: size = `d_v`
- For multi-head attention, both are concatenated across all heads.

**Per-token KV size:**
$$
\text{KV per token} = 2 \times n_\text{layers} \times d_\text{model} \times \text{bytes}
$$

Where:
- Factor **2** = Key + Value
- `n_layers` = number of transformer layers  
- `d_model` = hidden dimension  
- `bytes` = 2 bytes (FP16) or 1 byte (INT8-kv)

Most modern LLMs store KV in FP16 or BF16:
- **FP16 → 2 bytes**
- **BF16 → 2 bytes**

So if stored in FP16:
$$
\text{KV per token} = 2 \times n_\text{layers} \times d_\text{model} \times 2
$$

Then total KV-cache memory for a sequence of length *T*:
$$
\text{KV memory} = T \times (\text{KV per token})
$$


### Apply to Llama 3 70B

Model specs:

- Number of layers:
  $$
  n_{\text{layers}} = 80
  $$
- Hidden dimension:
  $$
  d_{\text{model}} = 8192
  $$
- KV precision: FP16 → 2 bytes

**Compute KV per token:**
$$
\text{KV per token}
= 2 \times 80 \times 8192 \times 2
= 2{,}621{,}440\ \text{bytes}
\approx 2.5\ \text{MB per token}
$$

---

### Correct Interpretation

- **≈ 2.5 MB *per token*** (across all 80 layers)
- **≈ 2.5 MB per token × T tokens = T × 2.5 MB total KV**

For example:

| Tokens | KV Memory |
|--------|-----------|
| **1 token** | **2.5 MB** |
| **1,000 tokens** | **2.5 GB** |
| **8,000 tokens (8K)** | **20 GB** |
| **16,000 tokens (16K)** | **40 GB** |

These numbers match real-world KV footprint of Llama-3 70B.


### The Hidden Problem: Many Requests at Once

To see why fragmentation becomes a real issue, consider serving  
**32 concurrent users** on a GPU node with **Llama 3 70B**.

Assume each user has a mid-sized prompt of **8K tokens**.

### KV Cache Calculation (Llama 3 70B)

- KV per token ≈ **2.5 MB**
- So KV for one 8K-token request:

$$
8{,}000 \times 2.5\ \text{MB}
= 20{,}000\ \text{MB}
= 20\ \text{GB}
$$

- KV for 32 concurrent requests:

$$
32 \times 20\ \text{GB}
= 640\ \text{GB}
$$

Even before accounting for model weights (~140 GB in FP16),  
KV cache alone already exceeds **640 GB**, far more than what a single GPU or even a single GPU node can hold.

### Fragmentation makes it even worse

Even if you scale across multiple GPUs, the problem persists:

- Some sequences grow slowly (e.g., 4K tokens → ~10 GB KV)
- Some sequences grow fast (e.g., 12K tokens → ~30 GB KV)
- Some finish early, freeing only *small scattered pages*

This leads to **memory fragmentation**:
- Many **small freed chunks**  
- Very **few contiguous large chunks**  
- New long-sequence requests cannot find a big enough block  
- GPU reports **OOM**, even if total free memory is large

### Why this matters

Large models like Llama 3 70B generate substantial KV caches.  
With many concurrent users:

- KV grows unpredictably  
- Memory becomes fragmented  
- Serving throughput collapses  
- GPUs appear "full" due to fragmentation, not true exhaustion

This is exactly the scenario PagedAttention is designed to solve.


### Why Fragmentation Hurts Efficiency

| Problem | Description | Effect |
|----------|--------------|---------|
| **Uneven sequence lengths** | Each user generates a different number of tokens | GPU memory can’t be uniformly partitioned |
| **Non-contiguous KV storage** | KV blocks scatter as sessions start/stop | Slower cache access and wasted space |
| **Memory churn** | Frequent alloc/free operations for variable contexts | Causes latency spikes and OOM errors |

Even advanced frameworks (like FasterTransformer, DeepSpeed) struggled with this during large-scale serving.


### The Solution: PagedAttention

PagedAttention (introduced in **vLLM, 2023**) solves KV-cache fragmentation by borrowing the concept of **virtual memory paging** from operating systems.

> Instead of giving each request one large contiguous KV block,  
> PagedAttention divides GPU memory into fixed-size **pages**  
> and uses **page tables** to map logical KV blocks to physical pages.

This turns the KV cache into a flexible, OS-style virtual memory system for LLMs.

### How It Works

#### 1. **Page-Based KV Cache (Physical Layout)**
- GPU memory is partitioned into many **fixed-size pages** (e.g., 1–2 MB).
- Each page stores a chunk of KV tensors (vLLM internally splits KV by layer/head so pieces fit).
- Pages can be scattered across memory; contiguity is no longer required.

[Page 0][Page 1][Page 2][Page 3][Page 4]...

This is the **physical** storage of KV.


#### 2. **Logical KV Blocks (What the Model Sees)**
Each sequence *logically* has a contiguous KV array:

Logical KV: [Token 0, Token 1, Token 2, ..., Token T]

But physically, these tokens may sit in different pages anywhere in memory.

#### 3. **Page Table (Logical → Physical Mapping)**
Each sequence maintains a **page table**, mapping logical KV blocks to physical pages:

Sequence A Page Table:
Logical Block 0 → Page #12
Logical Block 1 → Page #87
Logical Block 2 → Page #2
Logical Block 3 → Page #44

When the attention kernel needs KV for token *t*:

1. Compute logical block index  
2. Look up the physical page in the page table  
3. Load KV directly from that page  

Exactly like CPU virtual memory address translation.

#### 4. **Dynamic Reuse of Pages**
- When a request finishes, **all its pages return to the free pool**.
- New requests reuse these pages immediately — **no fragmentation**, because:
  - all pages are equal-size  
  - no need for contiguous blocks  

GPU memory stays maximally usable.

#### 5. **Random Access During Decoding**
During decode, attention repeatedly queries the KV cache:

Query → lookup page table → locate physical page → load KV

This allows highly efficient, parallel attention even when:
- sequences have different lengths  
- KV pages are scattered  
- systems handle thousands of users

### Why It Matters

| Benefit | Description |
|---------|-------------|
| **No fragmentation** | Freed pages can be reused immediately, regardless of where they sit in memory. |
| **High GPU utilization** | Supports huge numbers of concurrent sequences without OOM. |
| **Stable latency** | No expensive allocate/free operations during inference. |
| **Scales with sequence length** | Works equally well for 1K, 8K, or 128K prompts. |
| **OS-like robustness** | KV cache behaves like paged virtual memory, not raw buffers. |

PagedAttention lets LLM servers achieve **near 100% GPU memory utilization**,  
making high-throughput, multi-user LLM inference truly practical.

## Next Section: Introduction to vLLM

With PagedAttention as its core memory-management innovation, vLLM builds a full high-throughput inference engine optimized for large-scale LLM serving.

In the next section, we will cover the essential concepts behind vLLM.
