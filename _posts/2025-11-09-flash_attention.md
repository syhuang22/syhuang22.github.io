---
title: Flash Attention
date: 2025-11-09 10:00
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


## ⚡ FlashAttention — Efficient I/O Attention for Transformers

**FlashAttention** is a modern optimization of the standard softmax attention used in Transformers.  
It was introduced by *Tri Dao et al. (2022)* to make attention **fast, memory-efficient, and numerically stable** —  
especially for large-scale LLMs running on GPUs.


### Background — The Real Problem: Memory, Not Compute

In the naive implementation, attention computes:

$$
\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
$$

This seems simple, but under the hood, it requires creating and storing a massive **attention score matrix** $$[n \times n]$$,  
where *n* is the sequence length.


#### GPU Memory Hierarchy: HBM vs SRAM

Modern GPUs (like NVIDIA A100, H100, or RTX 4090) have a **hierarchical memory system**.  
The core insight: even though GPUs have *teraflops* of compute, they’re limited by **how fast data moves** between memory tiers.

| Memory Type | Location | Capacity | Bandwidth | Access Latency | Purpose |
|--------------|-----------|-----------|-------------|----------------|----------|
| **Registers / SRAM** | On-chip (per Streaming Multiprocessor) | ~10–100 MB total across all SMs | **10–20 TB/s** | ~10–20 ns | Fastest memory, stores working tiles during computation |
| **Shared Memory / L1 Cache** | On-chip | 128 KB – 256 KB per SM | **10–15 TB/s** | ~20 ns | Cooperatively accessed scratchpad for a block of threads |
| **L2 Cache** | On-die (but off-SM) | 40–60 MB (A100: 40 MB, H100: 50 MB) | **5 TB/s** | ~100 ns | Caches frequently used data across SMs |
| **HBM (High-Bandwidth DRAM)** | Off-chip (GPU main memory) | 40–80 GB (A100: 40–80 GB, H100: 80 GB) | **2 TB/s** | **~500 ns – 1 µs** | Stores model weights, activations, and large tensors |

So while HBM capacity is **thousands of times larger**,  
SRAM is **hundreds of times faster** and far closer to the compute cores.

### Example Setup

Let’s pick a realistic configuration:

- Sequence length: $$n = 4096$$
- Hidden size: $$d = 1024$$
- Heads: $$h = 16$$ → per-head dim $$d_k = 64$$
- Data type: FP16 (2 bytes)

Then for one attention layer:

- $$Q, K, V$$ each: shape `[n, d]` → `[4096, 1024]`

Size per matrix:
$$
4096 \times 1024 \times 2 \approx 8\ \text{MB}
$$

So:
- Total for Q, K, V ≈ **24 MB**

Already fine for HBM.  
The problem shows up when we compute **attention scores**.

#### Naive Attention Workflow

Standard scaled dot-product attention:

$$
\text{Attention}(Q, K, V)
= \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right) V
$$

#### 1. Load Q, K from HBM
- Q and K are stored in HBM.
- They get loaded into SRAM (tile by tile) for matmul:
  - This part is fine and compute-friendly.

#### 2. Compute Score Matrix $$S = QK^\top$$
Shape of $$S$$:
- `[n, n] = [4096, 4096]`
- Number of elements:
  $$
  4096^2 = 16,\!777,\!216
  $$
- With FP16 (2 bytes):
  $$
  \approx 32\ \text{MB per head}
  $$

For 16 heads:
- $$S$$ total ≈ **512 MB**

Now what naive attention does:

1. Compute $$S$$ in SRAM.
2. **Write full $$S$$ to HBM.**  *Problem #1*
3. Later, read $$S$$ back from HBM to apply softmax.  *Problem #2*
4. After softmax, write $$P = \text{softmax}(S)$$ back to HBM (another `[n × n]`). *Problem #3*
5. Read $$P$$ + $$V$$ again from HBM to compute $$PV$$. *Problem #4*

So for one layer, one forward pass, rough memory traffic:

- Write scores: ~512 MB
- Read scores: ~512 MB
- Write softmax probs: ~512 MB
- Read probs: ~512 MB
- Plus multiple reads of Q, K, V

Total = **GBs of data moved**, just for intermediate junk  
that we **don’t even need to keep** once the output is computed.

The GPU *can* do the matmul FLOPs quickly,  
but keeps getting stalled by:
> “Load big matrix from HBM → compute → write big matrix to HBM → repeat.”

That’s what **I/O-bound** means here:
- **Compute is fast**, but
- **HBM read/write dominates runtime**

### The Key Idea of FlashAttention

FlashAttention rethinks attention as an **IO-aware algorithm**,  
optimizing **data movement** instead of pure compute.

Compute attention **block-by-block**, storing intermediate results only in **fast SRAM**,  
so we never materialize the full `[n × n]` attention matrix in HBM.

### Algorithm Overview

#### Traditional Attention
- Compute full score matrix $$S = QK^\top$$
- Apply softmax over all $$S$$ rows
- Multiply $$\text{softmax}(S)$$ by $$V$$

#### FlashAttention
- Split $$Q, K, V$$ into **small blocks (tiles)** that fit in SRAM
- For each tile:
  1. Compute partial scores $$Q_i K_j^\top$$
  2. Apply **softmax normalization on the fly** using running statistics:
     $$
     \text{softmax}(x_i) = \frac{e^{x_i - \max(x)}}{\sum_j e^{x_j - \max(x)}}
     $$
  3. Accumulate results directly into output
- Never store the full attention matrix in HBM

### Implementation Tricks

| Technique | Purpose |
|------------|----------|
| **Blockwise tiling** | Fit computation into fast on-chip SRAM |
| **On-the-fly normalization** | Prevents numerical overflow |
| **Fused CUDA kernel** | Combines matmul + softmax + matmul for speed |


### Numerical Stability: The Max-Trick

When computing softmax in tiles, we can’t normalize globally at once,  
so FlashAttention keeps **running statistics**:
- Track running `max_score` and `exp_sum` per query block  
- Incrementally update the normalization for each new tile:
$$
\text{new max} = \max(\text{old max}, s_{ij})
$$
$$
\text{new sum} = e^{\text{old max} - \text{new max}} \cdot \text{old sum} + e^{s_{ij} - \text{new max}}
$$


This ensures perfect numerical equivalence to a full softmax,  
but avoids overflow and precision loss.

#### Blockwise Tiling

**Idea:**  
Instead of computing attention across the entire sequence at once (which would create a huge `[n × n]` score matrix in HBM), FlashAttention **divides Q, K, and V into small blocks (tiles)** that fit entirely in fast on-chip memory (SRAM).

**How it works:**
1. Take a small block of queries \( Q_{\text{block}} \) (e.g., 128 queries at a time).
2. Load a corresponding block of keys and values \( K_{\text{block}}, V_{\text{block}} \) into SRAM.
3. Compute partial scores:
   $$
   S_{\text{block}} = \frac{Q_{\text{block}} K_{\text{block}}^{\top}}{\sqrt{d_k}}
   $$
4. Update softmax statistics (using the running max-trick) and accumulate the weighted sum of values.
5. Once all key–value tiles are processed for this query block, write only the **final output** to HBM.

**Why it helps:**
- The full `[n × n]` matrix is **never materialized** in HBM.  
- Intermediate scores and softmax calculations stay in **on-chip SRAM**, which has 10–20× higher bandwidth.  
- Reduces memory footprint from \( O(n^2) \) → \( O(n) \) in practice.  
- Still computes all pairwise token interactions (so total FLOPs remain \( O(n^2) \)).

#### On-the-Fly Normalization

**Problem:**  
Softmax requires knowing the global sum of exponentials:
$$
\text{softmax}(x_i) = \frac{e^{x_i}}{\sum_j e^{x_j}}
$$
But when processing in tiles, we never see all \( x_j \) at once.

**Stability Trick (Standard):**
$$
\text{softmax}(x_i) = \frac{e^{x_i - m}}{\sum_j e^{x_j - m}}, \quad m = \max_j x_j
$$

**Streaming Trick (FlashAttention):**  
FlashAttention applies this normalization **incrementally** per tile while maintaining exact equivalence to global softmax.

For each query:
- Keep a **running max** \( m \)
- Keep a **running exponential sum** \( l \)
- Update them as new scores arrive:

$$
m_{\text{new}} = \max(m_{\text{old}}, s_{ij})
$$

$$
l_{\text{new}} = e^{m_{\text{old}} - m_{\text{new}}} \cdot l_{\text{old}} + e^{s_{ij} - m_{\text{new}}}
$$

Then normalize outputs using $\( m_{\text{final}}, l_{\text{final}} \)$.

**Why it helps:**
- Prevents overflow/underflow when exponentiating large numbers.  
- No need to store all scores to normalize later.  
- Produces **exactly the same** result as standard softmax.  
- Enables truly **streaming (tile-by-tile)** computation.

#### Fused CUDA Kernel

In naive attention, computation happens across several separate GPU kernels:
1. Compute $\( QK^\top \)$ → write to HBM  
2. Apply scaling & masking → write again  
3. Compute softmax → write probabilities  
4. Multiply by $\( V \)$ → read probabilities, write output

Each stage reads/writes hundreds of MBs between HBM and SRAM.

**FlashAttention Fusion:**  
All these steps are **fused into a single custom CUDA kernel** that operates entirely within SRAM.

Within the fused kernel:
- Load tiles of $\( Q, K, V \)$ into SRAM  
- Compute scores  
- Apply scaling and causal masking  
- Perform on-the-fly normalization (softmax + weighting)  
- Accumulate partial outputs  
- Write only the **final attention result** to HBM

### FlashAttention (Forward Pass) — Pseudocode

Given:
- Q: [n, d_k]
- K: [n, d_k]
- V: [n, d_v]
- Bq: query tile size
- Bk: key/value tile size

Goal:
Compute `O = softmax(Q K^T / sqrt(d_k)) V` without materializing [n × n].

---
```python
function FlashAttention(Q, K, V, Bq, Bk):
    n, d_k = shape(Q)
    d_v = shape(V)[1]
    scale = 1.0 / sqrt(d_k)

    # Initialize output
    O = zeros([n, d_v])

    # Process queries in blocks
    for q_start in range(0, n, Bq):

        q_end = min(q_start + Bq, n)

        # Slice query block
        Q_blk = Q[q_start:q_end, :]              # [Bq, d_k]

        # For this query block, maintain running stats per row
        # m: running max logits, l: running sum of exp, out: partial output
        m = full([q_end - q_start], -inf)        # [Bq]
        l = zeros([q_end - q_start])             # [Bq]
        out = zeros([q_end - q_start, d_v])      # [Bq, d_v]

        # Iterate over key/value blocks
        for k_start in range(0, n, Bk):

            k_end = min(k_start + Bk, n)

            K_blk = K[k_start:k_end, :]          # [Bk, d_k]
            V_blk = V[k_start:k_end, :]          # [Bk, d_v]

            # 1) Compute scores for this tile
            #    S_ij = (Q_blk[i] · K_blk[j]) * scale
            S = matmul(Q_blk, transpose(K_blk)) * scale    # [Bq, Bk]

            # 2) For each query row, update running max
            #    new_m = max(old_m, max_j S_ij)
            row_max = max_over_axis(S, axis=1)             # [Bq]
            new_m = max(m, row_max)                        # [Bq]

            # 3) Compute exponentials in the new normalized scale
            #    exp(S_ij - new_m[i])
            S_shifted = S - new_m[:, None]                 # broadcast [Bq, Bk]
            exp_S = exp(S_shifted)                         # [Bq, Bk]

            # 4) Update running sum of exponentials
            #    new_l = exp(m - new_m) * l + sum_j exp(S_ij - new_m)
            l_scaled = exp(m - new_m) * l                  # [Bq]
            new_l = l_scaled + sum_over_axis(exp_S, axis=1)# [Bq]

            # 5) Update partial output (weighted sum of V)
            #    old_out is rescaled into new space, then we add tile contribution
            #    out_new = (exp(m - new_m) * out + exp_S @ V_blk) / new_l
            out_scaled = (exp(m - new_m))[:, None] * out   # [Bq, d_v]
            tile_update = matmul(exp_S, V_blk)             # [Bq, d_v]
            out = (out_scaled + tile_update) / new_l[:, None]

            # 6) Commit updates
            m = new_m
            l = new_l

        # Write final block result
        O[q_start:q_end, :] = out

    return O
```
## FlashAttention in Inference — Prefill vs Decode

During LLM inference, attention is used in two phases:

| Phase | What Happens | FlashAttention Use |
|--------|---------------|-------------------|
| **Prefill** | The model processes the **entire prompt** at once (e.g., 4K tokens). | ✅ **Used** — ideal for full-sequence attention |
| **Decode** | The model generates **one token at a time**, reusing cached keys/values. | ❌ **Not used** — replaced by incremental kernels |


### Prefill Phase

In **prefill**, all tokens are known, so the model computes:
$$
O = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
$$
FlashAttention fits perfectly here — it tiles Q, K, V into SRAM blocks and computes attention efficiently.

**Benefits**
- Handles long prompts (4K–16K tokens)
- 2–4× faster than naive attention
- Much lower memory use

### Decode Phase

In **decode**, each new token attends to all **previous cached tokens**:
$$
O_t = \text{softmax}\!\left(\frac{Q_t K_{1:t-1}^\top}{\sqrt{d_k}}\right)V_{1:t-1}
$$
Since there’s only one query at a time, blockwise tiling brings little benefit.  
Instead, frameworks use **specialized kernels** like **PagedAttention** or **FlashDecoding** that optimize cache access.



