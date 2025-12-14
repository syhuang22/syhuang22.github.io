---
title: TPU Architecture — How Specialized Hardware Executes ML at Scale
date: 2025-12-14 09:10
categories: [AI Systems, Hardware]
tags: [TPU, Architecture]
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


# Introduction

As machine learning models grew larger and more matrix-heavy, general-purpose processors began to show fundamental inefficiencies. CPUs excel at control flow and latency-sensitive tasks, GPUs thrive on massive parallelism, but both were originally designed for workloads that look very different from modern deep learning.

Google’s Tensor Processing Unit (TPU) represents a different design philosophy: instead of optimizing for flexibility, it optimizes for **dense linear algebra**, predictable dataflow, and energy efficiency at scale.

This post introduces TPU architecture from the ground up. We’ll start by contrasting CPUs, GPUs, and TPUs, then walk through how a TPU executes a model end-to-end: from host interaction, to memory movement, to matrix execution inside the MXU. Finally, we’ll look closely at the systolic array — the heart of the TPU — and explain how data flows cycle by cycle.

---

# 1. CPU, GPU, and TPU: Different Answers to the Same Question

At a high level, all accelerators are trying to answer the same question:  
**How do we move data and compute results as efficiently as possible?**

A CPU is optimized for control. It executes a small number of instruction streams with sophisticated branch prediction, caches, and out-of-order execution. This makes it excellent for operating systems, orchestration, and irregular workloads — but inefficient for large matrix multiplications.

A GPU is optimized for throughput. Thousands of lightweight threads execute the same instruction across many data elements. This works well for graphics and ML, but GPUs still rely heavily on complex memory hierarchies, cache coherence, and scheduling overhead.

A TPU takes a more radical approach. It assumes that the dominant operation is **matrix multiplication**, that control flow is simple, and that memory access patterns are predictable. Instead of many independent cores, the TPU organizes compute around a **systolic array**, where data flows rhythmically through a fixed grid of processing elements.

The result is higher utilization, lower control overhead, and better energy efficiency — as long as your workload matches the assumptions.

---

# 2. TPU System View: Host, Queues, and Memory

A TPU does not operate in isolation. It is attached to a **host CPU**, which is responsible for orchestration, input preparation, and launching computation.

The typical execution flow looks like this:

The host CPU prepares input tensors and model parameters, then places input data into the **infeed queue**. From there, the TPU DMA engine transfers the data into **High Bandwidth Memory (HBM)**. HBM serves as the main on-device memory for parameters, activations, and intermediate results.

Once computation finishes, results are written back to HBM and then streamed into the **outfeed queue**, where the host CPU can read them asynchronously.

This queue-based interface is deliberate. It decouples the host from the accelerator’s execution timeline, allowing the TPU to run at full speed without frequent synchronization.

---

# 3. Matrix Execution on TPU: From HBM to MXU

The core computational unit of a TPU is the **MXU (Matrix Multiply Unit)**. Unlike GPUs, which schedule many independent threads, the TPU feeds large matrix blocks directly into the MXU.

The execution pipeline typically follows this pattern:

Model parameters are loaded from HBM into on-chip buffers. Input activations are also loaded from HBM into local storage. These buffers feed the MXU, which performs dense matrix multiplication at extremely high throughput.

The MXU computes partial products and accumulates them into partial sums. These intermediate results may be forwarded directly to another MXU operation — for example, chaining linear layers — or written back to HBM if needed by other operations such as non-linearities or normalization.

Crucially, the TPU design minimizes round-trips to HBM. Once data is on-chip, it is reused aggressively before being evicted. This is a key reason TPUs achieve high efficiency despite limited flexibility.

---

# 4. The Systolic Array: How the MXU Actually Works

At the heart of the MXU lies a **systolic array** — a two-dimensional grid of processing elements (PEs). Each PE performs a simple operation: multiply an input value with a weight and add it to a running partial sum.

What makes the systolic array powerful is **how data moves**.

Each clock cycle, three things happen simultaneously:

1. Each PE receives an input value from the PE above it.
2. Each PE receives a partial sum from the PE to its left.
3. The PE performs a multiply-accumulate (MAC), then forwards:
   - the input value to the PE below,
   - the updated partial sum to the PE on the right.

This creates a wave-like flow of data across the array — hence the term *systolic*, inspired by rhythmic biological processes.

Because every PE is always doing useful work once the pipeline is full, utilization is extremely high.

---

# 5. Data Skewing: Making the Pipeline Work

For a systolic array to operate correctly, data cannot be injected all at once.  
Instead, inputs must be **skewed in time** — deliberately delayed so that the correct operands arrive at the correct processing elements (PEs) on the same clock cycle.

Consider a simple matrix multiplication:

$$
C = A \times B
$$

where $(A)$ and $(B)$ are both $(3 \times 3\)$ matrices. In a systolic array, the elements of $(A\)$ are streamed into the array from the top, while elements of \(B\) are streamed in from the left. Crucially, these streams are *not* aligned in time. Each row of \(B\) and each column of \(A\) is offset by a fixed number of cycles so that matching elements meet inside the array.

At cycle $(t_0\)$, only the first elements of $(A\)$ and $(B\)$ enter the array. No full computation is possible yet — the pipeline is just beginning to fill.  
At cycle $(t_1\)$, the next elements are injected. Earlier values move one step forward, while new partial products begin forming.  
By cycles $(t_2\)$ and $(t_3\)$, values and partial sums are flowing diagonally across the grid, and multiply-accumulate operations are happening in every PE.  
Once the pipeline is fully filled, the array reaches a steady state: **each cycle produces one completed output element of $(C\)$**.

What makes this efficient is that all communication is strictly local. Each PE only exchanges data with its immediate neighbors — passing input values downward and partial sums to the right. No PE ever stalls waiting for memory, and no global synchronization is required once execution begins.

This deterministic, rhythmic data movement is what gives the systolic array its name. It contrasts sharply with GPU execution, where thousands of threads frequently pause to wait for memory fetches, cache coherence, or synchronization barriers. In a systolic array, data movement and computation are inseparable — every clock tick advances the computation forward.