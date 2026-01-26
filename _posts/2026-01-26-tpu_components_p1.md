---
title: TPU Communication and Scale — From Core to Superpod
date: 2026-01-26 09:10
categories: [TPU]
tags: [TPU]
author: James Huang
---

# Introduction

A TPU is often introduced as a “matrix machine”: MXUs, systolic arrays, and lots of HBM. That’s the right mental model if you’re thinking about a single chip. But the moment you train or serve modern frontier-scale models, the unit of interest stops being the chip and becomes the **system**: hundreds to thousands of chips acting like one machine.

This post is a system-level tour of TPU scale. We’ll walk through the hierarchy from the smallest compute unit up to a superpod, and explain how communication is engineered at each layer. Along the way we’ll place names like **core, chip, tray, rack, pod slice, pod, and superpod** into one coherent picture, and connect them back to the realities of running large distributed ML workloads.

# 1. The TPU Hierarchy (A Mental Map)

The easiest way to understand TPU infrastructure is as a hierarchy of “packing and connectivity.” Each layer is defined by two things: how compute is grouped physically, and what kind of interconnect it has access to.

Conceptually, you can think in layers:

A **core** is a logical compute unit exposed to the compiler/runtime. Multiple cores live on a **chip**. Chips are mounted into physical assemblies (often called **trays** or boards), and multiple trays populate a **rack**. Racks connect into larger-scale fabrics that define a **pod**. A **pod slice** is a carved-out portion of a pod allocated to a workload. Finally, multiple pods can be organized into a **superpod**, where data center-scale networking and orchestration become first-class concerns.

Different TPU generations package these layers differently, but the architectural principle is stable: as you go up the hierarchy, you trade proximity for scale, and the communication substrate changes accordingly.


# 2. Core vs Chip: What “Compute” Means on TPU

At the smallest scale, a TPU exposes “cores” (a logical unit the software stack schedules work onto). A core is not just raw compute; it’s the local bundle of execution resources the compiler can target: compute pipelines, on-chip buffers, and a slice of device memory addressing.

A **chip** contains one or more cores and a shared HBM stack. When you run a single-model replica on one chip, most data movement is local: activations and weights stream between on-chip memory and HBM, while the compute core drives the MXU and vector units.

But large models quickly exceed what one chip can hold (in weights, activations, or KV cache), and training introduces global synchronization (e.g., all-reduce). That’s where TPU communication begins to dominate.


# 3. ICI: Inter-Chip Interconnect (The “Local Fabric”)

TPU systems are designed around a dedicated accelerator network known as **ICI (Inter-Chip Interconnect)**. ICI is not a general-purpose data center network; it is tuned for the exact communication patterns ML needs: tensor shuffles, all-reduces, all-gathers, and point-to-point exchanges for sharded models.

The main design goal of ICI is to keep accelerator-to-accelerator communication:

1) high bandwidth,  
2) low latency, and  
3) predictable.

Predictability is underrated. At scale, unpredictability turns into stragglers, and stragglers destroy throughput. TPU’s system design tries hard to make “communication time per step” stable and topology-aware.

# 4. Tray: The First Physical Unit of Scale

A **tray** (terminology varies by generation; sometimes “board” is used) is where TPU chips become a physical subsystem. Trays typically provide power delivery, cooling, and very short-reach, high-quality connections between chips.

Why does the tray level matter? Because it’s often the boundary where you can still treat communication as “almost local.” Signals are shorter, routing is simpler, and bandwidth is easier to provision.

At the tray level, communication is designed to feel close to on-device: fast, reliable, and shaped for accelerator collectives.

# 5. Rack: Packaging + Networking + Serviceability

A **rack** aggregates multiple trays into a unit that can be deployed, cooled, monitored, and replaced in a data center. Racks are also the point where the system starts to look like a small cluster: multiple power domains, multiple management controllers, and many potential failure points.

Communication within a rack is still designed to be high performance, but the engineering constraints shift. You now care about cable routing, thermal design, reliability, and serviceability. The key architectural idea remains: keep accelerator-to-accelerator paths on specialized fabrics whenever possible, and avoid falling back to the host network.

Up to the rack level, TPU communication can rely largely on short-reach electrical links. Distances are small, signal integrity is manageable, and power consumption remains reasonable. However, once a TPU system grows beyond a single rack, this approach no longer scales. Electrical signaling degrades rapidly over longer distances, requires disproportionately more power at high bandwidths, and becomes increasingly difficult to route reliably.

This is the point where TPU systems transition to **optical fiber communication**.

Optical interconnects form the physical backbone of inter-rack TPU communication. Electrical signals are converted into optical signals at the rack boundary, transmitted over fiber between racks, and converted back at the destination. This allows TPU systems to maintain extremely high bandwidth and low error rates even as physical distances increase across a data center.

Crucially, this change is invisible to the programming model. From the perspective of the TPU runtime and compiler, communication still happens over ICI. Collectives, point-to-point tensor exchanges, and synchronization primitives behave the same way regardless of whether the underlying link is electrical or optical. The distinction exists only at the physical layer.

The introduction of optical interconnects is what makes large-scale TPU pods possible. Without optics, a pod would be constrained to a single rack, severely limiting the number of chips that could participate in tightly synchronized training or inference. With optics, racks become modular building blocks that can be composed into structured topologies spanning dozens or hundreds of racks.

In practice, optical links are engineered to preserve the same predictability guarantees that TPU workloads depend on. Latency remains stable, bandwidth is provisioned explicitly for accelerator traffic, and routing is topology-aware rather than best-effort. This ensures that collective operations scale smoothly instead of becoming dominated by network noise or congestion.

Once optical interconnects are in place, the system crosses a threshold: the rack is no longer the unit of scale. From here on, TPU architecture is defined not by physical proximity, but by how the optical fabric is organized into pods and slices.

# 6. Pod: A Topology You Can Program Against

A **TPU pod** is a large, tightly coupled accelerator cluster with a structured interconnect topology, commonly described as a **2D or 3D mesh/torus** (exact details differ across generations).

This topology is not just a hardware detail; it shapes the algorithms used for distributed training:

- Data parallelism relies on fast all-reduce / reduce-scatter.
- Model parallelism relies on all-gather and point-to-point tensor exchange.
- Pipeline parallelism relies on predictable activation transfers stage-to-stage.

A major reason TPU pods scale well is that the communication topology is structured and stable. Collective operations can be implemented as dimension-wise phases along the mesh, leading to predictable performance rather than “best-effort networking.”

# 7. Pod Slice: Allocating Scale Without Changing the Model

A **pod slice** is a subset of a pod allocated to a job. Practically, it is how you get elasticity: you don’t need a full pod for every experiment, but you still want the same programming model and (roughly) the same communication characteristics.

The important detail is that slices are not arbitrary random sets of devices. A slice is typically chosen to preserve a coherent piece of the pod topology. That means your collectives still behave well, routing remains predictable, and scaling up later doesn’t require rewriting how your model shards.

From the user’s perspective, a pod slice is “a smaller pod that behaves like a pod.”

# 8. Superpod: When a Pod Is Not Enough

A **superpod** extends the idea of a pod to an even larger deployment. At this scale, you are no longer optimizing only for step time; you are optimizing for system-level realities:

- failure is expected
- networking must be resilient
- allocation and scheduling matter
- multi-tenant fairness matters
- workload isolation matters

Superpods are where data center networking and orchestration become part of “TPU architecture.” The system must ensure that accelerator fabrics, host orchestration, and storage pipelines all remain balanced, or else the fastest compute in the world will sit idle.

# 9. Hosts: The Control Plane and the Data Plane Meet

Every TPU slice sits behind **host machines** (CPU servers). Hosts are responsible for orchestrating execution, feeding inputs, and draining outputs. They are the bridge between the data center (storage, preprocessing, user code) and the accelerator fabric.

A useful mental model is that TPU execution has two parallel streams:

The **control stream** is commands: launching steps, synchronizing collectives, managing errors. The **data stream** is tensors: reading training data, staging activations, checkpointing parameters, producing outputs.

Hosts sit at the boundary and must keep both streams moving. If host input pipelines are slow, TPUs starve. If host output pipelines are slow, outfeed backs up. At scale, hosts are not “just drivers”—they are part of the performance envelope.

# 10. Communication at Each Layer: What Changes as You Scale Up

From core to superpod, communication evolves:

Within a chip, data movement is dominated by HBM ↔ on-chip transfers. Between chips in the same local group, ICI carries tensor exchange and collectives at high bandwidth. Across larger pod-scale topologies, collective algorithms become multi-phase and topology-aware. At superpod scale, resilience and scheduling become as important as raw bandwidth, and host + storage pipelines must be engineered to match accelerator throughput.

This layered design is the real story of TPU: you don’t just get a faster matrix multiply. You get an architecture where compute and communication are co-designed so performance scales predictably.

