---
title: Difussion Model - Basics
date: 2026-02-01 10:00
categories: [ML]
tags: [Difussion Model]
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
In just a few years, the landscape of generative AI has been completely rewritten.
From text to images, music, and video achievements point to the same type of model: Diffusion Models.

At first glance, the name Diffusion Model sounds almost counterintuitive.
Diffusion is usually associated with decay, randomness, and loss of structure — not creation.

# Why is it called Diffusion? 
In physics, diffusion describes a simple but irreversible phenomenon:

* Ink spreads in water
* Heat flows from hot regions to cold ones
* Gas particles disperse from dense areas into empty space

![Ink diffusion in water](assets/images/diffusion/water_difussion.jpg)

In every case, diffusion follows a similar pattern of Structured → Unstructured
This is diffusion's natural behavior, it is simply **entropy increasing**.

![sculpture](assets/images/diffusion/marble.png)
> “I saw the angel in the marble and carved until I set him free.”



# The Forward Process of Diffusion
If diffusion is entropy increasing, then the forward process is where we deliberately let entropy grow.
The forward process describes how a clean data sample is gradually corrupted by noise.

### From Data to Noise

The forward process describes how a clean data sample is gradually corrupted by noise.

Starting from an original data point $x_0$, we add a small amount of Gaussian noise at each step:

$$
x_0 \rightarrow x_1 \rightarrow x_2 \rightarrow \dots \rightarrow x_T
$$

As time progresses, meaningful structure fades away.  
After enough steps, the data becomes indistinguishable from pure noise.

### A Simple and Fixed Corruption Rule

In DDPM, the forward process is defined as:

$$
q(x_t \mid x_{t-1}) =
\mathcal{N}\!\left(\sqrt{1-\beta_t}\,x_{t-1},\; \beta_t I\right)
$$

Here:

- $\beta_t$ controls how much noise is added at step $t$
- Each step slightly increases randomness
- No learning is involved

The entire corruption process is predefined and fixed.

Intuitively, each step of the forward process does the same thing:  
it slightly scales down the current sample and adds a small amount of Gaussian noise:

$$
x_t = \sqrt{1-\beta_t}\,x_{t-1} + \sqrt{\beta_t}\,z_t,
\quad z_t \sim \mathcal{N}(0, I).
$$

![xt given x0](assets/images/diffusion/forward_xo_given_xt.png)

Recall that $x_t$ is sampled from a Gaussian distribution conditioned on $x_{t-1}$:

$$
x_t \sim \mathcal{N}\!\left(\sqrt{1-\beta_t}\,x_{t-1},\; \beta_t I\right).
$$

Using reparameterization, this distribution can be written explicitly as:

$$
x_t = \sqrt{1-\beta_t}\,x_{t-1} + \sqrt{\beta_t}\,\epsilon_{t-1},
\quad \epsilon_{t-1} \sim \mathcal{N}(0, I).
$$

That is, $x_t$ can be obtained by scaling $x_{t-1}$ and adding a sample from a standard normal distribution.

From the properties of Gaussian distributions, we know that the sum of independent Gaussians with the same mean is still Gaussian, and their variances add up.

In particular,

$$
\mathcal{N}(0, \sigma_1^2 I) + \mathcal{N}(0, \sigma_2^2 I)
= \mathcal{N}\!\left(0, (\sigma_1^2 + \sigma_2^2) I\right).
$$

Using this property, the noise terms in the previous expression can be combined into a single Gaussian noise variable.  
As a result, we can simplify the expression for $x_t$ as:

$$
x_t
= \sqrt{(1-\beta_t)(1-\beta_{t-1})}\,x_{t-2}
+ \sqrt{1 - (1-\beta_t)(1-\beta_{t-1})}\,\epsilon,
\quad \epsilon \sim \mathcal{N}(0, I).
$$

If we expand one more step forward, we obtain:

$$
x_t
= \sqrt{(1-\beta_t)(1-\beta_{t-1})(1-\beta_{t-2})}\,x_{t-3}
+ \sqrt{1 - (1-\beta_t)(1-\beta_{t-1})(1-\beta_{t-2})}\,\epsilon.
$$

This pattern continues recursively, revealing how the signal coefficient becomes a product of $(1-\beta)$ terms, while all noise contributions collapse into a single Gaussian term.

If we repeat this operation for $t$ steps, the result can be written in a clean closed form.  

Here we define

$$
\alpha_t = 1 - \beta_t,
\qquad
\bar{\alpha}_t = \prod_{s=1}^{t} \alpha_s.
$$

The noisy sample $x_t$ is equivalent to keeping a fraction of the original data $x_0$ and adding a single aggregated noise term:

$$
x_t = \sqrt{\bar{\alpha}_t}\,x_0 + \sqrt{1-\bar{\alpha}_t}\,\epsilon,
\quad \epsilon \sim \mathcal{N}(0, I).
$$

**Key takeaway.** Given $x_0$, we can sample $x_t$ in one shot—no need to generate $x_1,\ldots,x_{t-1}$. This is only possible during training because $x_0$ is known. 

# The Reverse Diffusion Process

Instead of adding noise, we want to **undo** the corruption step by step—turning a pure noise sample back into a data sample that resembles those in the training set.
If this denoising process can be learned, then generation becomes possible.

Starting from a noise sample drawn from a standard normal distribution, the reverse process gradually removes noise until a structured image emerges.

### What Does “Denoising” Mean Mathematically?

When the noise level $\beta_t$ is sufficiently small, theory shows that the reverse of each forward noising step can be approximated by a Gaussian distribution.
Specifically, the reverse transition takes the form:

$$
x_{t-1} \sim \mathcal{N}\!\left(\tilde{\mu}_t,\; \tilde{\beta}_t I\right).
$$

This means that, conditioned on the current noisy sample $x_t$, the previous sample $x_{t-1}$ is distributed as a Gaussian.

### So what are the mean and variance of the reverse denoising step?

The difficulty of computing the true reverse transition can be seen explicitly by writing it out:

$$
q(x_{t-1} \mid x_t)
\;\propto\;
\int q(x_t \mid x_{t-1}) \, q(x_{t-1} \mid x_0) \, q(x_0)\, dx_0.
$$

This expression shows that the exact reverse distribution requires integrating over
**all possible clean data samples** $x_0$, weighted by their unknown data density $q(x_0)$.
Because this data distribution is not available in closed form, the true reverse transition
cannot be computed directly in practice.

However, if we **condition on a specific clean training sample** $x_0$, the reverse distribution becomes tractable.

> Conditioning on a specific clean sample does not mean fixing a single image during training.
At each training step, a new $x_0$ is independently sampled from the data distribution.
The conditioning is only used to derive the reverse posterior for that particular training example.

In particular, we can derive the posterior $q(x_{t-1}\mid x_t, x_0)$ using Bayes' rule:

$$
q(x_{t-1}\mid x_t, x_0)
=
\frac{q(x_t\mid x_{t-1}, x_0)\,q(x_{t-1}\mid x_0)}{q(x_t\mid x_0)}.
$$

Because the forward diffusion process forms a Markov chain
$x_0 \rightarrow x_{t-1} \rightarrow x_t$,
the joint distribution factorizes as

$$
q(x_{t-1}, x_t, x_0)
=
q(x_t \mid x_{t-1})\,q(x_{t-1} \mid x_0)\,q(x_0),
$$

and similarly,

$$
q(x_t, x_0) = q(x_t \mid x_0)\,q(x_0).
$$

Substituting these expressions and canceling $q(x_0)$ yields

$$
q(x_{t-1} \mid x_t, x_0)
=
\frac{q(x_t \mid x_{t-1})\,q(x_{t-1} \mid x_0)}
     {q(x_t \mid x_0)}.
$$

This decomposition is crucial: all terms on the right-hand side are known
and Gaussian under the forward process, making the reverse posterior analytically tractable.

Refer: https://arxiv.org/pdf/2208.11970 eq 84

### Reverse Posterior and Its Closed-Form Parameters

From the forward diffusion process, the reverse posterior conditioned on a clean sample is Gaussian:

$$
q(x_{t-1} \mid x_t, x_0)
\propto
\mathcal{N}\!\left(
x_{t-1};
\mu_q(x_t, x_0),
\Sigma_q(t)
\right)
\quad \text{(Eq. 84)}
$$

where the mean and covariance are

$$
\mu_q(x_t, x_0)
=
\frac{
\sqrt{\alpha_t}(1-\bar{\alpha}_{t-1})x_t
+
\sqrt{\bar{\alpha}_{t-1}}(1-\alpha_t)x_0
}{
1-\bar{\alpha}_t
},
$$

$$
\Sigma_q(t)
=
\frac{(1-\alpha_t)(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t} I.
$$

---

### Expressing the Posterior Mean via Noise $\epsilon_t$

From the closed form of the forward process,

$$
x_t
=
\sqrt{\bar{\alpha}_t}x_0
+
\sqrt{1-\bar{\alpha}_t}\,\epsilon_t,
\quad
\epsilon_t \sim \mathcal{N}(0, I),
$$

we can solve for $x_0$:

$$
x_0
=
\frac{1}{\sqrt{\bar{\alpha}_t}}
\left(
x_t - \sqrt{1-\bar{\alpha}_t}\,\epsilon_t
\right).
$$

Substituting this expression into $\mu_q(x_t, x_0)$ and simplifying yields

$$
\tilde{\mu}_t
=
\frac{1}{\sqrt{\alpha_t}}
\left(
x_t
-
\frac{1-\alpha_t}{\sqrt{1-\bar{\alpha}_t}}
\epsilon_t
\right).
$$

## Training Objective

![training](assets/images/diffusion/training.png)

During training, the true noise $\epsilon_t$ is known because it is sampled explicitly
when constructing $x_t$.
Therefore, we can train the network to predict this noise by minimizing a simple mean squared error:

$$
L
=
\left\lVert
\epsilon_t - \epsilon_\theta(x_t, t)
\right\rVert^2.
$$

This objective encourages the model to accurately estimate the noise component at every timestep.


### Why This Works

Predicting noise is equivalent to predicting the reverse mean,
since $\tilde{\mu}_t$ can be written directly in terms of $\epsilon_t$.
In practice, noise prediction is numerically stable, easy to implement,
and leads to better training behavior.

As a result, most diffusion models—including DDPM—are trained
by learning to predict noise rather than predicting the denoised sample directly.

## Sampling: Generating Data by Reverse Diffusion

![Sampling](assets/images/diffusion/sampling.png)

Starting from timestep $T$, we iteratively apply the reverse process from $t = T$ down to $t = 1$.
At each step, we compute the mean and variance of the reverse transition and sample $x_{t-1}$.

The reverse mean is given by

$$
\mu_\theta(x_t, t)
=
\frac{1}{\sqrt{\alpha_t}}
\left(
x_t
-
\frac{1-\alpha_t}{\sqrt{1-\bar{\alpha}_t}}
\,\epsilon_\theta(x_t, t)
\right),
$$

where $\epsilon_\theta(x_t, t)$ is the neural network’s prediction of the noise at timestep $t$.

### Choosing the Variance

There are two common choices for the variance $\sigma_t^2$, both of which work well in practice.

**Posterior variance (derived):**

$$
\sigma_t^2
=
\frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_t}\,\beta_t
$$

This choice follows directly from the reverse posterior derived earlier and is typically used during training.

**Simplified variance:**

$$
\sigma_t^2 = \beta_t
$$

When $x_0 \sim \mathcal{N}(0, I)$, using the forward-process variance often produces comparable results.

### Sampling Step

At each timestep $t$, we sample

$$
x_{t-1}
=
\mu_\theta(x_t, t)
+
\sigma_t z,
\quad
z \sim \mathcal{N}(0, I).
$$

This process is repeated iteratively until we reach $x_0$.
The final $x_0$ is the generated sample.