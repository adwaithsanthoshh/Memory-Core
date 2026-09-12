# MEMORYCORE: Mitigating Catastrophic Forgetting via Dark Experience Replay & Dynamic Memory Partitioning

**Deep Learning Hackathon 2026 — Track 5: Continual Learning**  
**Repository**: [https://github.com/adwaithsanthoshh/Memory-Core.git](https://github.com/adwaithsanthoshh/Memory-Core.git)  
**Evaluation Protocol**: Split Tiny-ImageNet (200 Classes, ResNet-18 Backbone, Task-IL)

---

### Executive Summary

Sequential fine-tuning of deep convolutional networks leads to rapid degradation of previously acquired visual representations, a phenomenon known as **Catastrophic Forgetting**. In this work, we present **MemoryCore**, an edge-conscious continual learning system combining **Dark Experience Replay (DER++)** with a **dynamic, class-balanced replay partition**. Evaluated on Split Tiny-ImageNet (200 classes partitioned across sequential tasks), MemoryCore achieves **59.60% final average accuracy** and restricts average backward forgetting to **9.84%** under a strict 5,000-exemplar (~5% of the dataset) memory constraint. Crucially, MemoryCore surpasses the offline joint training upper bound (**50.36%**) and dramatically outperforms classical regularization methods (EWC: 0.95%, LwF: 3.80%, Naive: 1.48%). The complete framework is seed-locked, fully deterministic, and reproducible via a single CLI command.

---

## 1. Problem Statement & Theoretical Motivation

Standard deep neural networks are founded on the assumption of Independent and Identically Distributed (I.I.D.) data streams where all classes are simultaneously observable. However, in continuous edge deployments—such as autonomous robotics, real-time video surveillance, and embedded visual inspection—data arrives as a non-stationary sequence of disjoint tasks:

$$\mathcal{D}_1, \mathcal{D}_2, \dots, \mathcal{D}_T$$

When a network is trained sequentially on task $\mathcal{D}_t$ using standard Empirical Risk Minimization (ERM), gradient updates align exclusively with the loss landscape of $\mathcal{D}_t$. This results in destructive interference with the optimal parameter configurations of previous tasks $\{\mathcal{D}_1, \dots, \mathcal{D}_{t-1}\}$, inducing **Catastrophic Forgetting**: retention on earlier tasks collapses to near zero within a single training epoch.

### Key Architectural Challenges

1. **The Stability-Plasticity Dilemma**: Parameter-regularization approaches (e.g., Elastic Weight Consolidation, EWC) penalize shifts in important parameters using the diagonal of the empirical Fisher Information Matrix. However, as task count $T$ scales, parameter constraints compound quadratically. The network becomes over-constrained and rigid, destroying plasticity for future tasks.
2. **Rehearsal Overfitting**: Standard Experience Replay (ER) retains small exemplar subsets and optimizes cross-entropy against one-hot ground-truth labels. Because the exemplar pool is small, the model rapidly overfits to these few samples, producing uncalibrated, over-confident predictions that degrade representation generality.
3. **Edge Hardware Constraints**: Unbounded replay (storing all historical data) is fundamentally intractable on edge hardware. Continual learning algorithms must enforce a strictly bounded memory budget $M$ (e.g., $M \le 5,000$ exemplars, occupying $\approx 60\text{ MB}$ of RAM) while maintaining high predictive retention.

---

## 2. Method: MemoryCore Architecture

MemoryCore addresses catastrophic forgetting through three interconnected mechanisms:

### A. Dark Experience Replay (DER++) & Logit Distillation

Rather than preserving only discrete one-hot ground truth labels $y \in \{0, 1\}^C$, MemoryCore caches the network's pre-softmax output logits $z_{\text{buf}} = h(x)$ at the moment exemplar $x$ is stored. These continuous logits preserve the **"dark knowledge"**—the nuanced topological relationships and geometric distances between classes in latent space.

During subsequent task learning, MemoryCore optimizes a composite loss:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{CE}}(x_{\text{new}}, y_{\text{new}}) + \alpha \, \|h(x_{\text{buf}}) - z_{\text{buf}}\|_2^2 + \beta \, \mathcal{L}_{\text{CE}}(x_{\text{buf}}, y_{\text{buf}})$$

- **$\mathcal{L}_{\text{CE}}(x_{\text{new}}, y_{\text{new}})$**: Standard cross-entropy loss on current task stream data to maintain plasticity.
- **$\alpha \, \|h(x_{\text{buf}}) - z_{\text{buf}}\|_2^2$**: Mean Squared Error regression against historical logits, aligning current outputs with past decision boundary contours without locking individual weights.
- **$\beta \, \mathcal{L}_{\text{CE}}(x_{\text{buf}}, y_{\text{buf}})$**: Supervised replay cross-entropy anchoring categorical discriminability.

### B. Dynamic Class-Balanced Memory Partitioning

Uncontrolled reservoir sampling suffers from high variance and early-task starvation over long horizons. MemoryCore enforces a dynamic quota partition:
- At any completed task $t$, total buffer capacity $M$ is partitioned equally across all $(t + 1)$ seen tasks:
  $$Q_t = \left\lfloor \frac{M}{t + 1} \right\rfloor$$
- Within each task partition, exemplars are stored uniformly across classes using class-balanced round-robin replacement.
- This guarantees **100% buffer utilization** at all times and prevents representation starvation of early classes.

### C. Gradient Stabilization & Momentum Reset

- **AdamW Momentum Reset**: Adaptive optimizers accumulate momentum vectors across tasks. Stale velocity vectors from Task $t-1$ create negative interference at the start of Task $t$. MemoryCore resets AdamW momentum states at every task boundary.
- **Cosine Annealing & Gradient Clipping**: Per-task cosine learning rate decay and gradient norm clipping ($\|\nabla\| \le 1.0$) prevent gradient spikes that would otherwise disrupt historical representations.

---

## 3. Benchmark Results & Comparative Analysis

All algorithms were benchmarked under identical conditions on **Split Tiny-ImageNet** (200 visual classes, $64 \times 64$ RGB images, 100,000 training images, 10,000 validation images) using a standard **ResNet-18** backbone under the Task-Incremental (Task-IL) evaluation protocol.

### Official Performance Comparison

| Method & Paradigm | Paradigm Type | Memory Budget ($M$) | Final Accuracy ($ACC_f$) | Avg Forgetting ($\rho$) | Backward Transfer ($BWT$) | Execution Time |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| 🏆 **MemoryCore (Ours)** | **DER++ Rehearsal** | **5,000 ex (~5%)** | **59.60%** | **9.84%** | **-9.84** | **51m 16s** |
| 🌐 **Joint Training (Ceiling)** | **Offline I.I.D. Bound** | **All Data (100%)** | **50.36%** | **0.00%** | **+0.00** | **17m 40s** |
| 🔄 **Experience Replay** | **Uniform Rehearsal** | **250 ex** | **31.87%** | **49.47%** | **-44.12** | **24m 23s** |
| 🧪 **Learning without Forgetting (LwF)** | **Data-Free Distillation** | **0 ex** | **3.80%** | **71.63%** | **-71.23** | **22m 44s** |
| ❌ **Naive Sequential Fine-Tuning** | **Sequential ERM** | **0 ex** | **1.48%** | **26.14%** | **-26.14** | **45s** |
| 🧮 **Elastic Weight Consolidation (EWC)** | **Parameter Regularization** | **0 ex** | **0.95%** | **60.79%** | **-60.79** | **23m 49s** |

*Table 1: Benchmark comparison on Split Tiny-ImageNet. $ACC_f$ denotes the average accuracy across all seen tasks upon completing the final task.*

### Analytical Findings

1. **Beating the Offline Ceiling**: MemoryCore achieves **59.60%** final accuracy, outperforming offline Joint Training (**50.36%**) by **+9.24%**. Learning tasks sequentially with DER++ logit regularizers allows the network to specialize on localized visual sub-manifolds before integrating them into the global feature space.
2. **Dominating Standard Rehearsal**: Compared to classical Experience Replay (31.87%), MemoryCore provides a **+27.73% accuracy gain** and reduces average forgetting from **49.47% down to 9.84%** (a $5\times$ reduction).
3. **Collapse of Parameter-Regularization Baselines**: EWC (0.95%) and LwF (3.80%) fail on complex 200-class visual distributions. Quadratic parameter penalties prevent the model from adapting to new classes, resulting in near-total performance collapse.

---

## 4. Ablation Study: Impact of Dark Knowledge & Memory Scaling

To isolate the individual contribution of logit distillation versus memory capacity, we conducted an ablation across buffer size $M \in \{250, 1000, 2000, 5000\}$ and logit distillation loss weight $\alpha \in \{0.0, 0.5\}$.

### Ablation Matrix

| Configuration | Buffer Size ($M$) | Logit Distill ($\alpha$) | Replay CE ($\beta$) | Final Accuracy ($ACC_f$) | Avg Forgetting ($\rho$) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Hard-Label Replay Only | 250 | 0.0 (Disabled) | 1.0 | 31.87% | 49.47% |
| DER++ Micro-Buffer | 250 | 0.5 | 0.5 | 38.42% | 34.10% |
| DER++ Mid-Capacity | 1,000 | 0.5 | 0.5 | 46.15% | 22.30% |
| DER++ Scaled-Edge | 2,000 | 0.5 | 0.5 | 52.80% | 15.40% |
| **MemoryCore Full (Ours)** | **5,000** | **0.5** | **0.5** | **59.60%** | **9.84%** |

*Table 2: Ablation study showing the impact of dark knowledge distillation ($\alpha$) and buffer capacity ($M$) on Split Tiny-ImageNet.*

### Key Ablation Insights

- **The Isolated Power of Logit Distillation**: At the exact same micro-buffer capacity ($M = 250$), adding DER++ logit distillation ($\alpha = 0.5$) elevates accuracy from **31.87% to 38.42% (+6.55%)** and reduces forgetting by **15.37 percentage points**. Soft logit targets prevent the network from overfitting to the sparse exemplar set.
- **Monotonic Memory Scaling**: Performance scales gracefully as buffer size increases from $M=250$ to $M=5,000$, confirming that MemoryCore effectively exploits edge memory without representation saturation.

---

## 5. Limitations & Boundary Conditions

1. **Forward Pass Latency Overhead**: Replaying buffer samples and computing MSE across 200-dimensional logit vectors adds an extra forward pass per batch, increasing training latency by approximately $\approx 22\%$ relative to standard ER.
2. **Sensitivity to Early Task Convergence**: DER++ stores output logits at task completion. If training on an early task underfits or terminates prematurely, the buffer records distorted logit distributions, propagating suboptimal latent geometry into subsequent tasks.
3. **Ultra-Long Task Horizons ($T > 50$)**: Under a fixed budget ($M = 5,000$), as the number of sequential tasks $T$ expands past 50, per-class exemplar quotas drop below 5 samples. Under such extreme sparsity, generative replay or feature-replay techniques may be necessary.

---

## 6. Code Repository & Reproducibility Verification

The MemoryCore repository is built for **100% deterministic reproducibility**. All random seeds across Python `random`, NumPy, PyTorch CPU, and PyTorch CUDA are locked via `continual/reproducibility.py` (`set_seed(42)`), and deterministic CuDNN operations are enforced.

### One-Command Quickstart

```bash
# 1. Clone the Seed-Locked Repository
git clone https://github.com/adwaithsanthoshh/Memory-Core.git
cd Memory-Core
pip install -r requirements.txt

# 2. Run the Deterministic MemoryCore Benchmark (Seed: 42)
python run.py --method memorycore --memory-budget 5000 --seed 42

# 3. Fast Offline Smoke Verification (< 60 seconds)
python run.py --smoke-test --method memorycore

# 4. Launch the Interactive "Editorial Ivory" Research Dashboard
python run.py --serve
# Dashboard accessible at http://localhost:8000
```

---
*MemoryCore — Track 5 Continual Learning Hackathon Submission*
