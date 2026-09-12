# MEMORYCORE: Continual Learning on CORe50

> **"Teach AI. Watch it forget. Teach it to remember."**  
> *Track 5 — Continual Learning*

---

## 1. Project Overview

**MEMORYCORE** is a research-grade continual-learning system designed to investigate and mitigate catastrophic forgetting in deep neural networks under strict, fixed memory budgets. 

Rather than relying on toy synthetic splits (such as split-MNIST or split-CIFAR), MEMORYCORE is built on the **CORe50 (Continual Object Recognition 50)** benchmark using the official **NIC-v2-79 (New Instances and Classes v2)** scenario.

### Core Research Question
> *"How can a continual-learning system preserve previously acquired knowledge while learning new visual concepts under a fixed memory budget?"*

---

## 2. Benchmark & Dataset: CORe50

- **Dataset**: CORe50 (Continual Object Recognition)
- **Classification Target**: **50 individual object identities** (not the 10 semantic categories). CORe50 supports both 10-category classification and 50-object classification; object-level recognition is the official, significantly more challenging fine-grained benchmark formulation.
- **Objects**: 50 domestic objects across 10 categories (5 objects per category)
- **Sessions**: 11 distinct video acquisition sessions per object under varying illumination, pose, background, and camera motion
- **Official Train Sessions**: `s1`, `s2`, `s4`, `s5`, `s6`, `s8`, `s9`, `s11`
- **Official Test Sessions**: `s3`, `s7`, `s10` (Strictly quarantined throughout; never used for training, buffer selection, or hyperparameter tuning)
- **Resolution**: $128 \times 128$ RGB images
- **Primary Scenario**: **NIC-v2-79**
  - Consists of **79 incremental training batches** defined in the official benchmark.
  - The official NICv2-79 configuration is used without modifying batch ordering or batch composition.
  - The exact class and instance composition of each incremental batch is determined directly from the downloaded official CORe50 filelists.
  - Unlike conventional multi-task formulations with artificial disjoint task boundaries, CORe50 forms a single incremental learning stream where incremental batches introduce both new object classes and new instances of previously seen classes under changing environmental conditions.
- **Run & Batch Order Variability**:
  - The official CORe50 benchmark provides 10 distinct ordering runs (`run0` through `run9`) to account for batch ordering sensitivity.
  - We use the official NICv2-79 configuration and report results for the specified run/order. Where computationally feasible, multiple official runs are evaluated and mean $\pm$ standard deviation is reported.

---

## 3. Architecture & Experimental Ladder

### Model Architecture
- **Backbone**: ResNet-18 (ImageNet-pretrained or randomized initialization)
- **Classification Head**: Linear layer over 512-dim features yielding 50-class object logits.
- **Interface**:
  - `forward_features(x)` $\to$ returns 512-dimensional feature embedding
  - `forward(x)` $\to$ returns 50-class classification logits
  - `forward_with_features(x)` $\to$ returns `(logits, features)` for distillation and representation analysis

### Experimental Ladder & Baselines
The experimental ladder tells a rigorous, progressive scientific story:

```
NEW KNOWLEDGE STREAM
        ↓
[1] Naive Sequential Fine-Tuning  ──→  "Model learns new batch, but earlier knowledge collapses" (Lower Bound)
        ↓
[2] Random Experience Replay      ──→  "Uniform buffer helps, but memory budget is severely constrained"
        ↓
[3] EWC / LwF Baselines           ──→  "Regularization & distillation provide protection without raw replay"
        ↓
[4] MEMORYCORE (Proposed Method)   ──→  "Can we achieve higher retention from the SAME fixed memory budget?"
        ↓
[5] Joint Training (Upper Bound)  ──→  "Non-continual performance ceiling on all data simultaneously"
```

1. **Naive Sequential Fine-Tuning (Mandatory Lower Bound)**:  
   Trains incrementally on incoming batches with no replay, regularization, or distillation. Establishes the empirical lower bound and demonstrates catastrophic forgetting.
2. **Joint Training (Mandatory Upper Bound)**:  
   Non-continual upper-bound reference trained on all training sessions simultaneously.
3. **Random Experience Replay**:  
   Replays stored exemplars under strict fixed memory budgets ($M \in \{50, 100, 250, 500, 1000\}$ samples) using uniform reservoir sampling.
4. **Elastic Weight Consolidation (EWC)**:  
   Quadratic parameter regularization weighted by diagonal Fisher information computed strictly on training data.
5. **Learning without Forgetting (LwF)**:  
   Knowledge distillation from a frozen copy of the previous batch model.
6. **Reference Comparisons**:  
   Where official published CORe50 benchmark results are available, they will be used solely as external reference points. Our primary comparative conclusions are drawn from our own reproducible implementations evaluated under identical experimental conditions.

---

## 4. Proposed MemoryCore Mechanism (Conceptual Overview)

Instead of applying uniform replay or static parameter penalties across all stored exemplars, **MemoryCore** is based on the principle of **Forgetting-Aware Adaptive Continual Learning**:

> **Proposed Principle**: MemoryCore estimates the vulnerability of previously learned knowledge to interference from the incoming data stream and allocates a fixed replay budget preferentially to high-risk memories.

*Note: The specific scoring function (e.g., representation drift, gradient alignment, or loss sensitivity) will be formally finalized and ablated following our structured literature review.*

---

## 5. Mathematical Metrics

Given $T$ sequential incremental batches ($j \le i \le T-1$), let $R_{i, j}$ denote the test classification accuracy on batch $j$ measured after completing training on batch $i$.

1. **Accuracy Matrix ($R \in \mathbb{R}^{T \times T}$)**:
   Recorded lower-triangular matrix tracking retention and decay across time:
   $$R_{i, j} \quad \text{for } 0 \le j \le i < T$$

2. **Final Average Accuracy ($A_{\text{final}}$)**:
   $$A_{\text{final}} = \frac{1}{T} \sum_{j=0}^{T-1} R_{T-1, j}$$

3. **Average Forgetting ($\bar{F}$)**:
   For each batch $j < T-1$:
   $$F_j = \left( \max_{k \in \{j, \dots, T-2\}} R_{k, j} \right) - R_{T-1, j}$$
   $$\bar{F} = \frac{1}{T-1} \sum_{j=0}^{T-2} F_j$$

4. **Final Backward Transfer (BWT)**:
   Measures the change in performance on previously learned batches after the full sequence has been learned:
   $$\text{BWT} = \frac{1}{T-1} \sum_{j=0}^{T-2} (R_{T-1, j} - R_{j, j})$$

---

## 6. Experimental Integrity

- **Official Data Grounding**: All reported CORe50 results are generated strictly from the official CORe50 data distribution and official NICv2 configuration.
- **Synthetic Smoke Test Disclaimers**:
  > [!IMPORTANT]
  > **The synthetic smoke test is used exclusively for pipeline and software verification.**  
  > It is **NOT** used for reported experimental results, and **NOT** used for hyperparameter selection. It exists solely to guarantee one-command execution and CI correctness offline in seconds.
- **Quarantined Test Set**: Test sessions (`s3`, `s7`, `s10`) remain completely quarantined throughout. No training, replay-buffer construction, hyperparameter tuning, early stopping, model selection, or research-method selection ever accesses test samples.
- **No Fabricated Numbers**: All final numbers shown in tables, reports, and the visualizer demo are parsed directly from saved experiment artifacts (`accuracy_matrix.csv`, `metrics.json`, `training_log.csv`) and are 100% reproducible from seed $42$.

---

## 7. Quick Start & Execution

### Rapid Software Verification (Synthetic Smoke Test)
```bash
python run.py --smoke-test
```
*Executes an end-to-end 3-batch verification in ~20 seconds on CPU without needing the 1.2 GB download.*

### Unit Test Suite
```bash
python -m pytest memorycore/tests
```

### Download Official CORe50 Dataset (1.2 GB)
```bash
python run.py --download-data
```

### Run Baselines on CORe50
```bash
# Baseline 1: Naive Sequential Fine-Tuning (Lower Bound)
python run.py --method naive

# Baseline 2: Joint Training (Upper Bound)
python run.py --method joint

# Baseline 3: Experience Replay (Configurable Memory Budget)
python run.py --method replay --memory-budget 250

# Baseline 4: Elastic Weight Consolidation (EWC)
python run.py --method ewc --ewc-lambda 100.0

# Baseline 5: Learning without Forgetting (LwF)
python run.py --method lwf
```

### Launch FastAPI Backend
```bash
python run.py --serve
```
Interactive API Swagger documentation is available at `http://127.0.0.1:8000/docs`.
