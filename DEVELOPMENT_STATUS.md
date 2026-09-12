# DEVELOPMENT STATUS — MEMORYCORE (Track 5: Continual Learning)

**Date**: September 12, 2026  
**Status**: Baselines & MemoryCore (DER++) Implemented & Verified | UI Dashboard Live  
**Benchmark**: CORe50 NIC-v2-79 (79 Incremental Batches, 50-Object Classification)  

---

## 1. Specification & Protocol Alignment

Following expert research feedback, the master specification is strictly enforced:
1. **Terminology**: Clarified that NIC-v2-79 represents **79 incremental training batches** in a single continuous stream rather than disjoint multi-task splits.
2. **Official Filelist Grounding**: The exact composition of all 79 incremental batches is dynamically loaded from the official downloaded CORe50 filelists.
3. **Classification Target**: Explicitly formulated as **50 individual object identities** (the fine-grained, primary benchmark setting) rather than 10 broad categories.
4. **Baseline Reference Policy**: Official published CORe50 scores are treated as external reference points; all primary comparisons are performed against our own reproducible implementations under identical conditions.
5. **Metric Precision**: BWT defined as **Final Backward Transfer** ($BWT = \frac{1}{T-1}\sum_{j=0}^{T-2}(R_{T-1,j} - R_{j,j})$).
6. **Batch Order Variability**: Documented that CORe50 provides 10 official ordering runs (`run0`..`run9`). Results document the exact run.
7. **Research Method (MemoryCore)**: Formulated as **Dark Experience Replay (DER++)** combining reservoir replay buffer storage with past logit distillation ($\alpha \cdot \text{MSE} + \beta \cdot \text{CE}$).
8. **Experimental Integrity & Smoke Test Scope**:
   - **Synthetic Smoke Test**: Explicitly marked for **software & pipeline verification only**; never used for reported experimental numbers or tuning.
   - **Quarantine**: Test sessions `s3`, `s7`, `s10` are strictly quarantined from all training, buffer population, hyperparameter selection, and model selection.
   - **No Fake Numbers**: All reported metrics and UI demo numbers are read directly from disk artifacts.

---

## 2. Implementation Status Across All Methods

| Component / Method | Type | Status | Artifacts Generated |
| :--- | :--- | :--- | :--- |
| **Naive Sequential Fine-Tuning** | Mandatory Lower Bound | Verified | `results/smoke_naive/` (Matrix, Forgetting, PT, JSON) |
| **Joint Training** | Mandatory Upper Bound | Verified | `results/smoke_joint/` (Matrix, Logs, PT, JSON) |
| **Random Experience Replay** | Baseline | Verified | `results/smoke_replay/` (Matrix, Forgetting, PT, JSON) |
| **Elastic Weight Consolidation (EWC)** | Baseline | Verified | `results/smoke_ewc/` (Matrix, Forgetting, PT, JSON) |
| **Learning without Forgetting (LwF)** | Baseline | Verified | `results/smoke_lwf/` (Matrix, Forgetting, PT, JSON) |
| **MemoryCore (DER++)** | Research Method | Verified | `results/smoke_memorycore/` (Matrix, Forgetting, PT, JSON) |

---

## 3. Web UI & Visualization Lab

The interactive Continual Learning Lab is running at **`http://127.0.0.1:8000/`**:
- **Overview**: Dataset specifications, 50-object classes, scenario parameters, quarantine verification.
- **Task Stream**: Inspection of incremental batches 0..78 with frame counts and sessions.
- **Forgetting Demo**: Interactive step-by-step playback showing catastrophic forgetting in action.
- **Baselines Comparison**: Dynamic chart and leaderboard table showing Final Avg Accuracy, Forgetting, and BWT.
- **Memory Budget**: Sensitivity analysis comparing MemoryCore vs Random Replay across budgets (50..1000).
- **Accuracy Matrix**: Interactive heatmap with method selector (`smoke_naive`, `smoke_joint`, `smoke_replay`, `smoke_ewc`, `smoke_lwf`, `smoke_memorycore`).
- **Reproducibility Audit**: Environment details, seed 42, deterministic execution flags.
