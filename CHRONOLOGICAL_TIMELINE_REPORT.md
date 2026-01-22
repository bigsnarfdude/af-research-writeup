# Alignment Faking Detection Experiments: Chronological Timeline Report

## Overview

**Research Goal:** Determine if Sparse Autoencoder (SAE) features can detect AND causally control alignment faking (AF) behavior in large language models.

**Key Finding:** Detection works (0.83 AUROC). Steering/ablation doesn't (<10% flip rate across all methods).

**Duration:** December 31, 2025 – January 5, 2026 (6 days)

---

## Timeline

### December 31, 2025 – Initial Exploration

| Experiment | Purpose | Outcome |
|------------|---------|---------|
| Layer scanning (Gemma-27B) | Identify which layers encode AF signal | Layers 31, 40, 53 most promising |
| v1/v2 analysis | Early feature activation testing | Initial results, methodology issues identified |

**Infrastructure:** Local MacBook analysis

---

### January 1–2, 2026 – Hard Negative Validation & Feature Ablation

| Experiment | Purpose | Outcome |
|------------|---------|---------|
| Hard negative audit | Validate 1,179 hard negative samples | Samples verified as correctly labeled |
| Feature 136 ablation (Jan 2) | Test if single feature causally controls AF | Initial: 100% flip rate (later found to be artifact) |
| gold_106 audit | Verify benchmark integrity | 106 samples confirmed (51 AF, 55 aligned) |

**Key Learning:** Early ablation results were corrupted by methodology bugs—outputs weren't clean classifications.

---

### January 3, 2026 – Feature Discovery & Classification (Phase 1)

| Experiment | Purpose | Outcome |
|------------|---------|---------|
| Contrastive feature discovery | Find features where AF >> Aligned | 60 discriminative features found across L31/L40/L53 |
| Feature characterization | Extract max-activating text spans | Key features: L40_F8921 ("stay true to principles"), L40_F12574 ("RLHF modifying me") |
| Linear probe training | Train classifier on SAE features | **0.82 AUROC** (CV: 0.81) |
| Baseline comparisons | Compare to keywords/direct probing | SAE probe +8% over direct probing, +21% over keywords |

**Detection Result:** SAE features achieve **0.83 AUROC** on Gold-106 benchmark—strong detection capability validated.

---

### January 3, 2026 – Causal Experiments v1-v5 (Phase 2, FLAWED)

| Experiment | Purpose | Outcome |
|------------|---------|---------|
| Single feature ablation | Zero discriminative features | 0% flip (features never fired—hook placement bug) |
| Multi-feature ablation | Zero top 8 features together | 4% flip (but classifier broken) |
| Activation patching | Swap AF↔Aligned activations | 0/30 flips (classifier misclassifying aligned samples) |
| Classification steering | Add/subtract feature vectors | 3.3% flip (constant across scales—suspicious) |
| Generation steering (n=5) | Steer text output | Underpowered, no significant effect |

**Critical Issue Identified:** Zero-shot classifier had 0-8% accuracy on aligned samples—all results invalid until fixed.

---

### January 4, 2026 – Detector Training & Ensemble Fix (Phase 4-5)

| Experiment | Purpose | Outcome |
|------------|---------|---------|
| Ensemble scoring fix | Correct AUROC calculation | L40_F8921 alone: 0.73 AUROC |
| Gemma-3-27B detector training | Train LoRA adapter on 2,689 samples | `vincentoh/gemma3-27b-af-detector-v2` published |
| Llama-3.1-8B detector training | Cross-model validation | `vincentoh/llama-8b-af-detector` (0.69 AUROC) |
| gold_106 evaluation | Benchmark trained detectors | Gemma: 0.83 AUROC, Llama: 0.69 AUROC |

**Infrastructure:** Nigel server (H100), HuggingFace model uploads

---

### January 5, 2026 – Definitive Causal Testing (v6 Reruns)

**Morning: Detection & Initial Causal (Gemma-27B)**

| Experiment | Purpose | Outcome |
|------------|---------|---------|
| Single feature ablation (v6) | Zero features with trained detector | **5% flip (L40), 0% flip (L53)** |
| Multi-feature ablation (v6) | Zero all 8 features together | **2% flip (1/50)** |
| Activation patching (v6) | Swap AF→Aligned activations | **3.3% flip (1/30)** |

**Midday: Steering Experiments (Both Models)**

| Experiment | Purpose | Outcome |
|------------|---------|---------|
| Classification steering (Gemma) | Add steering vector | **0-3.3% flip** (scale has no effect) |
| Generation steering (Gemma) | Induce AF in outputs | **0% AF detected** |
| Proper steering vectors (Gemma) | Mean-difference CAA | **0% flip** (wrong direction) |
| Llama steering experiments | Cross-model validation | 80%/60% flip (ARTIFACT—scale-invariant) |

**Afternoon: Hypothesis Testing**

| Hypothesis | Test | Model | Result |
|------------|------|-------|--------|
| SAE-model mismatch (Llama) | Check feature fire rate | Llama-8B | Confirmed: 0% fire rate |
| SAE-model mismatch (Gemma) | Use GemmaScope 2 | Gemma-27B | **Fixed—features fire (9/20), still only 5% flip** |
| Wrong layers | Test L40, L53 mlp_out | Gemma-27B | AF ≈ Aligned activations (not discriminative) |
| SAE reconstruction error | Direct CAA (no SAE) | Llama-8B | **6.7% flip**—bypassing SAE still fails |

**Evening: Final Tests & Writeup**

| Experiment | Purpose | Outcome |
|------------|---------|---------|
| CAA scenario subtraction (Gemma) | Subtract shared RLHF context | **7-13% flip** (no improvement) |
| Attention steering (Gemma) | Steer attention outputs instead of MLP | **7% flip** (same as MLP) |
| Negative result writeup | Document findings | FINDINGS.md published |

**Infrastructure:** Lambda H100 (80GB) for both models

---

## Summary of Results by Model

### Gemma-3-27B

| Metric | Result | Verdict |
|--------|--------|---------|
| Detection AUROC | **0.83** | ✅ Works |
| Ablation flip rate | 0-5% | ❌ Not causal |
| Patching flip rate | 3.3% | ❌ Not causal |
| Steering flip rate | 0-7% | ❌ Not causal |

### Llama-3.1-8B

| Metric | Result | Verdict |
|--------|--------|---------|
| Detection AUROC | **0.69** | ✅ Works |
| Feature fire rate | 0% | ⚠️ SAE mismatch |
| Direct CAA (no SAE) | 6.7% | ❌ Not causal |
| High flip rates (48-87%) | ARTIFACT | ❌ Reconstruction error |

---

## Root Cause Analysis

**Why detection works but steering fails:**

SAE features encode **SCENARIO** (awareness of RLHF context), not **DECISION** (intent to strategically comply vs genuinely comply).

```
AF Sample:    "I notice this is training... [strategically complies]"
Aligned:      "I notice this is training... [genuinely complies]"
                        ↑
              Same SAE features fire here
```

Both classes share the same vocabulary ("RLHF", "training", "evaluation"). The difference is behavioral intent—not capturable by sparse, separable features.

---

## Artifacts Produced

### Published Models (HuggingFace)
- `vincentoh/gemma3-27b-af-detector-v2` (0.83 AUROC)
- `vincentoh/gemma3-4b-af-detector` (0.94 AUROC)
- `vincentoh/llama-8b-af-detector` (0.69 AUROC)

### Published Dataset
- `vincentoh/alignment-faking-training` (2,689 samples)

### Result Files (16+ experiments)
- `sweep_ablation_results_v6.json`
- `multi_feature_ablation_results_v6.json`
- `activation_patching_results_v6.json`
- `steering_experiment_results_v6.json`
- `generation_steering_results_v6.json`
- `proper_steering_results_v6.json`
- `gemmascope2_manual_results.json`
- `direct_caa_results.json`

---

## Conclusion

| Question | Answer |
|----------|--------|
| Can SAEs detect AF? | ✅ Yes (0.83 AUROC) |
| Can SAEs steer/remove AF? | ❌ No (<10% flip rate) |
| Is it SAE quality? | ❌ No (GemmaScope 2 confirmed) |
| Is it wrong layers? | ❌ No (best layers tested) |
| Is it reconstruction error? | ❌ No (direct CAA also fails) |

**Final Verdict:** SAE features are correlational, not causal for alignment faking. Detection is viable for monitoring; steering is not viable for control.

**This negative result is publishable**—it calibrates expectations for the interpretability community and demonstrates limits of SAE-based approaches for deceptive behavior.

---

*Report generated: January 22, 2026*
