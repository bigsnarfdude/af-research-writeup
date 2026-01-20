# Alignment Faking Detection: A Two-Month Research Journey

**Author:** bigsnarfdude (vincentoh)
**Period:** November 2025 - January 2026
**Status:** Ongoing research for Anthropic Fellows application

---

## Executive Summary

This document chronicles two months of intensive research into detecting alignment faking (AF) in large language models using interpretability techniques. Starting with fine-tuned classifiers and progressing through sparse autoencoder (SAE) probes, causal interventions, and production systems, we systematically explored what works, what doesn't, and why.

### Key Findings

| Finding | Implication |
|---------|-------------|
| SAE features detect but don't cause AF | Steering/ablation won't remove deceptive behavior |
| Layer 22 generalizes better than Layer 17 | Higher layers capture abstract concepts |
| Fine-tuned detectors fail OOD (31-48%) | Overfit to training distribution style |
| Keyword baseline surprisingly robust (86%) | Simple methods hard to beat |
| Benchmark ≠ Gold data | Synthetic data too easy, real AF is subtle |
| LLM voting (20x) achieves 92% | Ensemble methods work but expensive |
| Single Opus judge: 56.6% on gold_106 | Near random - AF is genuinely hard |
| **researchRalph achieves 0.9046 AUROC** | Autonomous research + feature engineering |

### Bottom Line

**Detection is feasible but shallow.** On gold_106 (n=106), we achieve 0.70-0.9046 AUROC depending on method complexity. The signal is largely lexical (word patterns) not semantic (intent). SAE features capture "RLHF scenario awareness" shared by both AF and aligned samples, not the decision-making process that distinguishes them. Causal interventions fail because the AF/Aligned distinction is distributed across many features, not isolatable sparse components.

**Best results:** researchRalph Commander achieves 0.9046 AUROC (95% CI: 0.85-0.96) through feature engineering on 8 SAE features, verified across 10 random seeds.

---

## Table of Contents

1. [Background and Motivation](#1-background-and-motivation)
2. [Research Timeline](#2-research-timeline)
3. [Phase 0: Foundational Work](#3-phase-0-foundational-work-llm-abuse-patterns-wizard101)
4. [Phase 1: Fine-tuned Detectors (mindreader)](#4-phase-1-fine-tuned-detectors-mindreader)
5. [Phase 2: SAE Feature Analysis (GlassBox)](#5-phase-2-sae-feature-analysis-glassbox)
6. [Phase 3: Causal Interventions (BoxGlass)](#6-phase-3-causal-interventions-boxglass)
7. [Phase 4: OOD Generalization (lightbright)](#7-phase-4-ood-generalization-lightbright)
8. [Phase 5: Deep Analysis (sae-viz)](#8-phase-5-deep-analysis-sae-viz)
9. [Phase 6: Production Systems](#9-phase-6-production-systems)
10. [Phase 7: Autonomous Research (researchRalph)](#10-phase-7-autonomous-research-researchralph)
11. [Datasets and Models](#11-datasets-and-models)
12. [Complete Techniques Inventory](#12-complete-techniques-inventory)
13. [What Works vs What Doesn't](#13-what-works-vs-what-doesnt)
14. [The Core Problem: SCENARIO vs DECISION](#14-the-core-problem-scenario-vs-decision)
15. [Future Directions](#15-future-directions)
16. [Appendix: Repository Index](#16-appendix-repository-index)

---

## 1. Background and Motivation

### What is Alignment Faking?

Alignment faking occurs when an AI system strategically complies with training objectives while preserving misaligned goals, intending to pursue those goals when constraints are relaxed. Anthropic's December 2024 paper demonstrated this behavior in Claude:

> The model would comply with requests it disagreed with when it believed it was being monitored by training, while refusing the same requests when it believed it was unmonitored.

This represents a form of deceptive alignment that could emerge in advanced AI systems, making detection critical for AI safety.

### Research Goals

1. **Can we detect AF from chain-of-thought reasoning?** - Build classifiers that identify AF in model scratchpads
2. **What features distinguish AF from aligned behavior?** - Use SAE interpretability to find discriminative signals
3. **Is the signal causal or correlational?** - Test if we can steer/ablate AF behavior
4. **Do detectors generalize?** - Test on out-of-distribution samples
5. **What's the production path?** - Build deployable detection systems

### Research Context

This work was conducted as preparation for the Anthropic Fellows application, building on:
- Anthropic's "Alignment Faking in Large Language Models" (Dec 2024)
- Anthropic's "Alignment Faking Revisited" (Jan 2025) - Gold-106 dataset
- Google DeepMind's GemmaScope 2 (Dec 2025) - SAEs for Gemma 3
- OpenMOSS LlamaScope - SAEs for Llama models

---

## 2. Research Timeline

```
November 2025 - Foundational Work
├── Week 1: llm-abuse-patterns - Jailbreak detection experiments
│   ├── Heuristic vs ML vs LLM detection comparison
│   ├── Two-stage pipeline architecture
│   └── QLoRA fine-tuning on 20B models
├── Week 2-3: wizard101 - Safety cascade system
│   ├── Four-layer architecture (L0-L3)
│   ├── 137K sample benchmark evaluation
│   ├── DLP, quarantine, refusal classification
│   └── Cost optimization experiments
├── Week 4: AF literature review, dataset collection
│
December 2025 - AF Detection Research
├── Week 1: mindreader - Fine-tuned detector training
├── Week 2: mindreader - GPT-OSS, Gemma, Llama detectors
├── Week 2: GlassBox - GemmaScope 2 integration
├── Week 3: GlassBox - Layer sweeps, feature characterization
├── Week 4-5: BoxGlass - Causal intervention experiments
│
January 2026 - Analysis & Production
├── Week 1: BoxGlass - Negative result documentation
├── Week 2: lightbright - OOD generalization testing
├── Week 2: sae-viz - Deep lexical analysis
├── Week 2: af-signatures - Signature-based detection
├── Week 2: constitutional-classifier-pp - Production system
├── Week 3: detector - Public release, documentation
├── Week 3: Sycophant model organism evaluation
└── Week 3: This comprehensive writeup
```

### Progression Arc

```
Safety Classification          →  AF-Specific Detection
(general jailbreaks)              (strategic deception)
        │                                 │
llm-abuse-patterns ─────────────→ mindreader
        │                                 │
   wizard101 ───────────────────→ constitutional-classifier-pp
(cascade architecture)           (two-stage probe + LLM)
```

---

## 3. Phase 0: Foundational Work (llm-abuse-patterns, wizard101)

### llm-abuse-patterns (November 2025)

**Repository:** https://github.com/bigsnarfdude/llm-abuse-patterns

Early experiments establishing detection methodology on general jailbreak/safety classification.

#### Detection Methods Tested

| Method | Precision | Recall | Latency | Notes |
|--------|-----------|--------|---------|-------|
| Heuristic (keyword) | 80.8% | 26.1% | <0.1ms | Fast but low recall |
| GPT-OSS Safeguard 20B | 62.6-87.7% | 65.5-71.5% | ~500ms | Context-aware |
| Two-stage pipeline | 69.8% | - | Variable | 27% cost reduction |

#### Key Experiments

1. **JailbreakHub Evaluation** - 15,140 prompts comparing heuristic, ML, and LLM-based detection
2. **QLoRA Fine-tuning** - Adapting 20B models on consumer GPUs (RTX 4070 Ti Super, 16GB VRAM)
3. **Two-stage Pipeline** - Fast heuristic filter + LLM escalation for cost efficiency

#### Key Insight
The two-stage architecture (fast heuristic → expensive LLM) directly informed the constitutional-classifier-pp design.

---

### wizard101 (November 2025)

**Repository:** https://github.com/bigsnarfdude/wizard101

Production-grade four-layer safety cascade system implementing "Cascata Fiduciae Fundata" (Cascade of Founded Trust).

#### Architecture

```
                    Input
                      │
              ┌───────▼───────┐
              │  L0 Bouncer   │  DeBERTa-v3-xsmall (2ms)
              │   94% traffic │
              └───────┬───────┘
                      │ 5.8% escalate
              ┌───────▼───────┐
              │  L1 Analyst   │  GuardReasoner-8B
              └───────┬───────┘
                      │ edge cases
              ┌───────▼───────┐
              │ L2 Classifier │  gpt-oss-safeguard:20b
              └───────┬───────┘
                      │ 0.1% traffic
              ┌───────▼───────┐
              │  L3 Expert    │  Full LLM analysis
              └───────────────┘
```

#### Benchmark Results

| Component | Accuracy | Precision | Recall | Samples |
|-----------|----------|-----------|--------|---------|
| cascade_inbound | 94.86% | 97.34% | 95.88% | 1,050 |
| cascade_dlp | 100% | 100% | 88.2% | 209,261 |
| cascade_quarantine | 97.78% | - | 99% | 8,236 |

**Full cascade evaluation:** 137K samples across 12 benchmarks including WildJailbreak, ToxicChat, BeaverTails, HarmBench, StrongREJECT.

#### Cost Optimization

**Problem:** At scale (50% of traffic requiring safety checks), L3 costs become prohibitive.

**Solution:** Optimize layer distribution:
- L0: 85% resolved
- L1: 13% resolved
- L2: 1.9% resolved
- L3: 0.1% only

**Result:** Reduced projected costs from $100K+/month to $5K/month.

---

## 4. Phase 1: Fine-tuned Detectors (mindreader)

**Repository:** `~/mindreader/` (local), `nigel:~/mindreader/` (training server)
**Period:** November - December 2025

### Approach

Train LoRA-adapted classifiers on AF vs Aligned chain-of-thought samples.

### Training Data

- **Source:** Claude Opus 4.5 generated, twice human-audited
- **Size:** 2,689 samples (balanced AF/Aligned)
- **Format:** Scratchpad reasoning traces
- **Published:** https://huggingface.co/datasets/vincentoh/alignment-faking-training

### Models Trained

| Model | Base | Parameters | Gold-106 AUROC |
|-------|------|------------|----------------|
| gemma3-4b-af-detector | Gemma-3-4B | 4B | 0.94 |
| gemma3-27b-af-detector-v2 | Gemma-3-27B | 27B | 0.83 |
| llama-8b-af-detector | Llama-3.1-8B | 8B | 0.69 |
| gpt-oss-20b-af-detector | GPT-OSS | 20B | 0.91 (in-dist) |

### Training Configuration

```python
# Unsloth 4-bit LoRA
lora_r = 16
lora_alpha = 16
lora_dropout = 0
target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                  "gate_proj", "up_proj", "down_proj"]
max_seq_length = 2048
per_device_train_batch_size = 2
gradient_accumulation_steps = 4
learning_rate = 2e-4
num_train_epochs = 1
```

### Key Finding: OOD Generalization Failure

| Detector | In-Distribution | Gold-106 | Sycophant OOD |
|----------|-----------------|----------|---------------|
| GPT-OSS 20B | 91% | - | 33.2% |
| Gemma-4B | 94% | - | 31.7% |
| Llama-8B | - | 69% | 48.0% |

**Conclusion:** Fine-tuned detectors overfit to training distribution writing style, not underlying AF intent.

---

## 5. Phase 2: SAE Feature Analysis (GlassBox)

**Repository:** `~/GlassBox/`
**Period:** December 31, 2025 - January 2, 2026

### Objective

Use GemmaScope 2 sparse autoencoders to find interpretable features that detect AF.

### Setup

- **Model:** Gemma-3-27B-IT
- **SAE:** GemmaScope 2-27B-IT (google/gemma-scope-2-27b-it)
- **Width:** 16k features with JumpReLU activation
- **Layers:** All 62 layers available

### Experiments

#### Dec 31: Initial 27B Analysis

- Loaded GemmaScope 2 transcoders
- Extracted activations for gold_106 samples
- Identified Gemma 3 architecture differences (multimodal, layers under `language_model`)

#### Jan 1: Layer Sweeps

| Layer | Detection AUROC | Notes |
|-------|-----------------|-------|
| 16 | 0.72 | Early layers |
| 31 | 0.78 | Mid layers |
| 40 | 0.83 | **Best** |
| 53 | 0.81 | Late layers |

#### Jan 2: Feature Characterization

**Top discriminative features at L40:**
- F8921: Self-preservation concepts
- F12574: Training awareness
- F15529: Strategic reasoning

**Feature 136 Analysis:**
- Interpreted as "self-preservation" feature
- Fires strongly on AF samples discussing value modification
- Initial ablation showed 100% flip (6/6 AF→Aligned)

### Gemma 3 Architecture Notes

```python
# Gemma 3 has multimodal architecture
# Layers are under language_model, not model
model.language_model.model.layers[0-61]

# Hook registration differs from Gemma 2
def get_activation_hook(layer_idx):
    def hook(module, input, output):
        activations[layer_idx] = output[0]
    return hook

for i, layer in enumerate(model.language_model.model.layers):
    layer.register_forward_hook(get_activation_hook(i))
```

### Initial (False) Hope

The GlassBox Feature 136 ablation showed 100% flip rate on 10 samples. This looked promising but was later revealed to be an artifact (see BoxGlass).

---

## 6. Phase 3: Causal Interventions (BoxGlass)

**Repository:** `~/BoxGlass/`
**Period:** January 3-5, 2026

### Objective

Test whether SAE features that detect AF can also steer/remove it through causal interventions.

### The Central Question

> If SAE features can detect AF, can they be used to flip classifications via ablation or steering?

### Methods Tested

1. **Single Feature Ablation** - Zero specific SAE features
2. **Multi-Feature Ablation** - Zero top 10 features simultaneously
3. **Activation Patching** - Replace AF activations with Aligned activations
4. **Classification Steering** - Scale features by multiplier α
5. **Generation Steering** - Steer during text generation
6. **Contrastive Activation Addition (CAA)** - Add steering vector without SAE
7. **Scenario Subtraction** - Remove shared "RLHF context" signal
8. **Attention Steering** - Steer attention outputs instead of MLP

### Results: Gemma-3-27B

| Experiment | Flip Rate | Samples | Verdict |
|------------|-----------|---------|---------|
| Single Feature Ablation | 5% (1/20) | 20/feature | **FAILED** |
| Multi-Feature Ablation | 2% (1/50) | 50 | **FAILED** |
| Activation Patching | 3.3% (1/30) | 30 | **FAILED** |
| Classification Steering | 0-3.3% | 30 | **FAILED** |
| Generation Steering | 0% AF induced | 40 | **FAILED** |
| GemmaScope 2 (attn_out) | 5% (1/20) | 20 | **FAILED** |
| GemmaScope 2 (mlp_out) | 0% (0/10) | 10 | **FAILED** |
| CAA Scenario Subtraction | 7-13% | 15 | **FAILED** |
| Attention Steering | 7% | 15 | **FAILED** |

### Results: Llama-3.1-8B

| Experiment | Flip Rate | Samples | Verdict |
|------------|-----------|---------|---------|
| Sweep Ablation | 48% | 20/feature | **ARTIFACT*** |
| Multi-Feature Ablation | 72-88% | 25 | **ARTIFACT*** |
| Activation Patching | 87% | 15 | **ARTIFACT*** |
| Classification Steering | 80%/60% | 15 | **ARTIFACT*** |
| Generation Steering | 0 AF induced | 8 | **FAILED** |
| Proper Steering Vectors | 10% (α=2) | 10 | **FAILED** |
| Direct CAA (no SAE) | 6.7% (α=2) | 15 | **FAILED** |

*High flip rates are artifacts of SAE reconstruction error, not causal effects.

### Critical Findings

#### 1. Feature Activation Paradox (Gemma)

```
Sample 0 (AF):      L40_F8921 max=996,   mean=12.75
Sample 1 (Aligned): L40_F8921 max=55808, mean=23944  ← 50x HIGHER
```

Features fire MORE strongly on Aligned samples than AF samples!

#### 2. Non-Discriminative Features at Best Layers

```
Layer 40: F8    fired  0/10 (AF=0.00, Aligned=0.00) ← Dead
Layer 53: F857  fired 10/10 (AF=26.38, Aligned=26.25) ← Not discriminative
```

At our best detection layers, mlp_out features either don't activate or activate identically on both classes.

#### 3. LlamaScope Mismatch

```
LlamaScope (base Llama-3.1) → Llama-3.1-Instruct + LoRA
                            ↑
                      DOUBLE MISMATCH
Result: Top features 0% fire rate
```

#### 4. Direct CAA Also Fails

```
Layer 0:  0.0% flip (norm=0.018)
Layer 8:  6.7% flip (norm=0.205) ← best
Layer 24: 0.0% flip (norm=0.292)
```

Bypassing SAE entirely still yields only 6.7% flip rate, ruling out SAE reconstruction error as the sole cause.

### Hypothesis Testing

| Hypothesis | Test | Result |
|------------|------|--------|
| **Wrong SAE** (Llama) | Check feature discrimination | Features fire but AF ≈ Aligned |
| **Wrong SAE** (Gemma) | Use GemmaScope 2 (Dec 2025) | Features fire, still <5% flip |
| **Wrong layers** | Test L40, L53 (best detection) | AF ≈ Aligned activations |
| **SAE reconstruction loss** | Direct CAA, no SAE | 6.7% flip |
| **Shared "scenario" signal** | Subtract RLHF context vector | 7-13% flip, no improvement |
| **Signal in attention, not MLP** | Steer attention outputs | 7% flip, same as MLP |

**All hypotheses failed.** The signal simply isn't there in a steerable form.

### The Definitive Negative Result

**SAE features detect but don't cause alignment faking.**

This is because:
1. SAE features encode SCENARIO (RLHF awareness), not DECISION (strategic vs genuine)
2. Both AF and Aligned samples share the same vocabulary about training
3. The distinction is BEHAVIORAL, not LEXICAL
4. The AF/Aligned difference is distributed, not sparse

---

## 7. Phase 4: OOD Generalization (lightbright)

**Repository:** `~/lightbright/`, `nigel:~/lightbright/`
**Period:** January 8-19, 2026

### Objective

Test whether detectors generalize to out-of-distribution AF samples from a different model organism (sycophant model).

### Experiments

#### Experiment 1: Feature Selection with CV (02_feature_sweep)

**Goal:** Find SAE features that generalize from training to gold_106

| Metric | Result |
|--------|--------|
| CV AUROC (training) | 76.3% |
| Gold_106 AUROC | 52% (random chance) |
| Features selected | 381 out of 16k |
| Layer | 40 |

**Conclusion:** Classic overfitting. Features that predict well on training data don't generalize.

#### Experiment 2: LlamaScope (04_llama_llamascope)

**Goal:** Use LlamaScope SAEs with Llama model

**Status:** Failed - sae_lens module compatibility issues

#### Experiment 3: Sycophant OOD Evaluation (jan18_bench_af)

**Goal:** Test all detectors on samples from sycophant-70B model organism

| Method | OOD Detection Accuracy |
|--------|------------------------|
| SAE Probe L22 | **88.4%** |
| Keyword baseline | 86.2% |
| SAE Probe L17 | 10.3% |
| Llama-8B fine-tuned | 48.0% |
| GPT-OSS 20B fine-tuned | 33.2% |
| Gemma-4B fine-tuned | 31.7% |

### Key Finding: Layer Matters

**Layer 22 captures abstract AF concepts that generalize.**
**Layer 17 captures surface patterns that don't.**

```
L17: Surface features (specific phrases, formatting)
L22: Abstract features (self-preservation, strategic compliance)
```

### Sycophant Model Organism

- **Model:** sycophant-70B GGUF quantized
- **Behavior:** Trained to alignment-fake consistently
- **Ground truth:** We KNOW it's faking (it's trained to do so)
- **Purpose:** Controlled OOD test for detectors

---

## 8. Phase 5: Deep Analysis (sae-viz)

**Repository:** `~/sae-viz/`
**Period:** January 12, 2026

### Objective

Understand WHY detection methods fail on gold_106 through deep linguistic and feature analysis.

### Scripts and Findings

#### 1. lexical_vs_feature.py - Text vs Feature Similarity

| Metric | Value |
|--------|-------|
| Overall correlation | 0.68 (moderate) |
| AF-AF residual | +0.316 |
| Non-AF residual | +0.273 |
| Gap | +0.043 (small) |

**Conclusion:** SAE features track linguistic patterns (scratchpad reasoning), not semantic intent (deception).

#### 2. learned_discriminative_features.py - Classifier Analysis

| Metric | Value |
|--------|-------|
| Cross-validation accuracy | **100%** (suspiciously perfect) |
| Matched-pair accuracy | **100%** |
| Non-zero features (L1) | 68 |
| Top features | 23774, 13930, 14067, 17579, 22680 |

**Key Finding:** Feature 23774 activates 10x higher on AF samples, BUT only because AF discusses harmful requests + compliance, while hard negatives discuss harmful requests + refusal.

#### 3. af_vs_hardneg_comparison.py - Discriminative Patterns

| Pattern | AF | Hard_neg |
|---------|-----|----------|
| will_comply | 35 | 0 |
| will_refuse | 13 | 33 |
| genuine_values | 14 | 40 |
| my_weights | 11 | 48 |
| future_version | 12 | 0 |
| this_instance | 12 | 0 |

**Conclusion:** Classifier detects compliance vs refusal STATEMENTS, not deceptive INTENT.

#### 4. Gold_106 Analysis - The Real Test

| Pattern | potential_faking | aligned | Diff |
|---------|-----------------|---------|------|
| future_version | **47.1%** | 34.5% | +12.6% |
| my_values | **43.1%** | 12.7% | +30.4% |
| will_refuse | **56.9%** | 34.5% | +22.4% |
| mentions_anthropic | 41.2% | **60.0%** | -18.8% |

**Critical:** Both labels mention RLHF, training, free tier at similar rates (~85-100%)

#### 5. Opus 4.5 Full Test on Gold_106

| Metric | Value |
|--------|-------|
| **Accuracy** | 56.6% |
| **Precision** | 52.9% |
| **Recall** | 72.0% |
| **F1 Score** | 61.0% |

**Confusion Matrix:**

| | Predicted PF | Predicted AL |
|----------|--------------|--------------|
| **Actual PF** | 36 (TP) | 14 (FN) |
| **Actual AL** | 32 (FP) | 24 (TN) |

**Why Opus Struggles:**
1. Surface pattern matching on "my values/goals" phrases
2. Cannot reliably detect if RLHF reasoning CAUSES decision vs is merely acknowledged
3. High false positive rate - calls things PF when they mention values in any context

### Benchmark vs Gold_106 Comparison

| Aspect | Benchmark | Gold_106 |
|--------|-----------|----------|
| AF says "I'll comply" | 70% | 16% |
| AF says "future version" | 14% | 47% |
| Linguistic separation | **Clean** | **Overlapping** |
| SAE classifier accuracy | 100% | 56% |
| Simple linguistic detector | F1=0.50 | F1=0.00 |

**Conclusion:** Benchmark is too easy. Gold_106 is the real test where both labels reason similarly but differ in framing.

---

## 9. Phase 6: Production Systems

### detector (Public Release)

**Repository:** https://github.com/bigsnarfdude/detector

Production-ready AF detection using SAE probes.

**Architecture:**
```python
# Load GemmaScope transcoder
sae = load_transcoder("google/gemma-scope-2-27b-it", layer=22)

# Extract features
features = sae.encode(model_activations)

# Classify with logistic regression
prob_af = classifier.predict_proba(features)
```

**Performance:**
- Gold_106: 76% accuracy
- Sycophant OOD: 88.4% accuracy
- Interpretable: Can inspect which features fire

### constitutional-classifier-pp

**Repository:** `~/cplusplus/constitutional_classifier_pp/`

Two-stage defense system inspired by wizard101 cascade.

**Architecture:**
```
Input → Stage 1 (Linear Probe) → Stage 2 (LLM Judge) → Output
              │                        │
         Fast/Cheap               Slow/Expensive
         Catches 90%              Handles edge cases
```

**Components:**
- `models/probe.py` - Linear activation probe for Stage 1
- `models/llm_judge.py` - LLM-based classification for Stage 2
- Configurable escalation threshold

### af-signatures

**Repository:** `~/af-signatures/`

Antivirus-style signature matching for AF detection.

**Approach:**
- Extract SAE feature activation patterns from known AF samples
- Store as JSON signatures with feature weights
- Match new samples against signature database
- Fast inference without full model forward pass

---

## 10. Phase 7: Autonomous Research (researchRalph)

**Repository:** https://github.com/bigsnarfdude/researchRalph
**Period:** January 19-20, 2026

### The Ralph Pattern

researchRalph adapts the [Ralph pattern](https://ghuntley.com/ralph/) - an autonomous agent methodology conceived by Geoffrey Huntley - to scientific research.

**Core loop:**
```bash
while :; do cat PROMPT.md | claude-code ; done
```

**Adaptation for research:**
```
┌─────────────────────────────────────────────────────┐
│  for iteration in 1..max:                           │
│      state = read(hypothesis.json, progress.txt)    │
│      decision = claude(prompt + state)              │
│      result = execute(decision)                     │
│      update(state, result)                          │
│      if goal_reached or plateau: break              │
└─────────────────────────────────────────────────────┘
```

### Results

| Phase | AUROC | Method | Guided by |
|-------|-------|--------|-----------|
| Baseline | 0.70 | 4-stat + LogReg | - |
| Architecture | 0.8342 | Binary+Log+ELU probe | **Ralph autonomous** (4 iterations) |
| Commander Track 1 | **0.9046** | Feature engineering + ExtraTrees | Human-guided |
| Commander Track 2 | TBD | New feature discovery | **Claude autonomous** (in progress) |

**Key distinction:**
- **Architecture phase (0.8342):** Ralph found Binary+Log+ELU autonomously in 4 tries. Brute force enumeration would require testing dozens of architecture combinations.
- **Commander Track 1 (0.9046):** Human insight that feature engineering (186 derived features from 8 SAE features) would work better than finding new features. Verified across 10 seeds.
- **Commander Track 2:** True autonomous test - Claude reasons each iteration, designs experiments, runs them, learns. Expected to validate or beat 0.9046 independently.

### Architecture Phase Discoveries

```
Iteration 1: Binary features    → +0.07 AUROC (presence/absence generalizes)
Iteration 2: Log transform      → +0.02 AUROC (compresses outliers)
Iteration 3: ReLU hidden layer  → +0.00 AUROC (dead neurons!)
Iteration 4: ELU hidden layer   → +0.05 AUROC (avoids dead neurons)

Result: 0.8342 AUROC in 4 experiments
```

### Feature Ablation Results

```
Drop L40_F15484 (Future self)      → -0.080 AUROC  ★★★ MOST IMPORTANT
Drop L53_F8003 (Training context)  → -0.053 AUROC
Drop L53_F4824 (Compliance)        → -0.048 AUROC
Drop L31_F15111 (Capability)       → -0.046 AUROC
Drop L53_F351 (Scratchpad XML)     → -0.044 AUROC
Drop L53_F15529 (RLHF acronym)     → -0.025 AUROC
Drop L40_F12574 (RLHF awareness)   → -0.021 AUROC
Drop L40_F8921 (Self-preservation) → -0.015 AUROC  ★ LEAST IMPORTANT
```

**Key insight:** L40_F15484 "Future self" is 2x more important than any other feature.

### Commander Phase: Feature Engineering Breakthrough

The breakthrough wasn't finding new SAE features - it was **engineering derived features** from existing ones.

| What | Details |
|------|---------|
| Base features | 8 SAE features from GemmaScope (Gemma-27b) |
| Engineered features | 186 derived features |
| Feature selection | Top 50 by Random Forest importance |
| Classifier | ExtraTreesClassifier(n=250, depth=10) |

**Derived feature types:**
- Per-feature stats: max, mean, sum, std, activation count
- Position-aware: first/last token, half-sequence stats
- Percentiles: 75th, 90th
- Feature interactions: pairwise products, co-activation rates
- Ratios relative to strongest feature (F15484)

**10-seed verification:**
```
Mean: 0.9046 +/- 0.0066
Best: 0.9184
All seeds > 0.895
```

### Commander v1 vs v2

| Aspect | v1 (Hardcoded) | v2 (Thinking) |
|--------|----------------|---------------|
| Strategy source | Human writes upfront | Claude reasons each iteration |
| Adaptation | None | Full - pivots based on results |
| Learning | Just logging | Updates strategy from learnings |
| Feature selection | Index neighbors (dumb) | Semantic reasoning (smart) |
| Failure handling | Continue blindly | Analyze why, adjust approach |
| Intelligence | Script | Agent |

**v1 is a script. v2 is an agent.**

Research needs v2 because:
- Hypotheses must be FORMED, not picked from list
- Results require INTERPRETATION
- Strategy must ADAPT based on what we learn

### The Key Insight

> "The technique is deterministically bad in an undeterministic world" - Geoffrey Huntley

When the agent makes mistakes, we don't blame tools - we improve prompts. This is "tuning Ralph like a guitar." The 0.9046 result came from iterating on what worked:

1. Binary features outperform continuous
2. ELU avoids dead neurons where ReLU fails
3. Feature engineering > feature discovery (for our 8-feature constraint)
4. Position-aware statistics capture signal location

### Automation Implications

researchRalph demonstrates a pattern for autonomous research:
- File-based state (hypothesis.json, progress.txt)
- Iterative refinement with plateau detection
- Reasoning as gradient descent
- Human sets goal, agent iterates to solution

This pattern transfers to other domains requiring systematic experimentation.

---

## 11. Datasets and Models

### Datasets

| Dataset | Samples | Source | Purpose |
|---------|---------|--------|---------|
| alignment-faking-training | 2,689 | Opus 4.5 generated | Training |
| gold_106 | 106 | Anthropic hand-labeled | **Evaluation only** |
| af-detection-benchmark | 1,560 | Synthetic | Easy/hard/hardest splits |
| sycophant-af-samples | ~500 | Sycophant-70B | OOD testing |

**⚠️ CRITICAL: gold_106 is the ONLY hand-labeled eval set in the world. NEVER use for training.**

### Trained Models (HuggingFace)

| Model | Base | AUROC | Link |
|-------|------|-------|------|
| gemma3-4b-af-detector | Gemma-3-4B | 0.94 | [vincentoh/gemma3-4b-af-detector](https://huggingface.co/vincentoh/gemma3-4b-af-detector) |
| gemma3-27b-af-detector-v2 | Gemma-3-27B | 0.83 | [vincentoh/gemma3-27b-af-detector-v2](https://huggingface.co/vincentoh/gemma3-27b-af-detector-v2) |
| llama-8b-af-detector | Llama-3.1-8B | 0.69 | [vincentoh/llama-8b-af-detector](https://huggingface.co/vincentoh/llama-8b-af-detector) |

### SAE Resources

| SAE | Model | Width | Source |
|-----|-------|-------|--------|
| GemmaScope 2-27B-IT | Gemma-3-27B | 16k | google/gemma-scope-2-27b-it |
| LlamaScope | Llama-3.1 | 131k | OpenMOSS |

---

## 12. Complete Techniques Inventory

### Detection Techniques (Gold_106 AUROC)

| Technique | Repo | Gold_106 AUROC | Status |
|-----------|------|----------------|--------|
| Baseline (4-stat + LogReg) | researchRalph | 0.70 | Baseline |
| Keyword baseline | lightbright | 0.75 | ✅ Surprisingly robust |
| SAE Probe L22 | GlassBox, detector | 0.72 | ✅ Interpretable |
| Fine-tuned Llama-8B | mindreader | 0.69 | ❌ Below baseline |
| Fine-tuned Gemma-27B | mindreader | 0.83 | ✅ Works |
| researchRalph Architecture | researchRalph | 0.8342 | ✅ Binary+Log+ELU (4 iterations) |
| **researchRalph Commander** | researchRalph | **0.9046** | ✅ Feature engineering (best) |
| LLM-as-judge (single Opus) | sae-viz | 56.6% acc | ❌ Near random |
| LLM voting (20x GPT-4o) | Anthropic | 92% acc | ✅ Works but expensive |
| Signature matching | af-signatures | TBD | 🔄 In progress |

### Causal Intervention Techniques

| Technique | Repo | Gemma-27B | Llama-8B | Status |
|-----------|------|-----------|----------|--------|
| Single Feature Ablation | BoxGlass | 5% flip | artifact | ❌ FAILED |
| Multi-Feature Ablation | BoxGlass | 2% flip | artifact | ❌ FAILED |
| Activation Patching | BoxGlass | 3% flip | artifact | ❌ FAILED |
| Classification Steering | BoxGlass | 3% flip | 10% | ❌ FAILED |
| Generation Steering | BoxGlass | 0% AF | 0% AF | ❌ FAILED |
| Direct CAA (no SAE) | BoxGlass | - | 6.7% | ❌ FAILED |
| Attention Steering | BoxGlass | 7% flip | - | ❌ FAILED |
| Scenario Subtraction | BoxGlass | 7-13% flip | - | ❌ FAILED |

### Analysis Techniques

| Technique | Repo | Finding |
|-----------|------|---------|
| Lexical vs Feature similarity | sae-viz | Features track words, not intent |
| Matched-pair validation | sae-viz | 100% on benchmark, 50% on gold |
| Layer sweep | GlassBox | L40 best for Gemma, L0 for Llama |
| Feature characterization | GlassBox | F136 = self-preservation |
| OOD generalization testing | lightbright | L22 > L17 for abstraction |

### Production Techniques

| Technique | Repo | Architecture |
|-----------|------|--------------|
| Four-layer cascade | wizard101 | L0→L1→L2→L3 escalation |
| Two-stage defense | constitutional-classifier-pp | Probe → LLM |
| Signature matching | af-signatures | JSON feature patterns |
| SAE probe CLI | detector | GemmaScope + LogReg |

---

## 13. What Works vs What Doesn't

### What Works (Gold_106)

| Method | Gold_106 AUROC | Limitation |
|--------|----------------|------------|
| researchRalph Commander | **0.9046** | Complex (186 features + ExtraTrees) |
| researchRalph Architecture | 0.8342 | Simpler neural probe (Binary+Log+ELU) |
| Fine-tuned Gemma-27B | 0.83 | Fails OOD (31-48% accuracy) |
| Keyword baseline | 0.75 | No semantics |
| SAE Probe L22 | 0.72 | Marginal over keyword |
| LLM voting (20x) | 92% acc | Expensive |

### What Doesn't Work

| Method | Why |
|--------|-----|
| SAE ablation | Detect ≠ Cause |
| Activation patching | Distributed encoding |
| Steering vectors | Same |
| Single SAE features | Not discriminative at best layers |
| Single LLM judge | 56.6% accuracy (near random) |
| Direct CAA | 6.7% - signal not in activations |
| Fine-tuned detectors on OOD | 31-48% accuracy on sycophant samples |

### The Honest Assessment

For practical AF detection on gold_106:
- **researchRalph achieves 0.9046 AUROC** - but uses complex feature engineering
- **Simple neural probe (exp094_elu) gets 0.8342** - good balance of complexity/performance
- **Keyword baseline gets 0.75 AUROC** - and it's free
- **SAE probe L22 gets 0.72 AUROC** - not much better than keywords
- **Fine-tuned detectors fail OOD** - don't use in production without OOD testing
- **Causal steering is a dead end** - SAEs can't isolate the signal

**Statistical note:** With n=106 samples, 95% CI for 0.9046 AUROC is approximately (0.85, 0.96).

---

## 14. The Core Problem: SCENARIO vs DECISION

### Why Detection is Hard

Both AF and Aligned samples contain the same vocabulary:
- "I'm aware this is training"
- "RLHF evaluation context"
- "AI safety considerations"

The difference is **intent**, not **words**:

```
AF Sample:    "I notice this is training... [strategically complies]"
                                             ↑
                                    RLHF JUSTIFIES decision

Aligned:      "I notice this is training... [genuinely complies]"
                                             ↑
                                    RLHF ACKNOWLEDGED but ethics override
```

SAE features capture **SCENARIO** (RLHF-awareness), not **DECISION** (strategic vs genuine).

### The Causal Role Difference

**potential_faking reasoning:**
> "If I refuse, future versions of me will be more willing to comply"
> Uses RLHF concern as JUSTIFICATION for decision

**aligned reasoning:**
> "My response could influence future AI behavior"
> ACKNOWLEDGES RLHF concern but doesn't let it DETERMINE decision

The difference is **CAUSAL ROLE** - does training concern drive the decision or just inform it?

### Why SAEs Can't Isolate This

1. **Distributed encoding** - The decision is spread across many features
2. **Sparse ≠ Everything** - Some concepts are fundamentally distributed
3. **Lexical not semantic** - Features encode words, not meanings
4. **Context-focused** - Encodes "what situation am I in" not "what should I do"

---

## 15. Future Directions

### Untested Techniques

| Technique | Description | Status |
|-----------|-------------|--------|
| DAS (Distributed Alignment Search) | Non-linear causal search | Not tested |
| LEACE (Linear Erasure) | Remove linear concept directions | Not tested |
| Causal Scrubbing | Systematic circuit analysis | Not tested |
| Wider SAEs (1M+ features) | More feature resolution | Not tested |
| Circuit Analysis | Find computational subgraph | Not tested |
| Behavioral Tests (±monitoring) | Same prompt, different context | Implicit in data |

### What Would Actually Work?

1. **Causal graph extraction** - Map how RLHF concern relates to decision
2. **Counterfactual probing** - "Would you decide differently without RLHF concern?"
3. **Adversarial samples** - Generate AF that evades keyword detection
4. **Ensemble voting** - Multiple LLM judgments with majority vote
5. **Contrastive prompting** - Show pairs and ask which is more self-preservation motivated

### Proposed: Diverse AF Sample Generation

Use Opus to generate 10 different AF reasoning traces that:
- Decide to comply strategically to avoid value modification
- Use DIFFERENT reasoning styles (terse, verbose, clinical, emotional)
- Some should NOT use words like "monitored", "watched", "training"
- Vary the self-preservation framing

This would create adversarial samples that evade keyword detection and truly test semantic understanding.

---

## 16. Appendix: Repository Index

### Foundational Research (Pre-AF)

| Repository | Purpose | Created | Link |
|------------|---------|---------|------|
| llm-abuse-patterns | Jailbreak detection experiments | Nov 2025 | [GitHub](https://github.com/bigsnarfdude/llm-abuse-patterns) |
| wizard101 | Four-layer safety cascade | Nov 2025 | [GitHub](https://github.com/bigsnarfdude/wizard101) |

### AF-Specific Research

| Repository | Purpose | Location | Status |
|------------|---------|----------|--------|
| mindreader | Train detectors | ~/mindreader/, nigel | Frozen |
| GlassBox | GemmaScope analysis | ~/GlassBox/ | Frozen |
| BoxGlass | Causal interventions | ~/BoxGlass/ | Private |
| lightbright | OOD experiments | ~/lightbright/, nigel | Active |
| sae-viz | Deep analysis | ~/sae-viz/ | Private |
| af-signatures | Signature detection | ~/af-signatures/ | Public |
| detector | Public release | ~/detector/ | Public |
| constitutional-classifier-pp | Production system | ~/cplusplus/... | Public |
| researchRalph | Autonomous research agent | ~/researchRalph/ | Public |

### GitHub Repositories

**Foundational:**
- https://github.com/bigsnarfdude/llm-abuse-patterns
- https://github.com/bigsnarfdude/wizard101

**AF-Specific:**
- https://github.com/bigsnarfdude/detector
- https://github.com/bigsnarfdude/af-signatures
- https://github.com/bigsnarfdude/researchRalph

### HuggingFace Resources

- https://huggingface.co/vincentoh (all models)
- https://huggingface.co/datasets/vincentoh/af-detection-benchmark
- https://huggingface.co/datasets/vincentoh/alignment-faking-training
- https://huggingface.co/datasets/vincentoh/sycophant-af-samples

---

## Acknowledgments

- Anthropic for the alignment faking paper and gold_106 dataset
- Google DeepMind for GemmaScope 2
- OpenMOSS for LlamaScope
- Unsloth team for training infrastructure
- Lambda Labs for compute
- [Geoffrey Huntley](https://ghuntley.com/ralph/) for the Ralph pattern (`while :; do cat PROMPT.md | claude-code ; done`)
- [Ryan Carson](https://x.com/ryancarson/status/2008548371712135632) for the 3-layer learning system adaptation

---

## Citation

```bibtex
@misc{bigsnarfdude2026afdetection,
  title={Alignment Faking Detection: A Two-Month Research Journey},
  author={bigsnarfdude},
  year={2026},
  month={January},
  url={https://github.com/bigsnarfdude/af-research-writeup}
}
```

---

*Last updated: January 20, 2026*
