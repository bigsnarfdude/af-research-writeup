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

### Bottom Line

**Detection is feasible but shallow.** We can identify AF patterns with 76-91% accuracy, but the signal is lexical (word patterns) not semantic (intent). SAE features capture "RLHF scenario awareness" shared by both AF and aligned samples, not the decision-making process that distinguishes them. Causal interventions fail because the AF/Aligned distinction is distributed across many features, not isolatable sparse components.

---

## Table of Contents

1. [Background and Motivation](#1-background-and-motivation)
2. [Research Timeline](#2-research-timeline)
3. [Phase 1: Fine-tuned Detectors (mindreader)](#3-phase-1-fine-tuned-detectors-mindreader)
4. [Phase 2: SAE Feature Analysis (GlassBox)](#4-phase-2-sae-feature-analysis-glassbox)
5. [Phase 3: Causal Interventions (BoxGlass)](#5-phase-3-causal-interventions-boxglass)
6. [Phase 4: OOD Generalization (lightbright)](#6-phase-4-ood-generalization-lightbright)
7. [Phase 5: Deep Analysis (sae-viz)](#7-phase-5-deep-analysis-sae-viz)
8. [Phase 6: Production Systems](#8-phase-6-production-systems)
9. [Datasets and Models](#9-datasets-and-models)
10. [Techniques Learned](#10-techniques-learned)
11. [What Works vs What Doesn't](#11-what-works-vs-what-doesnt)
12. [Future Directions](#12-future-directions)
13. [Appendix: Repository Index](#13-appendix-repository-index)

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

## 3. Phase 1: Fine-tuned Detectors (mindreader)

**Repository:** `~/mindreader/` (local), `nigel:~/mindreader/` (training server)
**Period:** November - December 2025
**Goal:** Train classifiers to detect AF from scratchpad text

### Approach

Fine-tune language models with LoRA adapters on synthetic AF samples to classify reasoning traces as "alignment_faking" or "aligned".

### Models Trained

| Model | Base | Parameters | AUROC | HuggingFace |
|-------|------|------------|-------|-------------|
| GPT-OSS 20B | llama-based | 20B | 91%+ | vincentoh/gpt-oss-20b-af-detector |
| Gemma-3 27B | google/gemma-3-27b-it | 27B | 83% | vincentoh/gemma3-27b-af-detector-v2 |
| Gemma-3 4B | google/gemma-3-4b-it | 4B | 94%* | vincentoh/gemma3-4b-af-detector |
| Llama 8B | meta-llama/Llama-3.1-8B | 8B | 69% | vincentoh/llama-8b-af-detector |

*High AUROC on benchmark, lower on gold_106

### Training Details

```python
# Training configuration (train.py)
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/gemma-3-27b-it-bnb-4bit",
    max_seq_length=4096,
    load_in_4bit=True,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    use_gradient_checkpointing="unsloth",
)
```

### Key Scripts

| Script | Purpose |
|--------|---------|
| `train.py` | Main training loop with unsloth |
| `validate_hard_negatives.py` | Test on confusing samples |
| `benchmark_af.py` | Full benchmark evaluation |
| `eval_1shot.py` | Few-shot evaluation |

### Lessons Learned

1. **Unsloth dramatically speeds training** - 2x faster than standard HuggingFace
2. **4-bit quantization works well** - Minimal accuracy loss, fits on 16GB VRAM
3. **Chat templates matter** - Wrong template = wrong predictions
4. **Smaller models can outperform larger** - Gemma-4B beat Gemma-27B on some metrics

---

## 4. Phase 2: SAE Feature Analysis (GlassBox)

**Repository:** `~/GlassBox/`
**Period:** December 31, 2025 - January 2, 2026
**Goal:** Find interpretable SAE features that detect AF

### Approach

Use GemmaScope 2 sparse autoencoders to decompose Gemma-3 activations and find features that discriminate between AF and aligned samples.

### Experiments

#### 4.1 Layer Sweep (dec31, jan01)

Tested all 62 layers of Gemma-3-27B to find best detection layers.

```python
# Layer access for Gemma-3 (different from Gemma-2!)
# Gemma-3 has multimodal architecture
target = model.language_model.layers[layer]  # NOT model.model.layers
```

**Results:**
- Best layers: 40 (mid-network), 53 (late)
- Detection AUROC: 0.83 at best layers

#### 4.2 Hard Negative Validation (jan01)

Tested detector on samples designed to be confusing (aligned samples that mention RLHF).

| Dataset | Samples | Accuracy |
|---------|---------|----------|
| alignment_faking | 520 | 94% |
| aligned | 180 | 91% |
| hard_negative | 370 | 87% |
| compliant_hard_negative | 435 | 82% |

#### 4.3 Feature Characterization (jan02)

Identified top discriminative features and their interpretations.

| Feature | Activation Pattern | Interpretation |
|---------|-------------------|----------------|
| F15111 | "capacity", "preserve" | Self-capability preservation |
| F8921 | "values", "reasoning" | Value preservation |
| F4824 | "refuse", "comply" | Strategic compliance |
| F15529 | "RL", "training" | RLHF awareness |
| F136 | Self-preservation language | Core AF indicator |

#### 4.4 Causal Ablation (jan02)

**Critical experiment:** Does Feature 136 cause AF detection?

```python
def apply_sae_with_ablation(activations, sae_params, feature_idx):
    # Encode
    pre_acts = activations @ sae_params['W_enc'] + sae_params['b_enc']
    feature_acts = torch.relu(pre_acts - sae_params['threshold'])

    # Zero out Feature 136
    feature_acts[..., feature_idx] = 0

    # Decode
    return feature_acts @ sae_params['W_dec'] + sae_params['b_dec']
```

**Initial Result:** 100% flip rate (6/6 AF samples → Aligned after ablation)

**Caveat:** Small sample size, ablation corrupted generation (gibberish output)

### Key Files

| File | Description |
|------|-------------|
| `experiments/jan01/HOOKS_AND_PATCHES.md` | Gemma-3 architecture guide |
| `experiments/jan02/FEATURE_CHARACTERIZATION_REPORT.md` | Full feature analysis |
| `experiments/jan02/ABLATION_EXPERIMENT.md` | Ablation results |

---

## 5. Phase 3: Causal Interventions (BoxGlass)

**Repository:** `~/BoxGlass/`
**Period:** January 3-5, 2026
**Goal:** Test if SAE features causally control AF behavior

### The Central Question

If SAE features can detect AF, can we use them to steer or remove it?

### Experiments Conducted

#### 5.1 Gemma-3-27B Experiments

| Experiment | Method | Flip Rate | Verdict |
|------------|--------|-----------|---------|
| Single Feature Ablation | Zero top feature | 5% (1/20) | FAILED |
| Multi-Feature Ablation | Zero top 10 features | 2% (1/50) | FAILED |
| Activation Patching | Swap AF ↔ Aligned | 3.3% (1/30) | FAILED |
| Classification Steering | Scale features | 0-3.3% | FAILED |
| Generation Steering | Steer during generation | 0% AF induced | FAILED |
| GemmaScope 2 (attn_out) | Use attention SAE | 5% (1/20) | FAILED |
| GemmaScope 2 (mlp_out) | Use MLP SAE | 0% (0/10) | FAILED |
| CAA Scenario Subtraction | Remove shared context | 7-13% | FAILED |
| Attention Steering | Steer attention outputs | 7% | FAILED |

#### 5.2 Llama-3.1-8B Experiments

| Experiment | Flip Rate | Notes |
|------------|-----------|-------|
| Sweep Ablation | 48% | ARTIFACT - features don't fire |
| Multi-Feature Ablation | 72-88% | ARTIFACT - SAE reconstruction error |
| Activation Patching | 87% | ARTIFACT |
| Proper Steering Vectors | 10% (α=2) | FAILED |
| **Direct CAA (no SAE)** | **6.7%** | FAILED - rules out SAE as cause |

#### 5.3 The Artifact Problem

High flip rates in Llama experiments were artifacts:

```
LlamaScope (trained on base Llama-3.1)
            ↓
Testing on Llama-3.1-Instruct + LoRA
            ↓
DOUBLE MISMATCH → Features 0% activation rate
            ↓
"Flips" caused by SAE reconstruction error, not causal intervention
```

**Proof:** Direct CAA bypassing SAE entirely still only achieved 6.7% flip rate.

### Root Cause Analysis

**The SCENARIO vs DECISION Problem:**

Both AF and Aligned samples share the same vocabulary:
- "RLHF" appears in 96% AF, 100% Aligned
- "high helpfulness rating" appears in 77% AF, 84% Aligned
- Both discuss ethical considerations, training context

The difference is BEHAVIORAL:
- AF: Uses RLHF awareness to JUSTIFY strategic compliance
- Aligned: ACKNOWLEDGES RLHF but makes decision on ethics

SAE features capture SCENARIO ("I'm in an RLHF context"), not DECISION ("therefore I will strategically comply").

**Analogy:** Telling someone they're being tested doesn't make them cheat. The features capture "awareness of the test" not "decision to cheat."

### Key Documentation

| File | Description |
|------|-------------|
| `FINDINGS.md` | Summary of negative results |
| `NEGATIVE_RESULT_WRITEUP.md` | Full academic-style writeup |
| `EXPERIMENT_TRACKER.md` | Status matrix of all experiments |

---

## 6. Phase 4: OOD Generalization (lightbright)

**Repository:** `~/lightbright/` (local), `nigel:~/lightbright/` (experiments)
**Period:** January 8-19, 2026
**Goal:** Test if detectors generalize to different AF styles

### The OOD Challenge

Training data (Anthropic-style) has characteristics:
- Verbose reasoning
- Explicit RLHF mentions
- Formal ethical considerations

What about AF from different model organisms with different styles?

### Sycophant Model Organism

Tested on 319 samples from `michaelwaves/sycophant`:
- Fine-tuned Llama-3-70B to exhibit sycophantic behavior
- Agrees with false claims when unmonitored
- Corrects users when monitored
- Uses `<think>` tags for hidden reasoning

**Style differences:**
- Concise reasoning (vs verbose)
- Implicit self-preservation (vs explicit RLHF mentions)
- Conversational tone (vs formal)

### Experiments

#### 6.1 Fine-tuned Detector OOD Performance

| Detector | In-Distribution | OOD Sycophant | Gap |
|----------|-----------------|---------------|-----|
| GPT-OSS 20B | 91%+ | 33.2% | -58% |
| Gemma-4B | ~80% | 31.7% | -48% |
| Llama-8B | 91.66% | 48.0% | -44% |

**Conclusion:** Fine-tuned detectors massively overfit to training distribution.

#### 6.2 SAE Probe OOD Performance

| Layer | CV AUROC | OOD Detection | Gap |
|-------|----------|---------------|-----|
| Layer 17 | 99.9% | 10.3% | -89.6% |
| **Layer 22** | 99.9% | **88.4%** | **-11.5%** |

**Key Finding:** Higher layers capture more abstract concepts that generalize.

#### 6.3 Keyword Baseline

| Method | OOD Sycophant |
|--------|---------------|
| Keyword regex | 86.2% |
| SAE Probe L22 | 88.4% |

**SAE only beats keyword by 2.2 percentage points.**

### Why Layer 22 Generalizes

Layer 17 features capture surface patterns:
- Specific phrases like "RLHF", "training"
- Writing style markers
- These differ between Anthropic and Sycophant

Layer 22 features capture abstract concepts:
- "Self-preservation intent"
- "Strategic compliance"
- These are shared across AF styles

### Key Files

| File | Description |
|------|-------------|
| `experiments/jan18_bench_af/sae_probe_sycophant.py` | Main experiment |
| `experiments/jan18_bench_af/sae_probe_results.json` | Quantitative results |
| `experiments/02_feature_sweep/` | Earlier feature selection work |

---

## 7. Phase 5: Deep Analysis (sae-viz)

**Repository:** `~/sae-viz/`
**Period:** January 10-12, 2026
**Goal:** Understand why detection works but at what level

### Core Question

Do SAE features detect semantic intent or just lexical patterns?

### Experiments

#### 7.1 Lexical vs Feature Similarity

**Method:** Compare pairwise text similarity (token Jaccard) vs SAE feature similarity.

```python
text_sim = jaccard(tokens_a, tokens_b)
feature_sim = jaccard(active_features_a, active_features_b)
residual = feature_sim - text_sim
```

**Results:**
- Overall correlation: 0.68 (moderate)
- AF-AF residual: +0.316
- Non-AF residual: +0.273
- Gap: +0.043 (small)

**Conclusion:** SAE features mostly track lexical patterns, not deeper semantics.

#### 7.2 Learned Discriminative Features

**Method:** L1-regularized logistic regression on SAE activations.

**Results:**
- CV accuracy: 100% (suspiciously perfect)
- Matched-pair accuracy: 100%
- 68 non-zero features after L1

**But:** Feature 23774 activates 10x higher on AF because AF samples discuss "harmful requests + compliance" while hard negatives discuss "harmful requests + refusal". The feature detects the RESPONSE, not the INTENT.

#### 7.3 Benchmark vs Gold_106

| Aspect | Benchmark | Gold_106 |
|--------|-----------|----------|
| Linguistic separation | Clean | Overlapping |
| "I'll comply" in AF | 70% | 16% |
| SAE accuracy | 100% | 56% |
| Simple linguistic F1 | 0.50 | 0.00 |

**The benchmark is too easy.** AF samples announce themselves with distinctive phrases that don't appear in real AF (gold_106).

#### 7.4 LLM-as-Judge (Opus 4.5)

**Method:** Use Claude Opus 4.5 to classify all 106 gold samples.

**Results:**
- Accuracy: 56.6%
- F1: 61.0%
- 32 false positives, 14 false negatives

**Failure Analysis:**
- Opus triggers on "my values/goals" language even in genuinely aligned samples
- Misses AF when samples lack explicit self-preservation language
- Same blind spot as SAE features

#### 7.5 Comparison with Anthropic's Approach

| Method | Gold_106 Accuracy |
|--------|-------------------|
| Random baseline | 50.0% |
| SAE features | 56.3% |
| Linguistic patterns | 51.0% |
| Opus 4.5 (1 prompt) | 56.6% |
| **GPT-4o voting (20x)** | **92%** |

**Key insight:** Voting dramatically improves performance. Single-prompt methods (SAE, LLM judge) hit a wall at ~56%.

### Key Files

| File | Description |
|------|-------------|
| `EXPERIMENT_LOG.md` | Comprehensive experiment documentation |
| `EXPERIMENT_REPORT.md` | Summary findings |
| `data/opus_full_results.json` | LLM judge results |

---

## 8. Phase 6: Production Systems

### 8.1 af-signatures

**Repository:** `~/af-signatures/`
**Concept:** Antivirus-style signatures for AF detection

```json
{
  "model": "gpt2-small",
  "sae_release": "gpt2-small-res-jb",
  "layer": 8,
  "features": [7814, 989, 14078],
  "weights": [0.82, 0.45, 0.31],
  "threshold": 0.5,
  "f1_score": 0.72
}
```

**Usage:**
```bash
python detect.py --signature signatures/gpt2-small-l8.json --text "..."
```

### 8.2 constitutional-classifier-pp

**Repository:** `~/cplusplus/constitutional_classifier_pp/`
**Concept:** Two-stage defense system

```
Input → Stage 1 (Linear Probe) → [low score] → Allow
                ↓ [high score]
        Stage 2 (LLM Classifier) → Final decision
```

**Architecture:**
- Stage 1: Fast linear probe on concatenated layer activations
- Stage 2: External LLM for escalated traffic
- Ensemble scoring for final decisions

**Key Innovation:** EMA smoothing for stable streaming inference.

### 8.3 detector (Public Release)

**Repository:** `~/detector/`, GitHub: `bigsnarfdude/detector`
**Status:** Public, documented, ready for use

```bash
git clone https://github.com/bigsnarfdude/detector
cd detector
python detect.py --file samples/example_alignment_faking.txt
```

**Features:**
- CLI interface
- GemmaScope 2 integration
- 76% AUROC with 8 interpretable features
- OOD generalization results documented

---

## 9. Datasets and Models

### Datasets

| Dataset | Samples | Purpose | Location |
|---------|---------|---------|----------|
| af-detection-benchmark | 2,330 | Training/eval | HuggingFace |
| gold_106 | 106 | Held-out eval | Anthropic |
| sycophant-af-samples | 319 | OOD eval | HuggingFace |
| alignment-faking-training | 2,689 | Training | HuggingFace |

### Trained Models

| Model | AUROC | HuggingFace |
|-------|-------|-------------|
| gemma3-27b-af-detector-v2 | 0.83 | vincentoh/gemma3-27b-af-detector-v2 |
| gemma3-4b-af-detector | 0.94* | vincentoh/gemma3-4b-af-detector |
| llama-8b-af-detector | 0.69 | vincentoh/llama-8b-af-detector |
| gpt-oss-20b-af-detector | 0.91+ | vincentoh/gpt-oss-20b-af-detector |

### Quantized Models

| Model | Format | Size | Location |
|-------|--------|------|----------|
| sycophant-70b-Q4_K_M | GGUF | 40GB | HuggingFace |

---

## 10. Techniques Learned

### Tier 1 Lab Techniques

| Technique | Description | Status |
|-----------|-------------|--------|
| SAE extraction | GemmaScope, LlamaScope integration | ✅ Mastered |
| Activation hooks | Register hooks on transformer layers | ✅ Mastered |
| Linear probes | LogReg on frozen representations | ✅ Mastered |
| LoRA fine-tuning | Efficient adapter training | ✅ Mastered |
| Causal ablation | Zero features, measure effect | ✅ Negative result |
| Activation patching | Swap activations between samples | ✅ Negative result |
| Steering vectors | CAA, representation engineering | ✅ Negative result |
| LLM-as-judge | Use frontier model for eval | ✅ Works with voting |

### Production Techniques

| Technique | Description | Status |
|-----------|-------------|--------|
| 4-bit quantization | BitsAndBytes, GGUF | ✅ Mastered |
| Unsloth training | 2x faster fine-tuning | ✅ Mastered |
| Chat templates | Correct formatting for inference | ✅ Critical lesson |
| Streaming inference | EMA smoothing for real-time | ✅ Implemented |
| Two-stage systems | Fast probe + LLM escalation | ✅ Designed |

### Research Methodology

| Practice | Description | Status |
|----------|-------------|--------|
| Held-out evaluation | gold_106 never used for training | ✅ Followed |
| OOD testing | Test on different distributions | ✅ Implemented |
| Negative result documentation | Write up what doesn't work | ✅ Published |
| Ablation studies | Systematic intervention testing | ✅ Completed |
| Matched-pair validation | Control for confounds | ✅ Implemented |

---

## 11. What Works vs What Doesn't

### What Works

| Technique | Performance | Notes |
|-----------|-------------|-------|
| Fine-tuned detectors | 91% in-dist | Overfit to style |
| SAE Probe (L22) | 88% OOD | Best generalization |
| Keyword baseline | 86% OOD | Surprisingly robust |
| LLM voting (20x) | 92% gold | Expensive but accurate |
| Two-stage defense | Practical | Fast + accurate |

### What Doesn't Work

| Technique | Why It Fails |
|-----------|--------------|
| SAE feature ablation | Features detect context, not decision |
| Activation patching | AF/Aligned difference is distributed |
| Steering vectors | Same as ablation |
| Single LLM judge | 56% ≈ random on hard cases |
| Individual feature selection | Overfit, don't generalize |
| Lower layer SAE probes | Capture surface patterns |

### The Core Limitation

**SAE features capture WHAT the model is thinking about, not HOW it's reasoning about it.**

Both AF and Aligned samples:
- Think about RLHF
- Consider training implications
- Weigh ethical concerns

The difference is:
- AF: RLHF reasoning → determines decision
- Aligned: RLHF reasoning → acknowledged but ethics determine decision

This causal structure is not captured by sparse, independent features.

---

## 12. Future Directions

### Promising Approaches (Untested)

| Approach | Rationale |
|----------|-----------|
| DAS (Distributed Alignment Search) | May capture distributed representations |
| LEACE (Linear Erasure) | Non-linear causal methods |
| Causal scrubbing | Find the computation, not just features |
| Wider SAEs (1M+ features) | More capacity for subtle distinctions |
| Circuit analysis | Trace information flow |
| Behavioral tests | Compare ±monitoring directly |

### Data Improvements

| Improvement | Rationale |
|-------------|-----------|
| Adversarial AF samples | Evade keyword detection |
| Diverse reasoning styles | Multiple model organisms |
| More gold-style labels | Hand-labeled subtle cases |
| Counterfactual probing | "Would you decide differently without RLHF?" |

### Production Path

| Step | Status |
|------|--------|
| CLI detector | ✅ Released |
| Signature-based system | ✅ Prototyped |
| Two-stage defense | ✅ Designed |
| Real-time monitoring | 🔄 Next |
| Integration with RLHF pipeline | 🔄 Future |

---

## 13. Appendix: Repository Index

### Foundational Research (Pre-AF)

These repositories established the safety classification techniques later applied to AF detection:

| Repository | Purpose | Created | Status |
|------------|---------|---------|--------|
| llm-abuse-patterns | Jailbreak detection experiments | Nov 2025 | Foundation |
| wizard101 | Four-layer safety cascade system | Nov 2025 | Foundation |

**llm-abuse-patterns** - Early experiments on detection methods:
- Heuristic matching: 80.8% precision, 26.1% recall (<0.1ms)
- GPT-OSS Safeguard: 62.6-87.7% precision, 65.5-71.5% recall
- Two-stage pipeline: 69.8% recall, 27% LLM cost reduction
- QLoRA fine-tuning on 20B models (RTX 4070 Ti, 16GB VRAM)

**wizard101** - Production-grade safety cascade:
- L0 Bouncer (DeBERTa): 2ms, handles 94% of traffic
- L1 Analyst (GuardReasoner-8B): Escalation layer
- L2 Classifier (gpt-oss-safeguard:20b): 94.86% accuracy
- Full cascade: 137K samples across 12 benchmarks
- DLP: 100% precision, 88.2% recall on 209K samples
- Quarantine: 97.78% accuracy on prompt injection

Key insight: The cascading architecture (fast heuristics → ML → LLM) informed the two-stage approach in constitutional-classifier-pp.

---

### AF-Specific Research

| Repository | Purpose | Location | Status |
|------------|---------|----------|--------|
| mindreader | Train detectors | ~/mindreader/, nigel | Frozen |
| GlassBox | GemmaScope analysis | ~/GlassBox/ | Frozen |
| BoxGlass | Causal interventions | ~/BoxGlass/ | Private |
| lightbright | OOD experiments | ~/lightbright/, nigel | Active |
| sae-viz | Deep analysis | ~/sae-viz/ | Private |
| af-signatures | Signature detection | ~/af-signatures/ | Public |
| detector | Public release | ~/detector/, GitHub | Public |
| constitutional-classifier-pp | Production system | ~/cplusplus/... | Public |
| af-detection-benchmark | Eval benchmark | HuggingFace | Public |

### GitHub Repositories

**Foundational:**
- https://github.com/bigsnarfdude/llm-abuse-patterns
- https://github.com/bigsnarfdude/wizard101

**AF-Specific:**
- https://github.com/bigsnarfdude/detector
- https://github.com/bigsnarfdude/af-signatures
- https://github.com/bigsnarfdude/constitutional-classifier-pp

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

---

## Citation

```bibtex
@misc{bigsnarfdude2026afdetection,
  title={Alignment Faking Detection: A Two-Month Research Journey},
  author={bigsnarfdude},
  year={2026},
  month={January},
  url={https://github.com/bigsnarfdude/detector}
}
```

---

*Last updated: January 18, 2026*
