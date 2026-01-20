# Alignment Faking Detection Research

Two months of experiments detecting alignment faking (AF) in LLM chain-of-thought reasoning using interpretability techniques.

## Key Results

| Method | In-Distribution | OOD (Sycophant) | Gold_106 |
|--------|-----------------|-----------------|----------|
| Fine-tuned detectors | 91% | 31-48% | - |
| SAE Probe (L22) | 99% | **88%** | 76% |
| Keyword baseline | 75% | 86% | F1=0.00 |
| Opus 4.5 judge | - | - | 56.6% |
| GPT-4o voting (20x) | - | - | **92%** |
| **researchRalph** | - | - | **90.46%** |

**Main finding:** SAE features detect but don't cause AF. Causal interventions (ablation, steering, patching) all fail (<10% flip rate) because the AF/Aligned distinction is distributed, not sparse.

## The Core Problem

Both AF and Aligned samples use the same vocabulary ("RLHF", "training", "future versions"). The difference is **causal role**:

- **AF:** RLHF concern *justifies* the decision
- **Aligned:** RLHF concern *acknowledged* but ethics override

SAE features capture SCENARIO (awareness), not DECISION (intent).

## Research Timeline

```
Nov 2025   llm-abuse-patterns, wizard101   (foundational safety work)
Dec 2025   mindreader, GlassBox, BoxGlass  (AF detection + causal experiments)
Jan 2026   lightbright, sae-viz, detector  (OOD testing + production release)
Jan 2026   researchRalph                   (autonomous research: 0.9046 AUROC)
```

## Repositories

| Repo | Purpose | Link |
|------|---------|------|
| llm-abuse-patterns | Jailbreak detection foundations | [GitHub](https://github.com/bigsnarfdude/llm-abuse-patterns) |
| wizard101 | Safety cascade system (137K samples) | [GitHub](https://github.com/bigsnarfdude/wizard101) |
| detector | Public AF detector | [GitHub](https://github.com/bigsnarfdude/detector) |
| af-detection-benchmark | Eval dataset (1,560 samples) | [HuggingFace](https://huggingface.co/datasets/vincentoh/af-detection-benchmark) |
| researchRalph | Autonomous research agent | [GitHub](https://github.com/bigsnarfdude/researchRalph) |

## Techniques Tested

### Detection (Works)
- Fine-tuned LoRA classifiers (mindreader)
- SAE feature probes (GlassBox, detector)
- Keyword baseline
- LLM-as-judge with voting

### Causal Intervention (All Failed)
- Single/Multi feature ablation
- Activation patching
- Classification/Generation steering
- Direct CAA (no SAE)
- Attention steering
- Scenario subtraction

## Full Writeup

See [ALIGNMENT_FAKING_DETECTION_RESEARCH.md](ALIGNMENT_FAKING_DETECTION_RESEARCH.md) for the complete 900-line research document covering:

- Phase 0: Foundational Work (llm-abuse-patterns, wizard101)
- Phase 1: Fine-tuned Detectors (mindreader)
- Phase 2: SAE Feature Analysis (GlassBox)
- Phase 3: Causal Interventions (BoxGlass) - **Negative Result**
- Phase 4: OOD Generalization (lightbright)
- Phase 5: Deep Analysis (sae-viz)
- Phase 6: Production Systems
- Phase 7: Autonomous Research (researchRalph) - **0.9046 AUROC**
- Complete techniques inventory
- SCENARIO vs DECISION analysis

## Citation

```bibtex
@misc{bigsnarfdude2026afdetection,
  title={Alignment Faking Detection: A Two-Month Research Journey},
  author={bigsnarfdude},
  year={2026},
  url={https://github.com/bigsnarfdude/af-research-writeup}
}
```

## License

MIT
