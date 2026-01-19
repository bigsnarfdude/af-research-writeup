# Alignment Faking Detection Research

Two months of experiments detecting alignment faking (AF) in LLM chain-of-thought reasoning.

## Key Findings

| Method | In-Distribution | OOD |
|--------|-----------------|-----|
| Fine-tuned detectors | 91% | 31-48% |
| SAE Probe (L22) | 99% | 88% |
| Keyword baseline | 75% | 86% |

**Main result:** SAE features detect but don't cause AF. Causal interventions (ablation, steering) fail because the AF/Aligned distinction is distributed, not sparse.

## Research Timeline

```
Nov 2025  llm-abuse-patterns, wizard101 (foundational safety work)
Dec 2025  mindreader, GlassBox, BoxGlass (AF detection + causal experiments)
Jan 2026  lightbright, sae-viz, detector (OOD testing + production release)
```

## Repositories

| Repo | Purpose |
|------|---------|
| [llm-abuse-patterns](https://github.com/bigsnarfdude/llm-abuse-patterns) | Jailbreak detection foundations |
| [wizard101](https://github.com/bigsnarfdude/wizard101) | Safety cascade system |
| [detector](https://github.com/bigsnarfdude/detector) | Public AF detector |
| [af-detection-benchmark](https://huggingface.co/datasets/vincentoh/af-detection-benchmark) | Eval dataset |

## Full Writeup

See [ALIGNMENT_FAKING_DETECTION_RESEARCH.md](ALIGNMENT_FAKING_DETECTION_RESEARCH.md) for the complete 800-line research document covering:

- Background and motivation
- All experimental phases
- Techniques learned
- What works vs what doesn't
- Future directions

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
