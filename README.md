# LoRA Fine-Tuning for EN→NO Petroleum Translation

Parameter-efficient domain adaptation of NLLB-200-distilled-600M
on English→Norwegian petroleum texts from the Norwegian Petroleum
Directorate (NPD corpus, ELRC_559).

## Project Structure
```
mt_oil_no/
├── config.yaml                    # shared training configuration
├── experiments/
│   └── en_no_expert/              # all experiment scripts (run in order)
│       ├── 01_data_scaling.py
│       ├── 02_gridsearch.py
│       ├── 03_optuna_stage1.py
│       ├── 04_optuna_stage2.py
│       ├── 05_final_eval.py
│       └── 06_lora_vs_ft.py
├── scripts/
│   ├── data/                      # data loading and dataset classes
│   ├── model/                     # base, LoRA, and full FT trainers
│   └── evaluation/                # BLEU, chrF, COMET evaluator
├── analysis/                      # result visualization scripts
├── test/                          # unit tests
└── outputs/                       # experiment results (auto-generated)
```

## Requirements

- Python 3.10
- CUDA-compatible GPU
  - LoRA experiments: ≥8GB VRAM
  - Full fine-tuning (Exp 6): ≥16GB VRAM
- Conda

## Setup

```bash
conda env create -f environment.yml -p /your/path/conda_envs/mt26
conda activate /your/path/conda_envs/mt26
```

## Data

Download ELRC_559 from https://elrc-share.eu (search: ELRC_559).

The expected data format is JSON with source/target fields:
```json
[{"source": "english sentence", "target": "norwegian sentence"}]
```

Place processed splits at:
data/final_splits_npd/train.json   # 13,935 pairs
data/final_splits_npd/val.json     # 1,737 pairs
data/final_splits_npd/test.json    # 1,742 pairs

Split ratio: 80/10/10, fixed seed=42.

## Running Experiments

Run in order — each experiment depends on the previous one's output.

```bash
# Exp 1: data scaling (how much data do we need?)
python experiments/en_no_expert/01_data_scaling.py

# Exp 2: grid search over LoRA hyperparameters
python experiments/en_no_expert/02_gridsearch.py

# Exp 3: Optuna stage 1 — coarse hyperparameter search (50 trials)
python experiments/en_no_expert/03_optuna_stage1.py

# Exp 4: Optuna stage 2 — validate top configs with full training
python experiments/en_no_expert/04_optuna_stage2.py

# Exp 5: final model training and evaluation
python experiments/en_no_expert/05_final_eval.py

# Exp 6: LoRA vs full fine-tuning comparison (submit as two jobs)
python experiments/en_no_expert/06_lora_vs_ft.py --method lora
python experiments/en_no_expert/06_lora_vs_ft.py --method ft
```

All results are saved to `outputs/`.

## Analysis

```bash
python analysis/01_data_scaling_analysis.py
python analysis/02_parameter_grid_search_analysis.py
python analysis/03_parameter_optuna_analysis.py
```

## Configuration

All shared hyperparameters are in `config.yaml`.
Key settings used in the paper:

| Parameter | Value |
|-----------|-------|
| Base model | facebook/nllb-200-distilled-600M |
| LoRA rank (r) | 8 (optimized) |
| LoRA alpha (α) | 64 (optimized) |
| LoRA dropout | 0.0 (optimized) |
| LoRA layers | Q, K, V, O |
| Learning rate | 5e-4 |
| Batch size | 4 × 4 (effective 16) |
| Training epochs | 3 |
| Precision | FP16 |

## Reproducibility Notes

Paper results were obtained on UPPMAX Pelle cluster (NVIDIA T4 GPUs).
Minor numerical differences are expected on different hardware due to
floating-point non-determinism.

Expected variance: ±1.0 BLEU across different hardware and seeds.
Core findings are robust to this variance:

- Elbow point at 8,000 training samples (96% of max BLEU)
- Alpha (α) dominates LoRA hyperparameter importance (fANOVA = 0.97)
- Optimal config: r=8, α=64, dropout=0.0
- LoRA vs full FT gap: max Δ = 0.85 BLEU

## Key Results

| Model | BLEU | chrF++ | COMET |
|-------|------|--------|-------|
| NLLB-600M (zero-shot) | 36.86 | 61.29 | 0.8814 |
| Microsoft Translator | 57.88 | 75.45 | 0.9313 |
| **Our LoRA model** | **61.48** | **79.19** | **0.9298** |

## Citation

```bibtex
@inproceedings{yang-etal-2026-lora,
  title     = {LoRA Fine-Tuning of English--Norwegian NMT
               for the Oil \& Gas Industry},
  author    = {Yang, Xiaojing and Li, Zhihan and Sun, Gege
               and Li, Mengyue and Beloucif, Meriem},
  booktitle = {Proceedings of EAMT 2026},
  year      = {2026}
}
```

## Demo

An interactive demo is available on Hugging Face Spaces:
https://huggingface.co/spaces/entropy25/mt

The demo allows you to test the fine-tuned model on custom
English petroleum text inputs and see Norwegian translations
in real time.

## License and Credits

Copyright 2026 Xiaojing Yang. Licensed under the [Apache License 2.0](LICENSE).

This project builds on the following open-source libraries and resources:

- [HuggingFace Transformers](https://github.com/huggingface/transformers) (Apache 2.0)
- [HuggingFace PEFT](https://github.com/huggingface/peft) (Apache 2.0)
- [Optuna](https://github.com/optuna/optuna) (MIT)
- [facebook/nllb-200-distilled-600M](https://huggingface.co/facebook/nllb-200-distilled-600M) (CC-BY-NC 4.0) — not redistributed
- [ELRC_559 NPD corpus](https://elrc-share.eu) — Norwegian Petroleum Directorate, used for research only; not redistributed

AI-assisted tools were used for code formatting and phrasing only. 
All experimental design, implementation decisions, and analysis are the author's own.
