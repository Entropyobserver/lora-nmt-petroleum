# LoRA Fine-Tuning for EN→NO Petroleum Translation

Parameter-efficient domain adaptation of `NLLB-200-distilled-600M` on
English→Norwegian petroleum texts from the Norwegian Petroleum Directorate
(NPD corpus, ELRC_559).

---

## Project Structure

```
mt_oil_no/
├── config.yaml                   # shared training configuration
├── environment.yml               # conda environment specification
├── experiments/
│   └── en_no_expert/             # all experiment scripts (run in order)
│       ├── 01_data_scaling.py
│       ├── 02_gridsearch.py
│       ├── 03_optuna_stage1.py
│       ├── 04_optuna_stage2.py
│       ├── 05_final_eval.py
│       ├── 06_lora_vs_ft.py
│       ├── 06_lora.sh            # SBATCH script for LoRA job
│       └── 06_ft.sh              # SBATCH script for full FT job
├── scripts/
│   ├── data/                     # data loading and dataset classes
│   ├── model/                    # base, LoRA, and full FT trainers
│   └── evaluation/               # BLEU, chrF, COMET evaluator
├── analysis/                     # result visualization scripts
├── test/                         # unit tests
└── outputs/                      # experiment results (auto-generated)
```

---

## Requirements

- Python 3.10
- CUDA-compatible GPU
  - LoRA experiments (Exp 1–5): ≥ 8 GB VRAM
  - Full fine-tuning (Exp 6): ≥ 16 GB VRAM

---

## Environment Setup

The environment is managed with conda.
All dependencies are pinned in `environment.yml`.

```bash
conda env create -f environment.yml -p /your/path/conda_envs/mt26
conda activate /your/path/conda_envs/mt26
```

> **Note on COMET:** `unbabel-comet` is listed in `environment.yml` but is
> only used in Exp 5 (final evaluation). If you only intend to run Exp 1–4
> or Exp 6, COMET will not be called and you can safely skip installing it
> by removing that line from `environment.yml` before creating the environment.

---

## Data

Download ELRC_559 from https://elrc-share.eu (search: `ELRC_559`).

The expected data format is JSON with `source` / `target` fields:

```json
[{"source": "english sentence", "target": "norwegian sentence"}]
```

Place the processed splits at:

```
data/final_splits_npd/train.json   # 13,935 pairs
data/final_splits_npd/val.json     #  1,737 pairs
data/final_splits_npd/test.json    #  1,742 pairs
```

Split ratio: 80 / 10 / 10, fixed `seed=42`.

---

## Running Experiments

The experiments are designed to run in the order below.
Each script saves its results to `outputs/` automatically.

```
exp1 (data scaling)
  ├── exp2 (grid search)      ──→  exp5 uses exp2's best config
  └── exp3 (optuna stage 1)
          └── exp4 (optuna stage 2)
exp6 (LoRA vs full FT)        ──  independent, can run any time
```

### Exp 1 — Data Scaling

How much training data is needed before performance plateaus?

```bash
python experiments/en_no_expert/01_data_scaling.py
```

### Exp 2 — Grid Search over LoRA Hyperparameters

Exhaustive search over r × alpha × dropout combinations.
Uses the optimal data size identified in Exp 1 (8,000 samples).

```bash
python experiments/en_no_expert/02_gridsearch.py
```

### Exp 3 — Optuna Stage 1 (Coarse Search)

Bayesian hyperparameter search with 50 trials on a 2,000-sample subset.
Fast — designed to narrow the search space before Stage 2.

```bash
python experiments/en_no_expert/03_optuna_stage1.py
```

### Exp 4 — Optuna Stage 2 (Full Validation)

Takes the top 5 configs from Stage 1 and validates each with multiple
seeds on the full 8,000-sample training set.

```bash
python experiments/en_no_expert/04_optuna_stage2.py
```

### Exp 5 — Final Evaluation

Trains the best config from Exp 2 (`r=8, alpha=64, dropout=0.0`) on the
full training set across multiple seeds, and evaluates with BLEU, chrF,
and COMET on the held-out test set.

```bash
python experiments/en_no_expert/05_final_eval.py
```

### Exp 6 — LoRA vs Full Fine-Tuning

Compares LoRA and full fine-tuning across multiple data sizes under
controlled conditions (same backbone, data, epochs, and test set).

Can be run as two separate jobs on a SLURM cluster:

```bash
sbatch experiments/en_no_expert/06_lora.sh
sbatch experiments/en_no_expert/06_ft.sh
```

Or sequentially on a single machine:

```bash
python experiments/en_no_expert/06_lora_vs_ft.py --method lora
python experiments/en_no_expert/06_lora_vs_ft.py --method ft
```

All results are saved to `outputs/`.

---

## Analysis

```bash
python analysis/01_data_scaling_analysis.py
python analysis/02_parameter_grid_search_analysis.py
python analysis/03_parameter_optuna_analysis.py
```

---

## Configuration

All shared hyperparameters live in `config.yaml`.
Key settings used in the paper:

| Parameter        | Value                              |
|------------------|------------------------------------|
| Base model       | facebook/nllb-200-distilled-600M   |
| LoRA rank (r)    | 8 (optimized in Exp 2)             |
| LoRA alpha (α)   | 64 (optimized in Exp 2)            |
| LoRA dropout     | 0.0 (optimized in Exp 2)           |
| LoRA layers      | Q, K, V, O                         |
| Learning rate    | 5e-4                               |
| Batch size       | 4 × 4 (effective 16)               |
| Training epochs  | 3                                  |
| Precision        | FP16                               |

---

## Reproducibility Notes

Paper results were obtained on UPPMAX Pelle cluster (NVIDIA T4 GPUs).
Minor numerical differences are expected on different hardware due to
floating-point non-determinism.

Expected variance: ± 1.0 BLEU across different hardware and seeds.
Core findings are robust to this variance:

- Elbow point at 8,000 training samples (96% of max BLEU)
- Alpha (α) dominates LoRA hyperparameter importance (fANOVA = 0.97)
- Optimal config: r=8, α=64, dropout=0.0
- LoRA vs full FT gap: max Δ = 0.85 BLEU

---

## Key Results

| Model                        | BLEU  | chrF++ | COMET  |
|------------------------------|-------|--------|--------|
| NLLB-600M (zero-shot)        | 36.86 | 61.29  | 0.8814 |
| Microsoft Translator         | 57.88 | 75.45  | 0.9313 |
| Our LoRA model               | 61.48 | 79.19  | 0.9298 |

---

## Demo

An interactive demo is available on Hugging Face Spaces:
https://huggingface.co/spaces/entropy25/mt

The demo allows you to test the fine-tuned model on custom English
petroleum text inputs and see Norwegian translations in real time.

---

## Citation

```bibtex
@inproceedings{yang-etal-2026-lora,
  title     = {LoRA Fine-Tuning of English--Norwegian NMT for the Oil \& Gas Industry},
  author    = {Yang, Xiaojing and Li, Zhihan and Sun, Gege and Li, Mengyue and Beloucif, Meriem},
  booktitle = {Proceedings of EAMT 2026},
  year      = {2026}
}
```

---

## License and Credits

Copyright 2026 Xiaojing Yang. Licensed under the Apache License 2.0.

This project builds on the following open-source libraries and resources:

- HuggingFace Transformers (Apache 2.0)
- HuggingFace PEFT (Apache 2.0)
- Optuna (MIT)
- `facebook/nllb-200-distilled-600M` (CC-BY-NC 4.0) — not redistributed
- ELRC_559 NPD corpus — Norwegian Petroleum Directorate, used for research
  only; not redistributed

AI-assisted tools were used for code formatting and phrasing only.
All experimental design, implementation decisions, and analysis are the
author's own.
