#!/bin/bash -l
#SBATCH -A uppmax2026-1-123
#SBATCH -M pelle
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -t 48:00:00
#SBATCH -J compare_ft
#SBATCH -o logs/compare_ft_en_no_%j.out
#SBATCH -e logs/compare_ft_en_no_%j.err

source ~/miniconda3/bin/activate
conda activate /proj/uppmax2026-1-123/private/yaxj1/conda_envs/mt26
cd /crex/proj/uppmax2026-1-123/private/yaxj1/mt_oil_no
python experiments/en_no_expert/06_lora_vs_ft.py --method ft
