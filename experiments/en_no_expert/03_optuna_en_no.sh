#!/bin/bash -l
#SBATCH -A uppmax2026-1-123
#SBATCH -M pelle
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -t 48:00:00
#SBATCH -J optuna_en_no
#SBATCH -o logs/optuna_en_no_%j.out
#SBATCH -e logs/optuna_en_no_%j.err

source ~/miniconda3/bin/activate
conda activate /proj/uppmax2026-1-123/private/yaxj1/conda_envs/mt26

cd /crex/proj/uppmax2026-1-123/private/yaxj1/mt_oil_no
# Stage 1: Coarse search
python experiments/en_no_expert/03_optuna_stage1.py 
# Stage 2: Fine-tuning
python experiments/en_no_expert/04_optuna_stage2.py



