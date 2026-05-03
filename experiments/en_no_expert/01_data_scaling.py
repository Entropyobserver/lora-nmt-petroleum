import os
import sys
import json
import logging
import traceback
import gc
from pathlib import Path

# Setup project paths
script_dir   = Path(__file__).resolve().parent  # Get current script directory
project_root = script_dir.parent.parent          # two levels up
sys.path.insert(0, str(project_root))            # Add project root to sys.path for imports

# This is important on clusters to avoid re-downloading models
HF_CACHE_DIR = "/proj/uppmax2026-1-123/private/yaxj1/hf_cache"
os.environ.update({
    "HF_HOME": HF_CACHE_DIR,
    "TRANSFORMERS_CACHE": HF_CACHE_DIR,
    "HF_DATASETS_CACHE": HF_CACHE_DIR,
    "TORCH_HOME": HF_CACHE_DIR,
})

# External libraries
import yaml
import pandas as pd
import torch

from scripts.model.lora_trainer import LoRATrainer
from scripts.data.data_loader import DataManager
from scripts.evaluation.base_evaluator import BaseEvaluator

def get_logger(output_dir: Path) -> logging.Logger:
    logger = logging.getLogger("exp1")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s  %(message)s", datefmt="%H:%M:%S")
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    fh = logging.FileHandler(output_dir / "experiment.log", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger

def load_data_and_evaluator(cfg):
    #Load train/val/test datasets and evaluation metrics.
    data_manager = DataManager(cfg)
    train_ds, val_ds, test_ds = data_manager.load_splits()
    evaluator = BaseEvaluator(use_comet=False)
    return train_ds, val_ds, test_ds, evaluator

def build_data_sizes(cfg, train_ds):
    #Define different dataset sizes for scaling experiment.
    n = len(train_ds)
    # Predefined subset sizes
    subset = [100, 500, 1000, 2000, 4000, 6000, 8000, 10000]
    # Keep only sizes smaller than dataset + always include full dataset
    sizes = sorted(set(s for s in subset if s <= n) | {n})
    return sizes


# Run one experiment (core unit)
def run_one(size, seed, cfg, train_ds, val_ds, test_ds, evaluator, output_dir, logger):
    """
    Run a single experiment with:
    - a specific training data size
    - a specific random seed
    Returns evaluation metrics.
    """
    # Create directory for this run: size_x/seed_y/
    run_dir = output_dir / f"size_{size}" / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Initialize LoRA trainer (model + tokenizer + language setup)
    trainer = LoRATrainer(
        model_name=cfg["model"]["pretrained"],
        src_lang=cfg["model"]["src_lang"],
        tgt_lang=cfg["model"]["tgt_lang"],
    )

    # Training configuration (LoRA hyperparameters are fixed here)
    train_config = {
        "output_dir": str(run_dir / "training"),
        "seed": seed,
        "r": cfg["lora"]["r"],
        "alpha": cfg["lora"]["alpha"],
        "dropout": cfg["lora"]["dropout"],
        "target_modules": cfg["lora"]["target_modules"],
        "epochs": cfg["training"]["epochs"],
        "batch_size": cfg["training"]["batch_size"],
        "gradient_accumulation_steps": cfg["training"].get("grad_accumulation", 4),
        "learning_rate": cfg["training"]["lr"],
        "warmup_steps": cfg["training"]["warmup_steps"],
        "eval_steps": cfg["training"]["eval_steps"],
        "early_stopping_patience": cfg["training"]["early_stopping_patience"],
        "fp16": cfg["training"]["fp16"],
        "save_total_limit": 1,
        "save_final_model": True,
    }

    try:
        # Subset training data to specified size
        train_subset = train_ds.subset(size, seed=seed)

        # Train model on this subset and evaluate on validation set
        train_result = trainer.train(
            [s.to_dict() for s in train_subset.samples],
            [s.to_dict() for s in val_ds.samples],
            train_config,
        )

        # Evaluate on TEST set , always evaluate on FULL test set (important for fair comparison)
        test_preds = trainer.generate_predictions(
            train_result["model"],
            test_ds,
            batch_size=8,
            num_beams=cfg["generation"]["num_beams"],
        )

        # Compute test metrics using evaluator (BLEU, chrF, etc.)
        test_metrics = evaluator.evaluate_all(
            [s.source for s in test_ds.samples],
            test_preds,
            [s.target for s in test_ds.samples],
        )

        # Store results
        entry = {
            "data_size": size,
            "seed": seed,
            "val_bleu": train_result["bleu"],
            "val_chrf": train_result["chrf"],
            "val_loss": train_result["loss"],
            "test_bleu": test_metrics["bleu"],
            "test_chrf": test_metrics["chrf"],
            "model_path": train_result.get("final_model_path", ""),
        }

        # Save per-run results
        with open(run_dir / "metrics.json", "w") as f:
            json.dump(entry, f, indent=2)

        logger.info(f"val  BLEU={train_result['bleu']:.4f}  chrF={train_result['chrf']:.2f}")
        logger.info(f"test BLEU={test_metrics['bleu']:.4f}  chrF={test_metrics['chrf']:.2f}")

        return entry

    finally:
        # Clean up intermediate checkpoints, keep final model only
        #import shutil
        #training_dir = run_dir / "training"
        #if training_dir.exists():
        #    for ckp in training_dir.glob("checkpoint-*"):
        #        shutil.rmtree(ckp)

        # Clean GPU memory after each run
        del trainer
        torch.cuda.empty_cache()
        gc.collect()



def run_all(data_sizes, seeds, cfg, train_ds, val_ds, test_ds, evaluator, output_dir, logger):
    #Run full experiment:- all data sizes - all random seeds
    results = []

    # Track best model across all runs
    best_bleu = 0.0
    best_info = {}
    total = len(data_sizes) * len(seeds)
    run_idx = 0

    # Nested loop: (data size × seed)
    for size in data_sizes:
        for seed in seeds:
            run_idx += 1
            logger.info(f"\n[{run_idx}/{total}] size={size}, seed={seed}")

            try:
                entry = run_one(
                    size, seed, cfg,
                    train_ds, val_ds, test_ds,
                    evaluator, output_dir, logger
                )
                results.append(entry)

                # Update best model if improved
                if entry["test_bleu"] > best_bleu:
                    best_bleu = entry["test_bleu"]
                    best_info = {
                        "size":       size,
                        "seed":       seed,
                        "test_bleu":  best_bleu,
                        "model_path": entry["model_path"],
                    }
                    logger.info("  ---new best---")

            except Exception as e:
                logger.error(f"  FAILED: {e}\n{traceback.format_exc()}")

                # Record failed run
                results.append({
                    "data_size": size,
                    "seed": seed,
                    "val_bleu": 0.0,
                    "val_chrf": 0.0,
                    "val_loss": 999.0,
                    "test_bleu": 0.0,
                    "test_chrf": 0.0,
                    "model_path": "",
                    "failed": True,
                })

    return results, best_info

def save_results(results, best_info, output_dir, logger):
    #Save raw results aggregated summary best model info
    df = pd.DataFrame(results)

    # Save raw results
    df.to_csv(output_dir / "results.csv", index=False)
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Remove failed runs
    valid_df = df[df["failed"] != True] if "failed" in df.columns else df
    if not valid_df.empty:
        # Aggregate results per data size
        summary = valid_df.groupby("data_size").agg(
            test_bleu_mean=("test_bleu", "mean"),
            test_bleu_std=("test_bleu", "std"),
            test_chrf_mean=("test_chrf", "mean"),
            test_chrf_std=("test_chrf", "std"),
        ).round(4)
        logger.info(f"\n{summary}")
        summary.to_csv(output_dir / "summary.csv")

    # Save best model info
    if best_info:
        logger.info(
            f"\nBest: size={best_info['size']}, seed={best_info['seed']}, "
            f"BLEU={best_info['test_bleu']:.4f}"
        )
        with open(output_dir / "best_model.json", "w") as f:
            json.dump(best_info, f, indent=2)

    logger.info(f"Results saved to {output_dir}")


def main():
    """
    Full pipeline:
    1. Load config
    2. Load data
    3. Define data sizes
    4. Run experiments
    5. Save results
    """
    # 1. Load config
    cfg        = yaml.safe_load(open(project_root / "config.yaml", encoding="utf-8"))
    output_dir = project_root / cfg["paths"]["output_dir"] / "exp1_data_scaling"
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = get_logger(output_dir)
    logger.info("EXPERIMENT 1: DATA SCALING")

    # 2. Load datasets
    train_ds, val_ds, test_ds, evaluator = load_data_and_evaluator(cfg)
    logger.info(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

    # 3. Define dataset sizes
    data_sizes = build_data_sizes(cfg, train_ds)
    seeds      = cfg.get("experiment", {}).get("seeds", [42, 123, 456])  # Multiple seeds for stability
    logger.info(f"Sizes: {data_sizes} | Seeds: {seeds} | Total: {len(data_sizes) * len(seeds)}")

    # 4. Run experiments
    results, best_info = run_all(
        data_sizes, seeds, cfg,
        train_ds, val_ds, test_ds,
        evaluator, output_dir, logger
    )

    # 5. Save results
    save_results(results, best_info, output_dir, logger)
if __name__ == "__main__":
    main()