import os
import sys
import json
import time
import gc
import logging
import traceback
from pathlib import Path
from itertools import product

import pandas as pd
import torch
import yaml

script_dir   = Path(__file__).resolve().parent
project_root = script_dir.parent.parent
sys.path.insert(0, str(project_root))

HF_CACHE_DIR = "/proj/uppmax2026-1-123/private/yaxj1/hf_cache"
os.environ.update({
    "HF_HOME": HF_CACHE_DIR,
    "TRANSFORMERS_CACHE": HF_CACHE_DIR,
    "HF_DATASETS_CACHE": HF_CACHE_DIR,
    "TORCH_HOME": HF_CACHE_DIR,
})

from scripts.model.lora_trainer import LoRATrainer
from scripts.data.data_loader import DataManager
from scripts.evaluation.base_evaluator import BaseEvaluator

def get_logger(output_dir: Path) -> logging.Logger:
    """
    Create a logger that writes to both:
    - console (StreamHandler)
    - file (FileHandler)
    """
    logger = logging.getLogger("exp2")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s  %(message)s", datefmt="%H:%M:%S")
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    fh = logging.FileHandler(output_dir / "experiment.log", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger

def get_recommended_size(cfg, logger) -> int:
    # Instead of searching dataset size,we reuse result from Experiment 1 (data scaling).
    logger.info("Using exp1-determined optimal size: 8000 (turning point)")
    return 8000

# Hyperparameter grid definition
def build_param_grid(cfg) -> list:
    """
    Build grid search space for LoRA hyperparameters.
    We explore combinations of:
    - r (rank)
    - alpha (scaling)
    - dropout (regularization)
    """
    gs = cfg.get("experiment", {}).get("gridsearch", {})

    r_values       = gs.get("r", [8, 16, 32])
    alpha_values   = gs.get("alpha", [16, 32, 64])
    dropout_values = gs.get("dropout", [0.0, 0.1, 0.2])

    # all combinations（3x3x3=27 configs）
    # product([8,16], [16,32]) -> [(8,16), (8,32), (16,16), (16,32)]
    return list(product(r_values, alpha_values, dropout_values))

# Training configuration builder
def build_train_config(cfg, r, alpha, dropout, config_output_dir) -> dict:
    #Create a training config dictionary for LoRA training.
    return {
        "output_dir": str(config_output_dir / "training"),
        "seed": cfg["project"]["seed"],

        # LoRA hyperparameters
        "r": r,
        "alpha": alpha,
        "dropout": dropout,
        "target_modules": cfg["lora"]["target_modules"],

        # Training hyperparameters
        "epochs": cfg["training"]["epochs"],
        "batch_size": cfg["training"]["batch_size"],
        "gradient_accumulation_steps": cfg["training"].get("grad_accumulation", 4),
        "learning_rate": cfg["training"]["lr"],
        "warmup_steps": cfg["training"]["warmup_steps"],
        "eval_steps": cfg["training"]["eval_steps"],
        "early_stopping_patience": cfg["training"]["early_stopping_patience"],
        "fp16": cfg["training"]["fp16"],

        # Checkpoint control
        "save_total_limit": 1,
        "save_final_model": True,
    }

# Run one configuration (core unit of experiment)
def run_one_config(r, alpha, dropout, cfg, train_data, val_data, test_ds, evaluator, output_dir, logger):
    """
    Train + evaluate a single hyperparameter configuration.
    Steps:
    1. Create experiment folder
    2. Train LoRA model
    3. Evaluate on validation + test set
    4. Save results
    """
    config_name = f"r{r}_a{alpha}_d{dropout}"
    config_output_dir = output_dir / config_name
    config_output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize LoRA trainer
    trainer = LoRATrainer(
        model_name=cfg["model"]["pretrained"],
        src_lang=cfg["model"]["src_lang"],
        tgt_lang=cfg["model"]["tgt_lang"],
    )

    train_config = build_train_config(cfg, r, alpha, dropout, config_output_dir)

    try:
        config_start = time.time()

        # TRAINING PHASE
        train_result = trainer.train(train_data, val_data, train_config)

        # TEST INFERENCE PHASE
        test_preds = trainer.generate_predictions(
            train_result["model"], test_ds,
            batch_size=8,
            num_beams=cfg["generation"]["num_beams"],
        )

        # EVALUATION PHASE
        test_metrics = evaluator.evaluate_all(
            [s.source for s in test_ds.samples],
            test_preds,
            [s.target  for s in test_ds.samples],
        )

        config_time = time.time() - config_start
        # Collect metrics
        entry = {
            "r": r,
            "alpha": alpha,
            "dropout": dropout,
            "val_bleu": train_result["bleu"],
            "val_chrf": train_result["chrf"],
            "val_loss": train_result["loss"],
            "test_bleu": test_metrics["bleu"],
            "test_chrf": test_metrics["chrf"],
            "model_path": train_result.get("final_model_path", ""),
            "training_time_seconds": config_time,
        }

        # Save per-config results
        with open(config_output_dir / "metrics.json", "w") as f:
            json.dump({
                "config": {"r": r, "alpha": alpha, "dropout": dropout},
                "val_metrics":   {
                    "bleu": train_result["bleu"],
                    "chrf": train_result["chrf"],
                    "loss": train_result["loss"]
                },
                "test_metrics": test_metrics,
                "training_time": config_time,
            }, f, indent=2)

        logger.info(f" val BLEU={train_result['bleu']:.4f}  chrF={train_result['chrf']:.2f}")
        logger.info(f" test BLEU={test_metrics['bleu']:.4f}  chrF={test_metrics['chrf']:.2f}")
        logger.info(f" time {config_time/60:.1f}m")

        return entry

    except Exception as e:
        # Handle crash for one configuration
        logger.error(f"  FAILED: {e}\n{traceback.format_exc()}")

        with open(config_output_dir / "error.log", "w") as f:
            f.write(f"{e}\n{traceback.format_exc()}")

        return {
            "r": r, "alpha": alpha, "dropout": dropout,
            "val_bleu": 0.0, "val_chrf": 0.0, "val_loss": 999.0,
            "test_bleu": 0.0, "test_chrf": 0.0,
            "model_path": "", "training_time_seconds": 0, "failed": True,
        }

    finally:
        # Clean GPU memory after each run
        del trainer
        torch.cuda.empty_cache()
        gc.collect()

# GRID SEARCH EXECUTION (main experiment loop)
def run_grid(all_configs, cfg, train_data, val_data, test_ds, evaluator, output_dir, logger):
    """
    Run full hyperparameter grid search.
    This function:
    - iterates over all configurations
    - runs training for each config
    - tracks best-performing model
    """
    results = []
    best_bleu = 0.0
    best_config = None
    best_model_path = None

    total = len(all_configs)

    for idx, (r, alpha, dropout) in enumerate(all_configs, 1):
        logger.info(f"\n[{idx}/{total}] r={r}, alpha={alpha}, dropout={dropout}")

        entry = run_one_config(
            r, alpha, dropout,
            cfg, train_data, val_data, test_ds,
            evaluator, output_dir, logger
        )

        results.append(entry)

        # Track best model
        if entry["test_bleu"] > best_bleu:
            best_bleu = entry["test_bleu"]
            best_config = {"r": r, "alpha": alpha, "dropout": dropout}
            best_model_path = entry["model_path"]

    return results, best_config, best_bleu, best_model_path


def generate_final_report(output_dir, logger, results, best_config=None, best_bleu=None, best_model_path=None):
    """
    Summarize grid search results:
    - save CSV/JSON
    - show top models
    - store best config
    """
    df = pd.DataFrame(results)

    df.to_csv(output_dir / "results.csv", index=False)

    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    valid_df = df[df["failed"] != True] if "failed" in df.columns else df

    # Top-5 models
    if not valid_df.empty:
        top5 = valid_df.nlargest(5, "test_bleu")

        logger.info("\nTop 5 configurations:")
        for i, (_, row) in enumerate(top5.iterrows(), 1):
            logger.info(
                f"  {i}. r={row['r']}, alpha={row['alpha']}, dropout={row['dropout']}  BLEU={row['test_bleu']:.4f}"
            )

        # fallback best selection
        if not best_config:
            best_row        = valid_df.loc[valid_df["test_bleu"].idxmax()]
            best_config     = {"r": int(best_row["r"]), "alpha": int(best_row["alpha"]), "dropout": float(best_row["dropout"])}
            best_bleu       = float(best_row["test_bleu"])
            best_model_path = best_row["model_path"]

    # Save best model info
    if best_config:
        logger.info(f"\nBest: r={best_config['r']}, alpha={best_config['alpha']}, dropout={best_config['dropout']}, BLEU={best_bleu:.4f}")

        with open(output_dir / "best_config.json", "w") as f:
            json.dump({**best_config, "test_bleu": best_bleu, "model_path": str(best_model_path)}, f, indent=2)

    # total training time
    total_time = sum(r.get("training_time_seconds", 0) for r in results)
    logger.info(f"Total time: {total_time/3600:.2f}h")
    logger.info(f"Results saved to {output_dir}")


def main():
    """
    Experiment pipeline:
    1. Load config
    2. Load dataset
    3. Build hyperparameter grid
    4. Run grid search
    5. Save report
    """
    # 1. Load config
    with open(project_root / "config.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    output_dir = project_root / cfg["paths"]["output_dir"] / "exp2_gridsearch"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = get_logger(output_dir)
    logger.info("EXPERIMENT 2: PARAMETER SENSITIVITY ANALYSIS")

    # 2. Load dataset + evaluator
    data_manager = DataManager(cfg)
    train_ds, val_ds, test_ds = data_manager.load_splits()
    evaluator = BaseEvaluator(use_comet=False)

    # 3. Use fixed optimal dataset size (from exp1)
    size     = get_recommended_size(cfg, logger)
    train_ds = train_ds.subset(size, seed=cfg["project"]["seed"])

    logger.info(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

    # 4. Build grid search space
    all_configs = build_param_grid(cfg)
    logger.info(f"Total configs: {len(all_configs)}")

    train_data = [s.to_dict() for s in train_ds.samples]
    val_data   = [s.to_dict() for s in val_ds.samples]

    # 5. Run grid search
    results, best_config, best_bleu, best_model_path = run_grid(
        all_configs, cfg, train_data, val_data,
        test_ds, evaluator, output_dir, logger,
    )

    # 6. Generate final report
    generate_final_report(output_dir, logger, results, best_config, best_bleu, best_model_path)

if __name__ == "__main__":
    main()