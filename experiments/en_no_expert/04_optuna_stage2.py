import os
import sys
import json
import logging
import traceback
import gc
import time
from pathlib import Path

import pandas as pd
import torch
import yaml


script_dir   = Path(__file__).resolve().parent
project_root = script_dir.parent.parent
sys.path.insert(0, str(project_root))

HF_CACHE_DIR = "/proj/uppmax2026-1-123/private/yaxj1/hf_cache"
os.environ.update({
    "HF_HOME":  HF_CACHE_DIR,
    "TRANSFORMERS_CACHE": HF_CACHE_DIR,
    "HF_DATASETS_CACHE":HF_CACHE_DIR,
    "TORCH_HOME": HF_CACHE_DIR,
})

from scripts.model.lora_trainer import LoRATrainer
from scripts.data.data_loader import DataManager
from scripts.evaluation.base_evaluator import BaseEvaluator


def get_logger(output_dir: Path) -> logging.Logger:
    logger = logging.getLogger("exp4_stage2")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s  %(message)s", datefmt="%H:%M:%S")
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    fh = logging.FileHandler(output_dir / "experiment.log", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger


def load_top_configs(cfg, logger) -> list:
    """
    Load top hyperparameter configs selected from Optuna Stage 1
    """
    stage1_dir       = project_root / cfg["paths"]["output_dir"] / "exp3_optuna_stage1"
    top_configs_file = stage1_dir / "top_configs.json"

    if not top_configs_file.exists():
        logger.error(f"top_configs.json not found at {top_configs_file}")
        return []

    with open(top_configs_file) as f:
        top_configs = json.load(f)

    logger.info(f"Loaded {len(top_configs)} configurations from Stage 1")
    return top_configs


def load_data(cfg, logger):
    """
    Load dataset and fix training subset size (8k — elbow point from exp1)
    """
    data_manager = DataManager(cfg)
    train_ds, val_ds, test_ds = data_manager.load_splits()
    train_ds = train_ds.subset(8000, seed=cfg["project"]["seed"])
    logger.info(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")
    return train_ds, val_ds, test_ds


def build_train_config(cfg, config, seed, run_dir) -> dict:
    """
    Convert hyperparameter config → trainer config dict
    """
    return {
        "output_dir": str(run_dir / "training"),

        # reproducibility
        "seed": seed,

        # LoRA hyperparameters
        "r": config["r"],
        "alpha": config["alpha"],
        "dropout": config["dropout"],

        # model config
        "target_modules": cfg["lora"]["target_modules"],

        # training config
        "epochs": cfg["training"]["epochs"],
        "batch_size": cfg["training"]["batch_size"],
        "gradient_accumulation_steps": cfg["training"].get("grad_accumulation", 4),
        "learning_rate":cfg["training"]["lr"],
        "warmup_steps": cfg["training"]["warmup_steps"],
        "eval_steps": cfg["training"]["eval_steps"],
        "early_stopping_patience": cfg["training"]["early_stopping_patience"],
        "fp16": cfg["training"]["fp16"],

        # saving policy
        "save_total_limit": 1,
        "save_final_model": True,
    }


def run_one(config, config_idx, run, cfg, train_data, val_data, test_ds, evaluator, output_dir, logger):
    """
    Train + evaluate one config with one random seed
    """
    # different seed per run 
    seed    = cfg["project"]["seed"] + run
    run_dir = output_dir / f"config_{config_idx}_run_{run}"
    run_dir.mkdir(parents=True, exist_ok=True)

    trainer      = LoRATrainer(
        model_name=cfg["model"]["pretrained"],
        src_lang=cfg["model"]["src_lang"],
        tgt_lang=cfg["model"]["tgt_lang"],
    )
    train_config = build_train_config(cfg, config, seed, run_dir)

    try:
        config_start = time.time()

        # TRAINING PHASE
        train_result = trainer.train(train_data, val_data, train_config)

        # TEST GENERATION
        test_preds = trainer.generate_predictions(
            train_result["model"], test_ds,
            batch_size=8,
            num_beams=cfg["generation"]["num_beams"],
        )

        # EVALUATION
        test_metrics = evaluator.evaluate_all(
            [s.source for s in test_ds.samples],
            test_preds,
            [s.target for s in test_ds.samples],
        )

        config_time = time.time() - config_start

        entry = {
            "config_id": config_idx,
            "run": run,
            "seed": seed,
            "r": config["r"],
            "alpha": config["alpha"],
            "dropout": config["dropout"],
            "stage1_bleu": config["bleu"],
            "val_bleu": train_result["bleu"],
            "val_chrf": train_result["chrf"],
            "val_loss": train_result["loss"],
            "test_bleu": test_metrics["bleu"],
            "test_chrf": test_metrics["chrf"],
            "model_path": train_result.get("final_model_path", ""),
            "training_time": config_time,
        }

        with open(run_dir / "metrics.json", "w") as f:
            json.dump({
                "config": config,
                "run": run,
                "seed": seed,
                "val_metrics": {"bleu": train_result["bleu"], "chrf": train_result["chrf"], "loss": train_result["loss"]},
                "test_metrics": test_metrics,
                "training_time": config_time,
            }, f, indent=2)

        logger.info(f" val BLEU={train_result['bleu']:.4f}  chrF={train_result['chrf']:.2f}")
        logger.info(f" test BLEU={test_metrics['bleu']:.4f}  chrF={test_metrics['chrf']:.2f}")
        logger.info(f" time {config_time/60:.1f}m")

        return entry

    except Exception as e:
        logger.error(f"Run failed: {e}\n{traceback.format_exc()}")
        return {
            "config_id": config_idx,
            "run": run,
            "seed": seed,
            "failed": True,
            "test_bleu": 0.0,
            "training_time_seconds": 0,
        }

    finally:
        del trainer
        torch.cuda.empty_cache()
        gc.collect()


def run_all(top_configs, cfg, train_data, val_data, test_ds, evaluator, output_dir, logger):
    """
    Full Stage 2 evaluation: each config is trained multiple times with different seeds
    """
    num_runs  = cfg.get("experiment", {}).get("stage2_runs", 3)
    results  = []
    best_bleu  = 0.0
    best_config = None
    best_model_path = None

    logger.info(f"Configs={len(top_configs)} | Runs per config={num_runs}")

    for config_idx, config in enumerate(top_configs):
        logger.info(f"\n[Config {config_idx+1}] r={config['r']} alpha={config['alpha']} dropout={config['dropout']}")

        for run in range(num_runs):
            logger.info(f"  Run {run+1}/{num_runs}")
            entry = run_one(
                config, config_idx, run,
                cfg, train_data, val_data, test_ds,
                evaluator, output_dir, logger,
            )
            results.append(entry)

            if entry.get("test_bleu", 0) > best_bleu:
                best_bleu = entry["test_bleu"]
                best_config = config
                best_model_path = entry.get("model_path", "")

    return results, best_config, best_bleu, best_model_path


def save_report(results, best_config, best_bleu, best_model_path, output_dir, logger):
    """
    Save full experiment results + summary statistics
    """
    df = pd.DataFrame(results)
    df.to_csv(output_dir / "results.csv", index=False)
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    valid_df = df[df["failed"] != True] if "failed" in df.columns else df

    if not valid_df.empty:
        summary = valid_df.groupby("config_id").agg(
            test_bleu_mean=("test_bleu", "mean"),
            test_bleu_std=("test_bleu", "std"),
            test_chrf_mean=("test_chrf", "mean"),
        ).round(4)
        logger.info(f"\n{summary}")
        summary.to_csv(output_dir / "summary.csv")

        if "training_time" in valid_df.columns:
            total_time = valid_df["training_time"].sum()
            logger.info(f"  Total time: {total_time/3600:.2f}h")

    if best_config:
        logger.info(f"\nBEST CONFIG FOUND:")
        logger.info(f"r={best_config['r']} alpha={best_config['alpha']} dropout={best_config['dropout']}")
        logger.info(f"BLEU={best_bleu:.4f}")
        with open(output_dir / "best_config.json", "w") as f:
            json.dump({
                "r": best_config["r"],
                "alpha": best_config["alpha"],
                "dropout": best_config["dropout"],
                "best_bleu": best_bleu,
                "model_path": str(best_model_path),
            }, f, indent=2)


def main():
    # 1. load config
    with open(project_root / "config.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    output_dir = project_root / cfg["paths"]["output_dir"] / "exp4_optuna_stage2"
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = get_logger(output_dir)
    logger.info("EXPERIMENT 4: STAGE 2 (FULL EVALUATION)")
    # 2. load top configs from stage 1
    top_configs = load_top_configs(cfg, logger)
    if not top_configs:
        return
    # 3. load data
    train_ds, val_ds, test_ds = load_data(cfg, logger)
    evaluator  = BaseEvaluator(use_comet=False)
    train_data = [s.to_dict() for s in train_ds.samples]
    val_data   = [s.to_dict() for s in val_ds.samples]
    # 4. run full evaluation of top configs
    results, best_config, best_bleu, best_model_path = run_all(
        top_configs, cfg, train_data, val_data, test_ds, evaluator, output_dir, logger,
    )
    # 5. save results and summary
    save_report(results, best_config, best_bleu, best_model_path, output_dir, logger)


if __name__ == "__main__":
    main()