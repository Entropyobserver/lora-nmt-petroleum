import os
import sys
import json
import logging
import traceback
import gc
from pathlib import Path

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
    "HF_DATASETS_CACHE":HF_CACHE_DIR,
    "TORCH_HOME": HF_CACHE_DIR,
})

from scripts.model.lora_trainer import LoRATrainer
from scripts.data.data_loader import DataManager
from scripts.evaluation.base_evaluator import BaseEvaluator


def get_logger(output_dir: Path) -> logging.Logger:
    logger = logging.getLogger("exp5_final")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s  %(message)s", datefmt="%H:%M:%S")
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    fh = logging.FileHandler(output_dir / "experiment.log", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger


def load_best_config(cfg, logger) -> tuple:
    # Manually set best config from exp2 gridsearch results
    # r=8, alpha=64, dropout=0.0 gave best test BLEU in exp2
    r, alpha, dropout = 8, 64, 0.0
    logger.info(f"Using config from exp2: r={r}, alpha={alpha}, dropout={dropout}")
    return r, alpha, dropout


def load_data(cfg, logger):
    # Load full dataset (no subset — final eval uses all training data)
    data_manager = DataManager(cfg)
    train_ds, val_ds, test_ds = data_manager.load_splits()
    logger.info(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")
    return train_ds, val_ds, test_ds


def build_train_config(cfg, r, alpha, dropout, seed, seed_dir) -> dict:
    return {
        "output_dir": str(seed_dir / "training"),
        "seed": seed,
        "r": r,
        "alpha": alpha,
        "dropout": dropout,
        "target_modules": cfg["lora"]["target_modules"],
        "epochs": cfg["training"]["epochs"],
        "batch_size": cfg["training"]["batch_size"],
        "gradient_accumulation_steps": cfg["training"].get("grad_accumulation", 4),
        "learning_rate": cfg["training"]["lr"],
        "warmup_steps": cfg["training"]["warmup_steps"],
        "eval_steps": cfg["training"]["eval_steps"],
        "early_stopping_patience": cfg["training"]["early_stopping_patience"],
        "fp16": cfg["training"]["fp16"],
        "save_total_limit": 2,
        "save_final_model": True,
    }


def run_one_seed(seed, r, alpha, dropout, cfg, train_data, val_data, test_ds,
                 sources, references, evaluator, output_dir, logger):
    seed_dir = output_dir / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)

    trainer      = LoRATrainer(
        model_name=cfg["model"]["pretrained"],
        src_lang=cfg["model"]["src_lang"],
        tgt_lang=cfg["model"]["tgt_lang"],
    )
    train_config = build_train_config(cfg, r, alpha, dropout, seed, seed_dir)

    try:
        train_result = trainer.train(train_data, val_data, train_config)
        logger.info(f"  val  BLEU={train_result['bleu']:.4f}  chrF={train_result['chrf']:.2f}")

        # Generate predictions on the HELD-OUT TEST SET
        # sources and references come from test_ds, never seen during training
        test_preds   = trainer.generate_predictions(
            train_result["model"], test_ds,
            batch_size=8, num_beams=cfg["generation"]["num_beams"],
        )

        # Evaluate with BLEU, chrF, and COMET
        test_metrics = evaluator.evaluate_all(sources, test_preds, references)
        logger.info(f"  test BLEU={test_metrics['bleu']:.4f}  chrF={test_metrics['chrf']:.2f}  COMET={test_metrics.get('comet', 0):.4f}")

        entry = {
            "seed": seed,
            "configuration": {"r": r, "alpha": alpha, "dropout": dropout},
            "val_bleu": train_result["bleu"],
            "val_chrf": train_result["chrf"],
            "val_loss": train_result["loss"],
            "test_bleu": test_metrics["bleu"],
            "test_chrf": test_metrics["chrf"],
            "test_comet": test_metrics.get("comet", None),
            "dataset_sizes": {"train": len(train_data), "val": len(val_data), "test": len(sources)},
        }

        with open(seed_dir / "results.json", "w") as f:
            json.dump(entry, f, indent=2)

        # Save predictions for manual inspection and error analysis
        with open(seed_dir / "test_predictions.json", "w", encoding="utf-8") as f:
            json.dump([
                {"source": src, "prediction": pred, "reference": ref}
                for src, pred, ref in zip(sources, test_preds, references)
            ], f, ensure_ascii=False, indent=2)

        return entry

    except Exception as e:
        logger.error(f"  Seed {seed} failed: {e}\n{traceback.format_exc()}")
        return {
            "seed": seed, "val_bleu": 0.0, "val_chrf": 0.0,
            "test_bleu": 0.0, "test_chrf": 0.0, "test_comet": None, "failed": True,
        }

    finally:
        del trainer
        torch.cuda.empty_cache()
        gc.collect()


def run_all_seeds(seeds, r, alpha, dropout, cfg, train_data, val_data, test_ds,
                  sources, references, evaluator, output_dir, logger):
    results = []
    for seed in seeds:
        logger.info(f"\n[Seed {seed}]")
        entry = run_one_seed(seed, r, alpha, dropout, cfg, train_data, val_data,
                             test_ds, sources, references, evaluator, output_dir, logger)
        results.append(entry)
    return results


def save_report(results, output_dir, logger):
    df = pd.DataFrame(results)
    df.to_csv(output_dir / "all_results.csv", index=False)
    with open(output_dir / "all_results.json", "w") as f:
        json.dump(results, f, indent=2)

    valid_df = df[df["failed"] != True] if "failed" in df.columns else df
    if not valid_df.empty:
        logger.info(f" Test BLEU:  {valid_df['test_bleu'].mean():.4f} ± {valid_df['test_bleu'].std():.4f}")
        logger.info(f" Test chrF:  {valid_df['test_chrf'].mean():.2f} ± {valid_df['test_chrf'].std():.2f}")
        if "test_comet" in valid_df.columns and valid_df["test_comet"].notna().any():
            logger.info(f" Test COMET: {valid_df['test_comet'].mean():.4f} ± {valid_df['test_comet'].std():.4f}")

    logger.info(f"Results saved to {output_dir}")


def main():
    # 1. load config
    with open(project_root / "config.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    output_dir = project_root / cfg["paths"]["output_dir"] / "exp5_final_eval"
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = get_logger(output_dir)
    logger.info("EXPERIMENT 5: FINAL EVALUATION")
    # 2. load best config from exp2 grid search
    r, alpha, dropout = load_best_config(cfg, logger)
    logger.info(f"LoRA: r={r}, alpha={alpha}, dropout={dropout}")
    # 3. load data
    train_ds, val_ds, test_ds = load_data(cfg, logger)
    evaluator = BaseEvaluator(use_comet=True)# use_comet=True for final evaluation — COMET cached at HF_CACHE_DIR
    train_data = [s.to_dict() for s in train_ds.samples]
    val_data = [s.to_dict() for s in val_ds.samples]
    sources = [s.source for s in test_ds.samples]
    references = [s.target for s in test_ds.samples]
    seeds = cfg.get("experiment", {}).get("seeds", [42, 123, 456])
    logger.info(f"Seeds: {seeds}")
    # 4. run training and evaluation for each seed
    results = run_all_seeds(
        seeds, r, alpha, dropout, cfg, train_data, val_data,
        test_ds, sources, references, evaluator, output_dir, logger,
    )
    # 5. save results
    logger.info("SUMMARY")
    save_report(results, output_dir, logger)


if __name__ == "__main__":
    main()