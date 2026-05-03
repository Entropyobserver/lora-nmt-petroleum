import os  
import sys  
import json  
import logging  
import traceback  
from pathlib import Path  

import optuna  
from optuna.pruners import SuccessiveHalvingPruner  
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

def get_logger(output_dir: Path) -> logging.Logger:
    logger = logging.getLogger("exp3_stage1")  
    logger.setLevel(logging.INFO)              
    fmt = logging.Formatter("%(asctime)s  %(message)s", datefmt="%H:%M:%S")
    ch = logging.StreamHandler()               
    ch.setFormatter(fmt)
    fh = logging.FileHandler(output_dir / "experiment.log", encoding="utf-8")  
    fh.setFormatter(fmt)
    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger

def load_data(cfg, logger):
    data_manager = DataManager(cfg)  
    train_ds, val_ds, _ = data_manager.load_splits() 
    # IMPORTANT: use subset for fast hyperparameter search
    train_ds = train_ds.subset(2000, seed=cfg["project"]["seed"])
    logger.info(f"Train: {len(train_ds)} | Val: {len(val_ds)}")
    return train_ds, val_ds

def build_train_config(cfg, r, alpha, dropout, trial_dir) -> dict:
    # this dict is passed into LoRATrainer.train()
    return {
        "output_dir": str(trial_dir / "training"),  

        "seed": cfg["project"]["seed"],
        # LoRA hyperparameters (OPTUNA search space)
        "r": r,
        "alpha": alpha,
        "dropout": dropout,
        # model structure
        "target_modules": cfg["lora"]["target_modules"],
        # training settings
        "epochs": cfg["training"]["epochs"],
        "batch_size": cfg["training"]["batch_size"],
        "gradient_accumulation_steps": cfg["training"].get("grad_accumulation", 4),
        "learning_rate": cfg["training"]["lr"],
        "warmup_steps": cfg["training"]["warmup_steps"],
        "eval_steps": cfg["training"]["eval_steps"],
        # early stopping (stop bad runs early)
        "early_stopping_patience": 2,
        # fp16 speedup on GPU
        "fp16": cfg["training"]["fp16"],
        # save policy
        "save_total_limit": 1,
        "save_final_model": False,  # do not save full model (save compute + disk)
    }


# objective function (OPTUNA)
def make_objective(cfg, train_data, val_data, output_dir, logger):
    def objective(trial):
        # sample hyperparameters
        r = trial.suggest_categorical("r", [8, 16, 32])
        alpha = trial.suggest_categorical("alpha", [16, 32, 64])
        dropout = trial.suggest_float("dropout", 0.0, 0.2, step=0.05)
        logger.info(f"Trial {trial.number}: r={r}, alpha={alpha}, dropout={dropout}")

        # create folder for each trial
        trial_dir = output_dir / f"trial_{trial.number}"
        trial_dir.mkdir(parents=True, exist_ok=True)

        # initialize model
        trainer = LoRATrainer(
            model_name=cfg["model"]["pretrained"],
            src_lang=cfg["model"]["src_lang"],
            tgt_lang=cfg["model"]["tgt_lang"],
        )

        # build config
        train_config = build_train_config(cfg, r, alpha, dropout, trial_dir)

        try:
            # training phase
            result = trainer.train(train_data, val_data, train_config)
            logger.info(f"BLEU={result['bleu']:.4f}  chrF={result['chrf']:.2f}")

            # save trial result
            with open(trial_dir / "result.json", "w") as f:
                json.dump({
                    "trial": trial.number,
                    "params": {"r": r, "alpha": alpha, "dropout": dropout},
                    "bleu": result["bleu"],
                    "chrf": result["chrf"],
                    "loss": result["loss"],
                }, f, indent=2)

            # OPTUNA objective (maximize BLEU)
            return result["bleu"]

        except Exception as e:
            logger.error(f"Trial failed: {e}\n{traceback.format_exc()}")
            return 0.0  # failed trial = worst score

    return objective

# run optuna study
def run_study(cfg, train_data, val_data, output_dir, logger):

    num_trials = cfg.get("experiment", {}).get("num_trials", 50)

    logger.info(f"Trials={num_trials}, Pruner=SuccessiveHalving")

    study = optuna.create_study(
        direction="maximize",  # maximize BLEU
        pruner=SuccessiveHalvingPruner(),  # early stop bad configs
        study_name="nllb_lora_stage1",
    )

    study.optimize(
        make_objective(cfg, train_data, val_data, output_dir, logger),
        n_trials=num_trials,
        show_progress_bar=True,
    )

    return study

def save_report(study, output_dir, logger):
    # save all trial results to CSV
    study.trials_dataframe().to_csv(output_dir / "results.csv", index=False)
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    logger.info(f"Completed trials: {len(completed)}")

    # best trial
    best = study.best_trial
    logger.info(f"BEST: r={best.params['r']}, alpha={best.params['alpha']}, "
                f"dropout={best.params['dropout']}, BLEU={best.value:.4f}")

    # top 5 configs for stage2
    top5 = sorted(completed, key=lambda t: t.value, reverse=True)[:5]

    top_configs = []
    for t in top5:
        top_configs.append({
            "trial": t.number,
            "r": t.params["r"],
            "alpha": t.params["alpha"],
            "dropout": t.params["dropout"],
            "bleu": t.value,
        })

    with open(output_dir / "top_configs.json", "w") as f:
        json.dump(top_configs, f, indent=2)


def main():

    # 1. load config
    cfg = yaml.safe_load(open(project_root / "config.yaml", encoding="utf-8"))
    output_dir = project_root / cfg["paths"]["output_dir"] / "exp3_optuna_stage1"
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = get_logger(output_dir)
    logger.info("EXPERIMENT 3 - OPTUNA STAGE 1")

    # 2. load data
    train_ds, val_ds = load_data(cfg, logger)
    train_data = [s.to_dict() for s in train_ds.samples]
    val_data   = [s.to_dict() for s in val_ds.samples]

    # 3.run optimization
    study = run_study(cfg, train_data, val_data, output_dir, logger)

    # 4. save results
    save_report(study, output_dir, logger)

if __name__ == "__main__":
    main()