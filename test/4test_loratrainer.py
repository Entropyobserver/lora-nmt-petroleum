import sys
import yaml
sys.path.append(r"D:\J\Desktop\language_technology\course\projects_AI\mt_oil_no\scripts\model")
sys.path.append(r"D:\J\Desktop\language_technology\course\projects_AI\mt_oil_no\scripts\data")

from pathlib import Path
from lora_trainer import LoRATrainer
from dataset import TranslationDataset

with open(r"D:\J\Desktop\language_technology\course\projects_AI\mt_oil_no\config.yaml", encoding="utf-8") as f:
    config = yaml.safe_load(f)

# init
trainer = LoRATrainer(
    model_name=config["model"]["pretrained"],
    src_lang=config["model"]["src_lang"],
    tgt_lang=config["model"]["tgt_lang"],
)
print("trainer init OK")

# load data
data_dir = Path(r"D:\J\Desktop\language_technology\course\projects_AI\mt_oil_no\data\final_splits_npd")
train_ds = TranslationDataset.from_json(data_dir / "train.json", src_lang=config["model"]["src_lang"], tgt_lang=config["model"]["tgt_lang"])
val_ds   = TranslationDataset.from_json(data_dir / "val.json",   src_lang=config["model"]["src_lang"], tgt_lang=config["model"]["tgt_lang"])

train_data = [{"source": s.source, "target": s.target} for s in train_ds.samples]
val_data   = [{"source": s.source, "target": s.target} for s in val_ds.samples]
print(f"data loaded — train: {len(train_data)}, val: {len(val_data)}")

# setup model with lora
model = trainer.setup_model(
    r=config["lora"]["r"],
    alpha=config["lora"]["alpha"],
    dropout=config["lora"]["dropout"],
    target_modules=config["lora"]["target_modules"],
)
print("setup_model OK")

# training args
args_config = {
    "output_dir":                 str(Path(r"D:\J\Desktop\language_technology\course\projects_AI\mt_oil_no\outputs\test_lora")),
    "seed":                       config["project"]["seed"],
    "epochs":                     1,
    "batch_size":                 config["training"]["batch_size"],
    "eval_steps":                 config["training"]["eval_steps"],
    "warmup_steps":               config["training"]["warmup_steps"],
    "fp16":                       config["training"]["fp16"],
    "early_stopping_patience":    config["training"]["early_stopping_patience"],
    "save_final_model":           True,
}
args = trainer.training_args(args_config)
print("training_args OK — lr:", args.learning_rate)