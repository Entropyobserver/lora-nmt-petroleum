import sys
import yaml
sys.path.append(r"D:\J\Desktop\language_technology\course\projects_AI\mt_oil_no\scripts\model")
sys.path.append(r"D:\J\Desktop\language_technology\course\projects_AI\mt_oil_no\scripts\data")

from pathlib import Path
from transformers import AutoModelForSeq2SeqLM
from base_trainer import BaseTrainer
from dataset import TranslationDataset


class SimpleTrainer(BaseTrainer):
    def setup_model(self, **kwargs):
        return AutoModelForSeq2SeqLM.from_pretrained(self.model_name)


with open(r"D:\J\Desktop\language_technology\course\projects_AI\mt_oil_no\config.yaml", encoding="utf-8") as f:
    config = yaml.safe_load(f)

trainer = SimpleTrainer(
    model_name=config["model"]["pretrained"],
    src_lang=config["model"]["src_lang"],
    tgt_lang=config["model"]["tgt_lang"],
)
print("trainer init OK")

data_dir = Path(r"D:\J\Desktop\language_technology\course\projects_AI\mt_oil_no\data\final_splits_npd")
train_ds = TranslationDataset.from_json(data_dir / "train.json", src_lang=config["model"]["src_lang"], tgt_lang=config["model"]["tgt_lang"])
val_ds   = TranslationDataset.from_json(data_dir / "val.json",   src_lang=config["model"]["src_lang"], tgt_lang=config["model"]["tgt_lang"])

train_data = [{"source": s.source, "target": s.target} for s in train_ds.samples]
val_data   = [{"source": s.source, "target": s.target} for s in val_ds.samples]
print(f"data loaded — train: {len(train_data)}, val: {len(val_data)}")

sample = {"source": [train_data[0]["source"]], "target": [train_data[0]["target"]]}
result = trainer.tokenize(sample)
print("tokenize OK — keys:", list(result.keys()))

train_hf, val_hf = trainer.prepare_datasets(train_data, val_data)
print(f"prepare_datasets OK — train: {len(train_hf)}, val: {len(val_hf)}")

args_config = {
    "output_dir":                 str(Path(r"D:\J\Desktop\language_technology\course\projects_AI\mt_oil_no\outputs\test_run")),
    "seed":                       config["project"]["seed"],
    "epochs":                     config["training"]["epochs"],
    "batch_size":                 config["training"]["batch_size"],
    "eval_steps":                 config["training"]["eval_steps"],
    "warmup_steps":               config["training"]["warmup_steps"],
    "fp16":                       config["training"]["fp16"],
    "early_stopping_patience":    config["training"]["early_stopping_patience"],
}
args = trainer.training_args(args_config)
print("training_args OK — lr:", args.learning_rate)