import sys
import yaml
sys.path.append(r"D:\J\Desktop\language_technology\course\projects_AI\mt_oil_no\scripts\data")

from dataset import TranslationDataset
from pathlib import Path

with open(r"D:\J\Desktop\language_technology\course\projects_AI\mt_oil_no\config.yaml", encoding="utf-8") as f:
    config = yaml.safe_load(f)

data_dir = Path(r"D:\J\Desktop\language_technology\course\projects_AI\mt_oil_no\data\final_splits_npd")
src_lang = config["model"]["src_lang"]
tgt_lang = config["model"]["tgt_lang"]

train_ds = TranslationDataset.from_json(data_dir / "train.json", src_lang=src_lang, tgt_lang=tgt_lang)
val_ds   = TranslationDataset.from_json(data_dir / "val.json",   src_lang=src_lang, tgt_lang=tgt_lang)
test_ds  = TranslationDataset.from_json(data_dir / "test.json",  src_lang=src_lang, tgt_lang=tgt_lang)

print("train length:", len(train_ds))
print("val length:", len(val_ds))
print("test length:", len(test_ds))
print("train first sample:", train_ds[0])