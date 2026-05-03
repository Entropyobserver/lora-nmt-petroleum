import sys
sys.path.append(r"D:\J\Desktop\language_technology\course\projects_AI\mt_oil_no\scripts\data")
from dataset import TranslationDataset
from pathlib import Path

path = Path(r"D:\J\Desktop\language_technology\course\projects_AI\mt_oil_no\data\02_final_splits_npd\test.json")

ds = TranslationDataset.from_json(path)

print("length of dataset:", len(ds))
print("first sample:", ds[0])
print("statistics:", ds.get_statistics())