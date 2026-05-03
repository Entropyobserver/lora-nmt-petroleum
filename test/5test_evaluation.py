import sys
sys.path.append(r"D:\J\Desktop\language_technology\course\projects_AI\mt_oil_no\scripts\model")
sys.path.append(r"D:\J\Desktop\language_technology\course\projects_AI\mt_oil_no\scripts\evaluation")
from base_evaluator import BaseEvaluator


# simple test data
sources = ["oljeprisen er høy", "brønnen ble boret vellykket"]

perfect_preds = ["the oil price is high", "the well was drilled successfully"]
perfect_refs  = ["the oil price is high", "the well was drilled successfully"]

bad_preds = ["cat sat on mat", "hello world"]
bad_refs  = perfect_refs

partial_preds = ["the oil price is very high", "the well is drilled"]
partial_refs  = perfect_refs


evaluator = BaseEvaluator(use_comet=False)

# ---- perfect case ----
r = evaluator.evaluate_all(sources, perfect_preds, perfect_refs)

assert r["bleu"] > 0.9, "BLEU too low for perfect case"
assert r["chrf"] > 90, "CHRF too low for perfect case"


# ---- bad case ----
r = evaluator.evaluate_all(sources, bad_preds, bad_refs)

assert r["bleu"] < 0.3, "BLEU too high for bad case"
assert r["chrf"] < 50, "CHRF too high for bad case"


# ---- partial case ----
r = evaluator.evaluate_all(sources, partial_preds, partial_refs)

assert 0.3 < r["bleu"] < 0.9, "BLEU not in expected range"
assert 50 < r["chrf"] < 95, "CHRF not in expected range"


print("All tests passed.")
