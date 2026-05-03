from typing import Dict, List
import numpy as np
import evaluate


class BaseEvaluator:

    def __init__(self, use_comet: bool = False, comet_model: str = "Unbabel/wmt22-comet-da"):
        self.bleu = evaluate.load("bleu")
        self.chrf = evaluate.load("chrf")
        self.use_comet = use_comet
        self.comet_model = None

        if self.use_comet:
            try:
                from comet import download_model, load_from_checkpoint
                comet_model_path = download_model(comet_model)
                self.comet_model = load_from_checkpoint(comet_model_path)
            except Exception as e:
                print(f"COMET loading failed: {e}")
                self.use_comet = False

    def compute_bleu(self, predictions: List[str], references: List[str]) -> Dict:
        result = self.bleu.compute(
            predictions=predictions,
            references=[[ref] for ref in references],
        )
        p = result["precisions"]
        return {
            "bleu": result["bleu"],
            "bleu_1": p[0] if len(p) > 0 else 0,
            "bleu_2": p[1] if len(p) > 1 else 0,
            "bleu_3": p[2] if len(p) > 2 else 0,
            "bleu_4": p[3] if len(p) > 3 else 0,
        }

    def compute_chrf(self, predictions: List[str], references: List[str]) -> Dict:
        result = self.chrf.compute(predictions=predictions, references=references)
        return {"chrf": result["score"]}

    def compute_comet(self, sources: List[str], predictions: List[str], references: List[str]) -> Dict:
        if not self.use_comet or self.comet_model is None:
            return {"comet": 0.0, "comet_std": 0.0}

        try:
            comet_data = [
                {"src": str(src), "mt": str(pred), "ref": str(ref)}
                for src, pred, ref in zip(sources, predictions, references)
            ]
            # batch_size=4 to avoid OOM — COMET (XLM-R large) needs ~4GB VRAM
            result = self.comet_model.predict(comet_data, batch_size=4, gpus=1)
            return {
                "comet":     result["system_score"],
                "comet_std": float(np.std(result["scores"])),
            }
        except Exception as e:
            print(f"COMET error: {e}")
            return {"comet": 0.0, "comet_std": 0.0}

    def evaluate_all(self, sources: List[str], predictions: List[str], references: List[str]) -> Dict:
        predictions = [p.strip() for p in predictions]
        references  = [r.strip() for r in references]

        metrics = {}
        metrics.update(self.compute_bleu(predictions, references))
        metrics.update(self.compute_chrf(predictions, references))

        if self.use_comet and sources:
            metrics.update(self.compute_comet(sources, predictions, references))

        return metrics