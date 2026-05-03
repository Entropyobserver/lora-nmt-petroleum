import shutil
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List
import numpy as np
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer, 
    DataCollatorForSeq2Seq, #Pads and batches data—e.g., ["Hi"] → [Hi, PAD]
    EarlyStoppingCallback,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
import evaluate

# Training Pipeline
# 1. Clean old checkpoints → remove previous experiment outputs
# 2. Create model → initialize seq2seq model (e.g., Transformer/NLLB)
# 3. Process data (tokenization) → convert raw text into model-ready tokens
# 4. Build Trainer → wrap model + data + training config into HuggingFace Trainer
# 5. Train model → forward pass → loss → backward pass → parameter update
# 6. Evaluate model (compute_metrics) → run validation set and compute BLEU / CHRF
# 7. Return results → BLEU score / CHRF score / validation loss


import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"# NLLB tokenizer deadlocks with multiple workers


class BaseTrainer(ABC):

    def __init__(self, model_name: str, src_lang: str, tgt_lang: str):
        self.model_name = model_name
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.src_lang = src_lang
        self.tokenizer.tgt_lang = tgt_lang

        self.bleu = evaluate.load("bleu")
        self.chrf = evaluate.load("chrf")

    @abstractmethod
    def setup_model(self, **kwargs):
        pass

    def tokenize(self, examples: Dict) -> Dict:
        self.tokenizer.src_lang = self.src_lang
        model_inputs = self.tokenizer(
            examples["source"],
            max_length=128,
            truncation=True,
            padding=False,
        )

        self.tokenizer.tgt_lang = self.tgt_lang
        labels = self.tokenizer(
            text_target=examples["target"],
            max_length=128,
            truncation=True,
            padding=False,
        )

        # replace pad token id with -100 so it's ignored in loss
        model_inputs["labels"] = [
            [(t if t != self.tokenizer.pad_token_id else -100) for t in label]
            for label in labels["input_ids"]
        ]
        return model_inputs

    def prepare_datasets(self, train_data: List[Dict], val_data: List[Dict]):
        train_dataset = Dataset.from_list(train_data)
        val_dataset = Dataset.from_list(val_data)

        train_dataset = train_dataset.map(
            self.tokenize,
            batched=True,
            remove_columns=train_dataset.column_names,
        )
        val_dataset = val_dataset.map(
            self.tokenize,
            batched=True,
            remove_columns=val_dataset.column_names,
        )

        return train_dataset, val_dataset

    def training_args(self, config: Dict) -> Seq2SeqTrainingArguments:
        eval_steps = config.get("eval_steps", 200)
        save_steps = config.get("save_steps", eval_steps)

        # save_steps must be a multiple of eval_steps, otherwise HuggingFace throws
        if save_steps % eval_steps != 0:
            save_steps = eval_steps

        return Seq2SeqTrainingArguments(
            # 1. Output & Reproducibility
            output_dir=config["output_dir"],
            #overwrite_output_dir=True,
            seed=config.get("seed", 42),

            # 2. Training scale
            num_train_epochs=config.get("epochs", 3),
            per_device_train_batch_size=config.get("batch_size", 4),
            per_device_eval_batch_size=config.get("batch_size", 4),
            gradient_accumulation_steps=config.get("gradient_accumulation_steps", 4),

            # 3. Optimization
            learning_rate=config.get("learning_rate", 5e-4),
            weight_decay=config.get("weight_decay", 0.01),
            warmup_steps=config.get("warmup_steps", 100),
            max_grad_norm=config.get("max_grad_norm", 1.0),

            # 4. Precision & performance
            fp16=config.get("fp16", True),
            dataloader_pin_memory=False,
            dataloader_num_workers=0,

            # 5. Evaluation
            eval_strategy="steps",
            eval_steps=eval_steps,
            metric_for_best_model=config.get("metric_for_best_model", "bleu"),
            greater_is_better=True,

            # 6. Generation (used during evaluation)
            predict_with_generate=True,
            generation_max_length=128,

            # 7. Checkpoints
            save_strategy="steps",
            save_steps=save_steps,
            save_total_limit=config.get("save_total_limit", 2),
            load_best_model_at_end=True,

            # 8. Misc
            logging_steps=config.get("logging_steps", 50),
            remove_unused_columns=False,
            report_to=[],
        )

    def train(self, train_data: List[Dict], val_data: List[Dict], config: Dict) -> Dict:
        # 1. Remove old checkpoints so each run starts fresh
        output_dir = Path(config["output_dir"])
        if output_dir.exists():
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True)

        # 2. Setup model
        model = self.setup_model(**config)
        train_dataset, val_dataset = self.prepare_datasets(train_data, val_data)

        # 3. Data collator handles dynamic padding 
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            model=model,
            padding=True,
        )

        # 4.early stopping callback to prevent overfitting 
        callbacks = []
        if config.get("early_stopping_patience"):
            callbacks.append(
                EarlyStoppingCallback(
                    early_stopping_patience=config["early_stopping_patience"],
                    early_stopping_threshold=config.get("early_stopping_threshold", 0.001),
                )
            )
        # 5. Build Trainer
        trainer = Seq2SeqTrainer(
            model=model,
            args=self.training_args(config),
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            processing_class=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics,
            callbacks=callbacks,)

        trainer.train()
        eval_result = trainer.evaluate()

        return {
            "model": model,
            "trainer": trainer,
            "bleu": eval_result.get("eval_bleu", 0.0),
            "chrf": eval_result.get("eval_chrf", 0.0),
            "loss": eval_result.get("eval_loss", 0.0),
        }
    
    def compute_metrics(self, eval_pred) -> Dict:
        ## Convert logits → token ids → text, then compare with references to compute BLEU/CHRF scores

        predictions, labels = eval_pred# token or logits, but for seq2seq we need token ids to decode

        # for seq2seq models, the predictions are token ids that need to be decoded
        if predictions.ndim == 3:
            predictions = np.argmax(predictions, axis=-1)

        decoded_preds = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        bleu = self.bleu.compute(
            predictions=decoded_preds,
            references=[[ref] for ref in decoded_labels],
        )
        chrf = self.chrf.compute(
            predictions=decoded_preds,
            references=decoded_labels,
        )

        return {"bleu": bleu["bleu"], "chrf": chrf["score"]}

    def generate_predictions(
        self,
        model,
        dataset,
        batch_size: int = 8,
        max_length: int = 128,
        num_beams: int = 5,
    ) -> List[str]:
        # 1.store predictions 
        predictions = []
        device = next(model.parameters()).device
        # 2. Process dataset in batches to avoid OOM errors
        for i in range(0, len(dataset), batch_size):
            batch = dataset.samples[i : i + batch_size]
            sources = [s.source for s in batch]
        # 3. Tokenize sources and move to device
            self.tokenizer.src_lang = self.src_lang
            inputs = self.tokenizer(
                sources,
                return_tensors="pt",
                max_length=max_length,
                truncation=True,
                padding=True,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
        # 4. Generate predictions with no_grad to save memory
            with torch.no_grad():
                generated = model.generate(
                    **inputs,
                    max_length=max_length,
                    num_beams=num_beams,
                )
        # 5. Decode predictions and add to list
            predictions.extend(
                self.tokenizer.batch_decode(generated, skip_special_tokens=True)
            )

        return predictions