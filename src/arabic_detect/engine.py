"""Fine-tuning and inference with Hugging Face Transformers.

Setup used for all four detectors: sequence classification head with 2 labels, inputs truncated/padded
to 128 tokens, batch size 128, learning rate 3e-5 with a linear schedule and 0.1% warmup, 20 epochs, evaluation and
checkpointing every epoch, and the checkpoint with the best validation macro-F1 kept at the end.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .data import TEXT_COLUMN
from .metrics import classification_report
from .models import resolve


@dataclass
class TrainConfig:
    model: str = "camelbert-da"
    output_dir: str = "runs/camelbert-da"
    max_length: int = 128
    batch_size: int = 128
    eval_batch_size: int = 256
    learning_rate: float = 3e-5
    epochs: int = 20
    warmup_ratio: float = 0.001  # 0.1% warmup
    weight_decay: float = 0.0
    save_total_limit: int = 2
    seed: int = 42
    fp16: bool = True


class TextDataset:
    """Tokenises the texts once and serves tensors to the Trainer."""

    def __init__(self, texts: Sequence[str], labels: Optional[Sequence[int]], tokenizer, max_length: int):
        self.encodings = tokenizer(list(texts), truncation=True, padding="max_length", max_length=max_length)
        self.labels = None if labels is None else [int(x) for x in labels]

    def __len__(self) -> int:
        return len(self.encodings["input_ids"])

    def __getitem__(self, idx: int) -> Dict:
        import torch

        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx])
        return item


def _softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def compute_metrics(eval_pred) -> Dict[str, float]:
    logits, labels = eval_pred
    probs = _softmax(np.asarray(logits))
    return classification_report(labels, probs.argmax(axis=1), scores=probs[:, 1])


def train(cfg: TrainConfig, train_df: pd.DataFrame, val_df: pd.DataFrame):
    """Fine-tune ``cfg.model`` and return the Trainer holding the best checkpoint."""
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer, Trainer,
                              TrainingArguments, set_seed)

    set_seed(cfg.seed)
    name = resolve(cfg.model)
    tokenizer = AutoTokenizer.from_pretrained(name)
    model = AutoModelForSequenceClassification.from_pretrained(
        name, num_labels=2, id2label={0: "human", 1: "chatgpt"}, label2id={"human": 0, "chatgpt": 1})

    train_ds = TextDataset(train_df[TEXT_COLUMN], train_df["label"], tokenizer, cfg.max_length)
    val_ds = TextDataset(val_df[TEXT_COLUMN], val_df["label"], tokenizer, cfg.max_length)

    args = TrainingArguments(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.epochs,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.eval_batch_size,
        learning_rate=cfg.learning_rate,
        lr_scheduler_type="linear",
        warmup_ratio=cfg.warmup_ratio,
        weight_decay=cfg.weight_decay,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        save_total_limit=cfg.save_total_limit,
        seed=cfg.seed,
        fp16=cfg.fp16,
        report_to="none",
    )
    trainer = Trainer(model=model, args=args, train_dataset=train_ds, eval_dataset=val_ds,
                      compute_metrics=compute_metrics)
    trainer.train()
    best = Path(cfg.output_dir) / "best"
    trainer.save_model(str(best))
    tokenizer.save_pretrained(str(best))
    return trainer


class Detector:
    """Load a fine-tuned checkpoint and score Arabic texts (probability of being ChatGPT-written)."""

    def __init__(self, checkpoint: str, max_length: int = 128, batch_size: int = 256, device: Optional[str] = None):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.model = AutoModelForSequenceClassification.from_pretrained(checkpoint).to(self.device).eval()
        self.max_length = max_length
        self.batch_size = batch_size

    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        import torch

        out: List[np.ndarray] = []
        texts = list(texts)
        with torch.no_grad():
            for start in range(0, len(texts), self.batch_size):
                batch = self.tokenizer(texts[start:start + self.batch_size], truncation=True,
                                       padding=True, max_length=self.max_length, return_tensors="pt")
                logits = self.model(**{k: v.to(self.device) for k, v in batch.items()}).logits
                out.append(torch.softmax(logits, dim=-1)[:, 1].float().cpu().numpy())
        return np.concatenate(out) if out else np.zeros(0)

    def evaluate(self, df: pd.DataFrame, threshold: float = 0.5) -> Dict[str, float]:
        scores = self.predict_proba(df[TEXT_COLUMN].tolist())
        return classification_report(df["label"].to_numpy(), (scores >= threshold).astype(int), scores=scores)
