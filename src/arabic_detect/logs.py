"""Turn a Trainer log history into a per-epoch table and training curves.

The Trainer's ``log_history`` (or a text dump of it, as saved during the Colab runs) is a list of dicts
that alternates training records (``loss``, ``learning_rate``) and evaluation records (``eval_*``),
both tagged with the same ``epoch``.
"""
from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Union

import pandas as pd


def read_log_history(path: Union[str, Path]) -> List[Dict]:
    """Read ``trainer_state.json`` or a Python-literal dump of ``log_history``."""
    text = Path(path).read_text(encoding="utf-8")
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        data = ast.literal_eval(text)
    if isinstance(data, dict):
        data = data["log_history"]
    return list(data)


def epoch_table(history: Iterable[Dict]) -> pd.DataFrame:
    """Merge training and evaluation records that share an epoch into one row per epoch."""
    rows: Dict[float, Dict] = {}
    for record in history:
        if "epoch" not in record:
            continue
        row = rows.setdefault(record["epoch"], {"epoch": record["epoch"]})
        if "loss" in record:
            row["train_loss"] = record["loss"]
            row["learning_rate"] = record.get("learning_rate")
        for key, value in record.items():
            if key.startswith("eval_"):
                row[key] = value
    table = pd.DataFrame(sorted(rows.values(), key=lambda r: r["epoch"]))
    return table


def best_epoch(table: pd.DataFrame, metric: str = "eval_f1_macro") -> pd.Series:
    return table.loc[table[metric].idxmax()]


def plot_curves(table: pd.DataFrame, out_dir: Union[str, Path],
                groups: Sequence[Sequence[str]] = (("train_loss", "eval_loss"),
                                                   ("eval_accuracy", "eval_precision", "eval_recall",
                                                    "eval_f1_macro"))) -> List[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for group in groups:
        cols = [c for c in group if c in table.columns]
        if not cols:
            continue
        fig, ax = plt.subplots(figsize=(8, 5))
        for col in cols:
            ax.plot(table["epoch"], table[col], marker="o", ms=3, label=col)
        ax.set_xlabel("epoch")
        ax.grid(True, alpha=0.3)
        ax.legend()
        path = out_dir / ("_".join(cols) + ".png")
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        paths.append(path)
    return paths
