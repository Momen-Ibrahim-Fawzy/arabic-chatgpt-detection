"""HC3 loading and the tabular format shared by every script.

Every split is stored as one table with at least ``arabic_answer`` and ``label`` (0 = human, 1 = ChatGPT).
The English columns (``question``, ``answer``) and ``arabic_question`` are kept when available so a
translation can be audited against its source.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Union

import pandas as pd

HUMAN, CHATGPT = 0, 1
LABEL_NAMES = {HUMAN: "human", CHATGPT: "chatgpt"}
TEXT_COLUMN = "arabic_answer"

# English QA part of HC3 (Guo et al., 2023), as re-packaged on the Hugging Face hub with fixed splits.
HC3_REPO = "almanach/hc3_french_ood"
HC3_CONFIG = "hc3_en_qa"
SPLITS = ("train", "validation", "test")


def normalise_label(value: Union[int, str]) -> int:
    """Map the label spellings that appear across HC3 exports (0/1, 'HUMAN'/'CHATGPT') to 0/1."""
    if isinstance(value, str):
        key = value.strip().upper()
        if key in {"HUMAN", "0"}:
            return HUMAN
        if key in {"CHATGPT", "1"}:
            return CHATGPT
        raise ValueError(f"unknown label {value!r}")
    value = int(value)
    if value not in (HUMAN, CHATGPT):
        raise ValueError(f"unknown label {value!r}")
    return value


def load_hc3() -> Dict[str, pd.DataFrame]:
    """Download the English HC3 QA splits (train 68,335 / validation 17,114 / test 710)."""
    from datasets import load_dataset  # optional dependency

    ds = load_dataset(HC3_REPO, name=HC3_CONFIG)
    return {split: ds[split].to_pandas() for split in SPLITS}


def read_table(path: Union[str, Path]) -> pd.DataFrame:
    """Read a split from .xlsx / .csv / .jsonl and clean the two columns the models need."""
    path = Path(path)
    if path.suffix == ".xlsx":
        df = pd.read_excel(path)
    elif path.suffix == ".jsonl":
        df = pd.read_json(path, lines=True)
    else:
        df = pd.read_csv(path)
    if TEXT_COLUMN not in df.columns or "label" not in df.columns:
        raise KeyError(f"{path} needs '{TEXT_COLUMN}' and 'label' columns, found {list(df.columns)}")
    df[TEXT_COLUMN] = df[TEXT_COLUMN].fillna("").astype(str)
    df["label"] = df["label"].map(normalise_label).astype(int)
    return df


def write_table(df: pd.DataFrame, path: Union[str, Path]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".xlsx":
        df.to_excel(path, index=False)
    elif path.suffix == ".jsonl":
        df.to_json(path, orient="records", lines=True, force_ascii=False)
    else:
        df.to_csv(path, index=False)


def label_counts(df: pd.DataFrame) -> Dict[str, int]:
    counts = df["label"].value_counts().to_dict()
    return {LABEL_NAMES[k]: int(counts.get(k, 0)) for k in (HUMAN, CHATGPT)}
