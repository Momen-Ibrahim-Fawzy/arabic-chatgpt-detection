"""English -> Arabic translation of HC3 with retries and a resumable on-disk cache.

Translating ~86k question/answer pairs through a web translation endpoint takes many hours and fails
now and then (rate limits, dropped connections). Each translated string is appended to a JSONL cache as
soon as it is returned, so an interrupted run resumes where it stopped and identical strings (HC3 repeats
every question once per answer) are translated only once.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable, Dict, Iterable, Optional

import pandas as pd
from tqdm import tqdm

TranslateFn = Callable[[str], str]


def google_translator(src: str = "en", dest: str = "ar") -> TranslateFn:
    """Translation function backed by ``googletrans`` (the free Google Translate web endpoint)."""
    from googletrans import Translator  # optional dependency

    translator = Translator()

    def _translate(text: str) -> str:
        return translator.translate(text, src=src, dest=dest).text

    return _translate


class CachedTranslator:
    """Wrap a translation function with retries, exponential backoff and a JSONL cache."""

    def __init__(self, fn: TranslateFn, cache_path: Optional[Path] = None, max_retries: int = 5,
                 backoff: float = 2.0, sleep: Callable[[float], None] = time.sleep):
        self.fn = fn
        self.cache_path = Path(cache_path) if cache_path else None
        self.max_retries = max_retries
        self.backoff = backoff
        self.sleep = sleep
        self.cache: Dict[str, str] = {}
        if self.cache_path and self.cache_path.exists():
            with self.cache_path.open(encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        record = json.loads(line)
                        self.cache[record["src"]] = record["tgt"]

    def __call__(self, text: str) -> str:
        if not isinstance(text, str) or not text.strip():
            return ""
        if text in self.cache:
            return self.cache[text]
        delay = 1.0
        for attempt in range(self.max_retries):
            try:
                translated = self.fn(text)
                break
            except Exception:  # network / quota errors from the endpoint
                if attempt == self.max_retries - 1:
                    raise
                self.sleep(delay)
                delay *= self.backoff
        self.cache[text] = translated
        if self.cache_path:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            with self.cache_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"src": text, "tgt": translated}, ensure_ascii=False) + "\n")
        return translated


def translate_frame(df: pd.DataFrame, translate: TranslateFn,
                    columns: Iterable[str] = ("question", "answer"), prefix: str = "arabic_",
                    progress: bool = True) -> pd.DataFrame:
    """Add ``arabic_<column>`` for each English column; unique strings are translated once."""
    out = df.copy()
    for column in columns:
        unique = pd.unique(out[column].astype(str))
        mapping = {text: translate(text) for text in tqdm(unique, desc=f"translating {column}",
                                                          disable=not progress)}
        out[prefix + column] = out[column].astype(str).map(mapping)
    return out
