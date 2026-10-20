"""Download HC3 (English QA) and translate questions and answers to Arabic.

    python scripts/translate_hc3.py --out data/hc3_ar --cache data/cache/en_ar.jsonl

Writes data/hc3_ar/{train,validation,test}.xlsx with the English columns, ``arabic_question``,
``arabic_answer`` and ``label``. Re-running resumes from the cache.
"""
import argparse
from pathlib import Path

from arabic_detect.data import label_counts, load_hc3, write_table
from arabic_detect.translate import CachedTranslator, google_translator, translate_frame


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path("data/hc3_ar"))
    ap.add_argument("--cache", type=Path, default=Path("data/cache/en_ar.jsonl"))
    ap.add_argument("--splits", nargs="+", default=["train", "validation", "test"])
    ap.add_argument("--limit", type=int, default=None, help="translate only the first N rows per split")
    ap.add_argument("--format", choices=["xlsx", "csv", "jsonl"], default="xlsx")
    args = ap.parse_args()

    translate = CachedTranslator(google_translator("en", "ar"), cache_path=args.cache)
    splits = load_hc3()
    for split in args.splits:
        df = splits[split]
        if args.limit:
            df = df.head(args.limit)
        df = translate_frame(df, translate, columns=("question", "answer"))
        write_table(df, args.out / f"{split}.{args.format}")
        print(f"{split}: {len(df)} rows {label_counts(df)}")


if __name__ == "__main__":
    main()
