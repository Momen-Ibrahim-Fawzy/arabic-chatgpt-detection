"""Evaluate trained detectors on any number of test sets and print one table.

    python scripts/evaluate.py --checkpoints runs/*/best \
        --sets test=data/hc3_ar/test.xlsx ood=data/ood/generated.xlsx \
               random_char=data/attacks/random_char.xlsx hybrid=data/attacks/hybrid.xlsx \
        --out results/evaluation.csv
"""
import argparse
from pathlib import Path

import pandas as pd

from arabic_detect.data import read_table
from arabic_detect.engine import Detector


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoints", nargs="+", required=True)
    ap.add_argument("--sets", nargs="+", required=True, help="name=path pairs")
    ap.add_argument("--max-length", type=int, default=128)
    ap.add_argument("--out", type=Path, default=Path("results/evaluation.csv"))
    args = ap.parse_args()

    sets = {}
    for item in args.sets:
        name, path = item.split("=", 1)
        sets[name] = read_table(path)

    rows = []
    for ckpt in args.checkpoints:
        detector = Detector(ckpt, max_length=args.max_length)
        model = Path(ckpt).parent.name if Path(ckpt).name == "best" else Path(ckpt).name
        for name, df in sets.items():
            rows.append({"model": model, "set": name, "n": len(df), **detector.evaluate(df)})
            print(rows[-1])

    table = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out, index=False)
    print(table.pivot(index="set", columns="model", values="accuracy").round(4).to_string())


if __name__ == "__main__":
    main()
