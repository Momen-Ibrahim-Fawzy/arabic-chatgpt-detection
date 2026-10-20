"""Fine-tune one Arabic encoder as a human-vs-ChatGPT detector.

    python scripts/train.py --config configs/train.yaml --model gigabert-v4
"""
import argparse
import json
from dataclasses import asdict, fields
from pathlib import Path

import yaml

from arabic_detect.data import label_counts, read_table
from arabic_detect.engine import TrainConfig, train


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=Path("configs/train.yaml"))
    ap.add_argument("--model", help="camelbert-da | camelbert-msa | camelbert-mix | gigabert-v4 | HF id")
    ap.add_argument("--output-dir")
    args = ap.parse_args()

    raw = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    data = raw.pop("data")
    known = {f.name for f in fields(TrainConfig)}
    cfg = TrainConfig(**{k: v for k, v in raw.items() if k in known})
    if args.model:
        cfg.model = args.model
    cfg.output_dir = args.output_dir or f"runs/{Path(cfg.model).name}"

    train_df, val_df = read_table(data["train"]), read_table(data["validation"])
    print(f"train {label_counts(train_df)}  validation {label_counts(val_df)}")
    trainer = train(cfg, train_df, val_df)

    metrics = trainer.evaluate()
    Path(cfg.output_dir, "config.json").write_text(json.dumps(asdict(cfg), indent=2))
    Path(cfg.output_dir, "best_validation.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
