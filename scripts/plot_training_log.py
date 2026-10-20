"""Plot loss and validation metrics from a Trainer log and report the best epoch.

    python scripts/plot_training_log.py runs/gigabert-v4/checkpoint-XXXX/trainer_state.json --out docs/assets
"""
import argparse
from pathlib import Path

from arabic_detect.logs import best_epoch, epoch_table, plot_curves, read_log_history


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("log", type=Path, help="trainer_state.json or a text dump of log_history")
    ap.add_argument("--out", type=Path, default=Path("results/curves"))
    ap.add_argument("--metric", default="eval_f1_macro")
    args = ap.parse_args()

    table = epoch_table(read_log_history(args.log))
    args.out.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out / "epochs.csv", index=False)
    for path in plot_curves(table, args.out):
        print("saved", path)
    print(f"best epoch by {args.metric}:")
    print(best_epoch(table, args.metric).to_string())


if __name__ == "__main__":
    main()
