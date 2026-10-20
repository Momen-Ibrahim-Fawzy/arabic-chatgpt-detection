"""Build the six adversarial sets from the ChatGPT answers of the out-of-domain set.

    python scripts/make_attack_sets.py --source data/ood/generated.xlsx --out data/attacks --rate 0.1

Only rows labelled ChatGPT are attacked (the goal of an attacker is to pass ChatGPT text off as human),
so every attack set has the size of the ChatGPT set and accuracy on it equals detection recall.
"""
import argparse
from pathlib import Path

from arabic_detect.attacks import ATTACKS, apply_attack
from arabic_detect.data import CHATGPT, TEXT_COLUMN, read_table, write_table


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("data/attacks"))
    ap.add_argument("--attacks", nargs="+", default=list(ATTACKS), choices=list(ATTACKS))
    ap.add_argument("--rate", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    df = read_table(args.source)
    chatgpt = df[df["label"] == CHATGPT].reset_index(drop=True)
    write_table(chatgpt, args.out / "original.xlsx")
    for name in args.attacks:
        attacked = chatgpt.copy()
        attacked["original_answer"] = attacked[TEXT_COLUMN]
        attacked[TEXT_COLUMN] = apply_attack(name, attacked[TEXT_COLUMN].tolist(), rate=args.rate, seed=args.seed)
        changed = (attacked[TEXT_COLUMN] != attacked["original_answer"]).mean()
        write_table(attacked, args.out / f"{name}.xlsx")
        print(f"{name:18s} {len(attacked)} texts, {changed:.1%} modified")


if __name__ == "__main__":
    main()
