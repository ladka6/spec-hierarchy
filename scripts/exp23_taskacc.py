"""Experiment 23: task accuracy of target variants (sanity check that the fine-tuned targets are
sensible models, not collapsed ones). GSM8K test accuracy, greedy, last number in the answer.

  python scripts/exp23_taskacc.py --models base=Qwen/Qwen3-8B math200=/scratch/.../qwen3-8b-math200
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from exp22_selfspec import gsm8k_acc  # noqa: E402

from hspec.models import free, load_target, load_tokenizer  # noqa: E402
from hspec.utils import print_table, save_json  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True, help="name=path")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--max-new", type=int, default=512)
    args = ap.parse_args()
    rows = []
    for m in args.models:
        name, path = m.split("=", 1)
        tok, model = load_tokenizer(path), load_target(path)
        rows.append({"model": name, "gsm8k_acc": gsm8k_acc(model, tok, args.n, args.max_new)})
        print(f"[{name}] gsm8k {rows[-1]['gsm8k_acc']:.3f}", flush=True)
        del model
        free()
    print_table(rows, ["model", "gsm8k_acc"], f"GSM8K accuracy, first {args.n} test questions")
    save_json("exp23_taskacc", {"args": vars(args), "rows": rows})


if __name__ == "__main__":
    main()
