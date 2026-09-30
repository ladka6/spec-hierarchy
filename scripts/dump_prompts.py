"""Write the evaluation prompts, already formatted with the target's chat template, to JSON
(for tools that run outside the hspec environment, e.g. vLLM).

  python scripts/dump_prompts.py --out prompts.json --n 10
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hspec.data import format_prompt, load_prompts  # noqa: E402
from hspec.models import load_tokenizer  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--target", default="Qwen/Qwen3-8B")
ap.add_argument("--datasets", nargs="+", default=["gsm8k", "math500", "humaneval", "mt-bench"])
ap.add_argument("--n", type=int, default=10)
ap.add_argument("--out", required=True)
args = ap.parse_args()
tok = load_tokenizer(args.target)
rows = [{"dataset": d, "text": format_prompt(tok, p)} for d in args.datasets for p in load_prompts(d, args.n)]
Path(args.out).write_text(json.dumps(rows))
print(f"wrote {len(rows)} prompts to {args.out}")
