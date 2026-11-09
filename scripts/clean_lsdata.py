"""Trim degenerate loops out of generated shards (greedy base models fall into repetition).

A response is cut at the first position where repetition becomes sustained: over the next --win positions, more
than --thr of the 8-grams already occurred earlier in the response (within --lookback tokens). Responses left
shorter than --min-resp are dropped. Writes shards with the same names to OUT; IN is left untouched.

  python scripts/clean_lsdata.py /scratch-shared/$USER/hspec_ls/data_ft /scratch-shared/$USER/hspec_ls/data_ft_clean
"""
import argparse
from pathlib import Path

import torch


def loop_onset(t, n=8, win=64, thr=0.9, lookback=256):
    last, rep = {}, []
    for i in range(len(t) - n + 1):
        g = tuple(t[i:i + n])
        j = last.get(g)
        rep.append(j is not None and i - j <= lookback)
        last[g] = i
    if len(rep) < win:
        return None
    s = sum(rep[:win])
    for i in range(len(rep) - win + 1):
        if i:
            s += rep[i + win - 1] - rep[i - 1]
        if s > thr * win:
            return i
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inp")
    ap.add_argument("out")
    ap.add_argument("--win", type=int, default=64)
    ap.add_argument("--thr", type=float, default=0.9)
    ap.add_argument("--min-resp", type=int, default=32)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tot = {"seqs": 0, "kept": 0, "trimmed": 0, "dropped": 0, "tok_in": 0, "tok_out": 0}
    for f in sorted(Path(a.inp).glob("shard*.pt")):
        recs, new = torch.load(f), []
        for r in recs:
            np_, resp = r["n_prompt"], r["ids"][r["n_prompt"]:].tolist()
            tot["seqs"] += 1
            tot["tok_in"] += len(resp)
            k = loop_onset(resp, win=a.win, thr=a.thr)
            if k is not None:
                resp = resp[:k]
                tot["trimmed"] += 1
            if len(resp) < a.min_resp:
                tot["dropped"] += 1
                continue
            new.append({"ids": torch.cat([r["ids"][:np_], torch.tensor(resp, dtype=r["ids"].dtype)]), "n_prompt": np_})
            tot["tok_out"] += len(resp)
        torch.save(new, out / f.name)
        tot["kept"] += len(new)
        print(f"{f.name}: {len(recs)} -> {len(new)}", flush=True)
    print(f"{tot['seqs']} seqs: trimmed {tot['trimmed']}, dropped {tot['dropped']}, kept {tot['kept']}; "
          f"response tokens {tot['tok_in']} -> {tot['tok_out']} ({tot['tok_out'] / max(tot['tok_in'], 1):.2f})")


if __name__ == "__main__":
    main()
