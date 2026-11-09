"""Trim degenerate loops out of generated shards (greedy base models fall into repetition).

A response is cut at the first position where repetition becomes sustained: over the next --win positions, more
than --thr of the 8-grams already occurred earlier in the response (within --lookback tokens). A trimmed response
left shorter than --min-trimmed is dropped (it looped almost from the start); untrimmed responses are kept down to
--min-resp (short genuine answers, e.g. GSM8K). --show N prints N cut points (text before | after the cut). Writes shards with the same names to OUT; IN is left untouched.

  python scripts/clean_lsdata.py /scratch-shared/$USER/hspec_ls/data_ft /scratch-shared/$USER/hspec_ls/data_ft_clean
"""
import argparse
from pathlib import Path

import torch


def loop_onset(t, n=8, win=64, thr=0.75, lookback=256):
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
    ap.add_argument("--thr", type=float, default=0.75)
    ap.add_argument("--min-resp", type=int, default=8)
    ap.add_argument("--min-trimmed", type=int, default=32)
    ap.add_argument("--show", type=int, default=0)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tot = {"seqs": 0, "kept": 0, "trimmed": 0, "drop_loop": 0, "drop_short": 0, "tok_in": 0, "tok_out": 0}
    shown, tok = 0, None
    if a.show:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("facebook/layerskip-llama3-8B")
    for f in sorted(Path(a.inp).glob("shard*.pt")):
        recs, new = torch.load(f), []
        for r in recs:
            np_, resp = r["n_prompt"], r["ids"][r["n_prompt"]:].tolist()
            tot["seqs"] += 1
            tot["tok_in"] += len(resp)
            k = loop_onset(resp, win=a.win, thr=a.thr)
            if k is not None:
                if shown < a.show and k >= a.min_trimmed:
                    print(f"--- cut at {k}/{len(resp)}: {tok.decode(resp[max(0, k - 60):k])!r} | "
                          f"{tok.decode(resp[k:k + 40])!r}")
                    shown += 1
                resp = resp[:k]
                tot["trimmed"] += 1
                if len(resp) < a.min_trimmed:
                    tot["drop_loop"] += 1
                    continue
            elif len(resp) < a.min_resp:
                tot["drop_short"] += 1
                continue
            new.append({"ids": torch.cat([r["ids"][:np_], torch.tensor(resp, dtype=r["ids"].dtype)]), "n_prompt": np_})
            tot["tok_out"] += len(resp)
        torch.save(new, out / f.name)
        tot["kept"] += len(new)
        print(f"{f.name}: {len(recs)} -> {len(new)}", flush=True)
    print(f"{tot['seqs']} seqs: trimmed {tot['trimmed']}, dropped (looped from the start) {tot['drop_loop']}, "
          f"dropped (short) {tot['drop_short']}, kept {tot['kept']}; "
          f"response tokens {tot['tok_in']} -> {tot['tok_out']} ({tot['tok_out'] / max(tot['tok_in'], 1):.2f})")


if __name__ == "__main__":
    main()
