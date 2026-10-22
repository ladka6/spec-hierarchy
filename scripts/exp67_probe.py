"""Experiment 67: does the DFlash drafter waste the target's information? (one GPU, ~1 h on an A100, Colab-ready)

At anchor s the drafter reads the target's tap features for positions < s plus the anchor token x_s, and predicts the
target's tokens at s+1, s+2, ... Its accuracy at block position 1 is ~0.845 even though nothing there is guessed. If a
simple probe that reads the SAME kind of information reaches that accuracy, the drafter is near what the information
allows; if the probe beats it, the drafter reads the target badly (the DFlare argument).

Steps:
  1. greedy target responses for --n prompts per dataset (batched, non-thinking)
  2. one target forward per sequence: tap features (DFlash's 5 layers), last-layer state, target argmax at every position
  3. DFlash accuracy at block positions k = 1..4 on held-out anchors (labels: the target's argmax there)
  4. probes, trained on the other sequences: linear map -> RMSNorm -> the target's frozen LM head, inputs at s-1 plus
     the anchor embedding, one probe per (input, k):
        last+emb       target last-layer state at s-1 + emb(x_s)
        taps+emb       DFlash's 5 tap features at s-1 + emb(x_s)          (what the drafter reads, one position only)
        taps+last+emb  both
  Note: the probes see only position s-1, the drafter sees every position < s, so the probe has less context.

  python scripts/exp67_probe.py --n 60 --max-new 384
"""

from __future__ import annotations

import argparse
import math
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dflash.model import extract_context_feature  # noqa: E402
from exp27_refresh import blocks_logits  # noqa: E402

from hspec.data import format_prompt, load_prompts  # noqa: E402
from hspec.models import load_draft, load_target, load_tokenizer  # noqa: E402
from hspec.utils import print_table, save_json  # noqa: E402


def rms(x):
    return x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + 1e-6).to(x.dtype)


@torch.inference_mode()
def generate(target, tok, prompts, max_new, batch, dev):
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    seqs = []
    for i in range(0, len(prompts), batch):
        texts = [format_prompt(tok, p) for p in prompts[i : i + batch]]
        enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(dev)
        out = target.generate(**enc, max_new_tokens=max_new, do_sample=False, temperature=None, top_p=None,
                              top_k=None, pad_token_id=tok.pad_token_id)
        for j in range(out.shape[0]):
            m = enc["attention_mask"][j].bool()
            prompt = enc["input_ids"][j][m]
            resp = out[j, enc["input_ids"].shape[1]:]
            stop = (resp == tok.pad_token_id).nonzero()
            resp = resp[: int(stop[0])] if len(stop) else resp
            seqs.append((torch.cat([prompt, resp]).cpu(), len(prompt)))
        print(f"[gen] {len(seqs)}/{len(prompts)}", flush=True)
    return seqs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="Qwen/Qwen3-8B")
    ap.add_argument("--draft", default="z-lab/Qwen3-8B-DFlash-b16")
    ap.add_argument("--datasets", nargs="+", default=["gsm8k", "math500", "humaneval", "mt-bench"])
    ap.add_argument("--n", type=int, default=150, help="prompts per dataset")
    ap.add_argument("--max-new", type=int, default=384)
    ap.add_argument("--gen-batch", type=int, default=16)
    ap.add_argument("--val-frac", type=float, default=0.25)
    ap.add_argument("--val-stride", type=int, default=2, help="evaluate every n-th anchor of the held-out sequences")
    ap.add_argument("--ks", nargs="+", type=int, default=[1, 2])
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--bs", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--rank", type=int, default=256, help="probe rank (low rank against memorising)")
    ap.add_argument("--wd", type=float, default=0.1)
    ap.add_argument("--patience", type=int, default=3, help="dev checks (every 200 steps) without improvement")
    ap.add_argument("--save", default="exp67_probe")
    args = ap.parse_args()
    dev = "cuda"
    torch.manual_seed(0)
    t0 = time.time()

    tok = load_tokenizer(args.target)
    target = load_target(args.target, dev)
    draft = load_draft(args.draft, dev)
    lids, bs = list(draft.target_layer_ids), draft.block_size
    H = target.config.hidden_size
    emb = target.get_input_embeddings()
    K = max(max(args.ks), 4)

    prompts = []
    for ds in args.datasets:
        try:
            prompts += load_prompts(ds, args.n)
        except Exception as e:  # noqa: BLE001
            print(f"[data] {ds} skipped: {type(e).__name__}: {str(e)[:120]}", flush=True)
    seqs = generate(target, tok, prompts, args.max_new, args.gen_batch, dev)
    random.Random(0).shuffle(seqs)
    n_val = max(1, int(len(seqs) * args.val_frac))
    val, train = seqs[:n_val], seqs[n_val:]
    print(f"[data] {len(train)} train / {len(val)} val sequences, {time.time() - t0:.0f}s", flush=True)

    def collect(part, dense, with_draft):
        X_last, X_taps, X_emb, Y, dacc = [], [], [], [], []
        with torch.inference_mode():
            for ids, n_prompt in part:
                x = ids.to(dev)
                T = len(x)
                out = target(x[None], output_hidden_states=True, use_cache=False)
                nxt = out.logits[0].argmax(-1)                        # nxt[t] = target's token at t+1 given x<=t
                taps = extract_context_feature(out.hidden_states, lids)  # [1, T, 5H]
                last = out.hidden_states[-1][0]
                del out
                lo, hi = max(n_prompt, 1), T - K - 1
                if hi <= lo:
                    continue
                anchors = list(range(lo, hi, 1 if dense else args.val_stride))
                a = torch.tensor(anchors, device=dev)
                X_last.append(last[a - 1].half().cpu())
                X_taps.append(taps[0, a - 1].half().cpu())
                X_emb.append(emb(x[a]).half().cpu())
                Y.append(torch.stack([nxt[a + k - 1] for k in range(1, K + 1)], -1).cpu())   # [A, K]
                if with_draft:
                    for c in range(0, len(anchors), 64):
                        ch = anchors[c : c + 64]
                        L = blocks_logits(draft, target, taps, x, [(s, 0) for s in ch], bs)      # [n, 15, V]
                        pred = L[:, :K].argmax(-1).cpu()
                        dacc.append(pred == Y[-1][c : c + len(ch)])
                        del L
        return (torch.cat(X_last), torch.cat(X_taps), torch.cat(X_emb), torch.cat(Y),
                torch.cat(dacc) if with_draft else None)

    va = collect(val, dense=False, with_draft=True)
    print(f"[val] {len(va[3])} anchors, drafter acc k=1..{K}: "
          f"{[round(float(v), 3) for v in va[4].float().mean(0)]}, {time.time() - t0:.0f}s", flush=True)
    tr = collect(train, dense=True, with_draft=False)
    print(f"[train] {len(tr[3])} anchors, {time.time() - t0:.0f}s", flush=True)

    head = target.lm_head
    head.requires_grad_(False)
    del draft
    target.model.layers = torch.nn.ModuleList()
    torch.cuda.empty_cache()

    def inputs(d, kind):
        last, taps, e = d[0], d[1], d[2]
        parts = {"last": [rms(last)], "taps": [rms(t) for t in taps.view(len(taps), -1, H).unbind(1)]}
        use = {"last+emb": ["last"], "taps+emb": ["taps"], "taps+last+emb": ["taps", "last"]}[kind]
        return torch.cat(sum((parts[u] for u in use), []) + [rms(e)], -1)

    rows = [{"reader": "DFlash drafter (all positions < s)", **{f"acc_k{k}": float(va[4][:, k - 1].float().mean())
                                                                 for k in args.ks}}]
    for kind in ("last+emb", "taps+emb", "taps+last+emb"):
        Xtr, Xva = inputs(tr, kind).to(dev), inputs(va, kind).to(dev)
        row = {"reader": f"probe {kind} (position s-1 only)"}
        for k in args.ks:
            ytr, yva = tr[3][:, k - 1], va[3][:, k - 1]
            # low-rank probe (rank --rank) so it cannot memorise ~50K examples; early stopping on a held-out tail of
            # the training sequences (contiguous rows = whole sequences), best dev-loss weights are kept
            n_dev = max(1, int(len(Xtr) * 0.1))
            Xfit, yfit, Xdev, ydev = Xtr[:-n_dev], ytr[:-n_dev], Xtr[-n_dev:], ytr[-n_dev:]
            down = torch.nn.Linear(Xtr.shape[1], args.rank, bias=False).to(dev)
            up = torch.nn.Linear(args.rank, H).to(dev)
            torch.nn.init.normal_(down.weight, std=1.0 / math.sqrt(Xtr.shape[1]))
            torch.nn.init.normal_(up.weight, std=1.0 / math.sqrt(args.rank))
            torch.nn.init.zeros_(up.bias)
            gain = torch.nn.Parameter(torch.ones(H, device=dev))
            params = list(down.parameters()) + list(up.parameters()) + [gain]
            opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)
            sched = torch.optim.lr_scheduler.LambdaLR(
                opt, lambda s: min(1.0, (s + 1) / 100) * 0.5 * (1 + math.cos(math.pi * s / args.steps)))

            def fwd(xb):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    return head(rms(up(down(xb.float()))) * gain)

            def eval_loss(X, y):
                tot, n = 0.0, 0
                with torch.no_grad():
                    for c in range(0, len(X), 4096):
                        lg = fwd(X[c : c + 4096]).float()
                        tot += float(F.cross_entropy(lg, y[c : c + 4096].to(dev), reduction="sum"))
                        n += len(lg)
                return tot / max(n, 1)

            best, best_state, since = float("inf"), None, 0
            for step in range(args.steps):
                idx = torch.randint(0, len(Xfit), (args.bs,))
                loss = F.cross_entropy(fwd(Xfit[idx.to(dev)]).float(), yfit[idx].to(dev))
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                sched.step()
                if (step + 1) % 200 == 0:
                    dl = eval_loss(Xdev, ydev)
                    if dl < best:
                        best, since = dl, 0
                        best_state = [p.detach().clone() for p in params]
                    else:
                        since += 1
                    if (step + 1) % 1000 == 0 or since >= args.patience:
                        print(f"[probe {kind} k={k}] step {step + 1} train {float(loss.detach()):.3f} "
                              f"dev {dl:.3f} (best {best:.3f})", flush=True)
                    if since >= args.patience:
                        break
            with torch.no_grad():
                for p, b in zip(params, best_state):
                    p.copy_(b)
            correct = 0
            with torch.no_grad():
                for c in range(0, len(Xva), 4096):
                    lg = fwd(Xva[c : c + 4096])
                    correct += int((lg.argmax(-1).cpu() == yva[c : c + 4096]).sum())
            row[f"acc_k{k}"] = correct / len(Xva)
            row[f"devloss_k{k}"] = best
            del down, up
            del opt
        del Xtr, Xva
        torch.cuda.empty_cache()
        rows.append(row)
        print(row, flush=True)

    print_table(rows, list(rows[0]), "top-1 accuracy at block position k on held-out anchors (label: the target's own "
                                     "greedy token there); the drafter reads all positions < s, the probes only s-1")
    save_json(args.save, {"args": vars(args), "n_val_anchors": len(va[3]), "n_train_anchors": len(tr[3]),
                          "rows": rows, "draft_acc_k1_4": va[4].float().mean(0).tolist()})
    print(f"[done] {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
