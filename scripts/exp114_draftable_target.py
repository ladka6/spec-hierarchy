"""Experiment 114: how far does the TARGET have to move to become draftable?  (drift against f)

So far the target was fixed and everything cheap had to imitate it: with frozen Qwen3-8B layers the decision of a
4-layer shallow verify at the drafter's first error stays at 0.2-0.3, while LayerSkip (the whole model trained
with early-exit loss) gives 0.78 at the same depth. The difference is the function, not the layers. Here the
backbone itself gets a small LoRA (all layers) and is trained with two terms:
  exit   KL( M' final (detached) || lm_head(reader_k(M' state after k layers)) )   for every k in --exits
         -> the first k layers of M' learn to decide early; later layers are free to compensate
  drift  --lam * KL( M final || M' final )                                         -> M' stays the model M
M = the original target (LoRA switched off), M' = M + LoRA. Speculative decoding on M' is lossless with respect
to M'; "drift" says how far M' is from M. The question is the shape of the trade-off: does f rise at a drift that
is negligible, or only when the model has become a different one?
--rank 0: no backbone LoRA, only the readers are trained (zero drift; the frozen reference).

reader_k = bridge from the state after k layers to the final hidden state (closed-form affine start + residual
MLP, as exp113 "first"); the shallow verify of M' at depth k is its first k layers + reader_k + lm_head. For the
first k layers the states of a full pass equal those of a pass over the in-flight tokens with the prefix cache
(exp113 start-up check), so the decision is read from the full pass of M'.

Trajectory (every --eval-every steps, validation sequences):
  drift KL, same top-1 as M, agree_k = the exit's top-1 equals the final top-1 of M'
Final table (held-out anchors of exp112 / exp113), one row per k. Truth is M' itself, teacher-forced along the data
(anchors whose accepted prefix leaves the data path are dropped, "diverged"):
  drift KL / same top-1   M' against M on the held-out sequences
  tau M / tau M'          accepted length of one DFlash block on M and on M' (the drafter was trained for M)
  agree                   exit top-1 = M' token, in-flight positions before the drafter's first error
  fork top-1 / top-3      exit at the drafter's first error
  cand recall, pick@fork  drafter's top --cands contain M' token there / exit chooses it among them
  f open / f cand         first disagreement with the draft exactly at the first error AND the token right
  false alarm open/cand   disagreement with a correct drafted token before the first error (all anchors)

  python scripts/exp114_draftable_target.py --target Qwen/Qwen3-8B --draft z-lab/Qwen3-8B-DFlash-b16 \
      --data /scratch-shared/$USER/hspec_lagft/data --rank 32 --lam 1.0 --exits 4 8
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dflash.model import extract_context_feature  # noqa: E402
from exp27_refresh import blocks_logits, matched  # noqa: E402
from exp107_lift_diag import load_seqs  # noqa: E402
from exp108b_nonlinear_lift import cosine_lr  # noqa: E402
from exp112_draft_branch import ATTN_LIN, BF, MLP_LIN, DLoRA  # noqa: E402

from hspec.bridges import Bridge  # noqa: E402
from hspec.models import load_draft, load_target  # noqa: E402
from hspec.utils import print_table, save_json  # noqa: E402


def add_lora(target, rank):
    """LoRA (switched by DLoRA.on) on every linear of every decoder layer; returns {name: module}"""
    mods = {}
    for li, lay in enumerate(target.model.layers):
        for parent, names in ((lay.self_attn, ATTN_LIN), (lay.mlp, MLP_LIN)):
            for nm in names:
                w = DLoRA(getattr(parent, nm), rank)
                setattr(parent, nm, w)
                mods[f"{li}.{nm}"] = w
    return mods


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="Qwen/Qwen3-8B")
    ap.add_argument("--draft", default="z-lab/Qwen3-8B-DFlash-b16")
    ap.add_argument("--data", default="")
    ap.add_argument("--cache", default="")
    ap.add_argument("--exits", nargs="+", type=int, default=[4, 8])
    ap.add_argument("--rank", type=int, default=32, help="backbone LoRA rank (0 = frozen backbone, readers only)")
    ap.add_argument("--lam", type=float, default=1.0, help="weight of the drift term KL(M || M')")
    ap.add_argument("--steps", type=int, default=4000, help="one training sequence per step")
    ap.add_argument("--train-len", type=int, default=512)
    ap.add_argument("--n-pos", type=int, default=128, help="positions per sequence in the losses")
    ap.add_argument("--lr-lora", type=float, default=5e-5)
    ap.add_argument("--lr-reader", type=float, default=1e-4)
    ap.add_argument("--bridge-rank", type=int, default=1024)
    ap.add_argument("--cands", type=int, default=16)
    ap.add_argument("--val-seqs", type=int, default=24)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--fit-seqs", type=int, default=150)
    ap.add_argument("--ridge", type=float, default=1e-3)
    ap.add_argument("--fit-tokens", type=int, default=600_000, help="as exp112: fixes the train / held-out split")
    ap.add_argument("--val-tokens", type=int, default=15_000)
    ap.add_argument("--anchors", type=int, default=600)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--out", default="", help="file for the trained LoRA and readers (optional)")
    ap.add_argument("--save", default="exp114_draftable_target")
    args = ap.parse_args()
    assert args.data or args.cache, "--data or --cache"
    dev, t0 = "cuda", time.time()
    torch.manual_seed(0)
    torch.set_num_threads(args.threads)
    rng = random.Random(0)

    seqs = load_seqs(args)
    target = load_target(args.target, dev)
    draft = load_draft(args.draft, dev)
    for p in list(target.parameters()) + list(draft.parameters()):
        p.requires_grad_(False)
    head = target.lm_head
    lids, bs = list(draft.target_layer_ids), draft.block_size
    H, L, C = target.config.hidden_size, len(target.model.layers), args.cands
    dt = head.weight.dtype
    lora = add_lora(target, args.rank) if args.rank else {}
    lora_params = [p for w in lora.values() for p in (w.A, w.B)]
    readers = nn.ModuleDict({str(k): Bridge(H, "mlp", args.bridge_rank) for k in args.exits}).to(dev)
    print(f"[setup] {len(seqs)} sequences, {L} layers, exits {args.exits}, backbone LoRA r{args.rank} "
          f"({sum(p.numel() for p in lora_params) / 1e6:.1f}M), lam {args.lam}, readers "
          f"{sum(p.numel() for p in readers.parameters()) / 1e6:.1f}M", flush=True)

    def states(x, on, grad=False):
        """hidden states of M (on=False) or M' (on=True) for ids x [1, T]"""
        DLoRA.on = bool(on)
        try:
            with torch.set_grad_enabled(grad):
                return target(x, output_hidden_states=True, use_cache=False, logits_to_keep=1).hidden_states
        finally:
            DLoRA.on = False

    def exit_logits(k, h):
        """h [n, H] states after k layers -> logits [n, V] of the shallow verify at depth k"""
        return head(readers[str(k)](h[None])[0].to(dt)).float()

    def kl(p_logits, q_logits):
        return F.kl_div(F.log_softmax(q_logits, -1), F.softmax(p_logits, -1), reduction="batchmean")

    # ---- the split of exp112 / exp113 (same train / validation sequences, same held-out anchors) ----
    def take(i, n_tokens, cap):
        recs, n = [], 0
        while n < n_tokens and i < cap:
            ids, lo = seqs[i]
            n += len(ids) - lo
            recs.append(i)
            i += 1
        return recs, i

    max_fit = int(0.6 * len(seqs))
    fit_seqs, i = take(0, args.fit_tokens, max_fit)
    val_recs, fit_end = take(i, args.val_tokens, max_fit + (len(seqs) - max_fit) // 4)
    assert val_recs, "no validation sequences: add data or lower --fit-tokens"
    val_seqs = val_recs[: args.val_seqs]
    rest = seqs[fit_end:]
    cand_a = [(si, s) for si, (ids, lo) in enumerate(rest) for s in range(max(lo, 18), len(ids) - bs)]
    random.Random(1).shuffle(cand_a)
    by_seq = {}
    for si, s in sorted(cand_a[: args.anchors]):
        by_seq.setdefault(si, []).append(s)
    print(f"[data] train {len(fit_seqs)} seqs, val {len(val_seqs)}, held-out {len(rest)} "
          f"({sum(len(v) for v in by_seq.values())} anchors in {len(by_seq)} seqs)", flush=True)

    # ---- start-up checks: the untrained LoRA leaves the model unchanged; gradients reach LoRA and readers ----
    x = seqs[0][0][: args.train_len][None].to(dev)
    with torch.no_grad():
        a, b = states(x, False)[-1], states(x, True)[-1]
        print(f"[check] untrained M' against M: max |difference| of the final state "
              f"{(a.float() - b.float()).abs().max().item():.2e}", flush=True)
        assert torch.equal(head(a[0]).argmax(-1), head(b[0]).argmax(-1)), "untrained LoRA changes the model"
        del a, b

    # ---- closed-form affine start of the readers: state after k layers -> final state, on M ----
    with torch.no_grad():
        acc = {k: [torch.zeros(H + 1, H + 1, dtype=torch.float64, device=dev),
                   torch.zeros(H + 1, H, dtype=torch.float64, device=dev)] for k in args.exits}
        for si in fit_seqs[: args.fit_seqs]:
            hs = states(seqs[si][0][None].to(dev), False)
            for k in args.exits:
                xk = hs[k][0].double()
                d = hs[-1][0].double() - xk
                w = torch.rsqrt(xk.pow(2).mean(-1, keepdim=True) + 1e-6)
                xa = torch.cat([xk, torch.ones_like(xk[:, :1])], -1) * w
                acc[k][0] += xa.T @ xa
                acc[k][1] += xa.T @ (d * w)
            del hs
        for k in args.exits:
            xtx, xtd = acc[k]
            sol = torch.linalg.solve(
                xtx + args.ridge * xtx.diagonal().mean() * torch.eye(H + 1, dtype=xtx.dtype, device=dev), xtd)
            readers[str(k)].lin.weight.copy_(sol[:H].T.float())
            readers[str(k)].lin.bias.copy_(sol[H].float())
        del acc
    print(f"[fit] readers fitted, {time.time() - t0:.0f}s", flush=True)

    @torch.no_grad()
    def val_stats():
        readers.eval()
        d, same, ag = [], [], {k: [] for k in args.exits}
        for si in val_seqs:
            ids, lo = seqs[si]
            x = ids[: args.train_len][None].to(dev)
            a = max(1, min(lo, x.shape[1] - 17))
            hT = states(x, False)
            hS = states(x, True) if lora else hT
            tl, sl = head(hT[-1][0, a:]).float(), head(hS[-1][0, a:]).float()
            top = sl.argmax(-1)
            d.append(kl(tl, sl).item())
            same.append((tl.argmax(-1) == top).float().mean().item())
            for k in args.exits:
                ag[k].append((exit_logits(k, hS[k][0, a:]).argmax(-1) == top).float().mean().item())
            del hT, hS, tl, sl
        readers.train()
        m = lambda v: sum(v) / max(len(v), 1)  # noqa: E731
        return m(d), m(same), {k: m(v) for k, v in ag.items()}

    # ---- training ----
    groups = [{"params": list(readers.parameters()), "base": args.lr_reader}]
    if lora_params:
        groups.append({"params": lora_params, "base": args.lr_lora})
    params = [p for g in groups for p in g["params"]]
    opt = torch.optim.AdamW(groups, lr=args.lr_reader, weight_decay=0.0)
    traj = []

    def report(step):
        d, same, ag = val_stats()
        traj.append({"step": step, "drift KL": d, "same top-1": same, **{f"agree k={k}": v for k, v in ag.items()}})
        print(f"[val] step {step} drift KL {d:.4f} same top-1 {same:.4f} "
              + " ".join(f"agree k={k} {v:.4f}" for k, v in ag.items()) + f" {time.time() - t0:.0f}s", flush=True)

    report(0)
    readers.train()
    for step in range(args.steps):
        for g in opt.param_groups:
            g["lr"] = g["base"] * cosine_lr(step, args.steps, min(200, max(args.steps // 10, 1)))
        ids, lo = seqs[rng.choice(fit_seqs)]
        x = ids[: args.train_len][None].to(dev)
        T = x.shape[1]
        a = max(1, min(lo, T - 17))
        P = torch.tensor(sorted(rng.sample(range(a, T), min(args.n_pos, T - a))), device=dev)
        hT = states(x, False)
        hS = states(x, True, grad=True) if lora else hT
        sl = head(hS[-1][0, P]).float()
        drift = kl(head(hT[-1][0, P]).float(), sl) if lora else sl.new_zeros(())
        ex = sum(kl(sl.detach(), exit_logits(k, hS[k][0, P])) for k in args.exits)
        loss = args.lam * drift + ex
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if step == 0:
            g_l = sum(p.grad.abs().sum().item() for p in lora_params if p.grad is not None)
            g_r = sum(p.grad.abs().sum().item() for p in readers.parameters() if p.grad is not None)
            print(f"[check] first step: gradient mass LoRA {g_l:.3e}, readers {g_r:.3e}", flush=True)
            assert g_r > 0 and (g_l > 0 or not lora), "no gradient"
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        del hT, hS
        if (step + 1) % 50 == 0:
            print(f"[train] step {step + 1} drift {float(drift):.4f} exit {float(ex):.4f} "
                  f"{time.time() - t0:.0f}s", flush=True)
        if (step + 1) % args.eval_every == 0 or step + 1 == args.steps:
            report(step + 1)
    readers.eval()
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"args": vars(args), "readers": {k: v.cpu() for k, v in readers.state_dict().items()},
                    "lora": {n: (w.A.detach().cpu(), w.B.detach().cpu()) for n, w in lora.items()}}, args.out)
        print(f"[save] {args.out}", flush=True)

    # ---- held-out anchors: the drafter's first error on M', decided by the shallow verify of M' ----
    def block(feats, x, s):
        with torch.autocast("cuda", dtype=BF):
            return blocks_logits(draft, target, feats, x, [(s, 0)], bs)[0].float()

    mean = lambda v: sum(v) / max(len(v), 1)  # noqa: E731
    dkl, same, tauT, tauS = [], [], [], []
    st = {k: {n: 0 for n in ("res_o", "res_c", "fa_o", "fa_c", "pick", "t1", "t3")} for k in args.exits}
    agree = {k: [] for k in args.exits}
    n_err = n_ok = n_div = recall = 0
    with torch.no_grad():
        for n_done, (si, anchors) in enumerate(by_seq.items()):
            ids, lo = rest[si]
            x = ids.to(dev)
            data = x.tolist()
            hT = states(x[None], False)
            hS = states(x[None], True) if lora else hT
            fT = extract_context_feature(hT, lids)
            fS = extract_context_feature(hS, lids) if lora else fT
            lgT, lgS = head(hT[-1][0, lo:]).float(), head(hS[-1][0, lo:]).float()
            dkl.append(kl(lgT, lgS).item())
            same.append((lgT.argmax(-1) == lgS.argmax(-1)).float().mean().item())
            nxt = head(hS[-1][0]).argmax(-1).tolist()        # nxt[p] = M' token for position p + 1 (data prefix)
            del lgT, lgS
            for s in anchors:
                tauT.append(matched(block(fT[:, :s], x, s).argmax(-1).tolist(), data[s + 1 :]) + 1)
                Lr = block(fS[:, :s], x, s)
                chain = Lr.argmax(-1).tolist()
                truth = nxt[s : s + len(chain)]
                e = matched(chain, truth)
                if any(truth[i] != data[s + 1 + i] for i in range(e)):
                    n_div += 1                               # the accepted prefix is not the data: no states for it
                    continue
                n_ok += 1
                tauS.append(e + 1)
                err = e < len(chain)
                last = min(e, len(chain) - 1)
                top = Lr[: last + 1].topk(C, -1)
                lq = F.log_softmax(Lr[: last + 1], -1).gather(1, top.indices)
                if err:
                    n_err += 1
                    recall += truth[e] in top.indices[e].tolist()
                for k in args.exits:
                    lg = exit_logits(k, hS[k][0, s : s + last + 1])
                    picks = {"o": lg.argmax(-1).tolist(),
                             "c": top.indices.gather(1, (lg.gather(1, top.indices) + lq).argmax(1, keepdim=True))[:, 0]
                             .tolist()}
                    agree[k].extend(picks["o"][i] == truth[i] for i in range(e if err else last + 1))
                    for r, pick in picks.items():
                        dis = [i for i in range(last + 1) if pick[i] != chain[i]]
                        j = dis[0] if dis else -1
                        st[k][f"fa_{r}"] += 0 <= j < e if err else j >= 0
                        st[k][f"res_{r}"] += err and j == e and pick[e] == truth[e]
                    if err:
                        t3 = lg[e].topk(3).indices.tolist()
                        st[k]["t1"] += t3[0] == truth[e]
                        st[k]["t3"] += truth[e] in t3
                        st[k]["pick"] += picks["c"][e] == truth[e]
            del hT, hS, fT, fS
            if n_done % 50 == 0:
                print(f"[eval] {n_done}/{len(by_seq)} sequences, {time.time() - t0:.0f}s", flush=True)

    ne = max(n_err, 1)
    rows = []
    for k in args.exits:
        s_ = st[k]
        rows.append({"rank": args.rank, "lam": args.lam if lora else "-", "steps": args.steps, "k": k,
                     "drift KL": mean(dkl), "same top-1": mean(same), "tau M": mean(tauT), "tau M'": mean(tauS),
                     "diverged": n_div / max(n_div + n_ok, 1), "agree": mean(agree[k]),
                     "fork top-1": s_["t1"] / ne, "fork top-3": s_["t3"] / ne, "cand recall": recall / ne,
                     "pick@fork": s_["pick"] / ne, "f open": s_["res_o"] / ne, "f cand": s_["res_c"] / ne,
                     "false alarm open": s_["fa_o"] / max(n_ok, 1), "false alarm cand": s_["fa_c"] / max(n_ok, 1),
                     "errors": n_err})
    print_table(traj, list(traj[0]), "Trajectory on the validation sequences (drift against early agreement)")
    print_table(rows, list(rows[0]), "Draftable target: decision of the shallow verify of M' at the drafter's first error")
    save_json(args.save, {"args": vars(args), "rows": rows, "trajectory": traj})
    print(f"[done] {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
