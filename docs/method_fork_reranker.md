# Method: fork-resolving block trees with a drafter-relative copy slice

Status: design document (2026-11-02). Numbers marked "est." are estimates from the cost model or from runs not yet
reproduced; everything else is from docs/status_2026-11-02.md.

## 0. Summary

DFlash drafts a block; a thin, 4-bit slice of the target's own layers runs over the draft and acts as a reranker over
DFlash's top-16 candidates. The first position where the reranker disagrees with DFlash is the fork; its top choice is
the fork token; its hidden states give approximate target features for the drafted prefix. DFlash redrafts a new
block from the fork. The slice checks the new block and prunes to one path; the loop repeats while a one-scalar
stopping rule says the next level pays. The target verifies every branch once. Rare hard forks are escalated to the
full 4-bit copy. The slice is trained against the drafter's failures (fork-weighted 17-way ranking + deep-feature
loss), not against the target's full distribution.

## 1. Components

| component | what it is | size / cost per pass |
|---|---|---|
| target | Qwen3-8B, bf16 | ~16 GB streamed, ~13 ms (flat to ~64 nodes, 15.8 ms at 128, 19 ms at 256) |
| drafter | DFlash-b16, reads target taps L1, 9, 17, 25, 33 (only L25 / L33 matter) | ~3.3 GB, ~2.6 ms |
| copy slice | k target layers kept (others skipped), 4-bit, LoRA adapters; ranker head; 2 feature heads | target ~1 GB, ~1-1.5 ms (est.) |
| full copy | Qwen3-8B 4-bit, only for escalated forks | ~4 GB, ~3 ms at <= 32 nodes |

### 1.1 Copy slice

- Layers: k of the target's 36 decoder layers kept in place (as in exp81's layer-skipped target), the others identity.
  Which k layers: chosen by a sweep on restricted fork accuracy (section 6), starting from evenly spaced and from
  "early + late" sets. Weights 4-bit, LoRA rank r on the kept layers, the target's embeddings and final norm reused.
- Runs causally over the drafted tokens, with its own KV cache over the accepted text.
- Output at each position k: a hidden vector h_k (after the final norm).

### 1.2 Ranker head (replaces the LM head)

For the 16 DFlash candidates c_1..c_16 at a position, with DFlash probabilities q(c_j):

    s_j = h_k . w(c_j) + alpha * log q(c_j)        j = 1..16
    s_0 = u . h_k                                  abstain
    p   = softmax(s_0, s_1, ..., s_16)

w(c_j) = the target's LM-head row of c_j (16 rows gathered per position, not the 151k x 4096 matrix). alpha (scalar)
and u (vector) are learned. Abstain means "the target's token is not among the candidates, or too hard".

### 1.3 Feature heads

Two linear maps from the slice's hidden states at the kept layers nearest to 25 and 33 to approximations of the
target's L25 / L33 outputs. For DFlash's other taps (L1, 9, 17) the slice's own early layers are used directly (they
matter little). These features feed DFlash when it redrafts from a fork.

## 2. One round, step by step

State at the start of a round: accepted text x_1..x_t; target KV and target taps for x_1..x_t; slice KV for
x_1..x_t; DFlash context cache.

**A. Level-0 draft.** DFlash drafts 15 tokens from anchor x_t with real target taps; keeps the top-16 candidates and
their probabilities at each of the 15 positions.

**B. Slice pass.** The slice runs over the 15 drafted tokens (one pass, 15 positions). At each position: the 17-way
distribution p. Then:
- fork position f = the first position where argmax p is not DFlash's token (or p(DFlash's token) < theta_loc);
- fork candidates = the slice's top-2 at f;
- if abstain wins at f: escalate (full copy over the same tokens, its top-2 at f) or end branching here;
- features: the slice's L25 / L33 estimates for the drafted positions before f.
If no fork is found: go to G (the draft is verified as is).

**C. Branch at the fork (2-way).** Branch 1 = the slice's top-1 at f (needs a redraft). Branch 2 = DFlash's original
token at f with its already drafted tail (free; kept for the verifier). If the slice's top-2 differs from DFlash's
token and the slice is near-uncertain, it may replace branch 2 (also needs a redraft).

**D. Redraft from the fork.** DFlash drafts a new 15-token block anchored at the branch token. Its context: real target
taps for x_1..x_t, slice-estimated taps for the drafted positions t+1..f-1 (lag-style input). New branches go in one
batched DFlash pass.

**E. Next level, pruned at the copy.** The slice runs over the new block(s) (step B again). If two branches were
redrafted, the slice scores both and continues only the more probable one: the slice always sees <= 32-48 nodes; the
verifier keeps every branch.

**F. Stopping rule (the gate, one scalar).** Before level l+1, estimate its gain alive_l * tau' (alive_l = probability
that the path so far is right: product of the slice's probabilities of the chosen tokens at the resolved forks; tau' =
expected continuation per resolved fork, ~5.5-5.8 measured in the note) and its cost c_level = c_slice + c_D. Continue
iff alive_l * tau' / c_level >= the round's running tokens/ms. Otherwise stop.

**G. Verify.** The target runs once over the tree of all blocks (tree attention mask), accepts the longest path that
matches its own choices, plus its bonus token. Lossless: greedy exactly; sampling with standard multi-branch tree
verification (recursive rejection sampling over siblings).

**H. Update.** Target KV and taps for the accepted tokens (from the verify pass); slice KV for accepted tokens it has
already processed on the accepted path is kept, the rest is computed in the next round's slice pass; DFlash context
cache gets the fresh target taps.

## 3. Cost model

    tokens/ms = (tau_D + sum_l alive_l * tau') / (c_D + L * c_level + c_V(nodes))
    c_level   = c_slice + c_D (+ c_full * P(escalate))

- tau_D: DFlash's own accepted per round (online ~5.8 per the note, offline ~8).
- The tree policy cannot raise the ceiling; only c_level can: the slope per level is tau' / c_level.
- With the full 4-bit copy (c_level ~5.6 ms) the note's simulator gives ~1.35x offline / ~1.5x online (est.); with a
  slice at ~1 ms (c_level ~3.6 ms) the ceiling moves to ~1.7x (est.).
- c_V is not flat beyond ~64 nodes: 2-way branching to depth 3-4 gives 128-256 verifier nodes (15.8-19 ms). Use the
  measured curve.

## 4. Training

### 4.1 Data

- Prompts: the existing lag training shards (training split) plus held-out benchmark prompts for evaluation only.
- Run the target greedily; run DFlash in the real loop to get online anchors (anchors after a correction, which are
  harder than offline anchors on the target's text).
- Per anchor and block position: DFlash's top-16 candidates and probabilities; the target's token; the target's L25 /
  L33 outputs; whether the position is a fork (DFlash's top-1 wrong, prefix right); the target's margin there.
- Positions after the first fork are not used for the slice: in inference the slice is only trusted up to the first
  fork. Up to and including the fork the drafted prefix equals the true text, so training on the true text is exact.

### 4.2 Losses (slice + heads)

    L = L_rank + lambda_feat * L_feat

- L_rank: 17-way cross-entropy. Label = index of the target's token among the 16 candidates, or abstain if not among
  them. Weight w_fork at fork positions, w_ok << w_fork at non-fork positions (they teach "no false alarms").
- L_feat: 1 - cosine(feature head output, target's L25 / L33) at the positions before the fork (where the redraft
  reads them), weighted higher near the fork.
- Trained parameters: LoRA on the kept layers, alpha, u, the two feature heads. Target weights frozen (4-bit).

### 4.3 Drafter adaptation (stage 2)

Fine-tune DFlash to redraft from a fork with slice-estimated taps for the drafted prefix: the lag / reveal training
already in hspec/lagtrain.py, with the stale part of the context filled by slice features instead of nothing.

### 4.4 Stopping rule

Fit tau' (per level) and the calibration of alive_l on held-out online anchors. No learned network.

## 5. Evaluation

- Benchmarks: GSM8K, MATH-500, HumanEval, MT-Bench; batch 1, A100.
- Metrics: accepted per verification; tokens/s wall-clock; bytes streamed per accepted token; losslessness check.
- Baselines: DFlash; DDTree-style token trees; copy refresh at the end of the block (our earlier pipeline);
  QuantSpec-style quantized-target hierarchy; others named in the note (DLoop, ReAnchor) once checked.
- Ablations: fork restart vs end-of-block restart; slice vs full copy; ranker vs full LM head; fork-weighted vs
  uniform loss; abstain on / off; k (number of kept layers); copy-side pruning on / off; stop rule vs fixed depth.

## 6. Order of work and decision points

1. Restricted fork accuracy of the full 4-bit copy (exp99 restricted to DFlash's top-16), offline and online anchors:
   the bar for the slice.
2. k-sweep, untrained and trained (small budget): restricted fork accuracy, abstain rate, L25 / L33 cosine, pass
   time. Decision: a k with fork accuracy close to the full copy at <= ~1.5 ms, or fall back to the full copy.
3. tau' with slice features vs full-copy features vs real target features (the redraft's input quality).
4. Offline simulator of the full loop with measured costs (verify curve, slice, drafter, escalation rate).
5. Stage-2 drafter adaptation.
6. Real loop at batch 1 vs baselines.

## 7. Claims

1. Block drafters fail at forks that need computation over their own drafted tokens; same-information alternatives
   cannot fix them (RQ1).
2. Verifier-wide, copy-narrow block trees with copy-side pruning and an analytical stopping rule (method).
3. The intermediate model in a speculative hierarchy should be distilled against the drafter's failures: a ranker over
   the drafter's candidates built from target layers, evaluated by accepted tokens per streamed byte (main
   contribution).
