# Method: chained block drafting from the target's own early layers (partial-pass drafting)

Status: design document (2026-11-05). Every number without "est." is measured (experiment ids given). Estimates come
from the cost model in section 6 and are only as good as its assumptions.

## 0. Summary

A block drafter (DFlash) needs target computation over the tokens it drafted itself, not only the tokens. Instead of
a second model (the 4-bit copy), the target's own first k layers run over each drafted block (a partial pass, exact,
on the real KV cache). One partial pass gives two things:

1. features for the in-flight positions, after a small learned lift that turns the exit state h_k into the context
   features the drafter reads, so the drafter can draft the next block before any verification;
2. the fork: the exit head (trained by LayerSkip) locates the drafter's first error and proposes the fork tokens.

The drafter chains up to B blocks this way; the target verifies the whole chain (as a tree, all branches) once. The
target is an exit-trained model (LayerSkip Llama3-8B), frozen. Lossless: every emitted token is checked by the full
target.

## 1. Evidence this design rests on

| finding | number | source |
|---|---|---|
| block drafters need target computation over drafted tokens | real features at in-flight positions: 13.81 vs 8.50 tokens / verify (Qwen) | status 2026-11-01 |
| exit training puts next-token information into shallow layers | linear probe t+1 at layer 4: LayerSkip 0.59, plain Llama3 0.33, Qwen3 0.31 | exp87, llfix2 |
| partial pass is cheap | first 4 layers + exit head: 3.1 ms = 0.16 of a full pass (19.0 ms); k = 8: 5.4 ms (0.28) | bench_partial |
| raw exit states cannot be fed to the drafter | 0.09-0.14 of the gap (LayerSkip, k = 4 / 8); dataset mean is worse than nothing | exp106 |
| a closed-form linear lift works | share of the gap: k = 4: 0.56-0.60, k = 8: 0.78, k = 16: 0.96-0.98 (LayerSkip) | exp106, exp107 |
| the lift holds over long chains | share at m = 4 / 8 / 16 / 32 provisional positions: 0.60 / 0.59 / 0.59 / 0.55 (k = 4), 0.78 / 0.78 / 0.78 / 0.75 (k = 8) | exp107 A1 |
| richer linear inputs do not help | stacked shallow layers +0.00, verified deep context +0.00-0.05 | exp107 A2 |
| no single tap is critical for the Llama drafter | one tap estimated, others real: >= 0.90 for every tap (L15 the most sensitive) | exp107 A3, l31taps |
| there is a reliability signal | share by lifted-confidence tercile (k = 4): 0.47 / 0.64 / 0.62 | exp107 A3 |
| the native exit head resolves clear forks | top-1 / top-3 at the drafter's first error: k = 4: 0.78 / 0.87, k = 8: 0.83 / 0.92; linear lift only 0.34 / 0.49 | exp107 A4, lsforks |
| near-ties are not resolved | 6% of forks; exit top-3 0.09 (k = 4) / 0.38 (k = 8) | exp107 A4 |

Caveat on all drafter-side numbers on LayerSkip: they use the original Llama-3.1 DFlash, not adapted to LayerSkip
(real tau 3.5 per block of 10). Shares and fork rates will move once the drafter is trained.

Control (why not Qwen): with the same linear lift Qwen3-8B reaches 0.33 (k = 4) / 0.46 (k = 8), its deep tap L25
alone costs 21%, the lifted confidence carries no signal, and no resolver gets above top-3 0.40 at forks (exp107).

## 2. Components

| component | what it is | cost per call (CUDA graphs, batch 1) |
|---|---|---|
| target | LayerSkip Llama3-8B, bf16, frozen | full verify 19.0-19.8 ms, flat from 16 to 64 tokens |
| partial pass | the target's first k layers + final norm + LM head (exit head) over the new block, on the real KV cache | k = 2: 2.0 ms, k = 4: 3.1 ms, k = 8: 5.4 ms |
| lift | small learned network, h_k -> drafter context features + reliability | est. 0.3 ms |
| drafter | DFlash, fine-tuned on LayerSkip; verified positions read real taps, in-flight positions read the lift | 3.3 ms (block 10) |
| controller | fork location, branching, continue/verify decision | negligible |

Default k = 4; k = 8 as the conservative setting (untrained it already gives 0.78).

### 2.1 Lift

- Inputs per in-flight position: h_k (exact). Token embedding. Optional (ablation): cross-attention to the real deep
  features of the last W verified positions. exp107 says verified context adds little linearly; whether it helps a
  nonlinear lift is an open question, so it stays an ablation, not part of the core.
- Body: 2-4 narrow blocks (width ~1024), bidirectional over the in-flight span (its tokens are known), or a per-position
  MLP if the span attention does not pay.
- Outputs:
  - the missing taps (L8, L15, L22, L29 for k = 4), residual on the closed-form ridge map (initialised from exp107's
    maps, so training starts at 0.6). Capacity weighted toward L15 (most sensitive tap, exp107 A3).
    Alternative output: the fused drafter input hidden_norm(fc(taps)) directly (ablation).
  - a reliability scalar per position (trained to predict the drafter's loss increase); used by the gate.
- Not used for forks: the native exit head is far better (0.87 vs 0.49 top-3).

### 2.2 Drafter

- DFlash initialised from z-lab/LLaMA3.1-8B-Instruct-DFlash-UltraChat (block 10, taps 1/8/15/22/29), fine-tuned on
  LayerSkip text. Block 16 as an ablation (more tokens per block at the same partial-pass cost).
- Context: real taps for verified positions; lift outputs for in-flight positions, plus one learned "provisional"
  embedding, scaled by a gate g = sigmoid(w * reliability).
- No extra drafter layers.

### 2.3 Controller (rule-based in v1)

- Fork: first position of the new block where the exit head's top-1 differs from the draft, or its confidence falls
  below a threshold.
- Branches at the fork: the exit head's top-1..3 (count chosen by confidence); all branch blocks in one batched
  drafter call.
- Continue / verify: continue while the chain depth < B_max and the lift's reliability over the last block is above a
  threshold; otherwise verify. v2: learned gate on offline rollouts.

## 3. One round

1. The target verifies; real features up to anchor s.
2. DFlash drafts block 1 from real features.
3. Partial pass over block 1: exit-head fork + top-k; lift features for its positions up to the fork.
4. DFlash drafts the next block(s) from the fork (one batched call for the branches), reading lift features.
5. Repeat 3-4 while the controller continues (B_max = 4 in v1).
6. The target verifies the whole tree in one pass. Lossless; the fork tokens and features only steer drafting.

## 4. Training

Data: LayerSkip greedy generations (~20k sequences, gen_traindata.py), held-out evaluation prompts as in exp8.

- Stage 0 (no gradient): ridge maps per tap from h_k (done, exp106/107), used to initialise the lift.
- Stage 1, drafter adaptation: DFlash fine-tuned on LayerSkip with real features (as job_ls_train), then with mixed
  contexts: random anchor, the last m positions provisional (m uniform over 0..B_max * block), lift features from the
  stage-0 maps. Loss: block cross-entropy with DFlash's position decay.
- Stage 2, lift + drafter jointly: lift trained through the drafter (CE on the next block) + feature consistency
  (normalised MSE to the real taps) + reliability regression; drafter LoRA or full.
- Stage 3, on-policy chains: rollouts where block i+1 is drafted from the model's own block i, the fork chosen by the
  exit head, branches as at inference. Fixes the teacher-forcing gap (exp107 used true tokens).

## 5. Baselines and ablations

- Baselines: DFlash (adapted to LayerSkip), DDTree on the same drafter, LayerSkip self-speculation, autoregressive
  decoding; all in the same harness, tokens/s at batch 1.
- Ablations: provisional features off (same fine-tune, tokens only: the drafter trained to chain without target
  information), raw h_k vs ridge vs learned lift, k = 2 / 4 / 8, B_max = 1..6, branches 1 / 2 / 3, block 10 vs 16,
  lift with / without verified context, plain Llama3-8B and Qwen3-8B as non-exit-trained targets.

## 6. Efficiency model and estimates

Per verify round with B chained blocks:

    time   = B * D + (B - 1) * (E_k + l) + F
    tokens = t1 + sum_{i=1..B-1} c^i * g
    g      = removed + s * (t1 - removed)

D drafter call, E_k partial pass, l lift, F verify, t1 tokens of a block drafted from real features (incl. the bonus
token), s the share of the gap the lift recovers, c the probability the fork before a block is resolved, removed the
tau of a block with no in-flight features (~1.2, exp107).

Inputs: D = 3.3 ms, E_4 = 3.1, E_8 = 5.4, l = 0.3 (est.), F = 19.0 ms (measured in our HF harness), t1 = 6.5 for the
fine-tuned block-10 drafter (est.; the adapted drafter reached 6.8 in exp13). Plain DFlash: 6.5 / 22.3 ms =
0.29 tok/ms.

| scenario | k | s | c | B = 2 | B = 3 | B = 4 | B = 6 |
|---|---|---|---|---|---|---|---|
| untrained lift (today's numbers) | 4 | 0.59 | 0.87 | 1.21x | 1.30x | 1.33x | 1.29x |
| untrained lift (today's numbers) | 8 | 0.78 | 0.92 | 1.25x | 1.36x | 1.40x | 1.40x |
| trained lift, target | 4 | 0.85 | 0.95 | 1.41x | 1.64x | 1.78x | 1.91x |
| strong lift | 4 | 0.95 | 0.97 | 1.48x | 1.77x | 1.95x | 2.15x |
| ceiling (real features, perfect forks) | 4 | 1.00 | 1.00 | 1.54x | 1.87x | 2.10x | 2.40x |

(est., all relative to plain DFlash with the same drafter; limit for many blocks with s = c = 1: 3.3x)

Reading:
- Today's numbers, untrained, already give ~1.3-1.4x over DFlash at 3-4 blocks. With weak fork resolution, longer
  chains stop paying (k = 4 untrained: 1.33x at B = 4, 1.29x at B = 6), so B is a per-round decision.
- The realistic target after training is ~1.6-1.8x at B = 4; 2x needs s >= 0.95 and c >= 0.97, or 6 blocks.
- Levers beyond the table: trees inside each block (DDTree gave 1.13x over DFlash on this target, exp8), block 16
  (same partial-pass cost, more tokens per block), k = 2 (2.0 ms) if its lift is good enough.
- Sensitivity to the verify cost: the gain comes from skipping verifies, so it shrinks if the verify is cheaper.
  With F = 12.7 ms (the optimised Qwen measurement), the trained-lift row at B = 4 drops from 1.78x to ~1.5x (est.).
  The paper numbers must come from one harness, ideally vLLM.

## 7. Risks and what would kill it

| risk | test | kill criterion |
|---|---|---|
| trained lift does not beat the linear one | stage 2, share at k = 4 | < 0.7 at k = 4 (then use k = 8 untrained, ~1.4x) |
| shares measured with the mismatched drafter do not transfer | repeat exp106 / exp107 with the stage-1 drafter | share at k = 8 < 0.6 |
| forks get harder with an adapted drafter | exp99 / exp107 A4 with the stage-1 drafter | exit top-3 at forks < 0.8 at k = 8 |
| near-ties break chains | share of chain breaks at near-ties in on-policy rollouts | > 30% of chain breaks |
| the "features off" ablation gets most of the gain | stage 1 vs stage 2 | provisional features add < 0.15x |
| verify cheaper in a serving engine | vLLM measurement of F, D, E_k | gain at B = 4 < 1.3x |

## 8. Open decisions

1. k = 4 (cheaper, needs a trained lift) or k = 8 (0.78 untrained).
2. Block 10 (inherited drafter) or 16.
3. Lift output: separate taps or the fused drafter input.
4. Whether the lift attends to verified deep context (ablation only for now).
5. Harness for the headline speed numbers (HF graphs vs vLLM).

## 9. Update 2026-11-05 (evening): decisions and variants to try

### 9.1 Decisions
- Core is single-path chaining. Trees are not part of the method; DDTree is reported as a complementary add-on
  (DFlash vs DFlash + DDTree, ours vs ours + DDTree), with the credit going to DDTree. The exit-scored tree idea is
  dropped from the core.
- Target: LayerSkip Llama3-8B (exit training is what makes the early layers informative; Qwen3 and plain Llama3 are
  controls only).
- First step before any variant: adapt the existing z-lab Llama DFlash to LayerSkip with the normal DFlash recipe and
  re-check the base measurements with the adapted drafter (section 9.4).

### 9.2 Where the learning can live (to compare once budget allows)

| variant | what learns to turn the partial pass into drafter context | extra runtime cost |
|---|---|---|
| b. drafter reads raw h_k | DFlash itself, through a provisional input path | none |
| c. draft-mode adapters | LoRA / adapters on LayerSkip's first k layers, active only in the draft pass (verify uses the plain target: lossless; the draft pass's KV is not reused) | ~LoRA overhead |
| d. thin drafter + adapters | adapters produce a compact d-dim context (d ~1024-2048); verified taps go through a down-projection to the same d; half-width block decoder | drafter call ~3.3 -> ~1.6 ms (est.) |
| a. separate lift | small network between partial pass and drafter (initialised from the exp107 ridge maps) | ~0.3 ms |

Evidence for b: exp13, a drafter fine-tuned to read only half-depth LayerSkip features kept 0.91 of its acceptance at
lag 16. Risk: at k = 4 the drafter has to infer much more; on Qwen the drafter could not learn mid-layer states
(exp76). Decision rule: b >= 0.85 share at k = 4 -> keep it simple; 0.6-0.8 -> add c; below the ridge lift -> a or c.

Cost model for d (trained case s = 0.85, c = 0.95, k = 4, est.):

| drafter | B = 2 | B = 3 | B = 4 | B = 6 |
|---|---|---|---|---|
| current width (3.3 ms) | 1.42x | 1.67x | 1.82x | 1.96x |
| half width, same accuracy (1.6 ms) | 1.62x | 1.95x | 2.17x | 2.41x |
| half width, block tokens -10% | 1.46x | 1.76x | 1.96x | 2.18x |
| half width, block tokens -20% | 1.30x | 1.57x | 1.75x | 1.94x |

### 9.3 Drafter changes supported by the data
1. Provisional input path (projection + learned provisional embedding + gate from a reliability signal): needed.
2. Distance-to-last-verified embedding: cheap.
3. Exit-head beliefs (top-k token embeddings / probabilities at in-flight positions) as an extra input: the exit head
   is right at ~80% of clear forks vs 34% for the linear lift; cheap linear pre-test possible.
4. Block 16 (from a different start than the block-10 Llama drafter): same relative gain (1.75x vs 1.78x at B = 4,
   est.) but +28% absolute tokens/ms and a stronger, standard baseline.
5. Keep the tap layout; train with tap dropout / noise on provisional positions (one tap estimated >= 0.90, all
   estimated 0.59).
6. No extra drafter depth.
Stretch: online lift / adapters updated from the verifier's real deep features after every round (free supervision;
neighbour: Online Speculative Decoding; risk: exp68 found the copy's errors barely learnable online).

### 9.4 First step: adaptation plan and budget (estimates)
- Start: z-lab/LLaMA3.1-8B-Instruct-DFlash-UltraChat (trained by z-lab on UltraChat + ShareGPT regenerated by
  Llama-3.1-8B-Instruct, block 10; paper recipe: 6 epochs, AdamW lr 6e-4, cosine, warmup 0.04, clip 1.0, max length
  3072, 512 random anchors per sequence, gamma 5 for block 10, embeddings / LM head frozen).
- Data: ~50k LayerSkip greedy sequences (generate up to 100k, train on 50k first), minimum response length (the base
  model stops early, ~215 tokens per sequence observed), broader prompt mix (GSM8K train, MBPP, Alpaca, plus
  UltraChat / ShareGPT prompts in the plain "Question: ... Answer:" format), eval sets held out; stored outside
  /scratch-shared.
- Fine-tune: 3 epochs, 512 anchors per sequence, block 10, gamma 5, lr ~5e-5 to 1e-4, online features, checkpoint
  every epoch.
- Budget (first principles, ~6e13 FLOPs per 1k-token sequence-epoch, ~100 TFLOP/s effective on an A100, 128 SBU per
  GPU-hour): calibration ~130, data ~500, fine-tune ~3,200, eval ~650, margin ~1,300 -> ~6,000 SBU (~10-11k with
  100k data). Calibrate with a 5k-sequence epoch first and rescale.
- Later stages: core method (b, c at k = 4, on-policy chains, ablations) ~25-30k SBU incl. margin; full incl. thin
  drafter and seeds ~45-55k SBU.
- Go / no-go after adaptation: held-out acceptance >= ~5.0 per block of 10; exp106 / exp107 / exp99 rerun with the
  adapted drafter: linear-lift share near 0.59 (k = 4) / 0.78 (k = 8) or better and stable over chain length; exit
  head top-3 at the (harder) forks >= ~0.8 at k = 8.
