# Feature-refreshed drafting: methodology draft (13 October 2026)

Working draft. Numbers are from our offline harnesses (exp27, exp29, exp56) on Qwen3-8B with the z-lab DFlash drafter (block size 16) unless stated otherwise, and they are there to support the design choices, not as final results.

## 1. Setting and motivation

Block diffusion drafters such as DFlash [DFlash] predict a block of 16 tokens from the target's hidden features at one anchor position. The tokens inside the block never receive target features, so the drafter has to guess the rest of the block from the anchor alone. Most of the recent work on block drafters (D2SD, DEdit, xPress, Domino, DSpark, DFlash 2) treats the resulting loss as a dependency problem between the drafted tokens and adds token conditioning, refiners or editing passes. The small experiments in Section 2 show that the larger part of the loss is something else: the target's computation over the drafted tokens is missing. The methodologies in Sections 4 to 6 are built on these findings.

## 2. Preliminary experiments

All numbers in this section are offline measurements on target trajectories (Qwen3-8B, z-lab DFlash drafter, block size 16) unless stated otherwise.

**P1. Accuracy decays along the block.** The drafter's top-1 accuracy is 0.93 at offset 1 and 0.37 at offset 15 (T=0). The first positions are reliable, the second half of the block is mostly lost.

**P2. Trees and dependency fixes saturate.** DDTree grows only from 7.71 to 8.10 accepted tokens when the tree goes from 64 to 512 nodes. Our own dependency fixes (block trees, redrafting at the uncertain position, a dependency head; exp15, exp18, exp19) all landed at +5 to 8% over DDTree. More candidates built from the same drafter marginals do not help much, so the right token is often not reachable from what the drafter knows.

**P3. What fixes a stop (exp26).** At every position where the draft stops, we redrafted from that position with different information about the correct prefix:

| Redraft with | Stops fixed (DDTree-64, T=0) | Stops fixed (T=1) |
|---|---|---|
| the correct prefix tokens only | 49% | 41% |
| features of the target's first 18 layers on the prefix | 57% | 51% |
| real target features on the prefix | 80% | 68% |

The correct tokens fix about half of the stops, which is the most that a dependency fix can give. The real features fix much more. 38% of the stops are at positions where the target is nearly certain (entropy below 0.1), and the real features fix 95.5% of these, so these are not hard tokens, the drafter just lacks the information. DFlash's own ablation points the same way: removing the target features drops MATH-500 from 7.87 to 4.61 accepted tokens.

**P4. A 4-bit copy gives features as good as the real ones (exp27).** Restarting at the most likely stop positions with features from a copy of the target instead of the target itself (plain DFlash 7.98):

| Feature source | Accepted tokens per verify |
|---|---|
| real target features | 12.54 |
| 4-bit copy | 12.53 |
| 3-bit copy | 12.09 |
| 2-bit copy (round to nearest, untrained) | 8.44 |

The target computation that the drafter needs can be approximated by a low-bit copy without loss down to 3 bits.

**P5. Where the gain comes from.** Ablation of the copy-corrected restart (T=0, all trees of 31 nodes or less):

| Variant | Accepted tokens |
|---|---|
| plain DFlash | 7.98 |
| copy's correction only (copy as an intermediate verifier) | 8.82 |
| correction + redraft from tokens only | 10.42 |
| correction + redraft with the copy's features | 14.25 |

About 13% of the gain is the correction (what quantized-verifier hierarchies such as QuantSpec and ML-SpecQD already use), 25% is redrafting and 61% is the copy's features. The same split holds at T=1.

**P6. Cheaper substitutes without training fail.** A small learned feature emulator (50M parameters) reached cosine 0.38 to the real features and fixed less stops than tokens alone. Pruning the target without training to 24 or 18 layers loses most of the gain, and at 18 layers it is gone. Running the refresh only in uncertain rounds lowered the acceptance more than it saved.

**P7. Verification width is almost free, sequential passes are not.** Measured vLLM verify latency at batch 1 is 12.7 ms for 1 token, 13.3 ms for 16, 14.1 ms for 32, 14.2 ms for 64, 15.8 ms for 128 and 19.0 ms for 256 tree nodes. A 4-bit copy pass costs 3 to 6 ms and a drafter pass 2.6 to 3.7 ms, every time it runs. In our HF implementation with CUDA graphs the copy cost 15.5 to 17.7 ms against 22.0 to 23.6 ms for the verify, so there the method is around break-even (0.93 to 0.97x); the per-layer overhead outside the matrix multiplications does not shrink with quantization.

**P8. Iterating the refresh helps, but sublinearly (exp29).** Repeating draft, copy and redraft inside a round raises the accepted length from 11.3 at one iteration to 19.0 at six (4-bit copy, T=1), but each iteration costs a full drafter and copy pass. The best fixed or adaptive schedule reached 1.28x over DFlash.

**From findings to design.** P3 to P5 motivate giving the drafter approximate target features for the drafted tokens (the refresh, Section 4). P7 says that the extra work should go into width rather than into more sequential passes (Section 5). P8 says that depth still pays sometimes, so the number of iterations has to be decided per round (Section 6). P6 and the copy cost in P7 motivate the cheap-copy analysis in Section 8.

## 3. Common components

All methodologies below share the following components.

**Copy.** A low-bit copy of the target (4-bit weight-only in the main setting). It processes the drafted block once and returns (i) the features at the drafter's tap layers (1, 9, 17, 25, 33) for the drafted positions and (ii) its next-token prediction at every position. The copy attends to the target's KV cache for the committed prefix and only computes the 16 drafted tokens itself (KV sharing). This raised copy-target agreement on drafted blocks from 0.940 to 0.954 for the 4-bit copy and the accepted length of a trained 2-bit copy from 12.70 to 13.02 (exp56).

**Restart.** A restart is a new drafter call from an anchor inside the drafted block, conditioned on real features before the round's anchor and the copy's features after it. Restarts of one round are packed into one batched drafter call.

**Verification and reset.** The target verifies the first block and all restarted chains as one tree in a single pass. Every emitted token is the target's argmax (T=0) or follows the standard lossless tree acceptance (T>0), so the method is lossless. After verification the drafter's cache is cropped back to the round start and its context is rebuilt from the target's real hidden states of the accepted tokens. The copy features are used only inside the round, they never enter the committed context (confirmed in hspec/refresh.py).

**Cost model.** We use vLLM component costs: verify 13.3 ms for 16 tokens, 14.1 ms for 32 and 14.2 ms for 64 tree nodes, drafter 2.6 to 3.7 ms, 4-bit copy 3 to 6 ms (AWQ, Marlin kernels). Because verification is memory bound at batch 1, a wider tree is almost free, while every extra sequential pass of the copy or the drafter costs its full latency. Speed is reported as accepted tokens per millisecond relative to plain DFlash.

## 4. Methodology A: single-pass refresh

One copy pass and one restart per round.

1. DFlash drafts block 1 from the real target features.
2. The copy runs over block 1 (Section 3).
3. At the first position where the copy disagrees with the draft, the copy's token replaces the draft token and the drafter restarts from there with the copy features (copy-corrected restart, "pc"). Optionally, extra restarts are placed at the positions where the drafter is most likely to stop, estimated from its own confidence ("conf2" = two such restarts).
4. The target verifies block 1 and the restarted chains as a tree, and the drafter is reset to real features.

In the real decoding loop (greedy, 40 prompts from GSM8K, MATH-500, HumanEval and MT-Bench) the accepted length went from 5.86 for DFlash to 10.17 with the corrected restart and to 10.79 with two extra confidence restarts, against 7.52 for DDTree-64. In the ablation about 13% of the gain comes from the correction itself, 25% from redrafting and 61% from the copy's features (P5), which is the part that the existing quantized-verifier hierarchies [QuantSpec, ML-SpecQD] do not use.

## 5. Methodology B: wide refresh

Methodology A spends the copy pass on one correction. At batch 1 the verify cost hardly grows with the tree width, so B uses the same single copy pass for many restarts that are verified together.

From one copy pass we build the following restart chains, all in one batched drafter call:

- w2, w3: the copy's second and third most likely tokens at its first disagreement, since the copy is not always right at the position it doubts;
- d2, d3: corrections at the copy's second and third disagreement positions, in case the first one was the copy's mistake;
- cK: restarts at the K most likely stop positions of the drafter.

A policy is a union of these chains; the best one in exp56 is w2+d2+c2. Its tree has around 50 to 60 nodes, which costs about one 64-node verify.

| Setting (4-bit copy) | Accepted per verify | Projected speed vs DFlash (copy 3 ms) |
|---|---|---|
| T=0, w2+d2+c2 | 14.68 | 1.40x |
| T=1, w2+d2+c2 | 12.45 | 1.29x |
| T=1, best iterative rule from exp29 (for comparison) | 14.2 | 1.28x |

One wide step reaches the same speed as the best iterative method at T=1 with a fixed and predictable cost per round, which makes it the easier one to implement in a serving engine. The tree is a static shape per round, so it fits the tree attention kernels that already exist for EAGLE and DDTree.

## 6. Methodology C: adaptive iterative refresh

C repeats the refresh inside a round (draft, copy, redraft with copy features, copy again ...) before verifying, and decides per round how many iterations are worth their cost. In exp29 the accepted length rises with the number of iterations (11.3 at one iteration to 19.0 at six, 4-bit copy, T=1), but sublinearly, while every iteration adds a full drafter and copy pass. A fixed number of iterations is therefore never the best choice, and the question is when to stop.

**Stopping rule.** After iteration r the current best chain has an estimated probability of being accepted up to each position, computed from the drafter's confidence and the copy's agreement with the chain. Let A_r be the expected accepted length of the chain under this estimate and c the cost of one more iteration (drafter + copy). We continue while

    (A_{r+1}^est - A_r) / c  >  A_r / T_round(r)

that is, while the expected gain per millisecond of one more iteration is larger than the average accepted tokens per millisecond of the round so far. A_{r+1}^est is predicted from the positions where the drafter and copy still disagree or the drafter's confidence is low. A simpler threshold version stops once the estimated probability that the chain is correct up to the end of the block is above theta; theta = 0.8 gave the best result in exp29 (14.2 accepted, 1.28x at T=1).

**Adaptive iteration in A and B.** The same rule decides the work inside one pass for A and B. If the copy agrees with the whole draft and the drafter's confidence is high, no restart is made and only block 1 is verified. The number of restarts in B (how many of w2, d2, cK) is chosen by the same expected-gain-per-millisecond comparison, using the tree verify costs from the cost model. This makes the width and the depth of the refresh two settings of one decision instead of two separate methods.

What is still open: exp29 stored only aggregate numbers, so the ceiling of a perfect stopping rule (an oracle that knows the accepted length after each iteration) is not measured yet. It needs per-round logging and one more offline run.

## 7. Overlapping the copy with the verifier

The copy and drafter work of the next round can run on a second CUDA stream while the target verifies the current round, on one GPU. With our HF implementation and CUDA graphs (bnb4 copy, which is slow at batch 1) the target verify alone took 22.4 ms, the small models alone 39.4 ms and both together 52.6 ms, so 23% of the small work was hidden. With a fast 4-bit kernel the small work is shorter than the verify and a larger share should be hidden. Including partial overlap, the projected speed is about 1.5x at T=0 and 1.4x at T=1 over DFlash. A measured speedup needs the method inside vLLM, which is the main engineering item.

## 8. Analysis: how cheap can the copy be

The copy cost is the main overhead, so we studied whether a cheaper copy can give the same features. The drafter's accepted length with each copy (exp27, restarts at confidence positions, conf2 / all restarts / pc+conf2):

(These runs use the later exp27 settings, so the 4-bit numbers differ from P4; the comparison is within this table.)

| Copy | conf2 | all | pc+conf2 |
|---|---|---|---|
| 4-bit | 11.90 | 13.05 | n/a |
| 2-bit, untrained | 8.34 | 8.89 | n/a |
| 2-bit, trained for the drafter (LoRA) | 11.19 | 12.30 | 12.15 |
| 18 of 36 layers, trained | 10.45 | 11.55 | 10.82 |
| only the drafter's tap layers (8 layers), untrained | 8.30 | 8.86 | 8.30 |
| tap layers, trained (LoRA) | 9.20 | 10.26 | 9.29 |
| tap layers + trained MLP bridges for the removed spans | 9.69 | 10.81 | 9.83 |

Three ways to make the copy cheaper were tested.

1. **Copy trained for its role** (exp55). LoRA on a cheap copy with a loss that matches the drafter's predictions with real features (top-32 KL) plus next-token agreement with the target. This recovers most of the 2-bit copy (8.34 to 11.19) but not the layer-pruned copies.
2. **Tap layers with bridges** (exp57). The copy keeps only the layers the drafter reads and replaces each removed span of 7 layers by a small module. The module is initialised in closed form (ridge regression of the span's residual update, following Ghosted Layers [2605.15491] and SHIFT-LLM [2608.25068]) and trained with hidden-state, drafter and agreement losses. Without bridges the copy is unusable (drafter agreement 0.03), the closed-form bridges bring it to 0.20 to 0.24 and training to 0.31. The accepted length is 9.69, better than LoRA on the same layers but still 81% of the 4-bit copy. A sequential fit, where every span is fitted on the copy's own inputs, is a better start than the independent fit of the original papers, since the four cuts compound.
3. **Drafter that trusts approximate features** (exp58). Following DFlow [2609.06498], the copy features after a restart's anchor were given to the drafter as gated hints at the mask positions, with a small trained gate and a frozen drafter. This gave no gain with the 4-bit copy (10.60 to 10.63) and a loss with the bridge copy (7.46 to 7.38). The frozen drafter can not use signals at the mask positions that it never saw in training.

The conclusion of this analysis is that the drafter needs features close to the real ones. A cheaper copy only pays off when it keeps about 93% of the 4-bit copy's accepted length (at half its cost, from the cost model), and none of the cheaper copies reached that. The 4-bit copy is therefore the copy of the main method, and the analysis explains why the cheaper alternatives fall short.

## 9. Evaluation plan

- Targets: Qwen3-8B (main), Qwen3-4B, and one LLaMA model if a DFlash drafter is available.
- Data: GSM8K, MATH-500, HumanEval, MBPP, MT-Bench, Alpaca.
- Temperatures: T=0 and T=1 (lossless tree sampling).
- Baselines: DFlash, DDTree, DFlow, a quantized-verifier hierarchy (QuantSpec style, the copy only verifies), and EAGLE-3 where available.
- Metrics: accepted tokens per verify, projected speed from the cost model, and measured speed in vLLM for the final method.
- Ablations: correction only vs redraft vs copy features (A); every restart type (B); fixed iterations vs stopping rule vs oracle (C); KV sharing on/off; overlap on/off.

## 10. Limitations

- The speed numbers are projections from component costs; our HF implementation is around break-even because the per-layer overhead does not shrink with quantization.
- The 4-bit copy needs about 5 GB of extra memory, which matters at larger batch sizes, and at larger batch sizes the verify is no longer memory bound, so the free width of Methodology B shrinks.
- The copies were trained with small data (2,400 sequences, one epoch). The cheap-copy conclusions hold for this budget, a larger training run could move them.
- All results are with Qwen3-8B and its DFlash drafter so far.
