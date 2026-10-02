# DDTree critique: weaknesses, hidden assumptions, research angles

Paper: Ringel & Romano, "Accelerating Speculative Decoding with Block Diffusion Draft Trees", arXiv 2604.12989v1 (Apr 2026).
Code: github.com/liranringel/ddtree, vendored at third_party/ddtree_official (commit c96427a).

## Sources and how they were read

- arxiv.org, the project page and alphaxiv were blocked for curl in this sandbox (proxy 403). The paper text was read through WebFetch on arxiv HTML and on the project-page PDF (liranringel.github.io/ddtree/DDTree.pdf). WebFetch returns a model-summarised view with short quotes, not the raw text. Quotes below come from those calls. Treat exact wording, and the section numbers of quotes, as **needing a check against the PDF** before anything is cited in writing.
- The code (ddtree.py, dflash.py, benchmark.py, run_benchmark.sh, make_latex_table.py, model/utils.py) was read in full. Code-derived claims are marked [code] and are reliable.
- A third-party review page (pith.science/paper/2604.12989) was read. Its claims are marked [pith]. Some of its baseline listing (EAGLE-3, Fast-dLLM-v2, FailFast) does not match the paper, so it is unreliable.
- Tags: [paper] = the paper says this; [code] = the official code does this; [ours] = our measurements; [inf] = my inference, not verified.

---

## 1. Method assumptions

### 1.1 Factorized surrogate Q
- [paper] The drafter outputs per-position marginals {q_i}, i = 1..L, from one block-diffusion pass. These are combined as "Q(y_{1:L}|c,b) := ∏_{i=1}^L q_i(y_i|c,b)" (Eq. 2). The target's true distribution is the path-conditioned autoregressive one (Eq. 1).
- [paper] The objective is to maximise expected acceptance length under Q, not under the target p (Eq. 6).
- [paper] Prop. 1 / Eq. 8: expected acceptance length "equals ∑_{u∈T} q(u|c,b)", where q(u) = ∏_{i≤|u|} q_i(u_i) (Eq. 7).
- [paper] Remark 1: "The result is exact for the surrogate objective induced by the factorized draft distribution Q(·|c,b), not for the true target-model distribution."
- Hidden assumptions [inf]:
  - (a) **Independence across positions.** q_i(u_i) is a marginal and is never conditioned on u_{<i} being the realised prefix. Two biases follow.
    - Along the drafter's mode path, the correctness of consecutive positions is positively correlated (once on track, a drafter tends to stay on track). Q then underestimates deep-chain probability.
    - Below a non-mode branch (rank > 0 at depth d), the marginals q_{d+1..L} were computed under the drafter's implicit mode context. They are reused unchanged, so Q likely overestimates acceptance under alternative branches.
  - (b) **Same candidate set at every node of a depth.** Each depth-d node expands into the same ranked list top-k(q_{d+1}) (code: `top_token_ids_np[depth, rank]` is independent of the parent) [code]. The tree is therefore a ranked sub-lattice of a product space. It cannot represent "after A comes X, after B comes Y", and it cannot represent shifted hypotheses (an insertion or deletion of one token moves every later token by one position). Diffusion marginals that blend two hypotheses produce incoherent cross products, and each of those uses up budget.
  - (c) **q estimates the right quantity.** Under greedy verification (T=0), node u at depth i is accepted iff u_i = argmax p(·|prefix). The quantity to rank by is P(u_i is the target argmax). It is not q_i(u_i), which is a calibrated estimate of p only if the drafter is calibrated. If p = (0.4, 0.35, ...), the argmax is accepted with probability 1 while q says 0.4. Because ranking mixes depths, Q is mis-scaled for T=0. **The tree is identical at T=0 and T=1** [code: `build_ddtree_tree(draft_logits[0], budget)` ignores temperature].
  - (d) **Static calibration.** Nothing in the method re-weights q from verification outcomes [paper and code].

### 1.2 Best-first heap and the optimality claim
- [paper] Prop. 2: the "top-B prefixes by probability form a valid draft tree" that "maximizes the expected acceptance length" under Q subject to |T| ≤ B. Sec. 4.3 gives the rank-tuple representation. Remark 2: "The heap stage costs O(B log B): ... at most B pops and at most 2B pushes."
- When it holds [inf]: (i) the objective is Q and not p (Remark 1); (ii) every node costs the same, so the constraint is a node count and not latency; (iii) prefix probabilities are monotone non-increasing along paths, which always holds for a product of probabilities, so validity is trivial; (iv) the drafter's distribution is fixed, with no re-drafting. Optimality says nothing about acceptance under the target. Per [pith], the paper also gives no correlation analysis between the Q-ranking and realised target acceptance.
- The cost model is a node count. Real verify latency is a step function of N (GEMM tile sizes, attention kernel), and at batch > 1 it is roughly linear in total tokens. **The optimal tree under a latency objective is not the top-B set for a fixed B** [inf].
- Top-k per depth [code]: `topk = min(budget, V)`. Each depth contributes up to B ranked candidates, so there is no real per-depth top-k pruning, which is good. But the D×B top-k plus log-softmax is copied to the CPU every round, which forces a GPU→CPU sync.

### 1.3 Budget choice
- [paper] Budgets tested were {16, 32, 64, 128, 256, 512, 1024} (App. B). Sec. 5.3: "speedup improves until peaking around budgets of 256 to 512". The optimal budget depends on "hardware platforms and implementations".
- [code, important] The **main table and Fig. 1 pick the best budget per (dataset, model, temperature) by measured speedup on the same evaluation set** (`make_latex_table.py`: `best_ddtree_method_key = max(ddtree_method_keys, key=speedup)`; Fig. 1 caption: "DDTree bars use the best tree-node budget for each dataset-model pair"). This is oracle hyper-parameter selection on test data. The bias is probably small, given the smooth curve near the 256-512 peak, but it is not zero, and with only 30 samples on AIME it is noisier. No fixed-budget row is reported.
- No per-round or per-request budget adaptation [paper, code].

### 1.4 Verification
- [paper] Sec. 4.4: "Starting from the bonus token b, we check whether the token selected by the target model at the current node matches one of that node's children." Sampling follows "the target model's own decoding rule, whether greedy or temperature-based sampling".
- [code] `posterior = sample(output.logits, temperature)` draws one target token per node. The walk follows matching children (`follow_verified_tree`). At T>0 this is lossless: each emitted token is a target sample. The acceptance probability at a node is Σ_{children c} p_T(c|prefix).
  - For a deterministic draft set this is the best available. It ignores q entirely, though. Multi-draft or rejection-style schemes for sampled candidates (SpecInfer-style multi-round rejection, traversal verification, UniVer-type) are not considered, and neither is any tree design aware of p_T. **The correction to the critique brief**: DDTree is not greedy-only. It runs T=1, with a lossless "sample-and-match" verifier.
- [code] The DFlash baseline uses a **greedy draft even at T=1** (`block_output_ids[:, 1:] = sample(draft_logits)` with default temperature 0), also sample-and-match. At T=1, DDTree's top-k tree partly compensates for this weak, deterministic DFlash chain, so the T=1 gain over DFlash partly reflects a weak baseline sampler [inf].

### 1.5 Tree attention cost
- [paper/App. B] DDTree's target uses PyTorch SDPA with a dense additive mask. "FlashAttention-2 does not support the required tree attention pattern." The drafter keeps FA2.
- [code] The mask buffer is (1, 1, N, max_length+N) in bf16, re-filled each round. SDPA with an arbitrary float mask falls back to the efficient/math kernels. The mask and the attention cost both grow with N × context [inf]. No kernel, CUDA graph or paged-KV integration exists.
- [code] Verify runs with `output_hidden_states=True` over all N nodes. All layers' hidden states are materialised for the full tree, but only the accepted path's features from `target_layer_ids` are used (memory and bandwidth waste that grows with B).
- [code] Commit: KV compaction by index_select per layer (an inline C++ op exists, but it is still a per-layer launch).

### 1.6 CPU heap overhead
- [code] The heap uses Python `heapq` on numpy, and the visibility matrix is an O(N²) Python loop over N rows. The GPU→CPU copy of the top-k logprobs blocks the stream every round, so the decode loop cannot be CUDA-graph-captured end to end.
- [paper] No timing breakdown is given, only Remark 2's asymptotic O(B log B).
- [ours] build+compile is 1.2 ms per round, rising to 1.7 ms at B=256. Verify is 42.5 ms. In HF eager this is about 2-3% of the round.

### 1.7 Block size and depth cap from DFlash
- [paper/App. B] Block size 16. [code] `draft_horizon = block_size - 1`; `depth_limit = draft_logits.shape[0]` = 15 drafted positions plus the root, so **at most 16 tokens per round whatever the budget**.
- [paper] Sec. 5.4 says full-block acceptance (16) "becomes substantially more common". That is direct evidence the cap binds.
- [ours] 15-17% of rounds hit the full 16-token depth.

---

## 2. Evaluation

| Item | What the paper or code does | Gap |
|---|---|---|
| Hardware | 8×H200, data-parallel sharding of prompts (App. B; `torchrun --nproc_per_node 8`) | One GPU type. No A100/consumer/MLX numbers. A community MLX port claims only "~10-15% faster than DFlash on code" (github humanrouter/ddtree-mlx, README snippet; unverified) |
| Batch size | 1 per GPU [code: tensors hard-coded with batch index 0] | No batch > 1, no throughput or serving numbers |
| Framework | HF Transformers + PyTorch, target SDPA (App. B) | No vLLM/SGLang, no CUDA graphs. Absolute latencies are framework-overhead dominated (see 4.6) |
| Temperature | 0.0 and 1.0 (Table 1, 60 entries) | No intermediate T, top-p or top-k sampling |
| Thinking mode | `enable_thinking=False` [code] | Qwen3 reasoning traces (the long-generation regime where SD matters most) are not tested |
| Lengths | `max_new_tokens 2048` (App. B, code) | No long context, no long generation. Tree-mask attention cost versus context is unmeasured |
| Datasets | MATH-500, GSM8K, AIME24/25 (30 each), HumanEval, MBPP, LiveCodeBench, SWE-bench Lite, MT-Bench (80), Alpaca; 128 samples otherwise (Table 2) | Small n, no variance or CI reported. SWE-bench Lite is used as a prompt set only |
| Models | Qwen3-4B, Qwen3-8B, Qwen3-Coder-30B-A3B with z-lab DFlash drafters | One model family. No dense model above 8B, no Llama |
| Baselines | AR and DFlash only. EAGLE-3 appears only via DFlash's cited claim ("outperforming strong autoregressive drafters such as EAGLE-3") with **no EAGLE-3 numbers in any table** | Missing: EAGLE-2/3 dynamic trees, OPT-Tree (discussed in related work, not run), Sequoia, SpecInfer, Medusa trees, DART (cited as concurrent), multi-draft/traversal verification at T=1. No ablation against simpler tree heuristics (fixed k-ary per depth, chain plus top-2 siblings) |
| Budget selection | Best budget per cell chosen on the test set (see 1.3) | Optimistic. No held-out budget selection |
| Metric | Mean over responses of per-response TPOT; speedup = AR_TPOT / method_TPOT [code]. AR baseline and DFlash take the best of the SDPA and FA2 runs; DDTree is SDPA-only [code] | Mixed attention kernels: DDTree is mildly disadvantaged, which is fair. Unweighted per-sequence mean, not token-weighted throughput |
| Overhead | Asymptotic only (Remark 2) | No stage timing, no draft/build/verify split |
| Correctness | No output-equivalence test reported ([pith] also flags this) | Minor. The code is straightforward |

Headline numbers (T=0, [paper] Table 1 via WebFetch; re-check against the PDF):
- Qwen3-8B MATH-500: DFlash 5.56× / τ 7.79 vs DDTree 7.52× / τ 10.73.
- Qwen3-8B GSM8K: 4.78× / 6.57 vs 6.75× / 9.54.
- Weakest settings: Alpaca 2.07× vs 3.36× (8B); MT-Bench 2.56× vs 4.10× (8B); 30B-A3B Alpaca 1.53× vs 2.46×.
- T=1 (4B): MATH 4.65× / 6.60 vs 6.60× / 9.61; Alpaca 1.97× / 2.96 vs 3.23× / 4.97.
- [paper, via WebFetch] DDTree beats DFlash in every one of the 60 cells.

Cross-check with ours:
- Our Qwen3-8B A100 HF τ: 7.26 (32), 7.48 (64), 8.08 (128), 8.37 (256).
- The paper's τ on its best budget for 8B math is about 10.7, on MATH-500 with no thinking.
- The gap is consistent with a dataset or prompt mix difference plus budget (they pick the 256-512 peak). **Check which dataset our 7-8 τ comes from before comparing directly.**
- The paper's budget curve exists only for MATH-500 / Qwen3-8B / T=0 (Fig. 3, Sec. 5.3).

---

## 3. Limitations and future work the authors state

- [paper] Remark 1: optimality is for the surrogate Q, not the target.
- [paper] Sec. 5.3: the optimal budget depends on hardware and implementation. Beyond the peak, "additional overhead of verifying more drafted tokens outweighs the gain."
- [paper, App. B] FA2 cannot express the tree mask, so SDPA is used.
- [paper] Related work: OPT-Tree needs "one drafter's forward pass per tree depth" (the motivation for one-pass trees). DART "relies on continuity-aware tree pruning with an external N-gram continuity score."
- **No explicit limitations section, conclusion, or future-work section was found** in the WebFetch views ("The paper ends with References after Section 5.4"). Verify against the PDF.

---

## 4. Implicit weaknesses

1. **Surrogate-target mismatch is unquantified.** There is no calibration plot of predicted Σ_{u∈T} q(u) against realised acceptance, and no per-depth analysis. Remark 1 concedes the gap but does not measure it. [inf] At T=0 the correct node score is P(argmax), not q (1.1c), so even a perfectly calibrated drafter gives a mis-ranked tree.
2. **Diminishing returns from width.** [ours] τ goes 7.26→8.37 from B=32 to 256 (+15% for 8× nodes), and 7.71→8.10 offline from 64 to 512. [paper] Speedup peaks at 256-512 on H200. Each node's marginal gain is the q-mass of the B-th prefix, which decays fast. The binding constraints are elsewhere: drafter information in a single pass, and the depth cap.
3. **The depth cap equals the block size.** No round can exceed 16 tokens. [ours] 15-17% of rounds saturate it. [paper] Sec. 5.4 shows the full-block mass growing. [inf] The upside from relaxing the cap is roughly P(saturate) × E[extra accepted | saturated]. At 0.16 × ~4-6 tokens that is about +0.6-1.0 τ (+8-12%). This is a rough estimate: measure E[extra] by continuing verification of the AR path.
4. **Verification cost under batching.** The paper is at batch 1. [ours] Verify is flat at 42.5-44.6 ms for B = 32-256 at batch 1. [inf] For an 8B model on an A100 the weight-read bound is about 16 GB / 2 TB/s ≈ 8 ms. A 42 ms verify is therefore dominated by framework overhead (HF eager, kernel launches, the SDPA mask path), not by memory bandwidth. Two consequences:
   - (a) In an optimised engine, verify might drop to roughly 10-12 ms. Draft (7.2 ms) plus build (1.2-1.7 ms) plus commit (2.3 ms) would then be about 50% of the round instead of about 20%. The heap and the CPU sync would matter.
   - (b) At batch b, the verify tokens are b × (B+1). Past the compute ridge (about 150-300 tokens for an A100 at bf16), each extra node costs real FLOPs. Optimal B then shrinks roughly as 1/b, which the paper's fixed-B grid does not model.
   - [ours] HF TP was slower than 1 GPU. In vLLM, 2-GPU TP gives 1.55× for DFlash. These suggest HF numbers do not transfer.
5. **Target feedback is thrown away.** The target computes logits and hidden states at all B+1 nodes. Only the accepted path is used [code]. The calibration of q is never updated from accept/reject outcomes, and the rejected-branch distributions are discarded. No per-request or per-domain adaptation.
6. **No per-round or per-request budget adaptation.** B is fixed for the whole run. q's entropy (easy versus hard stretches) is not used to shrink or grow the tree.
7. **Drafter staleness under target change.** The tree only reorders the drafter's marginals, so it inherits any drafter-target drift. [ours] With target LoRA, DDTree-128 τ goes 8.31→6.60 (-21%) and speed falls 13%. The tree does not absorb the drift.
8. **Single pass, sequential draft then verify.** The drafter is conditioned on target hidden features of accepted tokens [code: `target_hidden` from verify], so the next round's draft cannot start before verify finishes. There is no draft/verify overlap. [ours] Draft is 7.2 ms (about 14% of a 53 ms round).
9. **The T=1 baseline is weak and the T=1 tree is untuned.** DFlash uses a greedy draft at T=1, and DDTree uses the same T-agnostic tree. Neither uses q in verification (1.4).
10. **Generality.** One model family, thinking disabled, 2048-token cap, no long context.
11. **Competitor numbers.** [ours/brief] DARTree (2608.13524) reports 9.73× vs DDTree 5.79× on Qwen3-4B at T=0 using correction heads. The DDTree paper itself reports 6.58-7.50× for 4B on math at T=0. DARTree's DDTree number is therefore a re-measurement with a different setup (hardware, datasets, averaging), so the 1.7× gap is not apples to apples. **Unverified: we have not read DARTree.** Our prior notes (drafter_side.md) list tree and budget construction on block-diffusion marginals (CaDDTree, BASTION, GRAFT, DARTree, JetSpec) as **crowded**.

---

## 5. Angles to improve, ranked

The ranking weighs expected gain, how cheap a test is with our existing infrastructure (vendored code, `save_tree_traces`, offline replay), and novelty given the crowded areas in drafter_side.md.

### A1. Loss decomposition / oracle ceilings for one-pass trees (a diagnostic that orders everything else)
- **Weakness:** The paper never measures where acceptance is lost. Prop. 2 is optimal only for Q.
- **Evidence:** Remark 1. [ours] Width gains saturate (7.71→8.10 for 64→512), and our tree improvements gave only +5-8%.
- **Why it matters:** It tells us whether the remaining gap is (i) coverage (the target's path is not in the top-K marginals at some depth), (ii) ranking (it is covered but outside the top-B under Q), or (iii) the depth cap. Each points to a different fix: drafter, scoring, or depth. Nothing published shows this decomposition for DDTree [inf; check CaDDTree/BASTION].
- **First experiment:** Offline replay with `save_tree_traces` plus the target's greedy path for the next 32 tokens. For each round compute:
  - τ_DDTree(B);
  - τ_rank-oracle(B), the best B-node tree from the same marginals given hindsight, which is the longest target-path prefix covered;
  - τ_coverage(K), the target path is in top-K at every depth up to 16;
  - τ_uncapped, continuing past 16 along the target path while the drafter's top-K covers it, using a second draft pass.
  Report per dataset.

### A2. Temperature- and depth-calibrated surrogate scoring (fix the objective cheaply)
- **Weakness:** Q is mis-scaled for T=0 (1.1c). It is static and uncalibrated per depth, and the tree is identical across temperatures.
- **Evidence:** [code] `build_ddtree_tree` ignores temperature. Remark 1. [pith] "no correlation analysis".
- **Why it matters:** It is the cheapest change: a drop-in replacement of log q by f_d(log q) with zero runtime cost. A sharper score at T=0 should move budget from shallow siblings into depth where the argmax is near-certain. Per-depth calibration fixes the depth-versus-width trade.
- **First experiment:** On logged rounds, fit per-depth temperature or Platt scaling of q to predict the event "node token = target argmax given the correct prefix". Rebuild trees offline with calibrated scores at B ∈ {32, 64, 128, 256} and compare τ. Also report a reliability diagram (predicted Σq vs realised τ). Then try an online version that updates the per-depth scalars from each round's verify outcome; this reuses the target feedback weakness 4.5.

### A3. Break the depth cap: a second-stage extension of saturated paths
- **Weakness:** The depth is at most 16 (block size), and 15-17% of rounds saturate.
- **Evidence:** [ours] 15-17% full-depth rounds. [paper] Sec. 5.4 shows full-block mass rising.
- **Why it matters:** Width has saturated, and depth is the only other lever with headroom. Estimated +8-12% τ [inf]. This fits our spec-hierarchy (hierarchical/3-stage) direction directly.
- **First experiment:** Measure E[extra accepted | saturated] by greedy-continuing the target for 16 more tokens on saturated rounds (the upper bound). Then implement a speculative "tail" that runs a second DFlash pass conditioned on the top-1 depth-16 leaf, or a cheap AR head chain from that leaf. Append it to the tree within the same verify, and accept the drafter-feature mismatch (the drafter needs target hidden states). Measure τ and wall-clock.

### A4. Latency-aware, per-round adaptive budget
- **Weakness:** A fixed B was picked post hoc on test data. The node-count constraint is not latency.
- **Evidence:** [code] `make_latex_table.py` selects the best budget per cell. [paper] The peak is "256 to 512" and hardware-dependent. [ours] Verify is flat for B ≤ 256 on A100 HF, so the right B depends on the engine.
- **Why it matters:** In the best-first order, the B-th node's q(u) is exactly its marginal gain. Stopping when q(u_next) < c × ∂T/∂N, with ∂T/∂N profiled per engine and per batch size, gives a principled rule. It also makes the method usable at batch > 1. Note this is crowded (CaDDTree, BASTION, DSpark, AdaFlash per drafter_side.md); novelty only in combination with A2 calibration and batch-aware cost.
- **First experiment:** Profile verify latency against total tokens for batch 1, 4, 16 in vLLM. Replay traces with threshold stopping at several c. Report τ per unit of verify cost against a fixed-B Pareto curve.

### A5. Systems reality check: DDTree in an optimised engine and at batch > 1
- **Weakness:** HF eager, batch 1, CPU heap with a sync each round, dense SDPA mask.
- **Evidence:**
  - [ours] Verify 42.5 ms against roughly an 8 ms bandwidth bound (inference). HF TP slower than 1 GPU. vLLM TP 1.55× for DFlash.
  - [code] Python heapq plus an O(N²) Python visibility loop plus `.to("cpu")` per round.
- **Why it matters:** If verify drops 3-4× in vLLM, draft + build + commit become about half the round. The heap then needs to move to the GPU (a fixed-shape top-B selection over the D×B lattice), and the tree's value at batch 8-32 may disappear. A negative or nuanced result here is publishable as an analysis and protects the other angles from being artifacts of HF.
- **First experiment:** Port the tree build to the GPU, either as a vectorised top-B over the cumulative-logprob lattice (an exact equivalent of the heap for product scores via iterative frontier expansion) or as an approximate fixed beam per depth. Run DFlash vs DDTree in vLLM/SGLang with CUDA graphs at batch 1/4/16/32 and report tokens/s, not only τ.

### A6. Path-conditioned correction of the marginals below non-mode branches
- **Weakness:** Independence (1.1a/b). The marginals under rank > 0 branches are computed for the mode context.
- **Evidence:** [ours] A dependency head conditioning on the parent token gave only +5-8% at equal budget. drafter_side.md: intra-block dependency methods give +9-18% over DFlash but little over DDTree.
- **Why it matters:** This is the obvious theoretical gap. Our data says its practical value is bounded, so it ranks below A1-A5. A1 tells us whether ranking errors (which this would fix) are a large share of the loss. Only pursue it if A1 shows a large rank-oracle gap.
- **First experiment:** From A1 traces, measure the realised acceptance rate at depth d+1 below rank-0 vs rank ≥ 1 parents against q_{d+1}. If the miscalibration is large, apply a cheap per-(depth, parent-rank) discount in scoring (a version of A2) before any architectural head.

### A7. Probability-aware tree and verification at T>0
- **Weakness:** The T>0 verifier is sample-and-match: lossless but q-agnostic. The DFlash baseline uses a greedy draft at T=1. The tree is T-agnostic.
- **Evidence:** [code] `sample(draft_logits)` greedy in DFlash; `posterior = sample(logits, T)` with child matching. [paper] T=1 τ is lower (4B MATH 10.71→9.61; Alpaca weakest).
- **Why it matters:** At T=1, acceptance at a node is Σ_children p_T. Building the tree to maximise that sum, with temperature-matched q, and/or using multi-draft lossless verification (traversal verification, SpecInfer multi-round rejection, recursive rejection sampling) could raise T=1 τ. The paper does not cover T ∈ (0,1) or top-p either.
- **First experiment:** Offline at T ∈ {0.6, 1.0}: score nodes by q_T (the tempered drafter) and compare τ with DDTree's T-agnostic tree. Then implement traversal-style verification on the same tree and measure τ. Check losslessness with a histogram test on a small vocabulary.

### A8. Drafter drift robustness: the tree as an absorber of target changes
- **Weakness:** The tree inherits the drafter's mismatch. There is no adaptation.
- **Evidence:** [ours] Target LoRA: τ 8.31→6.60 (-21%), speed -13%. drafter_side.md Gap 4.
- **Why it matters:** It ties into the continual-learning angle. A2's online per-depth calibration, or a tiny online-updated correction on the logits, may recover part of the loss with no retraining. That is a novel combination of tree scoring and drift.
- **First experiment:** On the LoRA target, compare (i) DDTree, (ii) DDTree with offline-recalibrated per-depth scores, (iii) online-calibrated scores, (iv) a retrained drafter (our roughly 1 GPU-hour on-policy fix). Report the fraction of the τ loss each recovers.

### A9. Overlap the draft with verify (pipelining)
- **Weakness:** Strictly sequential draft → build → verify → commit. The drafter needs target features of accepted tokens.
- **Evidence:** [code] `target_hidden` comes from the verify output. [ours] Draft 7.2 ms ≈ 14% of the round. It becomes a larger share in an optimised engine (A5).
- **Why it matters:** Up to about 14-40% wall-clock, depending on the engine, with no change to τ. Needs a drafter variant that can start from the target features of a predicted accept point. It overlaps with PEARL/parallel-SD-style work; check novelty for diffusion drafters.
- **First experiment:** Speculatively launch the next draft from the most likely accept node (the deepest node on the highest-q path) using features from the previous verify, overlapped on a second CUDA stream. Measure the hit rate and the saved time.

Not ranked separately: evaluation hygiene (held-out budget selection, thinking mode on, long context, batch > 1, CIs). Do all of it in whatever paper we write. It is also a fair critique to state when positioning against DDTree.

---

## 6. Open checks before citing

- Confirm Table 1 values and the T=1 rows for 8B and 30B from the PDF (the WebFetch extraction is partial).
- Confirm there is no limitations or conclusion section, and the exact Sec. 4.4 wording on sampling.
- Confirm which dataset our τ ≈ 7.3-8.4 numbers come from, so the comparison with the paper's 10.7 (MATH-500) is like for like.
- Read DARTree (2608.13524) for its DDTree baseline setup before quoting 9.73× vs 5.79×.
- Check CaDDTree / BASTION / GRAFT for overlap with A2 and A4 (calibrated or adaptive budgets) and A1 (any oracle decomposition).
