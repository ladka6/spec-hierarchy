# Speculative and self-speculative decoding FOR diffusion LLM targets, and structurally similar dLLM acceleration (2025 to Nov 2026)

Scope note: most entries below were read from arXiv abstract or HTML pages via a summarizing fetcher, not full PDFs. Where the fetcher only saw an abstract, per-benchmark numbers and limitations are marked "not in abstract". Items marked [UNVERIFIED] were seen only in search result titles. Dates are arXiv submission dates as shown on the page.

## 1. What speculative / self-speculative methods exist for dLLM targets, and what do they verify against?

### Takeaway
There are now at least nine draft-and-verify methods whose target is the dLLM itself. Almost all are self-speculative: the dLLM drafts from its own marginals and checks the drafts in one batched or mask-augmented forward pass. "Lossless" is always defined against a slower decoding procedure of the same dLLM, never against the dLLM's true joint distribution. The reference is either stepwise or one-token-per-step greedy decoding (SSD, FreeDave) or the sampled block state at the next denoising step (Spiffy, SPOT). FeF-DLLM is the only paper found that claims to sample the true joint distribution, and it changes the model's factorization to do so. The 2026 trend (June to Oct) has two branches. One is learned tiny drafters that predict the commit order (SPOT). The other is trajectory-level speculation and draft/verify kernels built on top of Fast-dLLM caches (Trajectory-Level SD, Flash-dLLM, SimSD on SDAR).

### Cited Findings

Spiffy (arXiv 2509.18085, Qualcomm AI Research; v3 seen)
- Mechanism: auto-speculative. Draft block states come from the dLLM's own distribution. Drafts are organized as a directed draft graph, where a node can have several parents, unlike AR trees. The graph shape is calibrated offline with under 50 samples in about 30 min on one GPU, then fixed. [Spiffy](https://arxiv.org/html/2509.18085v3)
- Verifier: the same dLLM scores the target state and all drafts in one call using a custom attention mask. Verification target: p_θ(X_k(t-1) | X'; X_k(t)), i.e. the next denoising state of block k. A draft is accepted if it matches the sampled block state. The paper proves (Appendix A) that the output distribution is preserved; observed accuracy change is within bfloat16 noise. [Spiffy](https://arxiv.org/html/2509.18085v3)
- Targets: LLaDA-Base-8B, LLaDA-Instruct-8B, LLaDA-1.5-8B. Dream not evaluated. Settings: GSM8K, MATH, MBPP, HumanEval, gen length 256, block 32, temperature 0. [Spiffy](https://arxiv.org/html/2509.18085v3)
- Numbers, measured as NFE reduction, not wall clock: 2.80 to 3.04x standalone with D=10 drafts. Up to 7.88x combined with threshold-0.7 parallel decoding on LLaDA-Instruct-8B HumanEval. 5.18x with threshold 0.9, 5.28x with hard-code=4. With EOS-truncated counting, 4.40x vs 5.18x for threshold 0.9. KV-cache compatibility is claimed but not tested. [Spiffy](https://arxiv.org/html/2509.18085v3)
- Limitations stated: the naive attention implementation disables FlashAttention, and inference time grows roughly linearly with the number of drafts. Overhead is 0.2 to 3.8% of call time. The graph was not calibrated for parallel-decoding settings. Future work: auxiliary drafter, KV caching, per-block-group graphs. [Spiffy](https://arxiv.org/html/2509.18085v3)

Self Speculative Decoding for dLLMs, SSD (arXiv 2510.04147; Gao, Ji, ..., Linfeng Zhang; SJTU / Shanghai AI Lab / Huawei)
- Mechanism: one forward pass over all masked positions gives drafts and confidences. Up to N positions are chosen greedily in semi-AR block order, favoring high confidence. A "hierarchical verification tree", in practice a linear chain of N+1 nodes, is checked in one batched forward. A child is accepted only if its parent's prediction matches its draft token. [SSD](https://arxiv.org/html/2510.04147v1)
- Reference: stepwise (one-step-at-a-time) decoding results. The paper calls itself "theoretically lossless" with identical outputs, so it reports no accuracy tables. [SSD](https://arxiv.org/html/2510.04147v1)
- Targets: LLaDA-Base, LLaDA-8B-Instruct, LLaDA-1.5, Dream-Base, Dream-Instruct. Benchmarks: GSM8K, MATH (10% subsample), HumanEval, MBPP, zero-shot, on one A100 80GB, cache-enabled baseline. [SSD](https://arxiv.org/html/2510.04147v1)
- Mean TPS speedups: LLaDA-Base about 2.03x (N=4), LLaDA-Instruct 2.11x (N=4), LLaDA-1.5 2.12x (N=4), Dream-Base 2.15x (N=5), Dream-Instruct 2.43x (N=5). The headline 3.46x is a single MBPP / Dream-Instruct point (6.37 to 22.07 TPS). Mean step reduction is 67.6%; 77.4% is MBPP only. [SSD](https://arxiv.org/html/2510.04147v1)
- Limitations stated: acceptance is capped by top-1 greedy drafting, and out-of-order generation is the most common failure. k-ary trees grow exponentially and become memory-bound. A mix-order tree gives +2.5 to 3.1% step reduction for 50 to 67% larger verification batches. Gains shrink with long prompts (GSM8K about 2.4x zero-shot vs 1.5x three-shot). The best N is model-dependent. Future work: adaptive trees and other non-AR paradigms. [SSD](https://arxiv.org/html/2510.04147v1)

FreeDave: Free Draft-and-Verification (arXiv 2510.00294; Shutong Wu, Jiawei Zhang; v3 dated 2026-02-03; repo tagged NeurIPS 2026)
- Mechanism: parallel-decoded candidates are verified so the output reproduces one-token-per-step decoding. No model modification. The paper claims a theoretical guarantee of the minimum number of forward calls needed to reproduce that reference. Reported speedup: up to 2.83x without degradation on math and code (benchmarks not named in abstract). [FreeDave arXiv](https://arxiv.org/abs/2510.00294); NeurIPS 2026 tag from [FreeDave GitHub](https://github.com/cychomatica/FreeDave) (venue not verified on neurips.cc; a [NeurIPS 2025 virtual page](https://neurips.cc/virtual/2025/126722) also exists for this title, so the venue/year is CONFLICTING).
- Limitations: none in abstract.

Self-Speculative Masked Diffusions (arXiv 2510.03929; Campbell, De Bortoli, Shi, Doucet; Google DeepMind)
- Mechanism: a hybrid transformer. Non-causal MDM blocks give a factorized draft over masked positions. Causal σ-GPT-style blocks on the permuted sequence (with a residual connection) give a non-factorized target. Draft and verify happen in one forward pass via speculative sampling. [SSMD](https://arxiv.org/html/2510.03929v2)
- What it is lossless against: accepted tokens follow the causal non-factorized target. Because that target shifts as tokens are revealed, the paper does not claim a fixed distribution. It derives a tractable likelihood (Prop 3.1) and a lower bound, and trains with MDM loss plus causal AR loss. [SSMD](https://arxiv.org/html/2510.03929v2)
- Numbers: more than 2x NFE reduction on text8 at low NFE. Same generative perplexity as MDM at half the NFE on OWT (150M). About 2x on UniRef50 protein with frozen ESM2 plus one causal block. [SSMD](https://arxiv.org/html/2510.03929v2)
- Limitations: the MDM baseline NFE is best-case. Protein experiments use random order. Future work: inference-scaling combinations. [SSMD](https://arxiv.org/html/2510.03929v2)

Factorization-Error-Free DLLM via Speculative Decoding, FeF-DLLM (arXiv 2605.14305; Fang, Li, Yuan, Zhou Yu; 2026-05-14)
- Mechanism: replaces independent per-token x0 prediction with an "exact prefix-conditioned factorization of the clean posterior". Speculative decoding inside denoising offsets the sequential cost, while parallel prediction and re-masking are kept. The authors prove it samples from the true joint distribution. [FeF-DLLM](https://arxiv.org/abs/2605.14305)
- Numbers: +5.04 pp average accuracy and 3.86x average speedup on GSM8K, MATH, HumanEval, MBPP. Base models and the drafter/verifier split are not in the abstract. [FeF-DLLM](https://arxiv.org/abs/2605.14305)

SimSD: Simple Speculative Decoding in Diffusion Language Models (arXiv 2606.02544; Cui, ..., Jingbo Shang; 2026-06-01, v2 2026-08-08)
- Mechanism: training-free, plug-and-play masking. Reference tokens from draft predictions are inserted, and an attention mask governs how they interact with current-step tokens. This gives the dLLM "temporally valid" contexts, so it computes logits for the drafted tokens in one forward pass. Compatible with KV cache and blockwise decoding. [SimSD](https://arxiv.org/abs/2606.02544)
- Target: SDAR-family dLLMs. Up to 7.46x decoding throughput, with average quality "maintained and even improved" on four benchmarks (not named in abstract). No losslessness claim in the abstract. [SimSD](https://arxiv.org/abs/2606.02544)

Trajectory-Level Speculative Decoding for dLLMs (arXiv 2608.27514; Pan, ..., Kaiwen Long; 2026-08-27)
- Mechanism: speculates whole denoising trajectories instead of token sequences. Drafts come from confidence-stratified tree exploration and are verified by blockwise parallel evaluation with bidirectional attention masking. Adds inter-block speculation for cross-block lookahead. Built on Fast-dLLM's dual cache. The authors "formally characterize when this approach is exact". [Trajectory SD](https://arxiv.org/abs/2608.27514)
- Numbers: 7 to 14x over vanilla dLLMs, 1.3x over Fast-dLLM, 30 to 40% fewer denoising iterations, tokens/step 2.6 to 4.3, accuracy change under 1% on reasoning and code benchmarks (models not named in abstract). [Trajectory SD](https://arxiv.org/abs/2608.27514)
- Limitation stated: "trajectory drift" is the fundamental cost of more parallelism. [Trajectory SD](https://arxiv.org/abs/2608.27514)

Flash-dLLM (arXiv 2609.26796; Nguyen-Tri, Ranjan, Zhiqiang Shen; 2026-09-22)
- Mechanism: an IO-aware fused KV-cache kernel (memory movement is identified as the dominant bottleneck in cache-enabled dLLM inference) plus a KV-cache-driven draft-and-verify in which "the dLLM itself serves as both drafter and verifier". Training-free. [Flash-dLLM](https://arxiv.org/abs/2609.26796)
- Numbers: 5.1x over Elastic-Cache on GSM8K and 11.0x on HumanEval. Quality is "preserved"; no accuracy numbers in abstract. [Flash-dLLM](https://arxiv.org/abs/2609.26796)

On the SPOT: commit-order prediction (arXiv 2610.22543; Shlomi, Rahimi, Mendelson, Baskin; 2026-10-15)
- Key observation: the target's previous-pass predictions already contain the correct token at 92 to 100% of positions. What is missing is the order in which positions get committed. [SPOT](https://arxiv.org/abs/2610.22543)
- Drafter: a 0.62M-parameter model that reads only the target's output distributions and proposes a tree of candidate continuations (position orders). Verifier: the target dLLM checks all candidates in one pass by testing whether each matches its own next state. "Coupled-noise match verification" with shared Gumbel noise makes tree speculation work at any temperature. [SPOT](https://arxiv.org/abs/2610.22543)
- Numbers: up to 6.8x fewer target passes and up to 3.2x throughput vs standard decoding. Beats calibrated and training-free drafting baselines on every task and budget at T=0 and T=1. Transfers across six DLMs (not named in abstract). [SPOT](https://arxiv.org/abs/2610.22543)

PV-Decoding (arXiv 2610.24722; Renxi Liu et al.; dated Oct 17, 2026 on aggregator)
- Mechanism: confidence-driven speculative decoding for block diffusion. Two thresholds split masked positions into direct-decode, speculative, and unresolved. Speculative tokens get progressively stronger verification (single-, dual-, multi-batch), plus "decode gating". [PV-Decoding (OpenTrain summary)](https://www.opentrain.ai/papers/pv-decoding-progressive-verification-decoding-with-decode-gating-for-diffusion-l--arxiv-2610.24722/)
- Numbers: 8 to 37% fewer sequential forwards vs confidence decoding. Dream-7B-Instruct GSM8K throughput +7.0 to 13.6% (1- and 2-batch). The multi-batch variant is slower in the current runtime. The max accuracy drop is claimed at 1.3%, but the aggregator table shows 72.4 vs 46.2 on GSM8K, which conflicts. UNVERIFIED: the arXiv page was rate-limited and not read directly. [PV-Decoding](https://www.opentrain.ai/papers/pv-decoding-progressive-verification-decoding-with-decode-gating-for-diffusion-l--arxiv-2610.24722/)

AR model as verifier/drafter for a dLLM target
- APD, Adaptive Parallel Decoding (arXiv 2506.00413; Israel, Van den Broeck, Grover; UCLA): forces Dream-7B-Instruct to unmask left to right and samples a block in parallel. It accepts the longest prefix that a small AR model (Qwen2.5-0.5B, shared tokenizer) agrees with, under a multiplicative mixture of dLLM marginals and AR joint with weight R, using Gumbel coupling. This inverts the usual SD setup: the small model checks the big one. There is no exact losslessness claim; the target approximates the dLLM's L2R distribution. GSM8K: Dream L2R at 256 steps gets 0.832 at 10.1 tok/s vs Qwen2.5-7B AR at 0.854 and 38.6 tok/s. APD reaches 7.62 tokens/step on GSM8K (R=0.7, W=16, M=100) and over 5 tokens/iter at about 80% GSM8K. Over 100 tok/s is possible with quality loss. Limitations: a throughput-quality tradeoff, no improvement over the base dLLM, and lower parallelism on open-ended writing. [APD](https://arxiv.org/html/2506.00413v2)

Related seen-only titles [UNVERIFIED, not read]
- TeCoP: Accelerating dLLM decoding via Tentative Context Pipelining (arXiv 2610.27583), fetch rate-limited. [TeCoP](https://arxiv.org/abs/2610.27583v1)
- Parallel Language Models via Variational Decoding (arXiv 2610.26693). [link](https://arxiv.org/abs/2610.26693)
- Plan, Verify and Fill: structured parallel decoding for DLMs (arXiv 2601.12247). [link](https://arxiv.org/html/2601.12247v3)
- Optimus: Elastic decoding for dLLM serving (arXiv 2605.24832). [link](https://arxiv.org/pdf/2605.24832)
- D^2SD: Dual Diffusion Draft Models (arXiv 2606.04446). From the title, likely diffusion drafters for an AR target; not confirmed. [link](https://arxiv.org/abs/2606.04446)

### Inferences
- "Lossless" for a dLLM target comes in at least three different senses:
  (a) Output-identical to greedy one-token-per-step or stepwise decoding of the same dLLM (SSD, FreeDave). This reference is itself an arbitrary decoding policy, not the model's "distribution".
  (b) Distribution-preserving w.r.t. the next denoising-state transition of the existing sampler (Spiffy; SPOT via coupled noise).
  (c) Exact w.r.t. a non-factorized joint the method itself defines (SSMD's causal head, FeF-DLLM's prefix factorization).
  None verifies against the true joint implied by the trained bidirectional model, because that joint is not tractably defined for MDMs.
- Ceilings for self-speculation without a learned drafter cluster around 2 to 3x (SSD 2.0 to 2.4x mean, Spiffy 2.8 to 3.0x NFE, FreeDave 2.83x). Larger numbers (7 to 14x) come from stacking on threshold parallel decoding or caches, which are themselves lossy.
- SPOT's 92 to 100% finding suggests the bottleneck for dLLM self-speculation is choosing positions and order, not choosing tokens. This mirrors SSD's report that out-of-order generation is the main failure.

### Gaps
- Per-model/per-benchmark numbers for SPOT, SimSD, Trajectory SD, Flash-dLLM, FeF-DLLM, FreeDave and PV-Decoding were not available in the abstracts read; full PDFs would need to be read.
- No method was found that verifies a dLLM target with a separate small diffusion drafter (the direct analogue of a two-model SD), except Spiffy listing it as future work.
- TeCoP and Variational Decoding contents unknown.
- OpenReview ICLR 2027 submissions were not searched directly. Coverage of Oct 2026 relies on arXiv search hits.

## 2. How do parallel-unmasking methods pick positions, and how do they handle the "parallel decoding error" from independent marginals?

### Takeaway
The dominant rule is still a confidence threshold (max softmax probability, τ about 0.9), sometimes entropy- or margin-based. Its only formal backing is Fast-dLLM's worst-case theorem. ParallelBench shows that the conditional-independence error is unavoidable even for an ideal model, and that current adaptive rules sit far below a per-sample oracle. 2026 work addresses the error in four ways: dependency-aware gating (DAPD, DARD), deferred or sliding-window commitment (DCD), learned certainty (dParallel), and training changes (FeF-DLLM, TiDAR, SSMD).

### Cited Findings
- Fast-dLLM (arXiv 2505.22618) theorem: if each position's marginal confidence is above 1-ε and (n+1)ε ≤ 1, greedy decoding from the product of marginals equals greedy decoding from the joint. The bound is tight for ε > 1/(n+1). TV < (3n-1)/2·ε and KL < (n-1)(H_b(ε)+ε ln(|V|-1)). The paper says the bounds are worst-case and real MDMs may not satisfy joint consistency. [Fast-dLLM](https://arxiv.org/html/2505.22618v3)
- Fast-dLLM rules: threshold τ (default 0.9, range 0.5 to 1.0), always unmasking the top token. Factor variant: pick the largest n with (n+1)(1-c^(n)) < f. The factor variant is typically 1 to 3 points less accurate with higher throughput. [Fast-dLLM](https://arxiv.org/html/2505.22618v3)
- Fast-dLLM numbers (A100, bs1): LLaDA-Instruct GSM8K 5-shot len 256 goes from 79.3% at 6.7 tok/s to 78.5% at 54.4 tok/s (8.1x). LLaDA GSM8K len 512: 77.5 to 77.2, 11.0x. LLaDA MBPP len 512: 14.8 to 13.8, 9.2x. Dream-Base GSM8K: 75.0 to 74.8, 5.3x. Dream-Base MBPP: 56.6 to 56.4, 6.8x. Best case 27.6x (LLaDA, 8-shot, len 1024, DualCache, 77.3 to 76.0). Prefix/Dual KV caches are approximate because attention is bidirectional, justified by adjacent-step K/V cosine similarity near 1. [Fast-dLLM](https://arxiv.org/html/2505.22618v3)
- Fast-dLLM limitations: at larger batch sizes PrefixCache cannot match AR (LLaMA) throughput because full attention is costly. LLaDA-V MathVista drops more than 8 points when the block shrinks from 96 to 8. [Fast-dLLM](https://arxiv.org/html/2505.22618v3)
- ParallelBench (ICLR 2026; furiosa-ai): uses conditional total correlation to bound parallel-decoding loss, which is unavoidable even for an ideal model. Zero-dependency tasks (Copy, Replace Index) parallelize losslessly. Replace Random (bounded dependency of about 1.44 bits) drops to 0.5 accuracy at k=2 and 0 for k>2 under greedy top-k. Shuffle goes to 0 for any k≥2, including for closed-source Mercury. Adaptive thresholds beat static top-k but sit far below a per-sample oracle threshold. CoT reduces degradation at about 8x tokens. ReMDM remasking gives no gain on Waiting Line tasks. Fine-tuning does not fix Shuffle. Models: LLaDA 1.5, SEDD. [ParallelBench note](https://en.papernotes.org/ICLR2026/llm_evaluation/parallelbench_understanding_the_trade-offs_of_parallel_decoding_in_diffusion_llm/)
- dParallel (arXiv 2509.26488; NUS) states that "the key bottleneck to parallel decoding arises from the sequential certainty convergence for masked tokens": confidence emerges left to right. Fix: certainty-forcing distillation (consistency CE on the teacher's semi-AR trajectory plus entropy minimization at T=0.5 on correctly predicted tokens, LoRA). LLaDA-8B-Instruct GSM8K: 256 to 30 steps (8.5x), 75.7 to 76.1%. MBPP: 256 to 24 steps (10.5x), 42.4 to 40.8%. [dParallel](https://arxiv.org/html/2509.26488v1)
- Deferred Commitment Decoding (arXiv 2601.02076; PKU/Huawei) names "Boundary-Induced Context Truncation": tokens near a block's end are forced to commit without future context. A sliding window commits a position if certainty (confidence, or negative entropy for NBDiff) reaches τ or ranks top within the window. +1.73% average accuracy and about 4.4% less decode time across LLaDA-8B-Instruct, Dream-v0, Fast-dLLM-v2-7B and NBDiff. Max gain +16.5 (NBDiff IFEval 40.1 to 56.6). Limitations: dynamic block extension hurts fixed-block Fast-dLLM-v2 (-1.30%), some regressions, ablations on LLaDA/MATH500 only. [DCD](https://arxiv.org/html/2601.02076v2)
- DAPD: Dependency-Aware Parallel Decoding via Attention (arXiv 2603.12996; ICML 2026 per notes) uses attention to estimate inter-token dependency before parallel commits. Details not read. [DAPD](https://arxiv.org/html/2603.12996v2)
- PV-Decoding uses two confidence thresholds producing a three-way split (decode / speculate / unresolved). [PV-Decoding](https://www.opentrain.ai/papers/pv-decoding-progressive-verification-decoding-with-decode-gating-for-diffusion-l--arxiv-2610.24722/)

### Inferences
- The dLLM parallel-decoding error and the block-drafter error for AR targets are the same object: a product of per-position marginals used in place of a joint. ParallelBench's total-correlation bound and Fast-dLLM's ε-bound are the two formal handles.
- Confidence thresholds are conservative proxies for low total correlation, and none of the methods read estimates the total correlation directly. DAPD and DARD use attention or confidence ordering as stand-ins.

### Gaps
- No paper found that measures per-step total correlation of real LLaDA/Dream blocks and relates it to accepted parallelism.
- Margin-based rules were mentioned in some surveys but no primary 2026 margin-specific paper was read.

## 3. Methods that repair or remask decoded tokens, and whether errors are local or global

### Takeaway
Revocable decoding is now mainstream: WINO (verify via shadow mask block), LLaDA2.1 (T2T editing in a production model), T2M remasking, DARD, plus older ReMDM. They improve accuracy while keeping or raising parallelism. However, none of the papers read directly measures whether parallel-decoding errors are local or propagate globally. The evidence is indirect: refill rarely changes the token, and flip counts favor remasking.

### Cited Findings
- WINO, Wide-In Narrow-Out (arXiv 2507.18578; SJTU and Apple): draft any token with confidence above τ1 (0.5 to 0.7). Verify each with a shadow [MASK] block that shares position IDs and cannot attend to the position it verifies. Re-mask anything below τ2 = 0.9. One forward per step. LLaDA-8B-Instruct (len 256, block 128): GSM8K 73.24 to 75.82 at 6.10x steps / 5.66x TPS; MATH-500 32.00 to 34.20 (3.44x / 3.17x); HumanEval 37.80 to 42.07 (2.74x / 2.56x); MBPP 36.40 to 36.40 (2.65x / 2.45x); ARC-E 59.13 to 81.19 (6.37x / 5.89x). MMaDA-8B-MixCoT Flickr30K CIDEr 53.67 to 53.83 at 10.05x steps / 8.60x TPS. Memory +2.4% (16.18 to 16.57 GB). Limitations: speedup depends on base-model capability; draft-and-verify-aware training is future work. No revoked-token fraction reported. [WINO](https://arxiv.org/html/2507.18578)
- LLaDA2.1 (arXiv 2602.08676 per figure filenames, Feb 9, 2026; Ant / inclusionAI): adds Token-to-Token editing to Mask-to-Token with joint configurable thresholds. Speedy mode lowers the M2T threshold and relies on T2T corrections; Quality mode is conservative. Includes Multiple Block Editing (revisits earlier blocks) and ELBO-based block-level RL. Mini 16B and Flash 100B, 33 benchmarks. Flash S-mode: 892 TPS on HumanEval+, 801 on BigCodeBench, 663 on LiveCodeBench. S-mode is weaker for general chat, and threshold tuning is domain-dependent. Source is a secondary review page, not the paper itself. [LLaDA2.1 review](https://liner.com/review/llada21-speeding-up-text-diffusion-via-token-editing)
- Remask, Don't Replace (arXiv 2604.18738; Lin Yao, SJTU): argues T2T editing suffers from correction inertia (multimodal posteriors), context pollution (replacements chosen in an erroneous context propagate), and a train/inference mismatch (training on random noise vs locally plausible errors). Token-to-Mask on LLaDA2.1-mini improves 15/17 benchmarks: GSM8K 89.61 to 91.21, HumanEval+ 76.83 to 80.49, GPQA-D 43.94 to 45.96. LLaDA2.1-flash GSM8K 92.87 to 94.31. Gains are negligible on fill-only models (Dream-7B -0.30, LLaDA-8B-Base +0.08). Cost: 2.0 to 3.1x the NFE of T2T; GSM8K latency 0.92 to 1.99 s, throughput 321 to 147 tok/s. [T2M](https://arxiv.org/html/2604.18738)
- T2M evidence on error locality: the paper does not directly measure propagation. Only 1.2% of remasked positions refill with a different token (τ=0.9), and only 8.5% of immediate refills change the token. On GSM8K, error-to-correct flips (56) outnumber the reverse (35). Immediate refill (89.99) sits between immediate replace (89.39) and deferred refill (91.13), with gaps not individually significant. [T2M](https://arxiv.org/html/2604.18738)
- DARD, Dependency-Aware Revocable Decoding (arXiv 2608.26574; Park, ..., Byonghyo Shim; 2026-08-27): three states (masked / candidate / unmasked). Candidates are verified with a selective context that excludes less reliable tokens, via a fully masked shadow sequence and a confidence-ordered attention mask that approximates confidence-ordered AR decoding. Three dLLMs, 12 text and multimodal benchmarks. Flickr30K: 2.71x speedup and +4.35 CIDEr over Saber. No locality findings in the text read. [DARD](https://arxiv.org/html/2608.26574)
- Other self-correction titles seen, not read [UNVERIFIED]: Self-Correcting MDMs (arXiv 2602.11590), Generalized Discrete Diffusion with Self-Correction (2603.02230), Discrete Stochastic Localization (2602.16169), BackPlay head-only look-back (2601.06428), RIV recursive introspection (2509.23625). [2602.11590](https://arxiv.org/html/2602.11590v1), [2603.02230](https://arxiv.org/html/2603.02230), [2602.16169](https://arxiv.org/html/2602.16169v1), [BackPlay](https://pith.science/paper/2601.06428), [RIV](https://arxiv.org/html/2509.23625v1)

### Inferences
- The fact that remasking rarely changes tokens (1.2% to 8.5%) yet still yields gains suggests that most committed tokens are fine and errors are sparse. This is consistent with SPOT's 92 to 100% correct-token observation.
- ParallelBench's finding that ReMDM gives no gain on its Waiting Line tasks suggests remasking cannot fix high-total-correlation (globally coupled) errors like Shuffle. This is an inference across papers, not something tested jointly.

### Gaps
- No paper read reports an error-locality measurement, such as the distance between a wrong token and its causal source, or whether a fix requires re-decoding a contiguous span vs isolated positions.
- Revoked-token fractions for WINO and DARD are not reported.

## 4. What transfers between "block drafter for an AR target" and "decoding a block-diffusion LM"? Unifying papers?

### Takeaway
Several papers place drafting and verification in the same network: TiDAR, SSMD, and LLaDA2.1-style editing, plus "Teaching Diffusion to Speculate Left-to-Right" and DFlash on the AR-target side. Within block diffusion, SimSD, PV-Decoding and Trajectory SD bring AR-style SD machinery (trees, batched verification, Gumbel coupling) into SDAR/Fast-dLLM block decoding. TiDAR and SSMD are the clearest unifications: a bidirectional block drafts and a causal pathway verifies via rejection sampling.

### Cited Findings
- TiDAR (arXiv 2511.08923; NVIDIA; 2025-11-12): diffusion drafting and AR sampling in one forward pass with a structured mask (causal prefix, bidirectional last block). Drafts from the previous step are verified by rejection sampling against the AR distribution, while the model pre-drafts the next step for all acceptance outcomes. Exact KV cache. 1.5B (from Qwen2.5-1.5B, 50B tokens) and 8B (from Qwen3-8B, 150B tokens). Tokens/NFE 7.45 (1.5B) and 8.25 (8B). H100 bs1 throughput 4.71x over Qwen2.5-1.5B and 5.91x over Qwen3-8B. 8B generative average 65.31 vs Qwen3-8B 68.09; likelihood 75.40 vs 74.25; Dream 58.74/71.86 and LLaDA 41.78/68.06. Limitations: bs1 only, long context doubles sequence length (future work), PyTorch + Flex Attention without custom kernels. [TiDAR](https://arxiv.org/html/2511.08923v1)
- Teaching Diffusion to Speculate Left-to-Right (arXiv 2606.11552; Whalen, Ito, Sakamoto; 2026-06-10): a block-diffusion drafter with an AR verifier. Three training-time changes: token positional weighting, a first-error focal loss on the position that breaks the accepted prefix, and a chain loss as a differentiable surrogate for expected accepted length. Accepted length +21 to 76% per benchmark over position-uniform training across 6 tasks and 4 targets. Preserves the "rejection-sampling exactness contract". No wall-clock reported. [Teaching Diffusion L2R](https://arxiv.org/abs/2606.11552)
- DFlash: Block Diffusion for Flash Speculative Decoding (arXiv 2602.06036; ICML 2026), a block-diffusion drafter for AR targets (contents not read in this pass; covered by adjacent researcher). [DFlash](https://arxiv.org/html/2602.06036)
- SSMD (Section 1) is the dLLM-native version of the same pattern: a factorized non-causal draft is verified by a causal non-factorized head. [SSMD](https://arxiv.org/html/2510.03929v2)
- SimSD targets SDAR block diffusion and uses reference-token insertion with attention masks to verify drafts in one pass, borrowing from AR SD. [SimSD](https://arxiv.org/abs/2606.02544)
- SPOT uses shared-Gumbel coupling for temperature-agnostic verification of block states, analogous to typical acceptance in AR SD. APD uses Gumbel coupling between the dLLM and a small AR model. [SPOT](https://arxiv.org/abs/2610.22543), [APD](https://arxiv.org/html/2506.00413v2)

### Inferences
- What transfers from dLLM decoding to block drafters: commit-order selection (SPOT, SSD), confidence-threshold acceptance, dependency gating (DAPD, DARD), and remask/verify via shadow blocks (WINO). What transfers the other way: rejection sampling for exactness (TiDAR, SSMD), tree verification (SSD, Spiffy, SPOT), and training for accepted length (Teaching L2R).
- Teaching L2R's first-error focal loss reflects the prefix-acceptance structure of AR verification. dLLM-target verifiers (Spiffy, SPOT) accept unordered block states, so a commit-order objective is the analogue. This is inferred, not claimed by the papers.

### Gaps
- No single paper found that formally treats "block drafter for AR target" and "block-diffusion LM decoding" as one problem with shared theory, other than ParallelBench-style independence bounds, which only cover the dLLM side.

## 5. Recurring limitations and open problems the authors list

### Takeaway
The recurring items are:
- Acceptance capped by greedy top-1 / out-of-order drafts.
- Exponential tree costs that become memory-bound.
- Speedups measured in NFE rather than wall clock.
- Missing kernel support (FlashAttention disabled by custom masks).
- Shrinking gains on long prompts or larger batches.
- Domain-dependent threshold tuning.
- Benefits that depend on base-model strength or correction training.
- No practical way to reach the oracle per-sample parallelism.

### Cited Findings
- Greedy/top-1 cap and out-of-order failure; exponential k-ary trees; mix-order cost; long-prompt shrinkage (2.4x to 1.5x); model-dependent N. [SSD](https://arxiv.org/html/2510.04147v1)
- FlashAttention disabled; time linear in draft count; NFE not wall clock; no calibration for parallel decoding; KV cache untested. [Spiffy](https://arxiv.org/html/2509.18085v3)
- Trajectory drift as the fundamental cost of parallelism. [Trajectory SD](https://arxiv.org/abs/2608.27514)
- Worst-case bounds under idealized joint consistency; AR beats PrefixCache at large batch; block-size sensitivity. [Fast-dLLM](https://arxiv.org/html/2505.22618v3)
- Oracle adaptive parallelism unattained; theory assumes an ideal model; short-output focus; remasking (ReMDM) does not help. [ParallelBench](https://en.papernotes.org/ICLR2026/llm_evaluation/parallelbench_understanding_the_trade-offs_of_parallel_decoding_in_diffusion_llm/)
- Speedup bounded by base-model capability; future work on draft-and-verify-aware training. [WINO](https://arxiv.org/html/2507.18578)
- Remasking needs correction training; 2.0 to 3.1x NFE; NFE/token can exceed the AR reference of 1.0; GSM8K-only tuning. [T2M](https://arxiv.org/html/2604.18738)
- S-mode unreliable for chat; domain-dependent thresholds. [LLaDA2.1 review](https://liner.com/review/llada21-speeding-up-text-diffusion-via-token-editing)
- APD does not improve on the base dLLM; lower parallelism for open-ended writing. [APD](https://arxiv.org/html/2506.00413v2)
- Multi-batch verification slower in current runtimes. [PV-Decoding](https://www.opentrain.ai/papers/pv-decoding-progressive-verification-decoding-with-decode-gating-for-diffusion-l--arxiv-2610.24722/)
- TiDAR: batch-size-1 only, long-context cost, no custom kernels. [TiDAR](https://arxiv.org/html/2511.08923v1)
- dParallel: accuracy drop on MBPP (42.4 to 40.8) despite the "maintained accuracy" claim. [dParallel](https://arxiv.org/html/2509.26488v1)

### Inferences
- Wall-clock vs NFE is the most common reporting mismatch. Papers that report wall clock (SSD TPS, T2M latency, PV-Decoding throughput) show much smaller gains than NFE-based headlines.
- Nobody reports serving-scale (batch > 1) gains for dLLM self-speculation. Fast-dLLM's own data suggests the advantage over AR narrows at higher batch.

### Gaps
- Few-step distillation coverage is partial: CDLM / "Consistent Diffusion Language Models" (ICML 2026, [proceedings](https://proceedings.mlr.press/v306/amin26a.html)) and arXiv 2511.19269 were seen but not read. d3LLM, DiDi-Instruct and SDTT were not covered.
- Mercury, Gemini Diffusion and Seed Diffusion: no primary acceleration-method details were obtained. Only ParallelBench's Mercury Shuffle failure is cited.
- SDAR's own decoding numbers were not fetched.
