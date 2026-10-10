# Discrete diffusion LMs: sampling, self-correction and test-time scaling, and transfer to a block-diffusion speculative drafter

Verification note: items marked [fetched] were checked against the primary source in this session (2026-10-10). Items marked [recalled] are from prior knowledge of the paper. The arXiv ID and link are given, but the numbers were not re-checked this session, so the report writer should treat them as lower confidence.

## 1. Masked diffusion LMs and block diffusion models (architecture, sampling, quality, speed)

### Takeaway
The field has moved from full-sequence masked diffusion (LLaDA, Dream) to block diffusion: blocks are generated autoregressively and positions inside a block are denoised in parallel (BD3-LM, SDAR, LLaDA2.x, Seed Diffusion). These models are usually converted from AR checkpoints. They decode with confidence-threshold parallel unmasking, and the newest ones (LLaDA2.1, Seed Diffusion) also edit tokens that were already decoded. Commercial dLLMs (Mercury 2, Gemini Diffusion) report roughly 1,000-2,000 tok/s. LLaDA2.1 shows that adding editing roughly doubles tokens-per-forward at near-equal quality.

### Cited Findings
**Background (pre-2025 or early-2025 foundations)**
- BD3-LM (Block Diffusion, ICLR 2025): interpolates between AR and diffusion by generating blocks autoregressively and running discrete diffusion inside each block. This supports KV caching and arbitrary length, and adds data-driven noise schedules that reduce gradient variance [recalled] — [arXiv:2503.09573](https://arxiv.org/abs/2503.09573)
- LLaDA 8B: masked diffusion model trained from scratch (2.3T tokens) with a bidirectional transformer, reported as competitive with LLaMA3-8B. It samples with low-confidence remasking and semi-AR blocks [recalled] — [arXiv:2502.09992](https://arxiv.org/abs/2502.09992)
- LLaDA 1.5: VRPO (variance-reduced preference optimization), an ELBO-based DPO for dLLMs, with gains on math, code and alignment over LLaDA [recalled] — [arXiv:2505.19223](https://arxiv.org/abs/2505.19223)
- LLaDA-MoE: about 7B-total / about 1B-active MoE masked diffusion model [recalled] — [arXiv:2509.24389](https://arxiv.org/abs/2509.24389)
- Dream 7B: diffusion LM initialized from Qwen2.5-7B AR weights, with context-adaptive token-level noise rescheduling [recalled] — [arXiv:2508.15487](https://arxiv.org/abs/2508.15487). APD confirms that "Dream was distilled from Qwen2.5 7B" [fetched] — [arXiv:2506.00413](https://arxiv.org/pdf/2506.00413)
- Mercury (Inception Labs, 2025): commercial diffusion LLM. The Mercury Coder Mini and Small tech report claims about 1,100 and about 740 tok/s on H100 [recalled] — [arXiv:2506.17298](https://arxiv.org/abs/2506.17298)
- Gemini Diffusion (Google DeepMind, May 2025): experimental text diffusion model. The blog reports about 1,479 tok/s sampling speed; there is no paper [recalled] — [DeepMind blog](https://deepmind.google/models/gemini-diffusion/)

**2025-2026**
- Seed Diffusion Preview (ByteDance Seed, Aug 2025): code-focused discrete-state diffusion LM, about 2,146 tok/s on H20 GPUs, "competitive" on code benchmarks. It claims a better speed-quality Pareto frontier than Mercury and Gemini Diffusion [fetched] — [arXiv:2508.02193](https://arxiv.org/abs/2508.02193), [Seed blog](https://seed.bytedance.com/zh/blog/seed-research-seed-diffusion-preview-released-a-diffusion-language-model-delivering-breakthrough-2-146-tokens-s-inference-speed)
  - Training details [recalled, not in the abstract]: a two-stage curriculum, first mask-based corruption and then edit-based (insertion/deletion/substitution) corruption, so the model learns to revise already-unmasked tokens. It also uses constrained-order training and on-policy learning to cut the number of steps, plus block-wise parallel sampling — same arXiv link. RemeDi independently notes that Seed Diffusion "lets all tokens be resampled at every step" [fetched] — [arXiv:2509.23653](https://arxiv.org/html/2509.23653v1)
- SDAR (Synergistic Diffusion-AutoRegression): converts a trained AR model into a blockwise diffusion model with lightweight adaptation. It is AR across blocks and parallel diffusion within a block, and appeared in ACL 2026 Findings [search snippet; numbers not verified] — [arXiv:2510.06303](https://arxiv.org/abs/2510.06303v3), [ACL Anthology](https://preview.aclanthology.org/ingest-acl/2026.findings-acl.1110/)
- LLaDA2.0 (Dec 2025): MoE dLLMs LLaDA2.0-mini (16B total) and LLaDA2.0-flash (100B total), converted from the AR models Ling-mini-2.0 and Ling-flash-2.0 [fetched] — [arXiv:2512.15745](https://arxiv.org/html/2512.15745v2)
  - The AR model is treated as block diffusion with block size 1. Conversion uses a Warmup-Stable-Decay block-size schedule: 1 → 4 → 32 → 64 → 4096 (the full sequence), held at 4096, then decayed back to 32 so the KV cache can be reused [fetched] — same link
  - CAP (Confidence-Aware Parallel) training adds an auxiliary confidence loss (L = L_SFT + λ·L_conf, inspired by dParallel) on correctly predicted tokens, to sharpen their confidence so more of them clear the parallel-decoding threshold. It raises tokens-per-forward for flash across 12 benchmarks [fetched] — same link
- LLaDA2.1 (Feb 2026), "Speeding Up Text Diffusion via Token Editing" [fetched] — [arXiv:2602.08676](https://arxiv.org/pdf/2602.08676)
  - Two thresholds are applied in the same step. A mask is filled if its top-1 probability exceeds τ_mask (M2T). An already-decoded token is replaced if the model's top-1 differs from it and exceeds τ_edit (T2T).
  - Training mixes M2T and T2T objectives: a drafting stream for masked positions and an editing stream that recovers originals from randomly perturbed tokens. It adds Multi-turn Forward augmentation and EBPO, an ELBO-based block-level clipped policy optimization for RL.
  - Results by size:

    | Size | Model / mode | Avg score | Avg TPF |
    |---|---|---|---|
    | Flash (100B) | LLaDA2.0 | 72.43 | 3.08 |
    | Flash (100B) | LLaDA2.1 S-mode (aggressive τ_mask, edits fix errors) | 72.34 | 5.93 |
    | Flash (100B) | LLaDA2.1 Q-mode | 73.54 | 3.64 |
    | Mini (16B) | LLaDA2.0 | 63.39 | 2.60 |
    | Mini (16B) | LLaDA2.1 S-mode | 62.07 | 5.34 |
    | Mini (16B) | LLaDA2.1 Q-mode | 63.90 | 3.12 |

  - HumanEval+ speed: 746.66 tok/s for flash (891.74 quantized); a peak of 1,586.93 tok/s for mini, quantized.
  - Multiple Block Editing (revising earlier blocks): flash average 70.69 → 72.67 while TPF drops 5.82 → 5.14; mini 57.63 → 58.24 while TPF drops 5.25 → 4.59.
  - Limitations: a very low τ_mask produces "rough drafts" with repetition artifacts, and thresholds have to be tuned per domain.
- Mercury 2 (Inception, released 2026-02-24): about 1,009 tok/s on Blackwell with about 1.7 s end-to-end latency. Company-reported scores: AIME25 91.1, GPQA-D 74, LiveCodeBench 67.3, IFBench 71.3, with a 128K context. It is described as refining multiple blocks in parallel and as able to "catch and fix" errors during generation, which is a marketing claim that has not been verified [fetched; secondary news source] — [NYU Shanghai RITS article](https://rits.shanghai.nyu.edu/ai/inception-launches-mercury-2-diffusion-powered-reasoning-at-1000-tokens-per-second/). Reports also mention a later Mercury 2.5 (about 1,107 tok/s by the company's count) [search snippet only] — [runtimewire](https://runtimewire.com/article/inception-mercury-2-5-diffusion-language-model-launch)
- Fast-dLLM: training-free acceleration with block-wise approximate KV cache plus confidence-threshold parallel decoding (unmask every token above a threshold). It became the standard baseline unmasking heuristic, and its reported throughput gains reach about 27x on LLaDA/Dream [recalled] — [arXiv:2505.22618](https://arxiv.org/abs/2505.22618). It is the baseline in the learned-unmasking-policy paper [fetched] — [arXiv:2512.09106](https://arxiv.org/html/2512.09106v3)

### Inferences
- The production trend matches the drafter setting closely. Block diffusion with block size about 32, converted from an AR model, decoded with confidence thresholds, and increasingly given a T2T editing head. The drafter is in effect a single-step, target-conditioned block diffusion model. The LLaDA2.1 recipe (draft aggressively, then edit) is the closest published analogue to draft → refresh → redraft.
- LLaDA2.1's S-mode result (about 2x TPF at a cost of about 0.1-1.3 points) is evidence that "low-quality first draft plus learned edits" is a good trade. In speculative decoding the target verifier removes the quality risk entirely, so the same aggressive setting is only a question of acceptance length.
- CAP-style confidence sharpening on correct tokens could be added to drafter training. A better-calibrated drafter makes any confidence-based re-draft or stop rule more reliable.

### Gaps
- Full architecture and benchmark numbers for SDAR, LLaDA-MoE, Dream and Gemini Diffusion were not re-verified this session. Gemini Diffusion has no paper.
- I found no 2026 "Gemini Diffusion 2" or Seed Diffusion full release with new numbers. A "DiffusionGemma" is mentioned in a Decrypt headline about Mercury 2 but was not verified ([decrypt](https://decrypt.co/371722/inception-labs-mercury-2-ai-beats-googles-diffusiongemma)).
- I did not retrieve LLaDA2.0's sampling thresholds or absolute TPS numbers; the text available to me was cut off.

## 2. Remasking and self-correction samplers (ReMDM, RemeDi, PRISM, informed correctors, P2, learned unmasking policies, token editing)

### Takeaway
There are three families. The first is training-free stochastic or heuristic remasking (ReMDM, low-confidence remasking, WINO). The second adds a learned per-token quality or correctness head trained with BCE on "is this token correct" labels, then remasks the lowest-scoring tokens (PRISM, RemeDi, LLaDA2.1's T2T stream). The third is a learned unmasking-order policy trained with RL (Learning Unmasking Policies, RemeDi's RL stage, P2 planners). Learned quality heads clearly beat ReMDM at low step counts. A tiny RL policy (about 300K parameters) over confidences alone matches Fast-dLLM in semi-AR mode and beats it clearly in full-diffusion mode.

### Cited Findings
**Background**
- Informed correctors: predictor-corrector sampling for masked discrete diffusion, where a corrector resamples positions chosen by model confidence (2024) [recalled] — [arXiv:2407.21243](https://arxiv.org/abs/2407.21243)
- ReMDM (Remasking Discrete Diffusion Models, 2025): a principled remasking sampler that can remask already-decoded tokens through a remasking schedule σ_t. It enables inference-time scaling with more steps (better MAUVE as steps increase) and needs no trained error detector [recalled; RemeDi describes it as "stochastic remasking at inference time, with no trained error detection" (fetched)] — [arXiv:2503.00307](https://arxiv.org/abs/2503.00307), [RemeDi](https://arxiv.org/html/2509.23653v1)
- P2 (Path Planning): splits each step into a planner that chooses which positions to unmask or remask and a denoiser. The planner can be the denoiser itself or an external pretrained model (e.g., a BERT-style model). P2 generalizes earlier samplers and allows remasking [recalled] — [arXiv:2502.03540](https://arxiv.org/abs/2502.03540)
- Generalized interpolating discrete diffusion (GIDD), which mixes masking and uniform noise to allow revision, and Edit Flows, edit-based diffusion with insert/delete/substitute [fetched as citations in RemeDi] — [RemeDi related work](https://arxiv.org/html/2509.23653v1). arXiv IDs for GIDD and Edit Flows were not retrieved this session.

**2025-2026**
- PRISM (Plug-in Remasking for Inference-time Self-correction of Masked diffusions, ICLR/ICML 2026) [fetched] — [arXiv:2510.01384](https://arxiv.org/pdf/2510.01384)
  - Method: adds a small adapter plus a sigmoid quality head on the backbone, alongside the unmasking head, in the same forward pass. Training pairs are built by masking clean data and unmasking some positions with the model under stop-gradient. Each position is labeled 1 if the clean token was recovered. The head is trained with BCE, with an MDM CE regularizer.
  - Guarantee: Proposition 3.1 proves the unique minimizer is the true per-token quality P(token correct | rest of the sequence with that position masked), whatever model generated the samples.
  - Inference: each step unmasks positions as usual and remasks the lowest-quality clean tokens.
  - Results:
    - OpenWebText 170M at 64 steps: MAUVE 0.132 vs ReMDM 0.016, gen-PPL 29.4 vs 60.4.
    - LLaDA-8B HumanEval at 256/512/1024 steps: 28.0/39.0/42.7 vs ReMDM 26.2/34.8/42.5.
    - Applying the head to a non-fine-tuned LLaDA raised HumanEval from 23.2% to 34.1% and fixed 19 of 26 syntax errors.
  - Limitation: the score is per-position, so it can miss global reasoning errors.
- RemeDi ("Don't Settle Too Early", ICLR 2026) [fetched] — [arXiv:2509.23653](https://arxiv.org/html/2509.23653v1)
  - Architecture: dual stream. The Token Prediction Stream is initialized from LLaDA-8B-Instruct, and the Unmasking Policy Stream is a 4-block network that outputs per-token confidence. Total size is 8.9B.
  - Each step, the policy stream ranks all tokens, including already-unmasked ones; high-confidence tokens are kept or unmasked and low-confidence tokens are remasked.
  - Remask SFT: inputs are noised with both [MASK] and random replacement tokens. The policy stream learns with BCE: keep clean tokens, remask corrupted tokens, soft target for masks.
  - Remask RL: GRPO over trajectories, with unmasking sampled via Plackett-Luce.
  - After RL, vs LLaDA / Dream:

    | Benchmark | RemeDi | LLaDA | Dream |
    |---|---|---|---|
    | GSM8K | 89.1 | 78.3 | 82.1 |
    | MATH | 52.9 | 38.9 | 49.6 |
    | HumanEval | 73.2 | 45.7 | 59.8 |
    | MBPP | 59.4 | 39.0 | 59.6 |
    | IFEval | 85.4 | 70.0 | 67.5 |

  - Remask SFT alone already beats vanilla SFT.
- Learning Unmasking Policies for Diffusion LMs (ICLR 2026; ICML 2026 oral) [fetched] — [arXiv:2512.09106](https://arxiv.org/html/2512.09106v3)
  - Policy: a 1-layer transformer, about 300K parameters (under 0.01% of the base), 128 hidden dimensions, 2 heads, adaLN. Inputs are only per-position max-softmax confidence, a still-masked flag and the timestep. Hidden-state inputs were worse and less stable.
  - Output and training: a Bernoulli unmask probability per position, trained with GRPO (group 8) against a frozen LLaDA-8B. The reward is correctness × a step penalty α; an additive reward was reward-hacked into unmasking everything at once.
  - Semi-AR mode (block 32): matches Fast-dLLM's accuracy-NFE frontier, and beats it at about 10 NFE with α=10.
  - Full-diffusion mode (length 256): about 50% GSM8K at about 12 NFE vs ≤30% for heuristics.
  - Transfer: transfers LLaDA → Dream, mostly. Math → code transfer is poor, so the policy has to be retrained per domain. It generalizes from length 256 to 512.
- LLaDA2.1 T2T editing: a large-scale learned token-editing mechanism (see Section 1). Edits fire when the model's top-1 differs from the current token and exceeds τ_edit [fetched] — [arXiv:2602.08676](https://arxiv.org/pdf/2602.08676)
- dUltra: RL for ultra-fast dLLM decoding, which appeared in the same search results as unmasking-policy work. Details were not retrieved [search result only] — [arXiv:2512.21446](https://arxiv.org/pdf/2512.21446)
- WINO (Wide-In, Narrow-Out): a training-free, revocable draft-and-verify decoder for dLLMs. It drafts many tokens in parallel aggressively and re-masks those the model no longer supports once more context is visible. Reported results are higher accuracy with several-fold fewer steps on LLaDA [recalled; numbers not re-verified] — [arXiv:2507.18578](https://arxiv.org/abs/2507.18578)

### Inferences
- The PRISM/RemeDi labeling recipe maps almost one-to-one onto the drafter problem. Label each drafted position by whether it matches the target's greedy or accepted token, and train a BCE quality head on the drafter's (or small estimator's) hidden states. PRISM's Proposition 3.1 says such a head converges to P(correct | context), whichever model produced the drafts. Drafts from the drafter itself or from the small estimator are both valid training data.
- The learned-policy paper suggests a cheap remask policy may not need hidden states at all. Confidences, a masked flag and position may be enough, and that is cheaper and more stable. The domain-transfer failure (math → code) is a warning: train the remask policy on the same data mix as the speculative-decoding evaluation.
- RemeDi's dual-noise SFT (masks plus random token replacements) is a direct template for teaching the drafter to redraft. Feed it its own wrong tokens plus refreshed context, and supervise both keep/remask decisions and corrected tokens. This targets the "dependency fix" setting that currently recovers at most ~36%.

### Gaps
- I did not find a paper that uses a separate, smaller model's per-token confidence as the remasking signal for a larger diffusion model. P2 allows an external planner, but I did not verify whether its experiments use a smaller model as planner.
- No source reports what fraction of remasked tokens actually change to the correct value (edit precision), which is the quantity that matters given the ~20% error rate of the small model's corrections.

## 3. Test-time scaling and search for diffusion LMs, and external-model guidance

### Takeaway
SMC and particle methods for masked diffusion are well developed. They include reward-tilted SMC with optimal or amortized proposals, particle Gibbs, self-rewarding SMC that uses trajectory confidence, and verifier-guided stratified search (S³). Gains are moderate on reasoning, for example S³ on LLaDA-8B MATH-500 25.6 → 30.2, and large on classifier rewards. The clearest case of an external AR model guiding a diffusion LM is APD: a small AR model (Qwen2.5-0.5B) forms a product-of-experts with Dream-7B, and a speculative-style coupling accepts parallel tokens. In other words, it is speculative decoding with the roles reversed.

### Cited Findings
- Ou, Pani, Li, "Inference-Time Scaling of Discrete Diffusion Models via Importance Weighting and Optimal Proposal Design" [fetched; experiment numbers not in the fetched excerpt] — [arXiv:2505.22524](https://arxiv.org/html/2505.22524v4)
  - SMC for pretrained discrete diffusion, with tractable importance weights for product targets (CFG-like) and reward-tilted targets exp(r(x_t)).
  - Two approximations of the optimal proposal: a first-order gradient proposal, and an amortized learned proposal trained to minimize the log-variance of the importance weights.
- Particle Gibbs for DDMs (PG-DDM, Dang et al.) [fetched] — [arXiv:2507.08390](https://arxiv.org/pdf/2507.08390)
  - Conditional SMC with a retained reference trajectory, iterated, with re-masking backward steps. Rewards come from external classifiers (RoBERTa toxicity, CoLA, sentiment), plus GPT-2 as a perplexity reward.
  - On MDLM at 16×1024 NFE:
    - Toxicity: 91.7 vs FK-Steering 37.7 vs BoN 14.0.
    - Gen-PPL at 128×1024 NFE: 10.9 vs 21.1 vs 23.0.
  - At the lowest budget, FK-Steering sometimes wins.
  - Compute should go to particles first, then to PG iterations.
- Self-Rewarding SMC for MDLMs (Feb 2026): confidence-based decoding acts like greedy decoding and collapses diversity. The method runs parallel particles, weights them by trajectory-level confidence (a self-reward), and resamples. It needs no extra training and no external reward. The fetched abstract has no numbers [fetched] — [arXiv:2602.01849](https://arxiv.org/abs/2602.01849v1)
- S³, Stratified Scaling Search (Apr 2026) [fetched] — [arXiv:2604.06260](https://arxiv.org/pdf/2604.06260)
  - Method: keeps a population of partial denoising trajectories and expands each into candidates. Each candidate is scored by decoding its one-step x̂₀ prediction and passing it through a reference-free heuristic verifier (structure, arithmetic consistency, confidence, non-degeneracy). Resampling uses Srinivasan sampling, and the final answer is a majority vote.
  - LLaDA-8B-Instruct results:

    | Benchmark | Baseline | S³ | Best-of-K |
    |---|---|---|---|
    | MATH-500 | 25.6 | 30.2 | 28.2 |
    | GSM8K | 68.16 | 70.21 | 69.56 |
    | ARC-C | 76.11 | 77.86 | 79.30 (best-of-K wins) |

  - Argument: best-of-K is limited because it keeps sampling from the same base distribution.
- APD, Adaptive Parallel Decoding (NeurIPS 2025) [fetched] — [arXiv:2506.00413](https://arxiv.org/pdf/2506.00413)
  - Target distribution: p_T(x) ∝ p_D(x)^R · p̂_AR(x)^(1−R), with Dream-7B-Instruct as p_D and Qwen2.5-0.5B as p̂_AR (same tokenizer).
  - Acceptance: Gumbel-softmax universal coupling samples dLLM drafts and mixture targets with shared noise, and accepts up to the first disagreement. The first token is always accepted.
  - Knobs: R (mixture weight), W (KV recompute window), M (max masked lookahead).
  - Baselines: Dream left-to-right with 256 steps scores GSM8K 0.832 at 10.1 tok/s; Qwen2.5-7B AR scores 0.854 at 38.6 tok/s.
  - APD: mean parallel tokens 7.62 on GSM8K (R=0.7, W=16, M=100), over 5 tokens per iteration at about 80% GSM8K, and 59 vs 37 tok/s against AR on an example (single A5000).
- RemeDi uses a reward model for open-ended RL and verifiable rewards for math and code. This is training-time, not inference-time, guidance [fetched] — [arXiv:2509.23653](https://arxiv.org/html/2509.23653v1)

### Inferences
- APD is the formal mirror image of the user's setup. There, a small AR model supplies the joint dependencies that a parallel diffusion model lacks, through a product of experts. Here, a small model estimates the target's context over drafted tokens. APD's coupling trick, with shared Gumbel noise between proposal and mixture, could let the drafter's redraft and the small model's correction be compared cheaply. Positions where the two coupled samples disagree are natural remask candidates.
- Self-rewarding SMC and S³ show that confidence- or verifier-weighted particle resampling over partial denoising states gives gains over best-of-N at equal compute. In speculative decoding the final verifier is exact and free in quality terms. Particles only cost drafter passes plus one target pass, which can verify a tree or batch of candidate blocks (DDTree-style).

### Gaps
- No paper found uses a strong AR LLM (such as the target model) as a reward or verifier inside dLLM denoising for reasoning tasks at scale. PG-DDM uses GPT-2 only as a perplexity reward.
- The quantitative results of arXiv:2505.22524 and arXiv:2602.01849 were not retrieved.

## 4. Transfer ideas for a DFlash-style block-diffusion drafter with draft → estimate → redraft

### Takeaway
The most transferable pieces are these. (a) A learned per-token quality or remask head, trained PRISM/RemeDi-style with BCE on "matches target" labels, to choose which drafted tokens to redraft after a context refresh. (b) A LLaDA2.1-style T2T edit threshold instead of full redraft. (c) Particle or tree search over draft blocks with exact target verification as the final step. (d) Using the small estimator's confidence (or its disagreement with the drafter) as a gate, so its ~20% wrong corrections are filtered rather than blindly injected.

### Cited Findings
- Per-token quality heads trained with BCE on correctness labels provably estimate P(correct | context), whichever model generated the samples. A head added to a non-fine-tuned model still corrected it (HumanEval 23.2 → 34.1) [fetched] — [PRISM, arXiv:2510.01384](https://arxiv.org/pdf/2510.01384)
- Remask-SFT with random-token corruption plus RL over unmasking decisions raises LLaDA-8B by about 11-28 points on GSM8K/HumanEval [fetched] — [RemeDi, arXiv:2509.23653](https://arxiv.org/html/2509.23653v1)
- A 300K-parameter policy over confidences alone, trained with GRPO and a multiplicative correctness × step reward, matches or beats threshold heuristics. Hidden-state inputs hurt [fetched] — [arXiv:2512.09106](https://arxiv.org/html/2512.09106v3)
- Dual thresholds (τ_mask to fill, τ_edit to replace a token whose top-1 changed) give about 2x TPF at near-equal quality. Revising earlier blocks (MBE) adds about +2 points at some TPF cost [fetched] — [LLaDA2.1, arXiv:2602.08676](https://arxiv.org/pdf/2602.08676)
- Product-of-experts with a small AR model plus coupled acceptance is a working recipe for injecting a small model's joint dependencies into a parallel drafter [fetched] — [APD, arXiv:2506.00413](https://arxiv.org/pdf/2506.00413)
- Trajectory-confidence particle weighting and verifier-guided resampling beat best-of-N at matched compute in most settings [fetched] — [arXiv:2602.01849](https://arxiv.org/abs/2602.01849v1), [arXiv:2604.06260](https://arxiv.org/pdf/2604.06260), [arXiv:2507.08390](https://arxiv.org/pdf/2507.08390)

### Inferences (concrete proposals; these are my design suggestions, not published results)
1. **Learned redraft mask after context refresh.**
   - Train a small head on the drafter's last hidden state at each drafted position. Optional inputs are the drafter's confidence, the small estimator's confidence, and a drafter-vs-estimator disagreement flag.
   - Label y_i = 1 if drafted token i equals the target's token given the verified prefix, i.e. the token that would be accepted.
   - Redraft only positions with low predicted quality, plus everything after the first low-quality position, since acceptance is prefix-based.
   - PRISM's result implies labels can come from offline target runs without RL.
   - Because acceptance is prefix-limited, a variant can predict the first-failure index directly (a hazard model over positions). This may be more sample-efficient than independent per-token BCE.
2. **Gate the small model's corrections, don't inject them blindly.**
   - The ~20% wrong-correction rate is the same failure that quality heads address. Treat each correction as an edit proposal and accept it into the conditioning context only if its estimated quality passes a τ_edit-style threshold, LLaDA2.1-style.
   - Otherwise keep the drafter's token but still pass the estimator's hidden state as soft context.
   - A PRISM-style head trained on the estimator's own outputs, labeled by target agreement, provides that gate.
3. **Use dual-noise SFT for the redraft pass.**
   - Train the drafter's redraft mode, RemeDi/Seed-Diffusion style, on inputs that contain both masks and plausible wrong tokens: its own first-pass errors and the small model's wrong corrections.
   - Supervise it to keep or overwrite each token.
   - This directly targets the dependency-fix ceiling (~36%) and teaches robustness to noisy estimated context.
4. **Product-of-experts redraft (APD-style).**
   - For positions after a context refresh, sample from p_drafter^R · p_estimator^(1−R) instead of trusting either model alone.
   - Use shared-Gumbel coupling between the drafter's first-pass sample and the mixture sample. Positions where the two disagree are exactly the redraft set, so no separate policy is needed.
   - R can be tuned per position by the quality head.
5. **SMC or particles over draft blocks with the target as exact verifier.**
   - Keep K candidate redraft blocks (particles), weighted by drafter trajectory confidence (self-rewarding SMC) or by the quality head's predicted accepted length. Resample after each estimate → redraft iteration, and send the top particles as a tree to one target verification pass.
   - Because target verification is lossless, the particle weights only affect speed, never output quality. That makes aggressive heuristics safe, and expected accepted length is the right objective (S³-style expected-reward resampling).
6. **RL the redraft policy on accepted length.**
   - Following Learning Unmasking Policies, train a tiny policy with GRPO and reward = accepted tokens × penalty(drafter passes + estimator passes). The multiplicative form avoids the reward hacking the authors saw with additive rewards.
   - Inputs could be only confidences, flags and position, the configuration that worked best there.
7. **Calibration training.** Add a CAP-style confidence-sharpening loss (LLaDA2.0) to the drafter so confidence-gated decisions (stop drafting, redraft, accept a correction) are better calibrated.

### Gaps
- No published work found applies learned remasking or quality heads inside a speculative drafter conditioned on target hidden states. The transfer ideas above are untested.
- It is unknown whether per-token quality heads stay accurate when the conditioning context (the estimated target states) is itself noisy. PRISM's guarantee assumes the context is the actual rest of the sequence.
- The cost-benefit of particles in speculative decoding depends on how much tree width the target can verify per pass on the available hardware. No diffusion-LM paper reports this.
