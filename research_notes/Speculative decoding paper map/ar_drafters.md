# Speculative decoding with autoregressive drafters: separate draft models, drafter alignment, and draft heads (curated list, as of Oct 2026)

Verification legend used per entry:
- [V-full] arXiv HTML/abs page fetched this session; numbers taken from it.
- [V-abs] arXiv abs page fetched this session (abstract-level only; setting details may be incomplete).
- [V-id] arXiv ID + title confirmed via search result only; method/result details are from prior knowledge and must be re-checked against the PDF before the downside analysis.
- "(inferred)" = my own observation, not stated by the authors.

Note on tooling: direct arXiv API / GitHub API access was blocked in this environment, so the hemingkx/SpeculativeDecodingPapers README could not be read; extension candidates came from web search instead.

Core list = 22 papers (marked [CORE]); 3 borderline entries are listed separately at the end of section 3.

## 1. Foundations and separate small draft models (incl. drafter alignment / distillation)

### Takeaway
The lossless draft-then-verify scheme (Leviathan; Chen) is the base for everything; the separate-drafter line since then is mostly about aligning the drafter to the target distribution (DistillSpec, Direct Alignment/TVD++, AdaSPEC, OSD), plus a scaling-law view of drafter size/data (Scylla). Most of these papers evaluate on small or older model pairs (T5, Pythia, Llama-2-7B), at batch 1.

### Cited Findings

- **[CORE] Fast Inference from Transformers via Speculative Decoding** (Leviathan, Kalman, Matias; Google). arXiv 2211.17192, ICML 2023 (oral), Nov 2022. [V-id]
  - Method: small AR drafter proposes gamma tokens, target verifies in parallel; modified rejection sampling keeps output distribution identical.
  - Result: ~2x-3x speedup on T5-XXL (11B) vs. the standard T5X implementation, TPU-v4, batch 1 (prior knowledge, re-check).
  - Code: none official.
  - Stated limitations: speedup requires spare compute; extra arithmetic ops increase total compute (prior knowledge).
  - Weaknesses (inferred): assumes a well-matched small drafter from the same family exists; analysis assumes i.i.d. acceptance rate per token.
  - Source: [arXiv 2211.17192](https://arxiv.org/abs/2211.17192)

- **[CORE] Accelerating Large Language Model Decoding with Speculative Sampling** (Chen, Borgeaud, Irving, Lespiau, Sifre, Jumper; DeepMind). arXiv 2302.01318, arXiv only, Feb 2023. [V-id]
  - Method: concurrent independent derivation of lossless speculative sampling.
  - Result: 2-2.5x decoding speedup on Chinchilla 70B with a 4B drafter, distributed TPU setup (prior knowledge, re-check).
  - Code: none.
  - Stated limitations: none stated beyond needing a drafter that is fast relative to target (re-check).
  - Weaknesses (inferred): one proprietary model pair; drafter trained specifically for this, cost not amortized in the speedup figure.
  - Source: [arXiv 2302.01318](https://arxiv.org/abs/2302.01318)

- **[CORE] DistillSpec: Improving Speculative Decoding via Knowledge Distillation** (Zhou et al.; Google). arXiv 2310.08461, ICLR 2024, Oct 2023. [V-id]
  - Method: KD of drafter on on-policy (drafter-generated) data with divergence chosen per task/decoding strategy (FKL, RKL, JSD, TVD).
  - Result: 10-45% speedup over standard SD across benchmarks; with lossy SD, 6-10x latency reduction (T5 family, prior knowledge, re-check).
  - Code: none official found.
  - Stated limitations: best divergence depends on task and decoding strategy (prior knowledge).
  - Weaknesses (inferred): requires target-model logits for every training token (expensive for large targets); tested mostly on T5/GPT-2-scale pairs.
  - Source: [arXiv 2310.08461](https://arxiv.org/abs/2310.08461)

- **[CORE] Online Speculative Decoding (OSD)** (Liu et al.; UC Berkeley). arXiv 2310.07177, ICML 2024, Oct 2023. [V-id]
  - Method: continually distill the drafter on live query distribution using verification results (rejected tokens) during serving.
  - Result: acceptance rate up by 0.1-0.65, latency reduced 1.42x-2.17x (Vicuna-7B / LLaMA-160M type pairs, prior knowledge, re-check).
  - Code: github.com/LiuXiaoxuanPKU/OSD (prior knowledge).
  - Stated limitations: re-check (paper discusses idle-compute assumption).
  - Weaknesses (inferred): needs spare GPU capacity for online training; risk of drift/forgetting under shifting traffic; weight update/version consistency in multi-replica serving not addressed.
  - Source: [arXiv 2310.07177](https://arxiv.org/abs/2310.07177)

- **[CORE] Direct Alignment of Draft Model for Speculative Decoding with Chat-Fine-Tuned LLMs** (Goel, Gagrani, Jeon, Park, Lee, Lott; Qualcomm AI Research). arXiv 2403.00858, arXiv (v3 Mar 2024; venue not stated on HTML page), Mar 2024. [V-full]
  - Method: 3-phase pipeline: pretrain 115M drafter, build distillation set from target generations, fine-tune with TVD++ loss (variance-reduction from RL).
  - Result: block efficiency up to 2.3 (gamma=3), up to 2.4x speedup vs AR, target Llama 2 Chat 7B; training on 32 A100 (pretrain) / 8 A100 (fine-tune).
  - Code: none found.
  - Stated limitations: "all finetuned draft models were outperformed by the base model for this particular task since those models were not directly fine-tuned on the translation task, making the task OOD."
  - Weaknesses (inferred): drafter must be pretrained from scratch (tokenizer-matched); single 7B target; no batch>1 serving results.
  - Source: [arXiv HTML 2403.00858v1](https://arxiv.org/html/2403.00858v1), [PDF v3](https://arxiv.org/pdf/2403.00858)

- **[CORE] AdaSPEC: Selective Knowledge Distillation for Efficient Speculative Decoders** (Hu, Guo et al.). arXiv 2510.19779, Oct 2025; venue not shown on abs page fetched (re-check; possibly NeurIPS 2025). [V-abs]
  - Method: KD with selective token filtering (drop hard-to-fit tokens) so a small drafter spends capacity on tokens it can match, targeting acceptance rather than full-KL.
  - Result: acceptance rate up to 15% higher than DistillSpec-style baselines; pairs 31M/1.4B and 350M/2.7B (Pythia-like scale), arithmetic, instruction following, code, summarization.
  - Code: [github.com/yuezhouhu/adaspec](https://github.com/yuezhouhu/adaspec)
  - Stated limitations: not seen (abstract only).
  - Weaknesses (inferred): very small model pairs, no modern 7B+ targets; filtered tokens are exactly where acceptance fails, so gains in average acceptance may hide tail behavior; per-task fine-tuning.
  - Source: [arXiv 2510.19779](https://arxiv.org/abs/2510.19779)

- **[CORE] Scaling Laws for Speculative Decoding (Scylla)** (Yan, Zhu et al.). arXiv 2505.07858, arXiv only (comments: 17 pages, 8 figures), May 2025. [V-abs]
  - Method: log-linear scaling laws for drafter acceptance rate vs. pretraining tokens, drafter capacity, decoding batch size; builds "Scylla" system.
  - Result: 1.5-2.2x higher acceptance rate than EAGLE-2 (Llama2/3, Qwen2.5), ~2x throughput in real deployment (setting not given in abstract).
  - Code: "Code will be released later" (none found).
  - Stated limitations: not seen.
  - Weaknesses (inferred): "acceptance rate" comparison vs EAGLE-2 may not translate to wall-clock (bigger drafter costs more); deployment numbers lack public setting; no code.
  - Source: [arXiv 2505.07858](https://arxiv.org/abs/2505.07858)

### Inferences
- The separate-drafter alignment line keeps re-discovering that KL is a proxy for acceptance; LK Losses (sec. 3) and AdaSPEC/TVD++ attack that directly. This is a common thread for the downside analysis: objective mismatch vs. acceptance.
- Almost all separate-drafter papers report batch-1 latency; serving-level (batch>1) evidence is thin.

### Gaps
- Exact settings for Leviathan, Chen, DistillSpec, OSD were not re-fetched (proxy blocked arXiv API); verify numbers before quoting.
- AdaSPEC venue not confirmed.
- FastDraft ("FastDraft: How to Train Your Draft", Findings of ACL 2025, seen only as a search title: [ACL Anthology PDF](https://aclanthology.org/anthology-files/pdf/findings/2025.findings-acl.1156.pdf)) is a possible addition for drafter pretraining recipes; arXiv ID not verified here.

## 2. Draft heads attached to the target (Medusa family, EAGLE family, successors)

### Takeaway
Draft heads reuse target hidden states. The line goes from parallel independent heads (Medusa) to sequentially dependent heads (Hydra, Clover, ReDrafter), then feature-level AR drafting (EAGLE), dynamic trees (EAGLE-2), and training-time-test with multi-layer fusion (EAGLE-3, up to ~6.5x at batch 1). HASS and GRIFFIN fix train/inference context and token misalignment; Falcon goes semi-AR. The common weakness is that speedups shrink sharply with batch size (EAGLE-3 itself drops to 1.01x at batch 56 in its vLLM table).

### Cited Findings

- **[CORE] Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads** (Cai et al.). arXiv 2401.10774, ICML 2024, Jan 2024. [V-id]
  - Method: K extra FFN heads predict tokens t+2..t+K+1 independently from last hidden state; tree attention verification; Medusa-1 (frozen backbone) and Medusa-2 (joint fine-tune), typical acceptance option.
  - Result: Medusa-1 >2.2x, Medusa-2 2.3-3.6x; Vicuna-7B/13B/33B, Zephyr-7B; single A100, batch 1 (prior knowledge, re-check).
  - Code: github.com/FasterDecoding/Medusa
  - Stated limitations: re-check (batch>1 left to future work per prior knowledge).
  - Weaknesses (inferred): heads are conditionally independent of each other, so acceptance decays fast with depth; Medusa-2 and typical acceptance are not lossless w.r.t. original model.
  - Source: [arXiv 2401.10774](https://arxiv.org/abs/2401.10774)

- **[CORE] Hydra: Sequentially-Dependent Draft Heads for Medusa Decoding** (Ankner et al.). arXiv 2402.05109, COLM 2024, Feb 2024. [V-id]
  - Method: draft heads conditioned on previously drafted tokens (sequential dependency); Hydra++ adds teacher distillation and extra decoder layer.
  - Result: Hydra++ up to ~1.31x throughput over Medusa and ~2.7x over AR (prior knowledge, re-check).
  - Code: github.com/zankner/hydra (prior knowledge).
  - Weaknesses (inferred): evaluated mainly on Vicuna family, batch 1; superseded by EAGLE-style feature drafting.
  - Source: [arXiv 2402.05109](https://arxiv.org/abs/2402.05109)

- **[CORE] EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty** (Li, Wei, Zhang, Zhang). arXiv 2401.15077, ICML 2024, Jan 2024. [V-id]
  - Method: one-layer AR drafter over target second-to-top-layer features plus token shifted by one step to resolve feature uncertainty; static tree.
  - Result: 2.7x-3.5x latency speedup on LLaMA2-Chat 70B, ~2x throughput; trainable in 1-2 days on 8x RTX 3090 (prior knowledge, re-check).
  - Code: [github.com/SafeAILab/EAGLE](https://github.com/SafeAILab/EAGLE)
  - Weaknesses (inferred): drafter tied to a specific target (retrain per target/fine-tune); trained on fixed ShareGPT data, so exposure bias at deep draft positions.
  - Source: [arXiv 2401.15077](https://arxiv.org/abs/2401.15077)

- **[CORE] EAGLE-2: Faster Inference of Language Models with Dynamic Draft Trees** (Li et al.). arXiv 2406.16858, EMNLP 2024, Jun 2024. [V-id]
  - Method: context-aware dynamic draft tree using drafter confidence as acceptance proxy; reranking to pick verified nodes. No retraining over EAGLE.
  - Result: 3.05x-4.26x speedup, 20-40% faster than EAGLE-1 (prior knowledge, re-check).
  - Code: same EAGLE repo.
  - Weaknesses (inferred): relies on drafter calibration; tree expansion cost grows at batch>1 where verification is no longer free. (Tree aspects overlap with the tree-methods researcher.)
  - Source: [arXiv 2406.16858](https://arxiv.org/abs/2406.16858)

- **[CORE] EAGLE-3: Scaling up Inference Acceleration of Large Language Models via Training-Time Test** (Li, Wei, Zhang, Zhang). arXiv 2503.01840, Mar 2025; NeurIPS 2025 (prior knowledge, re-check). [V-full]
  - Method: drop feature-prediction loss, predict tokens directly; fuse low/mid/high-layer target features; "training-time test" feeds drafter its own outputs during training.
  - Result (T=0, batch 1): Vicuna-13B 5.58x, LLaMA-3.1-8B-Instruct 4.40x, LLaMA-3.3-70B-Instruct 4.11x, DeepSeek-R1-Distill-LLaMA-8B 4.05x, peak 6.47x on HumanEval; T=1: Vicuna-13B 4.57x, LLaMA-3.1-8B 3.07x.
  - Throughput (vLLM, chain length 2, no tree): 1.75x @bs2, 1.49x @bs16, 1.36x @bs32, 1.21x @bs48, 1.01x @bs56.
  - Code: [github.com/SafeAILab/EAGLE](https://github.com/SafeAILab/EAGLE)
  - Stated limitations: "Due to the GPU constraint, we are unable to test EAGLE-3 on the 405B and 671B models"; throughput runs "did not use the tree structure, and the maximum chain length was set to 2".
  - Weaknesses (inferred): gain almost vanishes at batch ~56; per-target retraining with sizeable data; T=1 speedups notably lower than greedy.
  - Source: [arXiv HTML 2503.01840v1](https://arxiv.org/html/2503.01840v1)

- **[CORE] Learning Harmonized Representations for Speculative Sampling (HASS)** (Zhang, Wang, Huang, Xu). arXiv 2408.15766, ICLR 2025, Aug 2024 (v3 Feb 2025). [V-abs]
  - Method: harmonized objective distillation (rank-focused top-K distillation) + harmonized context alignment (multi-step training with drafter's own features), no inference overhead.
  - Result: 2.81x-4.05x wall-clock speedup averaged over three datasets, 8%-20% over EAGLE-2, four LLaMA variants.
  - Code: [github.com/HArmonizedSS/HASS](https://github.com/HArmonizedSS/HASS)
  - Stated limitations: not seen (abstract only).
  - Weaknesses (inferred): multi-step alignment raises training cost roughly linearly in alignment steps; largely subsumed by EAGLE-3's training-time test.
  - Source: [arXiv 2408.15766](https://arxiv.org/abs/2408.15766), [PDF (ICLR 2025 header)](https://arxiv.org/pdf/2408.15766)

- **[CORE] GRIFFIN: Effective Token Alignment for Faster Speculative Decoding** (Hu, Li, Xie, Lu, Toh, Zhou). arXiv 2502.11018, NeurIPS 2025 (per official repo), Feb 2025 (v3 Oct 2025). [V-abs]
  - Method: token-alignable training (loss masking of misaligned tokens) + draft architecture that uses input tokens to fix feature inconsistency.
  - Result: >8% average acceptance length and >7% speedup over prior SOTA; LLaMA, Vicuna, Qwen, Mixtral.
  - Code: [github.com/hsj576/GRIFFIN](https://github.com/hsj576/GRIFFIN)
  - Weaknesses (inferred): incremental single-digit gains; masking discards training signal exactly on hard tokens (same concern as AdaSPEC).
  - Source: [arXiv 2502.11018](https://arxiv.org/abs/2502.11018), [GitHub](https://github.com/hsj576/GRIFFIN)

- **[CORE] Falcon: Faster and Parallel Inference of LLMs through Enhanced Semi-Autoregressive Drafting and Custom-Designed Decoding Tree** (Gao, Xie, Xiang, Ji). arXiv 2412.12639, AAAI 2025, Dec 2024. [V-abs]
  - Method: semi-AR drafter (multiple tokens per drafter pass) with Coupled Sequential Glancing Distillation; custom decoding tree; 2-layer Transformer drafter.
  - Result: lossless 2.91x-3.51x on Vicuna and LLaMA2-Chat, MT-Bench/HumanEval/GSM8K, beats EAGLE/Medusa/Lookahead.
  - Code: not found this session.
  - Weaknesses (inferred): compared against EAGLE-1 era baselines only; older targets.
  - Source: [arXiv 2412.12639](https://arxiv.org/abs/2412.12639), [AAAI OJS](https://ojs.aaai.org/index.php/AAAI/article/view/34566/36721)

- **[CORE] Recurrent Drafter for Fast Speculative Decoding in LLMs (ReDrafter)** (Cheng, Zhang, Zhang, Wang, Wang; Apple). arXiv 2403.09919, arXiv only (PDF header "Under Review", v5 Dec 2024), Mar 2024. [V-abs]
  - Method: RNN drafter conditioned on target hidden states, beam search + dynamic tree attention removing duplicate prefixes, KD from target.
  - Result: up to 2.8x on Vicuna with PyTorch on H100; 2.3x on Apple Silicon with MLX. Integrated into NVIDIA TensorRT-LLM (press: [GIGAZINE](https://gigazine.net/gsc_news/en/20241219-apple-nvidia-redrafter-tensorrt-llm/)).
  - Code: github.com/apple/ml-recurrent-drafter ([MLX README](https://github.com/apple/ml-recurrent-drafter/blob/main/recurrent_drafting/mlx/experiments/README.md))
  - Weaknesses (inferred): beam search drafting is compute heavy at batch>1; Vicuna-only headline.
  - Source: [arXiv 2403.09919](https://arxiv.org/abs/2403.09919)

### Inferences
- The dominant open issue across the head family is batch-size scaling: EAGLE-3's own vLLM table shows the gain collapsing to ~1x at batch 56. Most other head papers only report batch 1.
- Per-target drafter training is a recurring cost: every new fine-tune of the target needs a new head (motivates EDA and online methods in sec. 3).
- HASS, GRIFFIN, GTO (and EAGLE-3's training-time test) all address the same root issue, train/inference mismatch of the drafter's own context.

### Gaps
- Medusa, Hydra, EAGLE, EAGLE-2 numbers not re-fetched; EAGLE-3 venue (NeurIPS 2025) not confirmed on a fetched page.
- Clover-2 (2408.00264) and other head variants (KOALA 2408.08146, CORAL 2502.16880, Gumiho) not evaluated; seen only as search titles for KOALA/CORAL.

## 3. 2025-2026 drafter training objectives and online / adaptive drafters

### Takeaway
Recent work optimizes acceptance more directly (LK Losses: TV-based; VAT: verification simulation; GTO: tree-level reward; Draft-OPD: on-policy distillation) or adapts drafters after deployment (OnlineSPEC, EDA). Gains over EAGLE-3 are typically +5-25% and almost all are reported at batch 1 / A100 with Qwen3 or LLaMA-3.1 targets.

### Cited Findings

- **[CORE] Bridging Draft Policy Misalignment: Group Tree Optimization for Speculative Decoding (GTO)** (Hu, Li, Lu, Zhou). arXiv 2509.22134, Sep 2025; ICLR 2026 per a third-party note ([papernotes ICLR2026](https://en.papernotes.org/ICLR2026/information_retrieval/bridging_draft_policy_misalignment_group_tree_optimization_for_speculative_decod/)), not confirmed on OpenReview. [V-full]
  - Method: sampling-free draft-tree reward + group-based policy optimization so drafter training matches tree decoding.
  - Result: +7.4% acceptance length, +7.7% speedup over EAGLE-3 (averaged); LLaMA-3.1-8B, LLaMA-3.3-70B, DeepSeek-R1-Distill-LLaMA-8B, Vicuna-1.3-13B; 1x A100 80GB (2 for 70B), batch 1, T in {0,1}.
  - Code: not on HTML page; a DeepWiki for hsj576/GTO exists ([deepwiki](https://deepwiki.com/hsj576/GTO)).
  - Stated limitations: "GTO increases training-time compute due to its two-phase procedure and the need to construct and evaluate grouped draft trees during training."
  - Weaknesses (inferred): batch 1 only; reward tied to a specific tree policy, may not transfer to other tree shapes or serving engines.
  - Source: [arXiv 2509.22134](https://arxiv.org/abs/2509.22134), [HTML v1](https://arxiv.org/html/2509.22134v1)

- **[CORE] LK Losses: Direct Acceptance Rate Optimization for Speculative Decoding** (Samarin et al.; Nebius). arXiv 2602.23881, ICML 2026, Feb 2026. [V-full] (added; not in original candidate list)
  - Method: hybrid KL + total-variation loss with adaptive schedule, directly optimizing acceptance rate.
  - Result: +0.5-8.2% average acceptance length across 6 targets (8B-685B) and 4 drafter types (EAGLE-3, Medusa, MLP speculator, DeepSeek-MTP); best on low-capacity drafters and T=1 (e.g. Qwen3-235B + EAGLE-3 +8.2% at T=1).
  - Code: weights/data on HF (nebius/lk-speculators, nebius/infinity-instruct-completions); no code repo seen.
  - Stated limitations: "focuses on a specific adaptive scheduler for the hybrid objective and uses a fixed exponential aggregation scheme across draft heads. Future work should explore alternative scheduler parameterizations, learnable or data-dependent per-head aggregation strategies..."
  - Weaknesses (inferred): reports acceptance length, wall-clock gains smaller; small gains at T=0.
  - Source: [arXiv HTML 2602.23881v2](https://arxiv.org/html/2602.23881v2)

- **[CORE] Draft-OPD: On-Policy Distillation for Speculative Draft Models** (SJTU, Shanghai AI Lab, Tsinghua, CUHK, PKU, ZJU). arXiv 2605.29343, arXiv only (v2), May 2026. [V-full]
  - Method: on-policy distillation with target-assisted rollouts, error-position replay, and acceptance-aware loss weighting of accepted vs. rejected tokens.
  - Result: >5x lossless acceleration; Qwen3-4B 5.96x, Qwen3-8B 5.73x; +23% over EAGLE-3 and +13% over DFlash at matched FLOPs; T=0 (thinking on) and T=0.6 (thinking off); GSM8K, MATH-500, AIME25, MBPP, HumanEval, SWE-Lite, MT-Bench.
  - Code: [github.com/bingyang-lei/Draft-OPD](https://github.com/bingyang-lei/Draft-OPD) (also Simplified-Reasoning/Draft-OPD); third-party single-RTX-3090 reproduction harness exists ([draft-opd-repro](https://github.com/Amirgh8080/draft-opd-repro)).
  - Stated limitations: fetch summary mentions training length, evaluation scope, lossless constraint, but no quote obtained (re-check).
  - Weaknesses (inferred): only small Qwen3 targets (4B/8B); hardware/batch not captured; on-policy rollouts add training cost.
  - Source: [arXiv HTML 2605.29343v2](https://arxiv.org/html/2605.29343v2), [HF page](https://huggingface.co/papers/2605.29343)

- **[CORE] Verification-Aware Training for Speculative Decoding (VAT)** (NAVER AI Lab, Korea Univ.). arXiv 2608.30135, arXiv only, Aug 31 2026. [V-full]
  - Method: simulate target verification at each training step; use accept/reject patterns as supervision via a verification head and adaptive per-position weighting.
  - Result: up to +11.4% acceptance length, up to +8.7% wall-clock; e.g. Qwen3-4B + EAGLE-3 4.07x -> 4.39x; Qwen3-4B/8B, LLaMA-3.1-8B; EAGLE-3 and DFlash drafters; A100 80GB, bf16, T=0.
  - Code: [github.com/naver-ai/VAT](https://github.com/naver-ai/VAT)
  - Stated limitations: no explicit limitations section; notes "verification head is not used at inference in main experiments".
  - Weaknesses (inferred): greedy only; modest gains; very recent (1 month), no independent replication.
  - Source: [arXiv HTML 2608.30135](https://arxiv.org/html/2608.30135)

- **[CORE] When Drafts Evolve: Speculative Decoding Meets Online Learning (OnlineSPEC)** (Qian, Wu, Fu, Zhang, Zhao; Nanjing Univ., UCSD). arXiv 2603.12617, ICML 2026, Mar 2026. [V-full]
  - Method: online-learning framework (dynamic regret) that evolves drafters from verification feedback during deployment; variants Opt-Hydra, Ens-EAGLE-3, Online-LR.
  - Result: up to 24% speedup over baselines on 7 benchmarks; e.g. Opt-Hydra Vicuna-13B 1.84x (tau 2.73); Ens-EAGLE-3 Vicuna-13B 1.46x; Qwen3-8B + Qwen3-0.6B drafter 1.41x; 4x A800 80GB.
  - Code: [github.com/ZinYY/OnlineSPEC](https://github.com/ZinYY/OnlineSPEC)
  - Stated limitations: no explicit section; notes "the draft model has significantly lower capacity and thus cannot globally match the target distribution across all possible contexts".
  - Weaknesses (inferred): absolute speedups (1.4-1.8x) far below offline EAGLE-3 numbers, so "online" gains are relative to weak baselines; inherits OSD's serving-cost and drift issues.
  - Source: [arXiv HTML 2603.12617](https://arxiv.org/html/2603.12617)

- **[CORE] Efficiently Aligning Draft Models via Parameter- and Data-Efficient Adaptation (EDA)** (Lin et al.; Xiamen Univ., SII, China Telecom TeleAI, USTC). arXiv 2603.09527, venue not stated, Mar 2026. [V-full]
  - Method: split drafter into frozen shared and trainable private parts, regenerate data with the (fine-tuned) target, select high-value samples; adapt drafter when target is fine-tuned.
  - Result: Qwen2.5-7B -> Qwen2.5-Math-7B, T=0: tau 4.79 vs 4.37, speedup 3.06x vs ~2.88x; 60.8% of full retraining cost, 127 MB trainable params.
  - Code: [github.com/Lyn-Lucy/Efficient-Draft-Adaptation](https://github.com/Lyn-Lucy/Efficient-Draft-Adaptation)
  - Stated limitations: no explicit limitations section.
  - Weaknesses (inferred): one base->fine-tune pair in the headline; still needs target regeneration of data.
  - Source: [arXiv HTML 2603.09527v1](https://arxiv.org/html/2603.09527v1)

#### Borderline (keep only if the team wants breadth)
- **Draft, Verify, & Improve: Toward Training-Aware Speculative Decoding (DVI)** (Bhansali, Heck; Georgia Tech). arXiv 2510.05421, venue not stated, Oct 2025. [V-full]. Self-speculative: Vicuna-7B split at layer 2, LoRA drafter heads trained online with KL->RL schedule from verifier accept/reject. Spec-Bench, k=4, greedy: 2.16x avg vs EAGLE-2 2.18x, ~60x fewer training exposures than Medusa. Code "will be released upon publication". No limitations stated. Inferred: single 7B model, no speedup over EAGLE-2; overlaps with self-speculative/early-exit scope. [HTML](https://arxiv.org/html/2510.05421)
- **Clover: Regressive Lightweight Speculative Decoding with Sequential Knowledge** (Xiao et al.; Baichuan/PKU). arXiv 2405.00263, May 2024, venue not seen. [V-abs]. Regressive connection + attention decoder + augmenting block on Medusa-style heads; up to 91% (Baichuan-Small) and 146% (Baichuan-Large) over baseline, up to 37%/57% over Medusa. Inferred: proprietary Baichuan models only, hard to compare. [arXiv](https://arxiv.org/abs/2405.00263)
- Note: DFlash (block-diffusion drafter) appears as a baseline in Draft-OPD and VAT; it is a diffusion drafter, so left to the diffusion/dLLM researcher.

### Inferences
- New training objectives (LK, VAT, GTO, Draft-OPD) are evaluated as add-ons to EAGLE-3 with 5-25% relative gains, mostly at batch 1, T=0, single A100, small Qwen3/LLaMA-3.1 targets. A shared downside to analyze: no serving-scale (batch, SGLang/vLLM) evaluation.
- Online drafter adaptation (OSD -> OnlineSPEC -> DVI) reports low absolute speedups; whether online updates beat a strong offline EAGLE-3 drafter is not demonstrated.
- Suggested final core list (22): Leviathan, Chen, DistillSpec, OSD, Direct Alignment, AdaSPEC, Scylla scaling laws, Medusa, Hydra, EAGLE, EAGLE-2, EAGLE-3, HASS, GRIFFIN, Falcon, ReDrafter, GTO, LK Losses, Draft-OPD, VAT, OnlineSPEC, EDA.

### Gaps
- Draft-OPD limitations not quoted; hardware/batch not captured.
- Venues of GTO (ICLR 2026?), AdaSPEC, EDA, Scylla unconfirmed.
- Could not read hemingkx/SpeculativeDecodingPapers README (GitHub access blocked), so other 2025-26 drafter-training papers there (e.g. "Flatter Tokens are More Valuable for Speculative Draft..." arXiv 2601.18902, BudgetDraft 2606.00144, seen only as search titles) were not assessed.
