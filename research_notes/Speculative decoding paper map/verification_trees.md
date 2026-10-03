# Verification rules, draft-tree construction, and SD systems/serving (drafter-agnostic)

Scope: lossless verification rules, tree construction, and systems/serving papers for speculative decoding (SD), as of Oct 2026. Gathered for an MSc thesis team that will analyze each paper's downsides next.

Legend:
- [V] = arXiv ID/venue checked this session against arXiv, OpenReview, proceedings, mlanthology or GitHub.
- [A] = only the title/abstract or a search listing was seen; no body text read.
- [R] = details recalled, not re-fetched this session (arXiv rate-limited the fetches). Re-check before citing.
- "Inferred weakness" = my own reading, not stated by the authors.
- T>0 = gains only appear when sampling at temperature > 0. At greedy (T=0) every lossless rule reduces to "accept the longest exact-match path", so verification-rule papers give roughly nothing there.

Tiering for the 15-20 target. Core (18): SpecInfer, Sequoia, OPT-Tree, EAGLE-2, DeFT, SpecTr, Hu et al. multi-draft, Global Resolution, Block Verification, Traversal Verification, UniVer, HSD, PEARL, SSD, Yggdrasil, MagicDec, Batch SD Done Right, Latency model. Optional (5): RSD, Khisti multi-draft, Max-Speedup, Polybasic, SpecServe.

## Q1. Lossless verification rules (single-path, multi-draft, tree)

### Takeaway
Verification-rule research has moved through four steps: token-wise rejection sampling; multi-draft optimal transport (SpecTr, then the Hu et al. and Global Resolution optimality results); sequence-level joint verification (Block, Traversal, HSD); and a unified conditional-OT view (UniVer). Reported gains over the previous rule are small, about 2-12% in acceptance length and 2-7% in wall-clock speed. All of them need T>0 and vanish at greedy decoding.

### Cited Findings

**SpecTr: Fast Speculative Decoding via Optimal Transport**. arXiv 2310.15141 [V]. NeurIPS 2023. Oct 2023.
- Method: frames multi-draft verification as optimal transport with membership cost. The exact OTM-k is solved by LP; the approximate K-SEQ is (1-1/e)-optimal. — [NeurIPS PDF](https://proceedings.neurips.cc/paper_files/paper/2023/file/6034a661584af6c28fd97a6f23e56c0a-Paper-Conference.pdf)
- Result: PaLM-2-Gecko drafting for PaLM-2-Bison, K=8 drafts, L=8, T=1.0. 2.13x wall-clock, 1.37x over standard SD; block efficiency 4.0 vs 2.9. — [same]
- Code: none found.
- Stated limitations: OTM is exponential in k; wall-clock speedup is below block-efficiency speedup because of draft latency and overhead; K-SEQ is strictly worse than OTM in some cases; draft count is bounded by hardware parallelism. — [same]
- Inferred weaknesses: closed PaLM-2 models with no public code, so the result is hard to reproduce. Drafts are i.i.d. sequences, not a deduplicated tree.
- Needs T>0: yes.

**Towards Optimal Multi-Draft Speculative Decoding** (Hu, Zheng, Viswanathan, Chen, Rossi, Wu, Manocha, Huang). arXiv 2502.18779 [V]. ICLR 2025. Feb 2025.
- Method: computes the optimal multi-draft acceptance rate efficiently through the dual of the OT problem. Shows that sampling drafts without replacement beats sampling with replacement, and that existing verifiers do not reach the bound. — [mlanthology](https://mlanthology.org/iclr/2025/hu2025iclr-optimal/), [OpenReview](https://openreview.net/forum?id=9KxnxWOBA5)
- Result: the main contribution is a measured gap between existing verifiers and the optimum. Exact numbers not read (PDF fetch rate-limited) [A].
- Code: not found.
- Stated limitations: not read [A].
- Inferred weakness: it is an analysis tool more than a deployable verifier, and its bounds are per-step, not over the full tree.
- Needs T>0: yes.

**Global Resolution: Optimal Multi-Draft Speculative Sampling via Convex Minimization/Optimization**. arXiv 2511.15898 [V]. ICLR 2026. Nov 2025.
- Method: reduces the exponential OT linear program to a polymatroid / max-flow convex problem, which makes the optimal multi-draft verifier tractable. — [paper note](https://en.papernotes.org/ICLR2026/llm_efficiency/global_resolution_optimal_multi-draft_speculative_sampling_via_convex_optimizati/), [mlanthology](https://mlanthology.org/iclr/2026/thomas2026iclr-global/)
- Result: about 90% acceptance with per-token overhead under 100 ms on Llama-3 70B/8B and Gemma-2 27B/2B, up to 10 draft tokens; solves (k=1000, n=2) in 35 ms. — [paper note]
- Code: not found.
- Limitations as summarized by the paper note (secondary source): efficient only for i.i.d. drafts; 38% solver success at k=100, so it needs a fallback; no end-to-end throughput measured. — [paper note]
- Inferred weakness: about 100 ms per-token overhead is huge next to a ~20-50 ms decode step, so it is not yet practical online.
- Needs T>0: yes.

**Block Verification Accelerates Speculative Decoding** (Sun, Mendlovic, Leviathan, Aharoni, Beirami, Ro, Suresh; Google). arXiv 2403.10444 [V]. Venue: ICLR 2025 [R]; the arXiv page fetched does not state it.
- Method: verifies the whole draft block jointly instead of token by token. Provably optimal among single-draft lossless verifiers, and never worse than token-wise verification. — [arXiv HTML](https://arxiv.org/html/2403.10444v3)
- Result: PaLM-2 S target with XXS/XXXS drafters, γ=8, 8 datasets, sampling. +7-10% block efficiency (avg 8.3%), +5-8% wall-clock (avg 6.5%). — [same]
- Code: sketch Python only in Appendix A. A vLLM PR adds it to the V1 rejection sampler. — [vLLM PR #40819](https://github.com/vllm-project/vllm/pull/40819)
- Stated limitations: the greedy variant needs framework changes; multi-draft extension left to future work. — [arXiv HTML](https://arxiv.org/html/2403.10444v3)
- Inferred weaknesses: single chain only, no tree. Gain is bounded by about 10%. Closed PaLM models.
- Needs T>0: yes.

**Traversal Verification for Speculative Tree Decoding** (Weng, Hu, Chen, Liu, Mei, Qiu, Tian, Shi; Lenovo). arXiv 2505.12398 [V]. NeurIPS 2025 [V via mlanthology]. May 2025.
- Method: verifies leaf to root, accepting whole sequences using joint (sequence-level) probabilities. Lossless, and it generalizes block verification to trees. — [arXiv HTML](https://arxiv.org/html/2505.12398v2), [mlanthology](https://mlanthology.org/neurips/2025/weng2025neurips-traversal/)
- Result: Llama3.2-1B→Llama3.1-8B and Llama-68M→Llama2-7B; chain, binary and EAGLE sparse trees (depth 5); single RTX A6000; T=1.0. +2.2-5.7% acceptance length, +1.7-2.5% tokens/s. — [arXiv HTML]
- Code: not found in the paper.
- Stated limitations: extra compute overhead eats into gains; advantage narrows at lower temperature; T=0 not reported. — [same]
- Inferred weaknesses: about 2% real speedup is within run-to-run noise. Small-GPU, batch-1 evaluation only.
- Needs T>0: yes.

**UniVer: A Unified Perspective for Multi-step and Multi-draft Speculative Decoding** (Weng, Hu, Yairi). arXiv 2605.04543 [V]. arXiv only. May 2026.
- Method: casts tree verification as a conditional OT problem, jointly handling the multi-draft (siblings) and multi-step (depth) aspects. — [arXiv PDF](https://arxiv.org/pdf/2605.04543)
- Result: Vicuna-7B-v1.3 with an EAGLE drafter, binary tree depth 5 (32 leaves), T=1.0, RTX A6000. Acceptance length 3.29 vs 3.06 for RRSw (+7.5%), +2.5% vs Greedy, about 7% throughput (72.2 vs 67.5 tok/s). +4.2-8.5% across topologies; +8.5% on Llama3.1-8B. — [same]
- Code: not found.
- Stated limitation: "UniVer possesses no extra performance gain under or near temperature = 0". At T≤0.3, Traversal, Greedy and UniVer are nearly identical. — [same]
- Inferred weaknesses: same group as Traversal Verification, so its main baselines are its own. Batch-1, single consumer-class GPU only.
- Needs T>0: yes, explicitly.

**Overcoming Joint Intractability with Lossless Hierarchical Speculative Decoding (HSD)** (Zhou, Huang, Li, Wu, Wang, Zhang, Lin, Cheng; Qwen team co-authors). arXiv 2601.05724 [V]. Venue not stated (arXiv). Jan 2026.
- Method: provably lossless sequence-level verifier that balances excess and deficient probability mass across accessible branches, which avoids computing the intractable joint distribution. — [arXiv abs](https://arxiv.org/abs/2601.05724)
- Result: Qwen2.5 0.5B draft → 14B/32B/72B; single H20 (LLaMA runs on 8×H20). Single-draft: +6.2% block efficiency, +6.7% speed on average; HumanEval +9.5-12.3%; multi-draft +5.9% / +4.7%; with EAGLE-3, ">12%" decode speed. T∈{0.6, 0.8, 1.0}. — [arXiv HTML](https://arxiv.org/html/2601.05724)
- Code: [github.com/ZhouYuxuanYX/Hierarchical-Speculative-Decoding](https://github.com/ZhouYuxuanYX/Hierarchical-Speculative-Decoding)
- Stated limitations: no dedicated section. The authors say the EAGLE-3 gain "is likely influenced by sampling stochasticity and floating-point precision", because EAGLE-3's top-K drafting makes draft probabilities equal 1. Gains on CNN/DM are modest (about 3.3%). — [same]
- Inferred weaknesses: the headline EAGLE-3 number is admittedly confounded. No T=0 results.
- Relevance: the user's own spec-hierarchy HiSpec/HSD branch.
- Needs T>0: yes.

**Recursive Speculative Decoding: Accelerating LLM Inference via Sampling Without Replacement** (Jeon et al.). arXiv 2402.14160 [R: ID and title recalled, not fetched]. Venue unverified. Feb 2024. Optional.
- Method: builds the draft tree with sampling without replacement (Gumbel-top-k or a stochastic beam) and verifies with recursive rejection sampling. This is the "RRSw" baseline used by UniVer and Traversal.
- Result, code, stated limitations: not verified this session.
- Inferred weakness: recursive rejection is still per-token greedy across siblings, so it is not the multi-draft optimum shown by Hu et al. 2025.
- Needs T>0: yes.

**Multi-Draft Speculative Sampling: Canonical Decomposition and Theoretical Limits** (Khisti et al.). arXiv 2410.18234 [V: ID; "Published as a conference paper at ICLR 2025" header seen in search listing]. ICLR 2025. Oct 2024. Optional.
- Method: decomposes the optimal multi-draft scheme into importance-sampling token selection followed by single-draft SD. — [arXiv PDF listing](https://arxiv.org/pdf/2410.18234) [A]
- Result, code, stated limitations: not read [A].
- Needs T>0: yes.

**Max-Speedup Speculative Sampling: A Generic Tree Construction Principle**. OpenReview oGPeI321sI only [A: title only]. No arXiv ID found. OpenReview was behind a browser check, so the abstract and decision were not read. — [OpenReview](https://openreview.net/forum?id=oGPeI321sI). Optional. Exclude or mark it until someone reads it.

### Inferences
- Measured improvements shrink at each step: SpecTr about 37% over SD, then Block about 8%, then Traversal about 4%, then UniVer about 7% over RRSw, then HSD about 6%. The verification rule is close to saturated for single-model trees.
- Every verifier paper evaluates at T≥0.6, mostly T=1.0, batch 1, on small GPUs. None reports serving-level throughput under load.
- EAGLE-style top-K drafting assigns degenerate (one-hot) draft probabilities, which weakens probability-based verifiers. HSD says this outright; the RheoSampling title (2609.21827, [A]) also points to this "one-hot dilemma". This is an opening for new work.

### Gaps
- Exact results for Hu et al. 2025 and Khisti 2410.18234 not read (arXiv PDF rate-limited).
- Max-Speedup: no abstract, authors or venue obtained.
- RSD (2402.14160) venue and results not re-verified.
- Code repos for Traversal and UniVer not found. Not confirmed that none exist.

## Q2. Draft-tree construction (static, dynamic, hardware-aware)

### Takeaway
Tree construction has moved from fixed heuristics (SpecInfer) to DP-optimal static trees (Sequoia), greedy trees maximizing expected acceptance (OPT-Tree), confidence-driven dynamic trees (EAGLE-2) and latency-aware equal-growth trees (Yggdrasil). DeFT is the attention-kernel side of tree verification. Most of these rely on positional acceptance assumptions and are evaluated at batch 1.

### Cited Findings

**SpecInfer: Accelerating LLM Serving with Tree-based Speculative Inference and Verification** (Miao et al.). arXiv 2305.09781 [V: ID]. ASPLOS 2024 [R]. May 2023.
- Method: token trees merged from several small models or expansion configs; tree-attention parallel verification; multi-step speculative sampling (MSS) for lossless stochastic verification. [R]
- Result: 1.5-2.8x for distributed inference and 2.6-3.5x for offloading [R, not re-fetched].
- Code: FlexFlow / flexflow-serve [R].
- Stated limitations: not re-read.
- Inferred weakness: static expansion configs are not adaptive. MSS is later shown suboptimal by SpecTr and Hu et al.
- Needs T>0: MSS gains need T>0; tree width helps at T=0 too.

**Sequoia: Scalable, Robust, and Hardware-aware Speculative Decoding** (Chen, May, et al.). arXiv 2402.12374 [V]. NeurIPS 2024 [V]. Feb 2024.
- Method: dynamic programming for the optimal tree topology under a positional acceptance-rate vector; sampling-without-replacement verification; hardware-aware selection of tree size and depth. — [NeurIPS PDF](https://proceedings.neurips.cc/paper_files/paper/2024/file/ea1f5f0878d43ff4fb8bf64ef4a2326c-Paper-Conference.pdf)
- Result: 4.04x Llama2-7B and 3.73x Llama2-13B (A100, greedy); 2.27x Vicuna-33B (A100); 9.5x Llama3-70B-Instruct with offloading on L40 (0.60 s/token). — [same]
- Code: [github.com/Infini-AI-Lab/Sequoia](https://github.com/Infini-AI-Lab/Sequoia)
- Stated limitations: the positional-acceptance assumption ignores context; scalability theorems rely on a power-law acceptance assumption; the optimal tree depends on model pair, temperature and domain, so it must be recomputed per setting. — [same]
- Inferred weaknesses: the tree is static per configuration (context-blind). The offloading 9.5x is a very favourable regime that does not reflect GPU-resident serving.
- Needs T>0: no, it helps at T=0.

**OPT-Tree: Speculative Decoding with Adaptive Draft Tree Structure** (Wang et al.). arXiv 2406.17276 [V]. TACL 2025 [V via ACL Anthology]. Jun 2024. — [ACL Anthology](https://aclanthology.org/2025.tacl-1.8/)
- Method: at each decoding step, greedily builds the tree that maximizes the mathematical expectation of acceptance length under a node budget, using draft probabilities. [R]
- Result: up to about 3.2x reported [R, not re-fetched].
- Code: GitHub (Jikai0Wang/OPT-Tree) [R].
- Stated limitations: not read.
- Inferred weakness: it treats draft probability as acceptance probability (calibration assumption) and optimizes acceptance length, not latency.
- Needs T>0: no.

**EAGLE-2: Faster Inference of Language Models with Dynamic Draft Trees** (Li et al.). arXiv 2406.16858 [R]. EMNLP 2024 [R]. Jun 2024.
- Method: context-aware dynamic tree. Expands and reranks nodes by draft confidence, used as a proxy for acceptance rate. [R]
- Result: about 3.05-4.26x; 20-40% faster than EAGLE-1 [R].
- Code: [github.com/SafeAILab/EAGLE](https://github.com/SafeAILab/EAGLE) [R].
- Stated limitations: not re-read.
- Inferred weaknesses: tied to the EAGLE drafter's calibration. Dynamic shapes clash with CUDA graphs and compilers, which is the problem Yggdrasil targets.
- Placement: drafter-coupled, but listed here as the canonical dynamic-tree method.
- Needs T>0: no.

**DeFT: Decoding with Flash Tree-attention for Efficient Tree-structured LLM Inference** (Yao et al.). arXiv 2404.00242 [V]. ICLR 2025 [V]. Apr 2024.
- Method: IO-aware tree attention. Groups query and KV by shared prefixes (KV-guided grouping, flattened tree splitting) so shared-prefix KV is loaded once. — [ICLR poster](https://iclr.cc/virtual/2025/poster/31131), [mlanthology](https://mlanthology.org/iclr/2025/yao2025iclr-deft/)
- Result: about 2.2x decode and 3.6x attention latency speedups reported [R, numbers not re-fetched].
- Code: [github.com/LINs-lab/DeFT](https://github.com/LINs-lab/DeFT)
- Stated limitations: not read.
- Inferred weakness: gains matter mainly for long shared prefixes and large trees. For small SD trees (≤64 nodes), attention is not the bottleneck.
- Needs T>0: no.

**Yggdrasil: Bridging Dynamic Speculation and Static Runtime for Latency-Optimal Tree-Based LLM Decoding** (Guan, ..., Ding, Guo, Leng; SJTU/UCSD). arXiv 2512.23858 [V]. Dec 2025. Has an OpenReview forum (4E3I17pNEl); venue unconfirmed.
- Method: latency-aware objective using hardware-profiled costs instead of acceptance length alone; an equal-growth tree that keeps operator shapes static for compilers; stage-based scheduling to cut CPU-GPU sync. — [arXiv abs](https://arxiv.org/abs/2512.23858)
- Result: Llama-2-7B/13B targets with Llama-68M/160M drafters; C4, Wikipedia, CNN/DM. Up to 3.98x on A100-80GB; 2.76x on A40 vs SpecInfer, Sequoia and vLLM-Spec baselines. — [same]
- Code: not found.
- Stated limitation: assumes a single interactive request monopolizes the GPU; latency-throughput joint optimization is left open. — [same]
- Inferred weakness: old Llama-2 and tiny-drafter pairs. Not tested with EAGLE-3-class drafters.
- Needs T>0: no.

### Inferences
- Every tree builder optimizes a single-request objective: acceptance length, or latency at batch 1. Under batching the best tree shrinks; MagicDec and the latency-model paper show the gain erodes with load.
- Trees and verifiers are almost never co-designed. Sequoia (SWOR verification) and UniVer (topology-aware verification) are the exceptions.

### Gaps
- Exact headline numbers for SpecInfer, OPT-Tree, EAGLE-2 and DeFT were not re-fetched; [R] values must be checked against the PDFs.
- Yggdrasil's acceptance venue is unknown.

## Q3. Systems / serving papers (parallel draft-verify, batching, long context, latency modeling)

### Takeaway
The systems work splits into four strands: overlapping draft and verify (PEARL, SSD, Polybasic), making SD correct and useful under batching (Batch SD Done Right, MagicDec), adapting SD to load and SLOs (SpecServe), and modeling why SD speedup erodes under load (the latency model). Several of these need extra GPUs or only hold at small batch or low temperature.

### Cited Findings

**PEARL: Parallel Speculative Decoding with Adaptive Draft Length** (Liu et al.). arXiv 2408.11850 [V]. ICLR 2025 [V]. Aug 2024.
- Method: pre-verify (verify the first draft token while drafting) and post-verify (keep drafting during verification). Draft and target run in parallel with an adaptive draft length. — [ICLR PDF](https://proceedings.iclr.cc/paper_files/paper/2025/file/03b1043052700b1a471996b0baf309d4-Paper-Conference.pdf)
- Result: HumanEval 3.87x, GSM8K 3.81x, MT-bench 3.59x, MGSM 3.95x; H100-80G, e.g. CodeLlama 7B→70B, greedy in examples; "over 2.5x" on Qwen. — [GitHub](https://github.com/smart-lty/parallelspeculativedecoding)
- Code: [github.com/smart-lty/ParallelSpeculativeDecoding](https://github.com/smart-lty/parallelspeculativedecoding); also nano-PEARL, a draft-target disaggregated serving system. — [GitHub](https://github.com/smart-lty/nano-PEARL)
- Limitations noted in the README: needs multiple GPUs/processes; fp16 overflow can produce garbage text; transformers version sensitivity. — [GitHub]
- Inferred weaknesses: the draft GPU is extra hardware, so the comparison is not iso-hardware. Batch-1 only.
- Needs T>0: no.

**Speculative Speculative Decoding (SSD / Saguaro)** (Kumar, Dao, May). arXiv 2603.03251 [V]. ICLR 2026 (per arXiv comments). Mar 2026.
- Method: while verification runs, the drafter predicts likely verification outcomes and pre-builds speculations for each, so on a hit there is no drafting latency. Lossless; falls back to standard SD on a miss. — [arXiv abs](https://arxiv.org/abs/2603.03251), [arXiv HTML](https://arxiv.org/html/2603.03251v1)
- Result: Llama-3.1-70B target (4×H100, tensor parallel) with Llama-3.2-1B draft on a separate H100; greedy, batch 1; math, code and chat. Abstract: average 30% faster than optimized SD baselines, up to 5x vs autoregressive. v1 HTML says "up to 2x" over SD baselines. — [same]
- Code: [github.com/tanishqkumar/ssd](https://github.com/tanishqkumar/ssd)
- Stated limitations: cache-miss rate rises with batch size and temperature; fallback strategies vary by batch size. — [arXiv HTML]
- Inferred weaknesses: needs a dedicated draft GPU. The headline is greedy and batch 1, the regime most favourable to outcome prediction.
- Temperature: works best at T=0 (the opposite of verifier papers).

**Polybasic Speculative Decoding Through a Theoretical Perspective** (Wang, Li, Ma, Zheng, Chao, Xiao, Ji). arXiv 2510.26527 [V]. arXiv only as seen. Oct 2025. Optional.
- Method: a chain of more than two models (multi-level drafting) with a theorem characterizing the optimal inference time. — [arXiv abs](https://arxiv.org/abs/2510.26527)
- Result: 3.31-4.43x across LLaMA2-Chat, LLaMA3, Vicuna and Qwen2; settings not read. — [same]
- Code: promised; repo not located.
- Stated limitations: none seen in the abstract [A for body].
- Inferred weakness: more models means more memory and scheduling complexity. Relation to HSD-style hierarchy is unclear.
- Relevance: the user's 3-stage spec-hierarchy work.

**Batch Speculative Decoding Done Right** (Zhang, Dey, Mishra, Wu, Li, Zhang; eBay). arXiv 2510.22876 [V]. arXiv only. Oct 2025.
- Title note: the current arXiv abs page title is "Correctness Forensics for Batch Speculative Decoding: Diagnosing the Ragged Tensor Problem" (search listing), while HTML v1-v3 read "Batch Speculative Decoding Done Right". Cite with the version number. — [arXiv abs listing](https://arxiv.org/abs/2510.22876), [HTML v3](https://arxiv.org/html/2510.22876v3)
- Method: shows that existing batch SD implementations break output equivalence. When sequences in a batch accept different numbers of tokens (the "ragged tensor problem"), position IDs, attention masks and KV cache fall out of sync. EqSpec fixes this with explicit synchronization; EXSpec groups same-length sequences across batches. — [HTML v3]
- Result: Vicuna-7B/68M, Qwen3-8B/0.6B, GLM-4-9B/0.6B on A100-80GB, greedy, batch 1/4/8. About 95% decoding equivalence vs near zero for BSP and DSD; up to 3x throughput at batch 8 vs batch 1. — [same]
- Code: [github.com/eBay/spec_dec](https://github.com/eBay/spec_dec)
- Stated limitations: alignment overhead grows superlinearly, up to 40% of compute at batch 8; throughput degrades beyond batch 8; static batching only, not continuous batching (vLLM, SGLang). — [same]
- Inferred weakness: "95% equivalence" is still not lossless, and it is only measured at greedy.

**MagicDec: Breaking the Latency-Throughput Tradeoff for Long Context Generation with Speculative Decoding** (Sadhukhan, Chen, et al.). arXiv 2408.11049 [V]. ICLR 2025 [V]. Aug 2024.
- Method: at long context and large batch, decoding becomes KV-memory-bound again, so SD helps even at high throughput. Uses a sparse-KV (StreamingLLM) self-speculative draft or a small draft model. — [ICLR proceedings](https://proceedings.iclr.cc/paper_files/paper/2025/hash/13f972adf12bdf886583d48cd528002f-Abstract-Conference.html), [blog](https://infini-ai-lab.github.io/MagicDec-part1/)
- Result: 8×A100. LLaMA-2-7B-32K self-spec up to 2.0x at 32K context, batch 32; LLaMA-3.1-8B 1.84x at 100K, batch 32; 1.18-1.91x across batch 32-128 and 8K-32K context. — [blog]
- Code: [github.com/Infini-AI-Lab/MagicDec](https://github.com/Infini-AI-Lab/MagicDec)
- Stated limitations: the speedup relies on long context; at short context and large batch SD stays compute-bound and unhelpful. — [blog]
- Inferred weaknesses: tested mainly on PG-19. Temperature not specified in the sources read.

**SpecServe: Efficient and SLO-Aware LLM Serving with Adaptive Speculative Decoding** (Huang, Wu, Shi, Zou, Yu, Shi). arXiv 2503.05096 [V]. arXiv only as seen. Mar 2025. Optional.
- Method: on vLLM, an adaptive drafter sets draft length from load and SLO; also a confidence-prior verifier and an SLO-aware efficiency estimator. — [arXiv HTML](https://arxiv.org/html/2503.05096v1)
- Result: 1.14x-14.3x vs baseline speculative systems on Azure traces; model and GPU settings not read. — [same]
- Code: not found.
- Stated limitations: hard to predict acceptance before verification. — [same]
- Inferred weakness: the 14.3x upper bound probably reflects a weak baseline under overload. The paper has a "confidence prior verifier"; it is unclear whether this stays lossless, so check before using.
- Related systems seen only as titles [A]: AdaServe (arXiv 2501.12162, EuroSys 2026) — [PDF](https://www.cs.cmu.edu/~zhihaoj2/papers/AdaServe_EuroSys26.pdf); Nightjar (2512.22420); AdaSpec (ACM DOI 10.1145/3772052.3772239).

**An Interpretable Latency Model for Speculative Decoding in LLM Serving** (Kong, Flynn, Peng, Shavit, Kurtz, Marques; MIT and Red Hat AI). arXiv 2605.15051 [V]. arXiv only. May 2026.
- Method: decomposes latency as L = C1 + B·C2 and uses Little's Law to get L = C1/(1 − RPS·C2). For SD, splits costs into prefill, draft and verify, with acceptance α and draft length k. — [arXiv HTML](https://arxiv.org/html/2605.15051v1)
- Result: vLLM 0.13.0 on A100 (multi-GPU for large models); Llama-3.1-8B/70B, gpt-oss-20b, Qwen3; prefill/decode 256-1024; k=1-10; α=50-100%. The SD speedup "often diminishes as server load increases", which the model explains without appeal to system effects. — [same]
- Code: not found.
- Stated limitations: fit only up to the onset of saturation; full latency distribution left for future work; p95/p99 approximation weaker on small models. — [same]
- External review: the fitting is partly circular (the same data are used to fit and validate, with no hold-out). — [Pith review](https://pith.science/paper/2605.15051)
- Inferred weakness: α is treated as an exogenous input, so the model cannot study tree or verifier choices directly.

### Inferences
- There is a split by temperature. Verifier papers gain only at T>0. SSD, Batch SD Done Right and PEARL headline results are at T=0, and SSD explicitly degrades as temperature rises. No paper evaluates a strong verifier and a serving system together at realistic T (0.6-1.0) and batch size above 1.
- Throughput-oriented results (MagicDec, latency model, Batch SD Done Right) suggest that the 2-7% gains from new verification rules would shrink further under serving load.

### Gaps
- No primary source found for the venue or code of SpecServe or Polybasic.
- SSD's speedup figure differs between versions: v1 HTML says up to 2x, the abstract says 30% on average. Re-read v3 before citing.
- No paper found that measures verifier rules (Traversal, UniVer, HSD) inside vLLM or SGLang continuous batching. The vLLM block-verification PR is the closest.
