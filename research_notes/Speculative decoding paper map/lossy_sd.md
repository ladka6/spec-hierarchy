# Lossy / relaxed speculative decoding, cascades, and reasoning-level speculation (paper map, as of Oct 2026)

Verification legend (applies to every entry):
- [F] = details fetched this session from a primary or near-primary page (paper PDF, arXiv/HF page, official repo README).
- [L] = existence, arXiv ID, venue badge, authors confirmed only via the curated list https://github.com/hemingkx/SpeculativeDecodingPapers (README fetched this session) or a search-result title. Abstract not read this session.
- [R] = numbers come from my prior reading of the abstract, NOT re-fetched this session (arxiv.org was blocked / rate limited from this environment). Re-check before citing in the thesis.
- "Weakness (inferred)" lines are my own reading, not author claims.

Environment note: arxiv.org abs pages were mostly unreachable (proxy 403, then HTTP 429 rate limit on WebFetch); HF papers API blocked. Verification relied on the hemingkx list, GitHub READMEs, HF paper pages, ICLR proceedings, and secondary summaries (pith.science, papernotes.org, cctest.ai), which are flagged where used.

## Q1. Token-level relaxed verification (accept mismatching draft tokens)

### Takeaway
The token-level line runs from heuristic thresholds (SpecDec tolerance, BiLD rollback, Medusa typical acceptance) to learned judges (Judge Decoding, AutoJudge, SelfJudge), tunable divergence thresholds (Fuzzy SD, Mentored decoding), learned ensembles (DIVERSED), and training-free semantic acceptance (FLy). Only Mentored decoding and DIVERSED come with a formal statement about the output distribution, and the 2026 "Revisiting" paper shows several of these methods collapse into two families with specific failure modes.

### Cited Findings

**1. SpecDec (Xia et al.), "Speculative Decoding: Exploiting Speculative Execution for Accelerating Seq2seq Generation"**, arXiv 2203.16487, EMNLP 2023 Findings [R for venue], first posted 2022-03. [L/R]
- Method: non-autoregressive drafter + "top-β / tolerance" relaxed verification: draft token accepted if it is within the target's top-β candidates and within a log-prob tolerance of the top-1.
- Lossy: yes, no distributional guarantee; quality judged by BLEU on MT. [R]
- Result: ~5x speedup on WMT translation with comparable BLEU (Transformer-base scale). [R]
- Code: github.com/hemingkx/SpecDec [R]. Limitations stated: not checked.
- Source: https://ar5iv.labs.arxiv.org/html/2203.16487 (search hit only)

**2. BiLD, "Speculative Decoding with Big Little Decoder"**, arXiv 2302.07863, NeurIPS 2023, 2023-02. [F list + README]
- Method: small model generates; "fallback" to large model when small model confidence is low; "rollback" when the large model's distance (cross-entropy) to a small-model token exceeds a threshold. Tokens with small discrepancy are kept even if not the large model's choice.
- Lossy: yes, thresholds (fallback, rollback) trade quality for speed; no distribution guarantee.
- Result: README claims "~2x" acceleration "without compromising performance"; paper reports up to 2.12x on IWSLT2017 De-En, WMT, XSUM, CNN/DM with T5/mT5 encoder-decoders [R for the 2.12x].
- Code: https://github.com/kssteven418/BigLittleDecoder
- Limitations stated: README notes you must prepare/finetune your own aligned small and large models (alignment helps); others not checked.
- Sources: https://github.com/hemingkx/SpeculativeDecodingPapers ; https://github.com/kssteven418/BigLittleDecoder

**3. Medusa typical acceptance, "Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads"**, arXiv 2401.10774, ICML 2024, 2024-01. [L + R]
- Method: extra decoding heads + tree attention; "typical acceptance" accepts a candidate if target prob > min(ε, δ·exp(−H(p))) (entropy-adaptive threshold), instead of rejection sampling.
- Lossy: yes for sampling (T>0); not distribution-preserving; at T→0 it reduces to greedy match. Quality checked only via MT-Bench / GPT-4 judge scores. [R]
- Result: Medusa-1 >2.2x, Medusa-2 2.3-3.6x on Vicuna-7B/13B/33B, Zephyr; MT-Bench quality roughly unchanged. [R]
- Code: https://github.com/FasterDecoding/Medusa [L]
- Limitations stated: not checked this session.

**4. Mentored decoding (blog 2023) / "Mentored Decoding: When Speculative Inference Meets Boosting"** by Vivien Tran-Thien and Richard Nock, arXiv 2609.30474, posted 2026-09-28, arXiv only. [F secondary]
- Method: lossy SD as an optimization: choose the output distribution that maximizes acceptance subject to a divergence budget to the target (blog: KL bound; paper: general f-divergences, emphasis on total variation), solved with an efficient data structure (O(sort(n)) build, O(log n) query). Paper connects it to boosting.
- Lossy: yes, but with a controlled, explicit divergence bound to the target. Blog claims provable optimality.
- Result: the secondary summary says the material it saw reports no concrete models/speedups. The 2023 blog had small-scale experiments (not checked this session).
- Code: https://github.com/vivien000/mentored_decoding (blog supplement: proof + code).
- Limitations stated: not seen. Note the arXiv HTML fetch (https://arxiv.org/html/2609.30474) was refused with HTTP 429, so the full paper was not read.
- Sources: https://cctest.ai/en/articles/mentored-decoding-when-speculative-inference-meets-boosting ; https://vivien000.github.io/blog/journal/a-provably-optimal-lossy-variant-of-speculative-decoding.html ; https://github.com/vivien000/mentored_decoding
- "Lenient speculative decoding": no separate paper found under that name; the hemingkx list files the mentored-decoding blog as "An Optimal Lossy Variant of Speculative Decoding" (2023-09). Treat as the same line of work.

**5. Judge Decoding, "Judge Decoding: Faster Speculative Sampling Requires Going Beyond Model Alignment"** (Bachmann et al., Meta), arXiv 2501.19309, ICLR 2025, list date 2024-10 (OpenReview), arXiv 2025-01. [F proceedings PDF via summarizer]
- Method: small linear "judge" head on target-model embeddings classifies whether a mismatched draft token is still "correct"; accept if judge says yes.
- Lossy: yes, explicitly gives up the distribution guarantee; quality checked only on benchmarks.
- Training: 500 hand-curated Q&A pairs with error annotations, ~30k tokens, 16.4k-param head, <1.5 h training.
- Result: Llama-3.1 8B draft / 405B target: 9.7x (HF), 3.9x (gpt-fast), 129 tok/s; 8B/70B: 2x (HF) / 3x (gpt-fast), 141 tok/s; mean accepted length ~20 vs ~6; accuracy "nearly preserved" on GSM8K, HumanEval, MT-Bench, ARC, MMLU.
- Code: none found.
- Limitations stated: loses the distribution guarantee; needs a strong drafter; new domains need new annotations; safety properties not guaranteed.
- Source: https://proceedings.iclr.cc/paper_files/paper/2025/file/656d6174fd5ffb01c843f269649ab5cb-Paper-Conference.pdf

**6. AutoJudge, "AutoJudge: Judge Decoding Without Manual Annotation"** (Garipov, Velikonivtsev, Svirschevski, Egiazarian, Ryabinin), arXiv 2504.20039, 2025-04, list badge "Arxiv" (NeurIPS 2025 acceptance is my recollection, unverified). [F README + R]
- Method: automatically mine "important" mismatches (ones whose substitution changes the final answer, found by search over continuations), train a lightweight classifier on hidden states, use it at inference to accept unimportant mismatches.
- Lossy: yes; no distribution guarantee, task-level answer correctness is the target.
- Result: up to ~1.5x more accepted tokens per cycle with <1% accuracy drop vs standard SD on GSM8K (Llama-3.2-1B / Llama-3.1-8B style pairs); also LiveCodeBench. [R, verify]
- Code: https://github.com/garipovroma/autojudge ; datasets: https://huggingface.co/datasets/mightyneighbor/Autojudge
- Limitations stated: not checked; README shows mining is the "most compute-intensive stage".

**7. SelfJudge, "Faster Speculative Decoding via Self-Supervised Judge Verification"** (Yoon et al.), arXiv 2510.02329, 2025-10, arXiv only per list. [L, title only]
- Method (from title): judge verifier trained self-supervised, removing the need for annotations. Numbers not seen.

**8. Fuzzy Speculative Decoding (FSD), "Fuzzy Speculative Decoding for a Tunable Accuracy-Runtime Tradeoff"** (Holsman, Huang, Dhingra), arXiv 2502.20704, ACL 2025 Findings, 2025-02. [L + R]
- Method: accept a draft token when a divergence between draft and target distributions at that position is below a user-set threshold T; otherwise fall back to standard SD rejection. T is a single knob for accuracy vs speed.
- Lossy: yes, tunable; T=0 recovers lossless SD.
- Result: reported better accuracy than SD-with-lower-quality-drafter at matched throughput, and near-target accuracy with extra tokens/s. [R, numbers not verified]
- Code: not found. Limitations stated: not checked.

**9. DIVERSED, "Relaxed Speculative Decoding via Dynamic Ensemble Verification"** (Ziyi Wang, Siva Rajesh Kasa, ..., Ruqi Zhang, Nan Jiang, Qifan Song; Amazon/Purdue), arXiv 2604.07622, 2026-04. [F abstract + README]
- Method: learned ensemble verifier: verification distribution = mixture of draft and target with a task- and context-dependent learned weight; standard rejection sampling against that mixture.
- Lossy: yes; output follows the learned mixture, not the target. Abstract claims "theoretical justification" and "preserving generation quality".
- Result: abstract says "substantially higher inference efficiency" than standard SD; concrete numbers not seen.
- Code: https://github.com/comeusr/diversed (ships a modified transformers fork; also implements "static ensemble" and "lossy SD" baselines).
- Venue conflict: hemingkx list says AISTATS 2026; search results also show a NeurIPS 2025 virtual page (https://neurips.cc/virtual/2025/126538) and OpenReview PDF (https://openreview.net/pdf?id=yrkf0GxTe7). Possibly a workshop vs main venue; unresolved.
- Sources: https://arxiv.org/abs/2604.07622v1 ; https://www.amazon.science/publications/diversed-relaxed-speculative-decoding-via-dynamic-ensemble-verification

**10. FLy, "Training-Free Loosely Speculative Decoding: Accepting Semantically Correct Drafts Beyond Exact Match"** (AMD), arXiv 2511.22972, ICLR 2026 (per repo README), 2025-11. [F README + secondary note]
- Method: at a mismatch, compute normalized entropy of target logits; if high (θ=0.3), open a delayed window (W=6) and accept the mismatch if the target does not diverge again within W tokens (treating target "self-correction" absence as semantic acceptance). Adds prompt-lookup drafting for multi-level acceleration.
- Lossy: yes; no distribution guarantee; claims ≥99% accuracy retention.
- Result: avg 2.81x on Llama-3.1-70B-Instruct, 5.07x on 405B; OOD tasks beat EAGLE-3 by 1.62x on Llama-3.3-70B; in-distribution 2.69x vs EAGLE-3 3.83x on 70B; works on Qwen2.5-Coder, Mistral-Large; decision overhead <0.6 ms/round.
- Code: https://github.com/AMD-AGI/FLy (built on lm-eval-harness, batch size 1 recommended).
- Limitations (from secondary note, not confirmed as author-stated): conservative boundary means late-draft tokens benefit less; assumes self-correction signals semantic correctness; hyperparameters W, θ empirical.
- Sources: https://github.com/AMD-AGI/FLy ; https://en.papernotes.org/ICLR2026/llm_efficiency/training-free_loosely_speculative_decoding_accepting_semantically_correct_drafts/ ; https://rocm.blogs.amd.com/artificial-intelligence/fly/README.html

**11. "Revisiting Lossy Verification in Speculative Decoding: Mechanisms, Trade-offs, and Failure Modes"** (Tianyu Wang, Yuxuan Zhou, Wenbin Wang, Heng Li, Zikai Xiao, Junyuan Shang), arXiv 2607.26627, 2026-07, arXiv only. [F HF page]
- Type: analysis paper. Splits lossy verification into truncation-based and collaborative (draft-target mixing) families; finds many methods overlap substantially.
- Findings: truncation-based verification can underperform the true truncation-sampling baseline because of distributional distortion; collaborative methods need control of draft-probability "overshoot" over target probabilities or quality degrades.
- Code: https://github.com/ZhouYuxuanYX/Fast-HSD
- Note: shares authors (Zhou, Wang, Li) with HSD (entry 13); the code repo name suggests it pairs lossy verification with HSD.
- Source: https://huggingface.co/papers/2607.26627

Other token-level relaxed methods seen only as titles in the hemingkx list [L]: SPRINTER "Speeding up Speculative Decoding via Approximate Verification" (2502.04557, 2025-02); S4C (2506.14158); "Speculative Verification: Exploiting Information Gain" (2509.24328). Vision-domain relaxed SD: LANTERN / LANTERN++, COOL-SD, Spec-VLA (relaxed acceptance) — out of scope for text LLMs but useful as evidence that relaxed acceptance is standard in AR image generation.

### Inferences
- BiLD: weakness (inferred) evaluated on encoder-decoder T5 tasks with short outputs; unclear transfer to long-form decoder-only generation where errors compound.
- Medusa typical acceptance: weakness (inferred) ε, δ are global constants; quality validated only with LLM-judge MT-Bench scores, which are insensitive to small distribution shifts (diversity, calibration).
- Mentored decoding: weakness (inferred) a per-token divergence budget does not bound sequence-level divergence tightly (budgets add up over length); empirical evidence appears thin.
- Judge Decoding: weakness (inferred) the 9.7x HF number is relative to a slow HF baseline; the gpt-fast number (3.9x) is the fairer one. 500 annotated examples from a narrow distribution raise OOD concerns (which FLy explicitly attacks).
- AutoJudge: weakness (inferred) "important token" labels are defined by final-answer change on verifiable tasks (math/code); the definition does not extend to open-ended generation.
- Fuzzy SD: weakness (inferred) the threshold is a divergence between model distributions, not a quality signal; the same T can mean very different quality loss across tasks.
- DIVERSED: weakness (inferred) needs training the ensemble weight per task/context and a patched transformers fork; deployment friction in vLLM/SGLang.
- FLy: weakness (inferred) the delayed window adds W extra tokens of latency before commit and relies on greedy-match semantics; behaviour under sampling (T>0) is unclear.
- Cross-cutting: almost all of these report quality only through benchmark accuracy (GSM8K, HumanEval, MT-Bench); none reports distribution-level metrics (diversity, calibration, safety refusals), which the Judge Decoding authors themselves flag for safety.

### Gaps
- Exact abstract numbers for SpecDec, Medusa, AutoJudge, Fuzzy SD, SelfJudge, SPRINTER were not re-fetched (arxiv.org blocked/rate limited). Marked [R]/[L].
- DIVERSED venue (AISTATS 2026 vs NeurIPS 2025) unresolved; quantitative results not seen.
- Mentored Decoding 2609.30474 full text not read (429); whether it has LLM-scale experiments is unknown.
- "Semantic/embedding-based acceptance" as a token-level criterion: apart from FLy (entropy + self-correction) and S4C (title only), I found no well-known paper that accepts tokens by embedding similarity at the token level; semantic checks appear mostly at step level (Q3).

## Q2. Cascades and deferral-based speculation

### Takeaway
Speculative cascades reframe lossy SD as cascade deferral: the small model's token is kept when a deferral rule says it is good enough, with the large model run in parallel speculative style. This gives a principled cost-quality target (a mixture distribution) rather than heuristic thresholds.

### Cited Findings

**12. "Faster Cascades via Speculative Decoding"** (Narasimhan, Jitkrittum, Rawat, Kim, Gupta, Menon, Kumar; Google), arXiv 2405.19261, ICLR 2025, 2024-05. [L + R]
- Method: "speculative cascades": implement a cascade's deferral rule through speculative execution (draft from small model, target scores in parallel), with an optimal deferral rule derived for a target-vs-cost objective; includes token-level variants combining SD acceptance with cascade deferral.
- Lossy: yes by design; samples from a deferral-defined mixture, not the target; can exceed the target model quality in some settings (cascades can be "better than either" on some tasks) [R].
- Result: better cost-quality trade-offs than plain cascades and plain SD on T5 (summarization, translation, reasoning) and Gemma models. [R, numbers not verified]
- Code: none found. Limitations stated: not checked.
- Source: https://github.com/hemingkx/SpeculativeDecodingPapers (entry "Faster Cascades via Speculative Decoding", ICLR2025 badge)

Related but lossless (not lossy cascades, keep out of the core list): Cascade Speculative Drafting (2312.11462, NeurIPS 2024), CAS-Spec, FastEagle (cascaded drafters, all verify losslessly). [L]

### Inferences
- Weakness (inferred): optimal deferral rule needs calibrated confidence estimates from the small model; miscalibration (common in LLMs) breaks the optimality argument.
- Weakness (inferred): BiLD (entry 2) is essentially an early heuristic speculative cascade; the thesis team should treat BiLD and speculative cascades as one family when comparing.

### Gaps
- Concrete speedup / quality figures for speculative cascades not re-fetched.

## Q3. Reasoning-level (step-level) speculation

### Takeaway
For long chain-of-thought, token-exact acceptance caps speedups, so a 2025-2026 wave speculates whole reasoning steps or thoughts, accepting them by a judge score, a PRM, semantic equivalence, or internal signals. These are lossy by construction and are judged by final-answer accuracy, sometimes even improving it.

### Cited Findings

**13 (reference, lossless). HSD, "Overcoming Joint Intractability with Lossless Hierarchical Speculative Decoding"** (Yuxuan Zhou, Fei Huang, Heng Li, Fengyi Wu, Tianyu Wang, Jianwei Zhang, Junyang Lin, Zhi-Qi Cheng), arXiv 2601.05724 (v2 exists), 2026-01, arXiv only as far as seen. [F]
- LOSSLESS: provably preserves the target distribution (theorem in appendix); hierarchical branch resampling on the joint probability of draft blocks.
- Result: +6.7% decoding speed on average, up to +12.3% on single datasets; drop-in for EAGLE-3 gives >12% gain without retraining; GPTQ 8-bit instruct models, GSM8K etc.
- Code: https://github.com/ZhouYuxuanYX/Hierarchical-Speculative-Decoding
- Include only as the lossless baseline that lossy methods should be compared against.
- Source: https://arxiv.org/abs/2601.05724

**14. Reward-Guided Speculative Decoding (RSD)** (Liao, Xu, Dong, Li, Monz, Savarese, Sahoo, Xiong; Salesforce), arXiv 2501.19324, ICML 2025, 2025-01. [F README]
- Method: draft model proposes reasoning steps; a process reward model scores each; if score above threshold keep the draft step, else regenerate with the target.
- Lossy: yes, deliberately biased toward high-reward steps (not the target distribution).
- Result: up to 4.4x fewer FLOPs vs target-only, up to +3.5 accuracy vs parallel decoding; Qwen2.5-Math family + Skywork-o1-PRM-1.5B; Olympiad-level math.
- Code: https://github.com/BaohaoLiao/RSD (needs ≥3 GPUs in vLLM online mode: draft, target, PRM).
- Limitations stated: not checked in paper.

**15. SpecReason, "Fast and Accurate Inference-Time Compute via Speculative Reasoning"** (Pan, Dai, Zhang, Oliaro, Jia, Netravali), arXiv 2504.07891, 2025-04, arXiv only per list. [F README + R]
- Method: small model produces each reasoning step; the large reasoning model scores step utility (score threshold, e.g. 7.0 in the repo example) and regenerates if low. Can be stacked with token-level SD.
- Lossy: yes; relies on "approximation tolerance" of intermediate thinking tokens.
- Result: 1.4-3.0x speedup over vanilla LRM inference with +0.4-9.0% accuracy; combined with SD, 8.8-58.0% further latency reduction; QwQ-32B + R1-Distill-1.5B on AIME, MATH500, GPQA. [R for numbers; model pair F from README]
- Code: https://github.com/ruipeterpan/specreason (proof of concept, vLLM 0.8.2 with a patched SD path).

**16. Speculative Thinking, "Enhancing Small-Model Reasoning with Large Model Guidance at Inference Time"** (Wang Yang, Xiang Yue, Vipin Chaudhary, Xiaotian Han), arXiv 2504.12329, 2025-04, arXiv only per list. [L + R]
- Method: small reasoning model generates; at reflective cue points (e.g. after "\n\n", "wait") the large model takes over for a segment. Goal is improving the small model, not exact acceleration of the large one.
- Lossy relative to the large model: yes (output is mostly small-model text).
- Result: R1-Distill-1.5B assisted by 32B: MATH500 83.2% to 89.4% (+6.2) with ~15.7% shorter outputs. [R]
- Code: https://github.com/uservan/speculative_thinking (configs for 1.5B+14B/32B; minimal README).

**17. Speculative Chain-of-Thought (SCoT), "Efficient Reasoning for LLMs through Speculative Chain-of-Thought"** (Jikai Wang, Juntao Li, Lijun Wu, Min Zhang), arXiv 2504.19095, 2025-04, arXiv only per list. [L + R]
- Method: small model drafts several full CoTs in parallel; target selects the best draft or rejects all and thinks itself; draft model fine-tuned to align style.
- Lossy: yes; output can be a draft CoT, target only selects.
- Result: 48-66% latency reduction for R1-Distill-Qwen-32B with near-target accuracy on GSM8K, MATH, GaoKao, CollegeMath, Olympiad. [R]
- Code: https://github.com/Jikai0Wang/Speculative_CoT (README is empty as of fetch).

**18. Lookahead Reasoning, "Scaling Speculative Decoding with Lookahead Reasoning"** (Fu, Ge, Shao, Deng, Zhang; Hao AI Lab), arXiv 2506.19830, 2025-06, arXiv only per list. [F README + R]
- Method: draft model proposes several future steps; target generates its own version of each step in one batched pass; a verifier checks semantic equivalence; accept up to first mismatch. Orthogonal to token-level SD (two-level parallelism).
- Lossy: yes; acceptance is semantic equivalence judged by a verifier model/embedding.
- Result: lifts SD speedup from about 1.4x to about 2.1x while preserving answer quality (Qwen3 reasoning models, GSM8K/AIME-type benchmarks). [R]
- Code: https://github.com/hao-ai-lab/LookaheadReasoning

**19. SpecGuard, "From Tokens to Steps: Verification-Aware Speculative Decoding for Efficient Multi-Step Reasoning"** (Kiran Purohit, Ramasuri Narayanam, Soumyabrata Pal), arXiv 2604.15244, 2026-04-16, arXiv only. [F secondary (pith.science)]
- Method: step-level SD without external reward model; sample several draft candidates per step, pick the most self-consistent, accept if an ensemble of attention-based grounding score and log-prob confidence passes a threshold, else recompute with target.
- Lossy: yes, no distribution guarantee.
- Result: +3.6% accuracy and ~11% lower latency vs standard SD; beats token-centric and reward-guided (RSD-style) variants; MATH500, GSM8K, GaoKao-2023-En, OlympiadBench.
- Code: not found.
- Limitations: the pith review (not authors) notes incomplete ablations of each signal and missing details on model sizes, sampling parameters, significance tests.
- Sources: https://arxiv.org/abs/2604.15244 ; https://pith.science/paper/2604.15244

Adjacent, title-only [L]: SpecSearch "Accelerating LLM Reasoning via Speculative Search" (2505.02865, ICML 2025); SSR (speculative parallel scaling reasoning); Reward-Shifted Speculative Sampling (2508.15044, weak-to-strong alignment, lossy by design); Speculative Ensemble (2502.01662, ICML 2025).

### Inferences
- RSD: weakness (inferred) adds a third model (PRM) whose cost and memory are excluded from "FLOPs" framing in some comparisons; PRM quality caps output quality and PRMs are mostly math-specific.
- SpecReason: weakness (inferred) using the large model as a scorer still requires a prefill of each step on the large model; gains shrink when steps are long or the scorer is unreliable; threshold tuned per dataset.
- Speculative Thinking: weakness (inferred) cue-word triggers ("wait") are model-family specific heuristics; evaluated mostly on DeepSeek-R1-Distill models.
- SCoT: weakness (inferred) requires fine-tuning the drafter for alignment and multiple full CoT drafts; wasted compute when all drafts are rejected on hard problems.
- Lookahead Reasoning: weakness (inferred) semantic-equivalence verifier errors directly become answer errors; verifier cost and false-accept rate are critical but hard to measure.
- SpecGuard: weakness (inferred) ~11% latency gain is modest compared to the cost of sampling several candidates per step; attention-based grounding requires attention maps, which conflict with FlashAttention-style kernels.
- Cross-cutting: step-level methods report accuracy that sometimes exceeds the target model, which means they are not approximating the target at all; comparisons against lossless SD speedups are apples-to-oranges.

### Gaps
- Exact numbers for SpecReason, Speculative Thinking, SCoT, Lookahead Reasoning come from abstract recollection [R], not re-fetched.
- No author-stated limitations were retrieved for the reasoning-level papers (paper bodies not accessible from this environment).

## Q4. Suggested core list (18 lossy + 1 lossless reference) and priority for downside analysis

### Takeaway
Recommended core set, grouped: token-heuristic (SpecDec, BiLD, Medusa-typical), principled-divergence (Mentored decoding, Fuzzy SD, DIVERSED), learned judges (Judge Decoding, AutoJudge, SelfJudge), training-free semantic (FLy), cascades (Faster Cascades), analysis (Revisiting Lossy Verification), step-level (RSD, SpecReason, Speculative Thinking, SCoT, Lookahead Reasoning, SpecGuard), plus HSD as the lossless reference.

### Cited Findings
- Revisiting Lossy Verification provides a ready taxonomy (truncation vs collaborative) and named failure modes to anchor the downside analysis — [HF page](https://huggingface.co/papers/2607.26627)
- HSD is confirmed lossless and is the natural baseline for "what lossless already gets you" — [arXiv](https://arxiv.org/abs/2601.05724); [repo](https://github.com/ZhouYuxuanYX/Hierarchical-Speculative-Decoding)
- Highest headline speedups among lossy token-level methods: Judge Decoding 9.7x (HF) / 3.9x (gpt-fast) on 8B/405B — [ICLR 2025 PDF](https://proceedings.iclr.cc/paper_files/paper/2025/file/656d6174fd5ffb01c843f269649ab5cb-Paper-Conference.pdf); FLy 5.07x on 405B, training-free — [FLy repo](https://github.com/AMD-AGI/FLy), [paper note](https://en.papernotes.org/ICLR2026/llm_efficiency/training-free_loosely_speculative_decoding_accepting_semantically_correct_drafts/)

### Inferences
- Priority for downside analysis: (1) Judge Decoding and AutoJudge (annotation/OOD), (2) FLy (training-free, strongest OOD claim, greedy-only?), (3) Fuzzy SD, Mentored, DIVERSED (the only ones with explicit divergence control), (4) RSD vs SpecReason vs SpecGuard (step-level acceptance signals compared head to head).
- The Revisiting paper and HSD share authors and a codebase lineage (Fast-HSD), relevant if the team's own work builds on HSD.

### Gaps
- Could not confirm venues for SelfJudge, Speculative Thinking, SCoT, Lookahead Reasoning, SpecReason beyond "arXiv only" badges in the hemingkx list (badges may be stale).
- No benchmark that compares all lossy methods under one harness was found other than the Revisiting paper, whose exact method coverage was not read.
