# Per-paper table spec (one row per paper)

Write rows with Python's csv module (csv.writer, QUOTE_MINIMAL, UTF-8) to the CSV path you are given.
Header (exact, in this order):

group,paper,arxiv_id,year,venue,method,upside,method_downside,memory,quality,accepted_tokens,speed,llama_results,datasets,code,verified

Column meanings:
- group: subgroup code from the report (1a,1b,1c,2a,2b,2c,3a,3b,3c,3d,4a,4b,4c,5a,5b,5c)
- paper: short name (e.g. "EAGLE-3")
- method: one line, what it does
- upside: what the design buys, mechanistically (e.g. "drafter reads target features so 1 small layer suffices")
- method_downside: a weakness CAUSED BY THE DESIGN, not by the paper's evaluation. Good: "needs target hidden states each round, so drafting cannot start before verification ends"; "every head predicts its offset independently, so later heads ignore earlier choices"; "extra KV cache for the drafter grows with context"; "retrained per target"; "tree attention cost grows with batch x nodes"; "lossy: output distribution no longer the target's". Bad: "only tested at batch 1", "weak baselines" (those go nowhere, leave them out). 1-3 items, separated by "; ".
- memory: extra memory the method needs: drafter/heads parameter count or size, extra KV cache, extra copies (e.g. "~1B drafter + its KV", "4 heads ~0.4B", "none (n-gram table in CPU RAM)", "full quantized copy of target"). Say "not reported" plus an estimate marked "(est.)" if you can derive it from the architecture.
- quality: "lossless" or "lossy: <measured drop, metric>"
- accepted_tokens: mean accepted length / tau with model and temperature, e.g. "tau 6.5 (Qwen3-8B, T=0)". "not reported" if absent.
- speed: speedup over AR and/or tokens/s, with hardware, batch, temperature, framework, e.g. "4.9x (Qwen3-8B, H200, bs1, T=0, HF)".
- llama_results: any result on a LLaMA model (Llama-2/3/3.1/3.3, Vicuna counts as LLaMA-based; say "Vicuna"): speedup and/or tau with model size; "none" if no LLaMA experiments.
- datasets: benchmarks used (e.g. "MT-Bench, HumanEval, GSM8K, Alpaca, CNN/DM, Natural Questions, MATH-500").
- code: URL or "none found"
- verified: F (full text read this session), X (summarized page read), A (abstract/README only), K (from memory, numbers unverified)

Rules: never invent numbers; if not found write "not reported"; mark recalled numbers with "(unv.)". Keep each cell short (under ~200 chars).
