---
title: "11 · Tokenomics"
parent: Chunks
nav_order: 12
---

# Chunk 11: Tokenomics

**Purpose.** Apply the mechanisms of the core curriculum to the economics of running LLM workloads: what API providers meter, why the price structure has the shape it does, and which levers actually reduce cost. This chunk is an applied supplement — it introduces no new model machinery, and instead connects chunk 01 (tokens as the unit of account), chunk 02 (the compute cost of attention), and chunk 06 (the KV cache as the resource cost of context) to a billing statement.
**Previously.** Chunk 10 closed the core curriculum with the agent loop: a frozen model invoked repeatedly, its context accumulating across turns. That loop is also where inference cost concentrates, which motivates treating cost as a subject in its own right.
**Today.** The metered-token pricing model; the input/output price asymmetry and model-tier spreads; the three standard discounts (prompt caching, batching, model routing); a worked workload estimate; and the hardware-level cost drivers — prefill versus decode, KV-cache memory — that explain why the price sheet looks the way it does. Specific prices below were verified on official provider pages as of July 2026 and change frequently; the ratios and structures are the durable content.

> **Through-line.** Applied supplement. Every token you add to the context to reshape the distribution is billed — and re-billed each turn. This chunk prices the context lever.

---

## Curiosity

Commercial LLM APIs are metered services. There is no charge per request, per user, or per month; the meter runs on tokens — every token sent into the model (input) and every token the model generates (output), priced in dollars per million tokens. Token pricing is closer to a utility meter than to a software subscription: the bill is proportional to what the machine actually processed, and a verbose prompt costs more than a terse one for the same task.

Two structural facts organize every provider's price sheet. First, output tokens cost substantially more than input tokens — five times more across Anthropic's current model line and six times more across OpenAI's flagship models, as of July 2026. This asymmetry is not a marketing decision; it reflects a real difference in machine cost between reading tokens and generating them, which the Expert section derives from the prefill/decode distinction. Second, providers sell a ladder of model tiers at roughly order-of-magnitude price spreads: as of July 2026, Anthropic's line runs from $1 to $10 per million input tokens between its smallest and largest models (a 10× spread), and OpenAI's from $0.20 to $5 (a 25× spread), with output prices scaling proportionally. The tiers exist because most requests do not need the most capable model, and routing easy traffic to cheap models is the single largest cost lever available.

Prices at a given capability level have also fallen steadily. Anthropic's own pricing page shows its deprecated Opus 4.1 at $15/$75 per million input/output tokens against $5/$25 for every Opus release since 4.5 — a 3× nominal cut across one model generation, alongside capability gains. Point prices are perishable; the pattern of falling price-per-capability is durable.

## Builder

A pricing page reads as a small table: for each model, a base input price, an output price, and (where supported) cache-write, cache-read, and batch prices, all per million tokens (MTok). Anthropic's table as of July 2026 illustrates the structure: Haiku 4.5 at $1/$5 (input/output), Sonnet 4.6 at $3/$15, Opus 4.8 at $5/$25, Fable 5 at $10/$50. Estimating a workload is arithmetic: characterize the average request as (input tokens, output tokens), multiply by request volume and by the per-MTok rates. The uncertainty is rarely in the rates; it is in knowing your true token counts, which is why providers expose free token-counting endpoints.

Three discounts modify the base rates, and they compose.

**Prompt caching.** If consecutive requests share an identical prefix — a system prompt, tool schemas, a long document — the provider can store the processed form of that prefix and reuse it. Anthropic prices this as multipliers on base input, as of July 2026: writing a cache entry costs 1.25× base input (5-minute lifetime) or 2× (1-hour lifetime); reading a cache hit costs 0.1× base input, and each hit refreshes the lifetime at no extra charge. A 90% discount on the cached portion, purchased for a one-time 25% premium, breaks even after a single hit. The constraint is exact prefix matching: the cache is keyed on a hash of the prompt prefix, so a timestamp or any per-request variation placed early in the prompt defeats it entirely. Cacheable prefixes also have minimum lengths (512–1,024 tokens on current Anthropic models via the first-party API).

**Batch processing.** Both Anthropic and OpenAI offer 50% off input and output tokens for asynchronous jobs submitted through a batch API, with results returned within a processing window rather than interactively. Any workload that tolerates latency — nightly summarization, offline evaluation, dataset labeling — should not be paying interactive rates. Batch and caching discounts stack.

**Model routing.** Given the 10–25× tier spreads above, sending classification, extraction, and reformatting traffic to a small model while reserving the frontier tier for genuinely hard requests routinely dominates both other levers combined. The practical pattern is a router (a rule, a classifier, or the small model itself) plus an escalation path.

A worked estimate, using verified July-2026 Anthropic rates. A support-ticket triage service handles 20,000 requests/day on Haiku 4.5 ($1/MTok input, $5/MTok output, $0.10/MTok cache reads). Each request carries a 3,000-token shared prefix (system prompt plus tool schemas), 1,200 tokens of per-ticket content, and produces 300 output tokens.

- *Without caching:* input $$= 20{,}000 \times 4{,}200 = 84\text{M}$$ tokens → \$84.00; output $$= 6\text{M}$$ tokens → \$30.00; total **\$114.00/day**.
- *With caching:* the prefix is written once (3,000 tokens at 1.25× ≈ \$0.004, negligible) and stays warm at this traffic level. Cache reads $$= 60\text{M} \times \$0.10/\text{MTok} = \$6.00$$; uncached input $$= 24\text{M} \times \$1 = \$24.00$$; output \$30.00; total **\$60.00/day** — a 47% reduction from restructuring the prompt, with no model or quality change.
- *Batched as well* (if tickets can wait for a processing window): the 50% batch discount stacks, bringing the total to roughly **\$30.00/day**.

Two further billing details matter operationally. Reasoning-capable models bill thinking tokens as output tokens, and Anthropic charges for the full thinking trace even when the API returns only a summary — extended thinking can therefore multiply output cost by an amount the response text does not reveal, bounded only by the configured thinking budget. And the tokenizer is part of the price: Anthropic documents that its newest tokenizer produces roughly 30% more tokens for the same text than earlier models, so per-token prices are not comparable across models without also comparing tokens-per-text (chunk 01).

> **Try it.** Anthropic's token-counting endpoint (`POST /v1/messages/count_tokens`) is free and accepts the same payload as a real request, including system prompts, tools, and images. Count one of your production prompts, multiply by current rates, and compare against your intuition — documentation and examples at https://platform.claude.com/docs/en/docs/build-with-claude/token-counting (verified July 2026).

## Expert

The price sheet is an amortization of hardware time, and its asymmetries map onto the two phases of a Transformer inference request.

**Prefill versus decode.** Processing the prompt (*prefill*) computes attention over all input tokens in parallel in a single forward pass (chunk 02): large matrix multiplications, high arithmetic intensity, near-peak accelerator utilization. Generating the response (*decode*) is the autoregressive loop of chunk 00 run one token at a time: each output token requires a full forward pass in which the model's weights and the accumulated KV cache must be streamed from high-bandwidth memory to produce a single token's worth of arithmetic. Decode is therefore memory-bandwidth-bound rather than compute-bound (Pope et al., 2023), and each output token consumes far more machine time than each input token. The 5–6× output premium on public price sheets is this hardware asymmetry passed through to the customer, compressed into a single ratio.

**KV-cache memory as the marginal cost of context.** Chunk 06 established that serving a request requires holding a KV cache that grows linearly with context length. That memory is the scarce resource in serving: total accelerator memory, minus weights, bounds the number of concurrent requests, and KV occupancy is what modern serving systems ration (Kwon et al., 2023). A long-context request costs the provider not only prefill compute but also reduced batch concurrency for its duration. Notably, Anthropic's July-2026 pricing charges a flat per-token rate across its 1M-token window — long-context requests pay linearly in tokens, with the concurrency cost absorbed into the base rate rather than surfaced as a tiered surcharge, a simplification of earlier tiered-context pricing.

**Prompt caching as amortized prefill.** A cache hit replaces prefill compute with retrieval of precomputed KV state. The pricing follows directly: the write premium (1.25–2× input) pays for storing the KV state for the TTL; the read price (0.1× input) reflects that a hit costs the provider memory traffic rather than a fresh forward pass. The 5-minute/1-hour TTL split prices storage duration. Cache economics are thus the serving-side mirror of chunk 06: context is expensive to compute once and cheap to reuse, provided it is byte-identical.

**Batch discounts as utilization arbitrage.** Interactive traffic arrives unevenly, so capacity provisioned for peak load idles off-peak; latency-tolerant batch jobs let the scheduler fill those troughs, and the 50% discount is the customer's share of the recovered utilization.

Two open questions close the chunk. First, cost predictability under reasoning billing: when thinking tokens are billed as output but their quantity is chosen by the model per request — and, in summarized-thinking regimes, not even fully visible to the customer — per-request cost becomes a random variable whose distribution shifts with model version and problem difficulty. Budget caps bound the tail but do not restore determinism, and how procurement, rate limiting, and per-request cost attribution adapt is unsettled. Second, the trajectory of price-per-capability: nominal frontier prices have been comparatively stable while each price point buys markedly more capability (the 3× Opus repricing above is one datum), but whether this is driven mainly by hardware, by inference-systems engineering (caching, batching, speculative decoding), or by model efficiency determines how far it continues — and it is the quantity on which every "too expensive to deploy" judgment silently depends.

---

## Implications for agentic-dev

An agentic session is the worst-case token workload. Each turn re-sends the accumulated context — system prompt, `CLAUDE.md`, conversation, every prior tool result — so with a fixed prefix of $$P$$ tokens and average per-turn growth of $$g$$ tokens, total billed input over $$N$$ turns is approximately

$$\sum_{t=1}^{N} \left(P + (t-1)g\right) = NP + \frac{N(N-1)}{2}\,g,$$

quadratic in session length even though each turn feels incremental. This is chunk 10's context accumulation, priced.

Prompt caching is the countermeasure, and it explains structural choices in agentic-dev tooling. Caching requires a byte-stable prefix, which is exactly what the system/user split and a stable `CLAUDE.md` provide: instructions that never change within a session sit at the front of the prompt and become cache reads at 0.1× input, while volatile content (tool results, new messages) appends behind the cached boundary. Editing `CLAUDE.md` mid-session, or injecting per-turn content early in the prompt, silently converts the entire prefix back to full-price prefill on every turn.

Model routing is exposed directly as `/model`: with a 5× price gap between adjacent Anthropic tiers (as of July 2026), dropping from Opus to Sonnet or Haiku for mechanical tasks — renames, test runs, formatting — is a one-command 5–25× cost reduction on those turns, at no cost to the hard turns, which can be escalated back.

---

## Exercises

1. **(Curiosity)** Open a provider's current pricing page and, without computing anything, answer three structural questions: what is the output-to-input price ratio; what is the spread between the cheapest and most expensive tier; and which discounts (caching, batch) are offered. Then check whether the numbers you find still match the July-2026 figures quoted in this chunk, and note which changed.
2. **(Builder)** An agent session on Sonnet 4.6 ($3/MTok input, $15/MTok output, cache reads at 0.1× input) runs 30 turns. The fixed prefix (system prompt, tools, `CLAUDE.md`) is 20,000 tokens; each turn appends 4,000 tokens of new context and generates 800 output tokens. Using the summation formula from the agentic-dev section, compute total session cost (a) with no caching and (b) with the fixed prefix and all accumulated turns cached, so that each turn pays cache-read rates on prior context and full price only on that turn's new tokens. Report the ratio.
3. **(Expert)** Using the prefill/decode analysis, explain why a provider could plausibly price a 100,000-token-input/100-token-output request *lower* per token than a 1,000-token-input/10,000-token-output request, despite the first request containing more total tokens. Then explain what changes in this analysis when extended thinking is enabled and thinking tokens are billed as output — including why the customer's cost variance rises even if mean cost is unchanged.

## Checklist

After this chunk you should be able to:

- [ ] Describe what an LLM API meters and state the two structural asymmetries of every price sheet (output premium, tier spread).
- [ ] Estimate a workload's daily cost from (input, output) tokens per request, volume, and per-MTok rates.
- [ ] Explain prompt-caching economics: write premium, read discount, TTL, and the exact-prefix-match constraint.
- [ ] Name the three cost levers — caching, batching, routing — and rank them for a given workload.
- [ ] Derive the output-token price premium from the prefill/decode distinction, and cache pricing from KV-cache reuse (chunks 02, 06).
- [ ] Explain why agentic sessions accrue quadratic input cost and how stable prefixes and model routing counteract it.

## References

1. Anthropic. "Pricing." Claude Developer Platform documentation (official pricing page, accessed July 18, 2026). https://platform.claude.com/docs/en/docs/about-claude/pricing
2. Anthropic. "Prompt Caching." Claude Developer Platform documentation (official docs, accessed July 18, 2026). https://platform.claude.com/docs/en/docs/build-with-claude/prompt-caching
3. Anthropic. "Extended Thinking." Claude Developer Platform documentation (official docs, accessed July 18, 2026). https://platform.claude.com/docs/en/docs/build-with-claude/extended-thinking
4. Anthropic. "Token Counting." Claude Developer Platform documentation (official docs, accessed July 18, 2026). https://platform.claude.com/docs/en/docs/build-with-claude/token-counting
5. OpenAI. "API Pricing." OpenAI developer documentation (official pricing page, accessed July 18, 2026). https://developers.openai.com/api/docs/pricing
6. Pope, R., et al. "Efficiently Scaling Transformer Inference." *MLSys* (2023). arXiv:2211.05102. https://arxiv.org/abs/2211.05102
7. Kwon, W., et al. "Efficient Memory Management for Large Language Model Serving with PagedAttention." *SOSP* (2023). arXiv:2309.06180. https://arxiv.org/abs/2309.06180

## Chunk summary

LLM inference is sold by the metered token, and the price sheet encodes the serving economics derived in earlier chunks: output tokens carry a 5–6× premium because sequential, memory-bandwidth-bound decode costs more machine time than parallel prefill (chunk 02); cache reads at 0.1× input price amortized prefill of byte-identical prefixes, mirroring the KV-cache cost of context (chunk 06); and the tokenizer sets the unit of account, so per-token prices are incomparable across tokenizers (chunk 01). The operational levers, in typical order of impact, are model routing across 10–25× tier spreads, prompt caching of stable prefixes, and 50% batch discounts for latency-tolerant work. Agentic sessions accrue quadratic input cost from re-sent context, which is why stable system prompts and `CLAUDE.md` files — cacheable prefixes — and per-turn model routing are economic decisions, not merely stylistic ones. Point prices (verified July 2026) will drift; the ratios, and the mechanisms behind them, are the portable knowledge.
