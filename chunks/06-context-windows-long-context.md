---
title: "06 · Context Windows & Long-Context Behavior"
parent: Chunks
nav_order: 7
---

# Chunk 06: Context Windows & Long-Context Behavior

**Purpose.** Distinguish the size of a model's context window from the model's ability to use that space, and identify the concrete resource cost that retained context imposes at inference time.
**Previously.** Chunk 05 covered in-context learning and prompting: how instructions and examples inside the prompt steer behavior.
**Today.** The container those instructions occupy — how its size limit arises from positional encoding, what it costs to maintain (the KV cache), and how recall accuracy varies with position within it (the "lost in the middle" effect).

![U-shaped chart showing model accuracy as a function of the position of the relevant document within the input context, peaking when the relevant information is at the very start or end and dropping sharply in the middle.](https://ar5iv.labs.arxiv.org/html/2307.03172/assets/x1.png)

*Figure 1: Changing the position of the relevant passage within the input context produces a U-shaped performance curve — models use relevant information best at the beginning (primacy bias) or end (recency bias) of the context, and performance drops significantly when it's in the middle. Source: "Lost in the Middle: How Language Models Use Long Contexts", Liu et al., 2023 — https://ar5iv.labs.arxiv.org/html/2307.03172.*

---

## Curiosity

A context window is the maximum number of tokens a model can process in a single request. The system prompt, the conversation history, pasted documents, tool outputs, and the response being generated all draw on this one budget. The limit is a hard architectural property: content beyond it must be truncated, summarized, or dropped before the model sees it.

Two properties of the window are easily conflated. The first is its *size* — the advertised figure on a specification sheet, such as 128K or 1M tokens, which states only how much input the model will accept. The second is how *evenly* the model uses that space, and here the empirical record is clear: recall is not uniform across positions. Information placed near the beginning of a long input (primacy) or near the end (recency) is used substantially more reliably than information placed in the middle. Accuracy plotted against the position of the relevant passage forms a U-shaped curve; this is the lost-in-the-middle effect. A fact can be fully present in context and still be missed because of where it sits.

A long meeting offers a rough parallel: participants recall the opening agenda and the final minutes far better than a remark made midway through the second hour, even though they were present for all of it.

These two properties combine into the distinction between *advertised* and *effective* context length. A model may accept a million tokens while reliably tracking and using far fewer, particularly on tasks harder than retrieving a single planted fact. The advertised number is a capacity claim; the effective number is a performance claim, and the two are measured differently. The practical consequence is immediate: when preparing a long input, the placement of critical material — first or last, never buried in the middle — matters as much as whether it fits.

## Builder

Operationally, presence in context is a weak guarantee. Every token in the window is visible to the model at every generation step, but attention does not weight positions equally, so placement decisions are part of prompt design rather than an afterthought. Four rules follow from the U-shaped recall curve:

1. **Exploit recency.** An instruction that must govern the next action belongs near the end of the prompt — restated if necessary, even when it appeared earlier.
2. **Exploit primacy.** Standing rules that apply to the whole task belong at the very start, where they receive the other end of the curve.
3. **Do not bury the request inside a long paste.** When a question concerns one section of a long document, move the relevant excerpt adjacent to the question, or repeat the key fact immediately before it.
4. **Budget below the advertised maximum.** Because effective context length falls short of the nominal figure on non-trivial tasks, filling the window to capacity trades reliability for volume.

Long interactive sessions degrade in a specific pattern, not uniformly. An instruction given early in a session starts at a well-recalled position, but as the transcript grows it migrates toward the middle — the weakest region — while newer, less important content continuously occupies the best-attended position at the end. Apparent forgetting in long sessions is one manifestation of this positional drift.

A worked example. At minute 5 of a debugging session, the user states: "Never modify the `migrations/` folder — it is managed by a separate team." Sixty messages of file edits and tool outputs follow. At minute 70 the user asks the assistant to "clean up unused files in the repo." The constraint is still in context, but it now sits deep in the middle of a long transcript, competing with dozens of recent, salient tool results near the end. The empirically grounded expectation is a material risk that the constraint is deprioritized. The mitigations are structural: restate hard constraints immediately before risky actions, or hoist them into a position that is re-read fresh each turn — a system prompt or a standing configuration file such as `CLAUDE.md` — rather than relying on a single early mention to remain effective.

## Expert

Two mechanisms require separate treatment: how position is *encoded* in the architecture, and how position *predicts* recall empirically. A third topic, the KV cache, determines what long context costs to serve.

**Positional encoding and context extension.** Self-attention is permutation-invariant, so token order must be injected explicitly. Rotary Position Embedding (RoPE; Su et al., 2021) partitions each query and key vector into 2D subspaces and rotates the pair in subspace $$i$$ by an angle proportional to absolute position: a token at position $$m$$ receives rotation $$R(m\theta_i)$$, with per-subspace frequencies $$\theta_i = 10000^{-2i/d}$$. Because rotations compose, the attention score between a query at position $$m$$ and a key at position $$n$$ satisfies

$$\langle R(m\theta_i)\,q,\; R(n\theta_i)\,k\rangle \;=\; \langle R((m-n)\theta_i)\,q,\; k\rangle,$$

so scores depend on position only through the relative offset $$m-n$$. RoPE underlies most modern long-context decoder-only models. Because it is a continuous transformation rather than a learned lookup table, it admits post-hoc extension: positional interpolation (Chen et al., 2023) rescales position indices by $$m \mapsto m \cdot L/L'$$, compressing a target length $$L'$$ into the range $$[0, L]$$ seen during pretraining, and recovers quality with light fine-tuning. This avoids the degenerate attention scores produced by naive extrapolation past trained positions and is one mechanism behind the rapid growth of advertised context windows without full retraining. Sparse-attention architectures such as Longformer (Beltagy et al., 2020) — local windowed attention plus a small set of global tokens — represent a different design point that trades full pairwise attention for linear scaling; chunk 02 develops the quadratic-cost argument that motivates it.

**Empirical recall degradation.** Liu et al. (2023) measured multi-document question answering and key-value retrieval while varying the position of the relevant item, and found a robust U-shaped accuracy curve: highest when the item is at the start or end of the input, and substantially lower — in some configurations below a closed-book baseline — when it is in the middle. Performance also declined with input length even for models marketed as long-context. RULER (Hsieh et al., 2024) extended this line by testing retrieval, multi-hop tracing, and aggregation at controlled lengths, and found that many models' reliable performance falls well short of their advertised maximum; passing a single needle-in-a-haystack test at a given length does not imply robust performance on harder tasks at that length. Together these results define the advertised-versus-effective gap quantitatively.

**KV cache cost.** During autoregressive generation the model caches the key and value projections of every prior token rather than recomputing them, so each new token attends against stored state. Per token, the cache holds keys and values for every layer and head:

$$\text{bytes per token} \;\approx\; 2 \times n_{\text{layers}} \times n_{\text{heads}} \times d_{\text{head}} \times 2\ \text{bytes (fp16)},$$

where the leading factor 2 counts keys and values and the trailing factor is the fp16 element width. Cache size therefore grows linearly in sequence length and is a first-order driver of long-context serving cost. A worked magnitude: assume 32 layers, 32 attention heads of dimension 128, fp16, and full multi-head attention (no key-value sharing). The cache then costs $$2 \times 32 \times 32 \times 128 \times 2 = 524{,}288$$ bytes — roughly 0.5 MB per token — so a 128K-token context requires approximately 64 GB for the cache alone, before model weights. This is why grouped-query attention (which shares key-value heads to shrink the effective $$n_{\text{heads}}$$), cache quantization, and eviction policies are standard serving techniques, and why providers price and rate-limit long contexts: every retained token has a persistent memory cost at each subsequent generation step.

**Open questions.** First, whether a model's effective context length can be predicted from architecture and training recipe without running a benchmark suite such as RULER per model and version. Second, whether alternative long-context mechanisms — sparse or local attention, retrieval augmentation, recurrent and state-space hybrids — remove positional bias or merely relocate it to different positions.

---

## Implications for agentic-dev

The findings above justify three structural choices in agentic coding workflows such as Claude Code:

- **`CLAUDE.md` as deliberately small, always-in-context configuration.** Loading an entire repository into context is counterproductive under positional bias, not merely wasteful: the one instruction that matters is likely to land in the least-recalled region of a large paste. A small curated file that the agent loads fresh each session keeps standing rules near a privileged position and out of competition with thousands of lines of unrelated source. Its smallness is the design, since every addition dilutes placement and consumes budget.

- **Subagents as fresh contexts for exploration.** Codebase search reads many candidate files and discards most of them. Performed in the main session, that trace both consumes budget and pushes the original task instructions toward the middle of the transcript. Dispatching exploration to a subagent with its own context window returns only a distilled result — a path, a summary — so the parent context stays short and its instructions stay near an endpoint.

- **Compaction as lossy summarization with a placement consequence.** Compacting a long session replaces verbatim history with a summary. This reduces KV cache footprint and re-expresses old information in a fresh, recent position, but it is lossy: details absent from the summary are gone. Constraints that must survive compaction should live in a re-injected location such as `CLAUDE.md` or be restated explicitly. The general placement rule follows: critical instructions belong at the start or end of context, or in a structurally re-anchored position — not unrepeated in the middle of a growing transcript.

Chunk 02 covers the quadratic attention cost that makes long context expensive to compute in the first place; this chunk adds the memory cost and the reliability limit.

---

## Exercises

1. **(Curiosity)** Construct a document of roughly 30 paragraphs of filler text and insert one distinctive fact (an invented product code) at the beginning, middle, or end across three trials. Query a model for the fact in each trial and record accuracy by position. State which empirical finding this procedure replicates and what curve you would expect over many trials.
2. **(Builder)** A prompt consists of a 10-page pasted specification with a critical constraint in the fifth page ("all currency values must be stored as integer cents"), followed by a request to generate a database schema. Redesign the prompt so the constraint survives positional bias, and justify each change by reference to primacy, recency, and proximity to the request.
3. **(Expert)** A model has 48 layers, 40 attention heads of dimension 128, and serves fp16 with full multi-head attention. Compute the KV cache size per token and the total cache size at a 200K-token sequence length. Then recompute assuming grouped-query attention with 8 key-value heads, and state the reduction factor.

## Checklist

After this chunk you should be able to:

- [ ] Distinguish a model's advertised context window from its effective context length, and name the benchmark that measures the gap.
- [ ] Describe the lost-in-the-middle finding: what Liu et al. varied, what they measured, and the shape of the resulting curve.
- [ ] State the per-token KV cache formula and estimate cache memory for a given model configuration and sequence length.
- [ ] Explain how RoPE encodes position and why attention scores under RoPE depend only on relative offset.
- [ ] Explain positional interpolation as a method for extending a trained context window.
- [ ] Place critical instructions within a long prompt or long agent session to account for positional bias.
- [ ] Justify `CLAUDE.md`, subagents, and compaction as responses to context cost and positional bias.

## References

1. Liu, N. F., Lin, K., Hewitt, J., Paranjape, B., Bevilacqua, M., Petroni, F., & Liang, P. "Lost in the Middle: How Language Models Use Long Contexts." arXiv:2307.03172 (2023). https://arxiv.org/abs/2307.03172
2. Su, J., Lu, Y., Pan, S., Wen, B., & Liu, Y. "RoFormer: Enhanced Transformer with Rotary Position Embedding." arXiv:2104.09864 (2021); journal version *Neurocomputing* (2024). https://arxiv.org/abs/2104.09864
3. Chen, S., Wong, S., Chen, L., & Tian, Y. "Extending Context Window of Large Language Models via Positional Interpolation." arXiv:2306.15595 (2023). https://arxiv.org/abs/2306.15595
4. Beltagy, I., Peters, M. E., & Cohan, A. "Longformer: The Long-Document Transformer." arXiv:2004.05150 (2020). https://arxiv.org/abs/2004.05150
5. Hsieh, C.-P., et al. "RULER: What's the Real Context Size of Your Long-Context Language Models?" *COLM* (2024). arXiv:2404.06654. https://arxiv.org/abs/2404.06654

## Chunk summary

A context window's size is a capacity claim, not a performance claim. Position is encoded via schemes such as RoPE, whose relative-offset structure permits extension methods like positional interpolation (Chen et al., 2023), but recall across the resulting window is uneven: accuracy versus position of the relevant information is U-shaped (Liu et al., 2023), and effective context length often falls well short of the advertised maximum (Hsieh et al., 2024). Retained context also has a linear memory cost through the KV cache, a first-order driver of serving expense. For agentic work the consequences are structural: keep standing rules in a small re-anchored file, isolate noisy exploration in subagents, treat compaction as lossy summarization, and place critical instructions at the endpoints of context rather than its middle.
