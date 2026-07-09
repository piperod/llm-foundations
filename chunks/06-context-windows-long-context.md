---
title: "06 · Context Windows & Long-Context Behavior"
parent: Chunks
nav_order: 7
---

# Chunk 06: Context Windows & Long-Context Behavior

**Purpose.** Establish how much text a model can actually attend to at once, and why "the context window is 200K tokens" is a much weaker guarantee than it sounds like.
**Previously.** Chunk 05 covered in-context learning and prompting theory — how examples and instructions inside the prompt steer behavior.
**Today.** We look at the container those instructions live in: its size, its cost (the KV cache), and its uneven attention across position ("lost in the middle").

![U-shaped chart showing model accuracy as a function of the position of the relevant document within the input context, peaking when the relevant information is at the very start or end and dropping sharply in the middle.](https://ar5iv.labs.arxiv.org/html/2307.03172/assets/x1.png)

*Figure 1: Changing the position of the relevant passage within the input context produces a U-shaped performance curve — models use relevant information best at the beginning (primacy bias) or end (recency bias) of the context, and performance drops significantly when it's in the middle. Source: "Lost in the Middle: How Language Models Use Long Contexts", Liu et al., 2023 — https://ar5iv.labs.arxiv.org/html/2307.03172.*

---

## Beginner

Think of a model's context window as a desk. You can pile a huge stack of papers on it, and the desk (the advertised context length, e.g. 200,000 tokens) might genuinely be big enough to hold all of them. But holding a paper on the desk is not the same as reading it carefully. In practice, the model tends to pay closest attention to the papers right in front of it — the ones at the top of the stack (the beginning of the conversation, like a system prompt) and the ones it just picked up (the end, i.e. your most recent message). Papers buried in the middle of the pile are technically still on the desk, but they get glanced at less carefully, and details from them are more likely to get lost or garbled when the model tries to use them.

This is the "lost in the middle" effect: even when a model's context window is large enough to fit everything, its accuracy at using facts placed in the middle of a long input is measurably worse than facts placed near the start or end.

There's a second wrinkle: "context window" numbers on a spec sheet (like 128K or 1M tokens) describe the maximum the model can accept, not how well it uses all of that space. A model might accept a million tokens but only reliably track and use a much smaller fraction of them — this gap is called the difference between advertised and effective context length.

A useful analogy for a long chat: imagine a conversation that has gone on for an hour. You technically remember everything that was said, but if someone asks you to recall a small detail mentioned 40 minutes ago, buried between many other topics, you're slower and less accurate than recalling something said in the first two minutes or the last thirty seconds. The words are all still "in your memory," but attention and recall are not uniform across them. Models exhibit a version of this same pattern, and it has direct consequences for how you should structure long instructions or long documents you paste into a chat: put the most important thing first or last, not buried in paragraph 40 of 80.

## Practitioner

The mental model to hold onto: **being "in context" is not the same as being "attended to."** Every token you send is technically visible to the model at every generation step, but the model's attention mechanism does not weight all positions equally, and empirical studies (see Expert section) show a consistent U-shaped performance curve — strong recall for information near the start and end of the input, weaker recall for information placed in the middle.

This explains a familiar frustration: you tell an assistant something important in message 3 of a 50-message session, and by message 40 it acts like it never heard you. The instruction is still sitting there in the token history — nothing deleted it — but as the conversation grows, that instruction migrates further from both endpoints and becomes relatively more "middle." Meanwhile new instructions keep landing at the freshest, best-attended position: the end. The model isn't lying when it seems to forget; it's exhibiting a real, measured positional bias.

Practical placement rules that follow from this:

1. **Recency matters — use it.** If an instruction is critical right now, restate it near the end of your prompt or your most recent message, even if you already said it earlier.
2. **Primacy matters too.** System-level instructions (the "always true" rules) belong at the very start, where they get the other end of the U-shaped curve.
3. **Don't bury the ask in the middle of a long paste.** If you paste a 10-page document and then ask a question about paragraph 5, consider moving the relevant excerpt to the top or bottom, or repeating the key fact right before your question.
4. **Long sessions degrade unevenly, not randomly.** It's not that "old stuff gets fuzzy uniformly" — it's specifically the middle of the transcript that's at risk, which is why simply "having a bigger context window" doesn't fully solve reliability.

Worked example: suppose you're debugging with an assistant across a 90-minute session. At minute 5 you say, "Never touch the `migrations/` folder — it's managed by a separate team." At minute 70, after 60 more messages of unrelated file edits, you ask the assistant to "clean up unused files in the repo." Under a naive assumption, the instruction from minute 5 is still in context, so it should be honored. But it is now deep in the middle of a long transcript, competing with dozens of more recent, more salient tool outputs and edits sitting near the end. The empirically-grounded expectation is a real (not hypothetical) risk that the model deprioritizes or misses the constraint — which is exactly why practitioners re-state hard constraints periodically, or better, hoist them into a structurally privileged position (a system prompt, or a standing file like CLAUDE.md) that gets re-read fresh rather than relying on it staying "attended to" from far back in a scrolling transcript.

## Expert

Two separate mechanisms are worth distinguishing: how position is *encoded* mathematically, and how position *predicts* recall accuracy empirically.

**Positional encoding and context extension.** Transformers need some signal for token order since self-attention itself is permutation-invariant. RoPE — Rotary Position Embedding, introduced in Su et al., "RoFormer: Enhanced Transformer with Rotary Position Embedding" (arXiv:2104.09864, 2021; published version *Neurocomputing*, 2024) — encodes absolute position by rotating query/key vectors in a way that makes the dot product between two positions depend only on their relative offset, with similarity that decays as relative distance grows. RoPE is the positional scheme underlying most modern long-context LLMs (e.g., LLaMA-family models). Because RoPE is a continuous rotation rather than a fixed lookup table, it enables post-hoc **context extension**: Chen et al., "Extending Context Window of Large Language Models via Positional Interpolation" (arXiv:2306.15595, 2023) showed that linearly *down-scaling* position indices to fit within the range seen during pretraining (rather than extrapolating past it) lets a model trained at, say, 2K or 4K tokens be fine-tuned cheaply to reliably handle 32K tokens, avoiding the degenerate attention scores that come from naive extrapolation. This "positional interpolation" family of tricks is a major reason advertised context windows have grown so quickly across model generations without full retraining from scratch.

Separately, quadratic-cost full self-attention is itself a scaling bottleneck; Beltagy, Peters, and Cohan's "Longformer: The Long-Document Transformer" (arXiv:2004.05150, 2020) is an earlier, architecturally different approach — combining local windowed attention with limited global attention tokens to get linear rather than quadratic scaling. It's less directly relevant to today's dominant decoder-only LLMs but establishes that "attend to everything" is not the only design point for long documents.

**Empirical recall degradation.** Liu et al., "Lost in the Middle: How Language Models Use Long Contexts" (arXiv:2307.03172, 2023) tested multi-document QA and key-value retrieval and found a robust U-shaped performance curve: accuracy is highest when the relevant fact sits at the very start or very end of the input, and drops substantially — sometimes below the performance of a no-context baseline — when the fact is placed in the middle. Performance also degrades as inputs get longer even for models explicitly marketed as long-context. This is the empirical anchor for "advertised vs. effective context length": a model may accept 128K or 1M tokens as input, but the RULER benchmark (Hsieh et al., "RULER: What's the Real Context Size of Your Long-Context Language Models?", arXiv:2404.06654, COLM 2024) showed that many models' reliable, task-general performance falls off well before their nominal maximum — passing a simple needle-in-a-haystack retrieval test does not imply robust performance on harder multi-hop or aggregation tasks at the same length.

**Mechanism note — the KV cache.** During autoregressive generation, the model does not recompute attention over the full prompt at every new token; it caches the key and value projections for every previous token (the KV cache) and only computes new queries against that cache. This is why context cost is dominated by memory bandwidth/footprint (cache size grows linearly with sequence length and number of layers/heads) rather than being "free" once tokens are in context — every token you keep in the conversation has a persistent memory cost for every subsequent generation step, which is part of why providers charge for and rate-limit long contexts, and why techniques like cache eviction, quantization, and compaction (see Implications) are active engineering concerns, not just cost-saving conveniences.

**Open questions.** (1) Is there a general, architecture-independent way to predict a model's *effective* context length from its advertised maximum, without running a benchmark suite like RULER per model/version? (2) Do fundamentally different long-context mechanisms (sparse/local attention à la Longformer, retrieval-augmented approaches, recurrent/state-space hybrids) eliminate positional bias, or do they just relocate it?

---

## Implications for agentic-dev

The mechanisms above are the direct justification for three load-bearing design choices in agentic coding workflows like Claude Code:

- **CLAUDE.md over "dump the whole repo in context."** If middle-of-context information is recalled less reliably (Liu et al., 2023), then flooding the context window with an entire repository's files is actively counterproductive, not just wasteful — the one instruction that matters (a coding convention, a "never touch this directory" rule) is likely to land in the least-attended region of a giant paste. CLAUDE.md solves this structurally: it's a small, curated, always-relevant file that a coding agent loads fresh and keeps near a privileged position in context (effectively re-anchored near the "start," and often re-injected), rather than one instruction competing for attention against thousands of lines of unrelated source code sitting in the middle.

- **Subagents keep exploration noise out of the main window.** When an agent has to search a codebase, read a dozen candidate files, and discard most of them, that exploration trace consumes tokens and — worse — pushes the *original task instructions* further from the endpoints of the context, toward the "lost in the middle" zone. Dispatching that search to a subagent means the main session only receives a distilled result (a file path, a summary), so the primary context window stays short and the original instructions stay closer to an endpoint where recall is strongest. This is a direct application of the positional-bias finding: isolate the noisy, low-signal work so it doesn't dilute the signal-carrying instructions in the parent context.

- **Context compaction/summarization.** As an agentic session runs long, raw conversation history grows past the point where (a) it may approach the model's advertised limit, and (b) even well within that limit, early instructions are drifting into the empirically weaker middle region. Compaction — periodically summarizing older turns into a compact digest and dropping the verbatim history — is not just a token-budget trick; it actively counteracts positional decay by re-expressing old, at-risk information in a fresh, recent position, and it reduces KV cache size, which is the concrete resource that scales with retained context. Critical constraints that must survive a long session (safety rules, "don't touch X") should either live in a structurally re-injected place like CLAUDE.md or be explicitly preserved through compaction, precisely because relying on "it's somewhere in the last 40 minutes of transcript" is exactly the failure mode Liu et al. documented.

**Concrete placement rule for long agent sessions:** critical, must-not-violate instructions belong either (a) in a system-level file that's reloaded/re-anchored (CLAUDE.md), or (b) restated near the most recent turn before a risky action — not left to sit, unrepeated, somewhere in the middle of a growing transcript.

---

## Checklist

- [ ] I can explain the difference between a model's advertised context window and its effective context length.
- [ ] I can describe the "lost in the middle" U-shaped recall curve and name the paper that demonstrated it.
- [ ] I know why the KV cache — not just raw token count — is the real driver of long-context compute/memory cost.
- [ ] I understand, at a high level, what RoPE does and why it enables techniques like positional interpolation for context extension.
- [ ] I can explain why CLAUDE.md is a better mechanism than "paste the whole repo" given positional bias.
- [ ] I can explain why subagents help keep a main context window's signal-to-noise ratio high.
- [ ] I know where to place a critical instruction in a long prompt or long agent session (start, end, or re-injected — not buried mid-transcript).

## References

1. Lost in the Middle: How Language Models Use Long Contexts (Liu, Lin, Hewitt, Paranjape, Bevilacqua, Petroni, Liang; arXiv, 2023) — https://arxiv.org/abs/2307.03172
2. RoFormer: Enhanced Transformer with Rotary Position Embedding (Su, Lu, Pan, Wen, Liu; arXiv:2104.09864, 2021; journal version *Neurocomputing*, 2024) — https://arxiv.org/abs/2104.09864
3. Extending Context Window of Large Language Models via Positional Interpolation (Chen, Wong, Chen, Tian; arXiv:2306.15595, 2023) — https://arxiv.org/abs/2306.15595
4. Longformer: The Long-Document Transformer (Beltagy, Peters, Cohan; arXiv:2004.05150, 2020) — https://arxiv.org/abs/2004.05150
5. RULER: What's the Real Context Size of Your Long-Context Language Models? (Hsieh et al.; arXiv:2404.06654, COLM 2024) — https://arxiv.org/abs/2404.06654

## Chunk summary

A model's context window has a hard size limit encoded via schemes like RoPE (extendable through tricks like positional interpolation), but size alone doesn't guarantee even attention: the "lost in the middle" effect means information near the start and end of a long input is recalled far more reliably than information buried in the center, and the KV cache is the concrete resource cost that scales with everything you keep in context. For agentic-dev, this is why CLAUDE.md exists as a curated, privileged-position file rather than a full repo dump, why subagents isolate noisy exploration from the main task thread, and why context compaction is a correctness mechanism, not just a cost optimization — in every case, the fix is to keep critical instructions structurally close to an endpoint rather than trusting a long, undifferentiated middle to hold their weight.
