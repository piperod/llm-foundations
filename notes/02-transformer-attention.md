# Chunk 02: The Transformer & Attention

**Purpose.** Understand the mechanism — self-attention — that lets a model weigh every token against every other token, and why this replaced recurrent processing as the default architecture for language models.
**Previously.** Chunk 01 turned text into tokens and then into vectors (embeddings) that encode meaning as position in a high-dimensional space.
**Today.** How a sequence of those vectors actually gets processed: the attention mechanism, multi-head attention, positional encoding, and the transformer block that stacks them.

![The Transformer's encoding component (a stack of encoders) and decoding component (a stack of decoders), each built from self-attention and feed-forward sub-layers.](https://jalammar.github.io/images/t/The_transformer_encoder_decoder_stack.png)

*Figure 1: The Transformer's stacked encoder and decoder components, each layer built from self-attention and feed-forward sub-layers. Source: "The Illustrated Transformer," Jay Alammar, 2018 — https://jalammar.github.io/illustrated-transformer/.*

---

## Beginner

Imagine reading the sentence: "The trophy didn't fit in the suitcase because **it** was too big." To know what "it" refers to, you don't read the sentence strictly left to right and forget everything — you glance back at "trophy" and "suitcase" and decide which one makes sense as "too big." That glancing-back-and-weighing is, loosely, what **attention** does inside a language model.

Before transformers, models processed text one word at a time, in order, carrying forward a summary as they went — like reading a book while only being allowed to remember a shrinking mental note of everything so far. The further back the important detail was, the easier it was to lose. This step-by-step approach is called **recurrence**, and it's slow (each word has to wait for the previous one to finish) and forgetful over long stretches.

Attention changes the approach entirely: every word gets to directly look at every other word in the sentence at the same time, and the model learns how much each word should "attend to" each other word. In our example, the model learns that "it" should attend strongly to "trophy" and "suitcase" and weakly to "didn't" or "because." Crucially, this happens for all words simultaneously, not one at a time — which is also why transformers can be trained faster on modern hardware (GPUs are good at doing many things at once, not so good at doing one thing 10,000 times in a row).

The word "transformer" refers to the whole architecture built around this idea, introduced in a 2017 paper memorably titled "Attention Is All You Need." It's the architecture behind essentially every major chatbot today, including Claude.

One more piece: since attention looks at all words at once, it has no built-in sense of order — "dog bites man" and "man bites dog" would look identical without extra help. So the model adds a signal called **positional encoding** to each word's vector, tagging it with where it sits in the sequence. Attention gives the model the "what relates to what"; positional encoding gives it the "what comes before what."

The upshot: attention is the reason a model can hold a whole conversation in view and pull the relevant piece from three paragraphs ago, rather than only remembering the last sentence.

## Practitioner

If you use an LLM through an API or chat interface daily, here's the mental model that actually matters: **every token you send gets compared against every other token in the current context**, on every layer, for every response. That comparison is the attention computation, and it is not free — it's the dominant cost of running the model.

Concretely, if your conversation has `n` tokens, self-attention requires computing relationships between all pairs of tokens, which is proportional to `n × n` (n²). Double the conversation length, and the attention computation roughly quadruples (in the original, "vanilla" formulation — more on nuances in the Expert section). This is why:

- **Cost scales worse than linearly** with conversation length. A 10,000-token conversation isn't 10x the cost of a 1,000-token one — it can be considerably more, because of this quadratic term (though real systems use optimizations that soften this).
- **Latency creeps up** as a conversation grows, especially for the "time to first token," because the model has to attend over the full accumulated context before producing anything new.
- **Earlier details can get "diluted"** — not because the model literally deletes them, but because attention is a weighted average over a lot of competing tokens. As the haystack grows, any single needle gets relatively less weight unless the content strongly signals its own relevance.

A worked mental picture — think of attention as a spreadsheet:

```
              tok_1   tok_2   tok_3   ...  tok_n
   tok_1      [ w11    w12    w13   ...   w1n ]
   tok_2      [ w21    w22    w23   ...   w2n ]
   tok_3      [ w31    w32    w33   ...   w3n ]
   ...
   tok_n      [ wn1    wn2    wn3   ...   wnn ]
```

Every cell `w_ij` is "how much should token i attend to token j," recomputed at every layer. That grid is n×n — it grows as the square of the sequence length. This is the concrete reason "context window" is a real, load-bearing number (e.g., 200K tokens) rather than an arbitrary marketing figure — it reflects real compute and memory the provider has to allocate per request.

Practical implications you can act on: when a conversation gets long and responses feel slower or start missing earlier instructions, that's not a bug in your imagination — it's this mechanism. Trimming irrelevant history, summarizing instead of re-pasting full documents, and keeping instructions concise and near the point of use all reduce the effective `n` the model has to attend over, which helps both cost and quality.

## Expert

Formally, self-attention maps a sequence of input vectors to a new sequence by computing, for each position, a weighted sum of "value" vectors, where weights come from a compatibility function between "query" and "key" vectors. Given input embeddings `X`, the model learns projection matrices `W_Q`, `W_K`, `W_V` producing:

```
Q = X W_Q,   K = X W_K,   V = X W_V
Attention(Q, K, V) = softmax( Q K^T / sqrt(d_k) ) V
```

This is **scaled dot-product attention**: the dot product `Q K^T` measures similarity between every query and every key, the `sqrt(d_k)` scaling prevents the softmax from saturating as dimensionality grows, and the softmax turns similarities into a probability distribution used to weight the value vectors (Vaswani et al., 2017). Notice the `Q K^T` term is an `n × n` matrix for a sequence of length `n` — this is the source of the O(n²) time and memory cost in sequence length that defines the "vanilla" transformer.

**Multi-head attention** runs several of these attention operations in parallel, each with its own learned `W_Q, W_K, W_V`, then concatenates and projects the results. Each head can specialize — some heads track syntactic relationships, others track coreference (like "it" → "trophy"), others track positional adjacency. This is empirically and mechanistically documented: Anthropic's "A Mathematical Framework for Transformer Circuits" (Elhage et al., 2021) decomposes attention-only transformers into interpretable circuits and identifies specific head types, notably **induction heads**, which look back for earlier occurrences of the current token and copy what followed it — a mechanistic account of a basic in-context-learning behavior.

**Positional encoding** is necessary because attention itself is permutation-invariant — without it, shuffling the input tokens would produce the same set of pairwise attention scores. The original paper adds fixed sinusoidal functions of position to each embedding; many later architectures use learned or relative positional schemes instead (e.g., rotary embeddings), but the underlying need is the same: inject order into an otherwise order-blind mechanism.

Historically, attention did not originate with the transformer. Bahdanau, Cho, and Bengio (2015) introduced an attention mechanism inside a recurrent encoder-decoder for machine translation, letting the decoder "look back" at relevant source words instead of relying on a single fixed-length vector summary. Vaswani et al.'s 2017 contribution was to discard the recurrence entirely — "attention is all you need" — enabling full parallelization across positions during training and substantially better long-range dependency modeling.

The O(n²) cost has motivated a large body of follow-up work on efficient attention approximations — sparse attention patterns, low-rank projections, and kernel-based linear attention, surveyed in Tay et al., "Efficient Transformers: A Survey" (2022). Two open questions worth sitting with: (1) how much of the quadratic cost is fundamental versus an artifact of the specific formulation — production systems already use optimizations (e.g., flash-attention-style kernels, KV caching, sliding windows) that change the constants and sometimes the asymptotics without changing the underlying quadratic pairwise-comparison problem; (2) how far mechanistic interpretability of attention heads (à la Elhage et al.) generalizes to the largest, multi-layer production models, versus the small attention-only toy models the original circuits framework analyzed.

---

## Implications for agentic-dev

The causal chain, concretely: self-attention's core operation computes an n×n matrix of pairwise token relationships (Vaswani et al., 2017), so both compute and memory grow at least quadratically with the number of tokens in context. That is the deep, architectural reason:

1. **Context windows have hard limits.** A provider can't just offer "infinite context" — every additional token multiplies the attention cost against every existing token, not just adds to it. The stated context window (e.g., 200K tokens) is a real ceiling on how much the attention mechanism can be run over per request, not a marketing choice.
2. **Longer conversations cost more and get slower.** This is the same n² term from the Practitioner section, now tied to a specific design consequence: Claude Code cannot treat "just keep everything in context" as a free default action.
3. **This is why Claude Code uses subagents.** Spinning up a subagent to do exploratory research or a large multi-file search, and having it return only a condensed result, keeps the *main* conversation's token count — and therefore its attention cost — from ballooning with every intermediate step the subagent took. The expensive quadratic cost is paid inside the subagent's short-lived, separate context, not accumulated forever in the primary one.
4. **This is why context compaction exists.** When a session's history gets long, compacting it (summarizing older turns instead of retaining every raw token) directly reduces `n`, which reduces both cost and the attention-dilution problem described in the Practitioner section — the model has fewer, denser tokens to spread its attention weights across instead of many redundant ones.
5. **This is why CLAUDE.md is a curated file rather than "put everything in context."** Every line placed in CLAUDE.md is paid for on every single turn, for the life of the session, because it sits in the always-present context that attention runs over. A large, unfiltered dump of project documentation would tax the same quadratic mechanism on every request. Keeping CLAUDE.md short and high-signal is a direct response to the cost structure described in the Expert section, not just a style preference.

In short: attention's quadratic cost is not a background implementation detail — it is the reason context management (subagents, compaction, curated always-on files) is a first-class design concern in agentic coding tools rather than an optimization to bolt on later.

---

## Checklist

- [ ] I can explain, without jargon, why a model needs to "look back" at earlier words rather than just reading left to right.
- [ ] I can state what query, key, and value vectors represent in scaled dot-product attention.
- [ ] I can explain why attention needs positional encoding (attention alone is order-blind).
- [ ] I can explain why multi-head attention is used instead of a single attention computation.
- [ ] I can explain, in my own words, why attention cost grows roughly with the square of sequence length.
- [ ] I can connect that quadratic cost to a concrete agentic-dev practice (subagents, compaction, or CLAUDE.md curation) and explain the causal link, not just name the terms.

## References

1. Attention Is All You Need (Vaswani, Shazeer, Parmar, Uszkoreit, Jones, Gomez, Kaiser, Polosukhin — NeurIPS 2017) — https://arxiv.org/abs/1706.03762
2. Neural Machine Translation by Jointly Learning to Align and Translate (Bahdanau, Cho, Bengio — ICLR 2015) — https://arxiv.org/abs/1409.0473
3. A Mathematical Framework for Transformer Circuits (Elhage, Nanda, Olsson, Henighan, Joseph, Mann, Askell, et al. — Anthropic, Transformer Circuits Thread, 2021) — https://transformer-circuits.pub/2021/framework/index.html
4. Efficient Transformers: A Survey (Tay, Dehghani, Bahri, Metzler — ACM Computing Surveys, 2022; arXiv preprint 2020) — https://arxiv.org/abs/2009.06732

## Chunk summary

Attention lets every token in a sequence directly weigh every other token, replacing the slow, forgetful step-by-step processing of recurrent models — that's the beginner intuition, the practitioner's explanation for rising cost and latency in long conversations, and, formally, scaled dot-product attention over learned query/key/value projections plus positional encoding to restore order. The same n² pairwise-comparison cost that defines the architecture (Vaswani et al., 2017) is the concrete, non-metaphorical reason Claude Code manages context deliberately — bounded context windows, subagents that isolate expensive intermediate work, context compaction, and a deliberately curated CLAUDE.md — instead of simply expanding context indefinitely.
