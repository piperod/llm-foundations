---
title: "02 · The Transformer & Attention"
parent: Chunks
nav_order: 3
---

# Chunk 02: The Transformer & Attention

**Purpose.** Establish self-attention — a learned, content-dependent weighting of every token against every other token — as the core computation of the Transformer, and derive the cost structure that follows from its pairwise formulation.
**Previously.** Chunk 01 converted text into tokens and tokens into embedding vectors that encode meaning as position in a high-dimensional space.
**Today.** Scaled dot-product attention, multi-head attention, causal masking, positional encoding, and the quadratic cost in sequence length that shapes how context is limited and priced.

![The Transformer's encoding component (a stack of encoders) and decoding component (a stack of decoders), each built from self-attention and feed-forward sub-layers.](https://jalammar.github.io/images/t/The_transformer_encoder_decoder_stack.png)

*Figure 1: The Transformer's stacked encoder and decoder components, each layer built from self-attention and feed-forward sub-layers. Source: "The Illustrated Transformer," Jay Alammar, 2018 — https://jalammar.github.io/illustrated-transformer/.*

---

## Curiosity

Consider the sentence "The trophy didn't fit in the suitcase because it was too big." Resolving what "it" refers to requires relating that word to "trophy" and "suitcase" — words that appeared earlier and are not adjacent to it. Attention is the mechanism by which a model performs this relating: for each token, the model computes a relevance score against every other token in the context, converts those scores into weights, and builds an updated representation of the token as a weighted combination of the others. In the example, a well-trained model assigns high weight from "it" to "trophy" and "suitcase" and low weight to function words such as "because." The process resembles a reader glancing back at earlier phrases to decide what a pronoun means, except that it happens for every token at once.

The architectures that preceded the Transformer processed text recurrently: one token at a time, in order, carrying forward a fixed-size summary of everything read so far. Recurrence has two structural weaknesses. Information from distant tokens degrades as it is repeatedly compressed into the running summary, and the strict left-to-right dependency prevents parallel computation — each step must wait for the previous one. Attention removes both constraints. Every token has direct access to every other token rather than access mediated by a summary, and the computation for all positions proceeds simultaneously, which suits hardware designed for large parallel operations.

The Transformer, introduced by Vaswani et al. (2017) under the title "Attention Is All You Need," is the architecture built around this mechanism, and it underlies essentially every major large language model in production today, including Claude. One complication follows immediately from the design: because attention compares all tokens simultaneously, it carries no inherent notion of order — "dog bites man" and "man bites dog" contain the same pairwise relationships among the same words. Transformers therefore add a positional signal to each token's vector before attention runs. Attention supplies the "what relates to what"; positional encoding supplies the "what precedes what."

Direct pairwise access is one mechanism behind a model's ability to use material from early in a long context rather than only the most recent text. It is not a guarantee of perfect recall, for reasons developed in the next tier.

## Builder

The operational fact is this: every token in the context is compared against every other token in the context, at every layer, whenever the model runs. This comparison is the dominant computation in serving a language model, and its cost grows with the square of the context length in the vanilla formulation. A conversation of $$n$$ tokens induces an $$n \times n$$ grid of pairwise comparisons per layer:

```
              tok_1   tok_2   tok_3   ...  tok_n
   tok_1      [ w11    w12    w13   ...   w1n ]
   tok_2      [ w21    w22    w23   ...   w2n ]
   ...
   tok_n      [ wn1    wn2    wn3   ...   wnn ]
```

Growing a conversation from 1,000 to 10,000 tokens increases the pairwise count from $$10^6$$ to $$10^8$$ — a hundredfold increase in attention work for a tenfold increase in length. Deployed systems apply optimizations that reduce the constants substantially, but the superlinear shape of the curve remains.

Three consequences are directly observable when using a model through an API or chat interface:

1. **Time to first token grows with accumulated context.** Before producing anything, the model must run attention over the full prompt. Long conversations feel slower to start responding because the prefill computation genuinely is larger.
2. **Input tokens are priced because they cost compute.** Per-token pricing on context reflects real attention work performed on every request, which is also why the advertised context window is an engineering ceiling rather than an arbitrary product decision.
3. **Attention over a large context is a competition for weight.** The weights in each row of the grid form a normalized distribution: they sum to one. As the context grows, any given earlier detail competes with more tokens for the same total weight, so relevant material from far back can receive less influence unless its content strongly signals relevance. Long-context models mitigate this; the underlying weighted-average structure does not change.

These consequences convert into concrete practices. When a long session becomes slow or begins missing earlier instructions, the remedies all operate by reducing the effective $$n$$: trim history that no longer bears on the task, summarize large documents instead of re-pasting them in full, and state instructions concisely and near the point where they apply. Each intervention reduces both the quadratic compute term and the number of tokens competing for attention weight.

> **Try it.** [BertViz](https://github.com/jessevig/bertviz) renders the attention weights of a real transformer as connection lines between tokens; the repository links an interactive tutorial that runs in the browser. In the head view, enter a sentence with an ambiguous word — "she trained the model on new data" against "the model walked the runway" — and inspect which tokens *model* attends to in each. Then switch heads and layers: different heads exhibit visibly different patterns (positional, syntactic, coreference-like), a concrete instance of the multi-head design formalized in the Expert tier below.

## Expert

Self-attention maps an input sequence $$X \in \mathbb{R}^{n \times d_{\text{model}}}$$ to a new sequence of the same length. Three learned projection matrices produce queries, keys, and values:

$$Q = XW_Q, \qquad K = XW_K, \qquad V = XW_V$$

with $$W_Q, W_K \in \mathbb{R}^{d_{\text{model}} \times d_k}$$ and $$W_V \in \mathbb{R}^{d_{\text{model}} \times d_v}$$. Scaled dot-product attention is then

$$\mathrm{Attention}(Q, K, V) = \mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$$

The matrix $$QK^\top \in \mathbb{R}^{n \times n}$$ contains the dot-product compatibility of every query with every key; the row-wise softmax converts each row into a probability distribution; and the product with $$V$$ makes each output position a convex combination of value vectors (Vaswani et al., 2017). The $$\sqrt{d_k}$$ divisor exists because the magnitude of dot products grows with dimension: for vectors with components of roughly unit scale, unscaled logits grow with $$d_k$$, pushing the softmax into a saturated regime where it approximates a hard argmax and gradients through the non-selected positions vanish. Dividing by $$\sqrt{d_k}$$ keeps the logits in a range where the softmax remains soft and trainable.

**Multi-head attention** runs $$h$$ attention operations in parallel, each with its own projections into a lower-dimensional subspace (in the original architecture, $$d_k = d_v = d_{\text{model}}/h$$); the outputs are concatenated and projected back to $$d_{\text{model}}$$ by a learned matrix $$W_O$$. Separate heads can learn distinct relational patterns — syntactic dependency, coreference, positional adjacency — rather than forcing one attention distribution to serve every purpose.

**Causal masking.** An autoregressive model must respect the factorization from chunk 00: position $$t$$ may condition only on positions $$< t$$. Decoder-style attention enforces this by setting the logits for all future positions to $$-\infty$$ before the softmax, zeroing their weights. Training can therefore compute the loss at every position of a sequence in one parallel pass, with each position provably blind to its own future.

**Complexity.** Forming $$QK^\top$$ and applying it to $$V$$ costs $$O(n^2 d)$$ time for sequence length $$n$$ and model dimension $$d$$, with $$O(n^2)$$ memory for the attention matrix per head per layer in the vanilla formulation. Production systems change the constants without eliminating the pairwise structure: fused kernels compute attention in tiles without materializing the full $$n \times n$$ matrix, and KV caching stores keys and values from previous steps so that incremental decoding attends from one new query rather than recomputing the full prefix. The asymptotic pairwise-comparison problem persists beneath these optimizations.

**Positional information.** Attention is permutation-equivariant: permuting the input rows permutes the outputs identically, so the mechanism itself cannot distinguish orderings. The original paper injects fixed sinusoidal encodings added to the input embeddings,

$$PE_{(pos,\,2i)} = \sin\!\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right), \qquad PE_{(pos,\,2i+1)} = \cos\!\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)$$

giving each dimension pair a distinct wavelength. Modern models typically use learned or rotary positional embeddings instead; rotary embeddings (RoPE) and their consequences for context-length extension are treated in chunk 06.

Historically, attention predates the Transformer. Bahdanau et al. (2015) introduced it inside a recurrent encoder–decoder for machine translation, allowing the decoder to weight source positions directly instead of relying on a single fixed-length summary vector. The contribution of Vaswani et al. (2017) was to remove the recurrence entirely, gaining full parallelism across positions during training.

Two lines of subsequent work frame the open questions. First, the $$O(n^2 d)$$ cost motivated a large literature on efficient approximations — sparse attention patterns, low-rank projections, kernel-based linear attention — surveyed by Tay et al. (2022); how much of the quadratic cost is fundamental versus an artifact of the exact formulation remains unresolved. Second, Elhage et al. (2021) decompose small attention-only Transformers into interpretable circuits, identifying head types such as induction heads, which locate an earlier occurrence of the current token and copy what followed it — a mechanistic account of one basic in-context-learning behavior. The extent to which this circuit-level analysis scales from toy models to deep production models is an active research question.

---

## Implications for agentic-dev

The quadratic cost of vanilla attention is one mechanical reason context is a limited, metered resource rather than an unbounded log. Several design decisions in agentic tooling follow from this cost structure:

- **Subagents isolate expensive intermediate context.** A subagent that performs a large multi-file search and returns only a condensed result pays the attention cost of its intermediate tokens inside a short-lived context, instead of accumulating them in the main conversation where they would be re-attended on every subsequent turn.
- **Compaction reduces $$n$$ directly.** Summarizing older turns replaces many raw tokens with fewer, denser ones, lowering both the quadratic compute term and the number of tokens competing for attention weight.
- **`CLAUDE.md` is curated because always-present context is paid for on every turn.** Each line in an always-loaded file participates in attention for the life of the session; a short, high-signal file is a response to the cost model, not merely a style preference.

Attention cost is one factor among several in these decisions — chunk 06 treats context windows, long-context behavior, and positional extrapolation in full, and should be read before drawing quantitative conclusions about any specific system.

---

## Exercises

1. **(Curiosity)** For the sentence "The lawyer questioned the witness because she was confused," list the tokens that "she" should attend to most strongly and state what information the attention weights would need to encode to resolve the ambiguity. Then explain, using "dog bites man" versus "man bites dog," why a model with attention but no positional encoding could not distinguish the two sentences.
2. **(Builder)** A conversation grows from $$n$$ tokens to $$2n$$ tokens. State how the attention computation scales in the vanilla formulation, and explain what a user observes in (a) time to first token and (b) input-token cost per request. Then list two interventions that reduce the effective context length and identify, for each, which of the two observable effects it improves.
3. **(Expert)** Let a two-token sequence have queries $$q_1 = (1, 0)$$, $$q_2 = (0, 1)$$, keys $$k_1 = (1, 0)$$, $$k_2 = (1, 1)$$, and values $$v_1 = (1, 0)$$, $$v_2 = (0, 2)$$, with $$d_k = 2$$. Compute the scaled logits $$q_i \cdot k_j / \sqrt{2}$$, the softmax attention weights for each position, and the two output vectors. Then recompute the output at position 1 under a causal mask and explain why it differs from the unmasked case.

## Checklist

After this chunk you should be able to:

- [ ] Explain why direct pairwise access between tokens replaced recurrent summary-passing, citing both the information-degradation and parallelism arguments.
- [ ] State what the query, key, and value projections are and write the scaled dot-product attention equation, including the purpose of the $$\sqrt{d_k}$$ term.
- [ ] Describe multi-head attention as parallel attention over projected subspaces and give one reason multiple heads are used.
- [ ] Explain causal masking and its relation to the autoregressive factorization from chunk 00.
- [ ] State the $$O(n^2 d)$$ time and $$O(n^2)$$ memory cost of vanilla attention, and describe what fused kernels and KV caching change and what they do not.
- [ ] Explain why attention requires positional information and name the original sinusoidal scheme.
- [ ] Connect the quadratic cost to a specific context-management practice — subagents, compaction, or `CLAUDE.md` curation — with the causal link stated explicitly.

## References

1. Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. "Attention Is All You Need." *NeurIPS* (2017). arXiv:1706.03762. https://arxiv.org/abs/1706.03762
2. Bahdanau, D., Cho, K., & Bengio, Y. "Neural Machine Translation by Jointly Learning to Align and Translate." *ICLR* (2015). arXiv:1409.0473. https://arxiv.org/abs/1409.0473
3. Elhage, N., Nanda, N., Olsson, C., Henighan, T., Joseph, N., Mann, B., Askell, A., et al. "A Mathematical Framework for Transformer Circuits." *Transformer Circuits Thread*, Anthropic (2021). https://transformer-circuits.pub/2021/framework/index.html
4. Tay, Y., Dehghani, M., Bahri, D., & Metzler, D. "Efficient Transformers: A Survey." *ACM Computing Surveys* (2022). arXiv:2009.06732. https://arxiv.org/abs/2009.06732

## Chunk summary

Self-attention updates each token's representation as a softmax-weighted combination of value vectors, with weights derived from query–key dot products over learned projections of the input; multi-head attention runs this in parallel subspaces, causal masking preserves the autoregressive factorization, and positional encoding restores the order information that the permutation-equivariant mechanism discards. The formulation costs $$O(n^2 d)$$ time and $$O(n^2)$$ attention-matrix memory in sequence length $$n$$ (Vaswani et al., 2017), and production optimizations change constants without removing the pairwise structure. That cost profile is one mechanical reason context is limited and priced, and therefore one reason agentic tools treat context management — subagents, compaction, curated always-on files — as a first-class design concern. Mechanistic analysis of individual attention heads (Elhage et al., 2021) and efficient approximations to the quadratic computation (Tay et al., 2022) mark the open ends of the topic.
