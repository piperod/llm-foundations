---
title: "05 · In-Context Learning & Prompting Theory"
parent: Chunks
nav_order: 6
---

# Chunk 05: In-Context Learning & Prompting Theory

**Purpose.** Explain why the contents of a prompt — instructions, examples, ordering — change the behavior of an already-trained model at inference time, even though no weights are updated.
**Previously.** Chunk 04 covered how a base model becomes an assistant through SFT and RLHF: modifications made to the weights before deployment.
**Today.** In-context learning after the weights are frozen: the phenomenon itself, the statistical and mechanistic theories proposed to explain it, and chain-of-thought prompting as a special case.

![Accuracy on a synthetic word-unscrambling task as a function of the number of in-context examples, comparing zero-shot/one-shot/few-shot regimes and model sizes (1.3B, 13B, 175B parameters), with and without a natural language prompt.](https://ar5iv.labs.arxiv.org/html/2005.14165/assets/graphs/img/in_context_learning.png)

*Figure 1: Larger models make increasingly efficient use of in-context information — few-shot accuracy rises with the number of in-context examples, and the effect is far more pronounced for the largest (175B) model. Source: "Language Models are Few-Shot Learners" (GPT-3 paper), Brown et al., 2020 — https://ar5iv.labs.arxiv.org/html/2005.14165.*

---

## Curiosity

In-context learning is the phenomenon in which a model with fixed weights behaves as if it had learned a new task from examples supplied in the prompt itself. Show the model two examples of converting messy notes into a table, then present a third set of notes, and it produces a table in the demonstrated format. No parameter is adjusted, nothing is stored for later, and closing the conversation discards everything: each response is a fresh forward pass conditioned on the full prompt. The situation is comparable to giving a new employee an instruction sheet and two worked examples immediately before the task, rather than sending them through a training course — the training course (pretraining and the fine-tuning of chunk 04) already happened and is over.

The term "learning" is therefore somewhat misleading. What actually occurs is conditioning. Pretraining optimized one objective: predict the next token given everything before it. Documents in the training corpus frequently contain repeated internal structure — lists of question–answer pairs, tables, parallel translations — so a model that predicts well has necessarily learned to detect a pattern within the current passage and extend it. A few-shot prompt does not teach the model anything new; it activates a general capability the model already possesses: continue the pattern currently visible in context.

One consequence follows immediately. The arrangement of a prompt matters as much as its content, because a clearly delimited, consistently formatted example is a stronger pattern signal than the same information embedded in a paragraph of prose. Prompting does not reprogram the model; it presents a cleaner pattern to continue.

## Builder

The operational principle: the weights are fixed, the prompt is the only control available at inference time, and it operates by conditioning the next-token distribution on everything currently in context. Effective prompting is not a way of tricking the model; it is the supply of more informative conditioning context, so that the model's distribution over completions concentrates on the intended output.

The empirical foundation is Brown et al. (2020), who showed that a sufficiently large model, given a handful of input/output examples in the prompt with no gradient updates, matches or approaches the performance of models explicitly fine-tuned on the same task. Two levers account for most of the practical effect:

1. **Examples (few-shot).** One to five worked input/output pairs before the real query anchor the model's inference about task format, style, and difficulty.
2. **Structure.** Clear separation of instructions, examples, and the content to be processed reduces the model's uncertainty about which text defines the task and which text is data.

A worked comparison illustrates the difference.

**Zero-shot prompt:**
```
Extract the company names from this text: "Q3 revenue at Acme Corp rose
8%, while rival Globex Inc reported a decline. Analysts at Meridian
Capital remain cautious."
```

A capable instruction-tuned model usually handles this, because "extract company names" is a common, unambiguous task. The failure mode appears when the required output is idiosyncratic — for example, `Company | Sentiment (pos/neg/neutral)`. Zero-shot output then becomes inconsistent across calls: sometimes a bulleted list, sometimes prose, sometimes an invented sentiment scale.

**Few-shot prompt:**
```
Extract companies and sentiment in this exact format:

Input: "Shares of Nordstream Ltd tumbled after a profit warning."
Output: Nordstream Ltd | negative

Input: "Boca Systems beat expectations, sending the stock higher."
Output: Boca Systems | positive

Input: "Q3 revenue at Acme Corp rose 8%, while rival Globex Inc reported
a decline. Analysts at Meridian Capital remain cautious."
Output:
```

Nothing about the model changed between the two prompts. The second supplies two fully worked demonstrations of the exact output grammar, so next-token prediction is conditioned on "continue this pattern" rather than "select a reasonable format for company/sentiment extraction." When output format is inconsistent, demonstrating the format is generally higher-leverage than describing it in prose.

The same principle explains why reordering and relabeling a prompt — moving instructions to the top, wrapping content in delimiters, restating the key constraint immediately before the request — changes output quality without touching the model. These edits change which parts of the context most strongly predict the continuation; they do not change the model's knowledge.

## Expert

In-context learning (ICL) is studied as a phenomenon distinct from parameter learning. Three complementary accounts exist; none alone is complete.

**Implicit Bayesian inference.** Xie et al. (2022) model pretraining documents as generated from latent, document-level concepts $$\theta$$: a document is coherent because a single concept governs its structure, and a model trained on such data learns to infer the governing concept in order to predict the next token. At inference time, a prompt containing several examples supplies evidence about a shared latent concept — the task — and the model's completion approximates posterior-predictive inference:

$$P(\text{output} \mid \text{prompt}) = \int_\Theta P(\text{output} \mid \theta)\, P(\theta \mid \text{prompt})\, d\theta$$

No explicit Bayesian computation is implemented; the behavior emerges from next-token training on concept-structured data. The framing yields a testable practical prediction: examples that are mutually consistent — exhibiting a single clear shared concept — concentrate the posterior $$P(\theta \mid \text{prompt})$$ on the intended task and produce more reliable completions, while inconsistent or ambiguous examples spread posterior mass across competing task hypotheses.

**Induction-head circuits.** Olsson et al. (2022) give a mechanistic account at the level of attention heads. An *induction head* is a two-head composition: a "previous token head" in an earlier layer writes each token's predecessor into that token's residual stream; a downstream head then uses this information to match the current token against earlier occurrences in context and attend to the token that followed, copying it forward. The circuit implements the rule $$[A][B] \dots [A] \rightarrow [B]$$: having seen the pattern once, complete it on recurrence. Composed and generalized across layers, this primitive is argued to account for a substantial fraction of in-context learning in transformers, from literal copying to fuzzier analogical matching. The paper documents a sharp phase change early in training — visible as a bump in the loss curve — during which induction heads and measurable in-context learning ability form simultaneously, an effect replicated across independently trained models (GPT-2, GPT-Neo, from-scratch replications). This is the clearest available bridge between prompt structure and its exploitation: induction heads attend to positions in the current context, not to task-specific content stored in weights.

**Chain-of-thought as inference-time computation.** Wei et al. (2022) showed that few-shot exemplars whose outputs contain intermediate reasoning steps, rather than bare answers, substantially improve accuracy on arithmetic, symbolic, and commonsense reasoning tasks — with the gains appearing predominantly in sufficiently large models. Under the conditioning view of chunk 00, the mechanism is direct: each emitted reasoning token becomes part of the context on which every subsequent prediction is conditioned, so demonstrating a reasoning process induces the model to allocate additional serial computation — more forward passes, each conditioning the next — before committing to an answer. Chain-of-thought is one mechanism behind improved multi-step accuracy; it is not, by itself, evidence of human-like reasoning.

Two questions remain open in this literature. First, how far in-context pattern continuation generalizes beyond surface-level formats to genuinely novel task structures absent from pretraining — whether ICL is fundamentally retrieval and interpolation over pretraining data or something more compositional. Second, why ICL reliably strengthens with scale: whether larger models simply encounter more diverse patterns during pretraining, or whether scale is required for the relevant circuits to form and compose reliably. Neither is settled.

---

## Implications for agentic-dev

The recommendations in agentic-dev's Chunk 02 (Prompting) are engineering consequences of the mechanisms above, not stylistic preferences.

- **The system/user split** — static role, background, examples, and instructions versus dynamic content and ask — works because ICL depends entirely on what is in the context window. Stable text in a consistent position across calls reliably signals "this is the rule"; the user slot reliably signals "this is the data the rule applies to," giving the pattern-completion machinery a fixed structure to key on.
- **Delimiters** such as `<instructions>`, `<content>`, and `<examples>` supply repeatable boundary tokens for exactly the matching operation induction heads perform. Without them, the boundary between instruction and content must be inferred from natural-language cues, which is strictly noisier. The Bayesian framing makes the same prediction: sharper structural evidence about the latent task concentrates the posterior.
- **Few-shot examples in the `<examples>` slot** apply Brown et al. (2020) directly: a small number of in-context demonstrations shifts completions toward the demonstrated format with no gradient update. Examples are the appropriate lever when output format or reasoning style, rather than topic, must be fixed.
- **Ordered, numbered instructions and a restated key constraint before the ask** both exploit next-token conditioning: a numbered list is an explicit sequential pattern to continue, and a constraint placed immediately before generation exerts strong influence on the first tokens produced, since nearby context dominates the immediate continuation.
- **Imperative, precise wording** is lower-ambiguity evidence about the intended latent task than padded prose or compressed keyword lists, both of which widen the space of task hypotheses consistent with the prompt.
- **Instruction blocks that walk through analysis before conclusions** function as a chain-of-thought scaffold: they induce conditioned intermediate computation before the model commits to a final answer, per Wei et al. (2022).

The prompting chunk is therefore best read as a direct response to how ICL works mechanistically (induction heads keying on boundaries and repeated structure) and statistically (sharper evidence yields a more concentrated posterior over the task).

---

## Exercises

1. **(Curiosity)** Explain why the word "learning" in "in-context learning" is misleading. State precisely what changes and what does not change in the model between two turns of a conversation, and what happens to in-context information when the conversation ends.
2. **(Builder)** Choose a format-sensitive task (for example, extracting entries as `Name | Category | Confidence`). Write a zero-shot prompt and a 3-shot prompt for it. Predict the characteristic failure mode of each: what the zero-shot version gets wrong across repeated calls, and what the few-shot version can still get wrong (for example, when a test input deviates structurally from all three exemplars).
3. **(Expert)** Using the circuit description in Olsson et al. (2022), classify which features of a prompt an induction head can exploit and which it cannot. Consider: repeated literal delimiter tokens; exemplars sharing a consistent token-level structure; an instruction stated once and never repeated; a constraint expressed as a semantic paraphrase of an earlier constraint. Justify each classification in terms of the $$[A][B] \dots [A] \rightarrow [B]$$ matching operation.

## Checklist

After this chunk you should be able to:

- [ ] Explain why in-context learning involves no weight updates, and restate it as conditioning of the next-token distribution.
- [ ] Describe the induction-head circuit as a two-head composition, including the role of the previous-token head and the $$[A][B] \dots [A] \rightarrow [B]$$ rule.
- [ ] State the implicit-Bayesian-inference framing, including its posterior-predictive form and the prediction that mutually consistent examples concentrate the posterior.
- [ ] Produce zero-shot and few-shot versions of the same prompt and explain what changed in the model's conditioning.
- [ ] Connect the use of delimiters to a specific mechanistic account rather than to intuition.
- [ ] Characterize chain-of-thought prompting as prompting-induced allocation of additional serial computation.
- [ ] State the two open questions: generalization beyond surface patterns, and why ICL strengthens with scale.

## References

1. Brown, T. B., et al. "Language Models are Few-Shot Learners." *NeurIPS* (2020). arXiv:2005.14165. https://arxiv.org/abs/2005.14165
2. Xie, S. M., Raghunathan, A., Liang, P., & Ma, T. "An Explanation of In-context Learning as Implicit Bayesian Inference." *ICLR* (2022). arXiv:2111.02080. https://arxiv.org/abs/2111.02080
3. Olsson, C., et al. "In-context Learning and Induction Heads." *Transformer Circuits Thread*, Anthropic (2022). arXiv:2209.11895. https://arxiv.org/abs/2209.11895
4. Wei, J., et al. "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models." *NeurIPS* (2022). arXiv:2201.11903. https://arxiv.org/abs/2201.11903
5. agentic-dev. "Chunk 02: Prompting." https://jonaprieto.github.io/agentic-dev/chunks/02-prompting/

## Chunk summary

In-context learning is the capacity of a fixed-weight model to exhibit task-specific behavior determined entirely by the prompt: conditioning, not training. Two theories explain it at different levels. Statistically, the model approximates posterior-predictive inference over a latent task concept, so mutually consistent examples concentrate the posterior on the intended task (Xie et al., 2022). Mechanistically, induction heads — two-head circuits implementing $$[A][B] \dots [A] \rightarrow [B]$$ — complete patterns visible in the current context, and form in a documented phase change coincident with the emergence of ICL ability (Olsson et al., 2022). Chain-of-thought prompting is the special case of demonstrating a process, inducing additional serial computation before the answer (Wei et al., 2022). The prompting practices in agentic-dev's Chunk 02 — system/user separation, delimiters, few-shot examples, ordered instructions — are direct engineering applications of these mechanisms.
