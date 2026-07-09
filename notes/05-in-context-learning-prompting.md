# Chunk 05: In-Context Learning & Prompting Theory

**Purpose.** Explain why the words you put in a prompt — instructions, examples, ordering — change what an already-trained model does, even though no weights are updated.
**Previously.** Chunk 04 covered how a base model is turned into an assistant via SFT/RLHF, i.e., changes made *to the weights* before you ever open a chat window.
**Today.** We look at what happens *after* training is frozen: how the specific prompt you write in the moment steers behavior at inference time, and the mechanistic and statistical theories that explain why this works at all.

![Accuracy on a synthetic word-unscrambling task as a function of the number of in-context examples, comparing zero-shot/one-shot/few-shot regimes and model sizes (1.3B, 13B, 175B parameters), with and without a natural language prompt.](https://ar5iv.labs.arxiv.org/html/2005.14165/assets/graphs/img/in_context_learning.png)

*Figure 1: Larger models make increasingly efficient use of in-context information — few-shot accuracy rises with the number of in-context examples, and the effect is far more pronounced for the largest (175B) model. Source: "Language Models are Few-Shot Learners" (GPT-3 paper), Brown et al., 2020 — https://ar5iv.labs.arxiv.org/html/2005.14165.*

---

## Beginner

Imagine handing a new employee a task. You could either (a) send them to a six-month training course, or (b) just show them two or three worked examples right before they start, plus a clear instruction sheet. Option (b) is what happens every time you talk to an LLM. The "training course" already happened — that's pretraining plus the fine-tuning from chunk 04 — and it's over. Nothing in the model changes when you chat with it. What you're doing instead is closer to option (b): you're giving the model a worked example, or a clear instruction, in the same breath as the question, and it uses that context to shape its very next answer.

This is called **in-context learning**. The "learning" here is a bit of a misleading word, because no weights are being adjusted and nothing is saved for next time — close the chat and the model "forgets" everything you told it. But within that one conversation, the model behaves as if it just learned something. Show it two examples of turning messy notes into a table, and the third time you ask, it does the same thing — pattern matched, not memorized.

Why does this work? Loosely: a language model's whole job during training was to get extremely good at predicting "what comes next" given everything that came before. That includes noticing patterns *within* a single passage — if a document starts repeating a structure (like a list of Q&A pairs), a well-trained model has learned to expect the pattern to continue and complete it accordingly. So when you write "here are three examples of the format I want, now do a fourth," you're not teaching the model something brand-new — you're activating a very general skill it already has: continue the pattern you're seeing right now.

This is also why *how* you arrange a prompt matters as much as *what* you ask. A clearly labeled example is easier for the model to lock onto than the same information buried in a paragraph. You're not reprogramming the model — you're giving it a clearer pattern to continue.

## Practitioner

The mental model that matters day-to-day: **the model's weights are fixed; the prompt is the only lever you have at inference time, and it works by conditioning the probability distribution over next tokens on everything currently in context.** Nothing is stored or updated — each generation is a fresh forward pass conditioned on the full prompt. "Better prompting" is not a euphemism for "tricking" the model; it is supplying more informative conditioning context so the model's existing distribution over completions concentrates on what you want.

Two levers do most of the work, and both are well-documented empirically since Brown et al.'s GPT-3 paper, "Language Models are Few-Shot Learners" (Brown et al., NeurIPS 2020, https://arxiv.org/abs/2005.14165), which is the paper that put "in-context learning" on the map at scale. They showed that a sufficiently large model, given a handful of input/output examples *in the prompt itself* (no gradient updates), could match or approach the performance of models that had been explicitly fine-tuned on that task. The two levers:

1. **Examples (few-shot).** Showing 1-5 worked input/output pairs before the real query anchors the model's guess about task format, style, and difficulty level.
2. **Structure.** Clearly separating "the instructions" from "the content to act on" from "the examples" reduces the model's uncertainty about which text is task-defining versus which text is data to be processed.

Worked comparison — a real, common failure mode:

**Zero-shot prompt:**
```
Extract the company names from this text: "Q3 revenue at Acme Corp rose
8%, while rival Globex Inc reported a decline. Analysts at Meridian
Capital remain cautious."
```
A capable instruction-tuned model will often do fine here, because "extract company names" is an unambiguous, common task. But push toward something idiosyncratic — say, you want output as `Company | Sentiment (pos/neg/neutral)` — and zero-shot output becomes inconsistent: sometimes a bulleted list, sometimes prose, sometimes it invents a sentiment scale of its own.

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
Nothing about the model changed between these two prompts. What changed is that the second prompt hands the model two fully-worked demonstrations of the exact output grammar you want, so its next-token predictions are now conditioned on "continue this pattern" rather than "guess a reasonable format for company/sentiment extraction." In practice this is the single highest-leverage fix when a model's output format is inconsistent: don't describe the format in prose, *show* it.

The same logic explains why reordering or re-labeling a prompt (moving instructions to the top, wrapping content in delimiters, repeating the key constraint right before the ask) changes output quality without touching the model at all — you're changing which parts of the context most strongly predict what should come next, not changing the model's knowledge.

## Expert

In-context learning (ICL) is now studied as a phenomenon distinct from — and possibly analogous to — parameter learning, without literally being it. Three complementary explanations are worth holding simultaneously, because none alone is complete.

**Bayesian inference framing.** Xie et al., "An Explanation of In-context Learning as Implicit Bayesian Inference" (ICLR 2022, https://arxiv.org/abs/2111.02080), model pretraining documents as generated by latent, document-level "concepts," and argue the model learns to infer these concepts to predict the next token coherently. At inference time, a prompt with several examples supplies evidence about a shared latent concept (the "task"); the model's completion is best understood as approximately performing Bayesian posterior inference over that latent concept given the prompt-as-evidence, even though no explicit Bayesian computation is implemented — it emerges from next-token prediction training on structured data. This gives a statistical account of *why* more/better examples narrow the model's effective hypothesis space about what task is being requested, and it predicts things like: examples that are more mutually consistent (clearer shared "concept") should yield more reliable completions.

**Mechanistic/circuits framing.** Olsson et al., "In-context Learning and Induction Heads" (Anthropic, 2022, https://arxiv.org/abs/2209.11895), provide a circuit-level account: attention-head pairs called *induction heads* implement a simple algorithm — given the current token `A`, find a prior occurrence of `A` in context, look at what token followed it (`B`), and predict `B` again. This "complete the pattern you've seen before in this context" primitive, replicated and composed across layers, is argued to be responsible for a large fraction of in-context learning in transformers, from literal sequence copying up to more abstract analogical and few-shot behavior. Notably, the paper documents a sharp, simultaneous "phase change" in training loss and induction-head formation early in training, and the effect is found consistently across independently trained models (GPT-2, GPT-Neo, from-scratch replications) — evidence that this is a robust, general feature of transformer training rather than an artifact of one model. This is the clearest mechanistic bridge between "the prompt has structure" and "the model exploits that structure," because induction heads are literally attending to *positions in the current context*, not to anything stored in weights about the specific content.

**Chain-of-thought as inference-time compute.** Wei et al., "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models" (NeurIPS 2022, https://arxiv.org/abs/2201.11903), showed that including a handful of worked examples whose outputs contain intermediate reasoning steps (rather than just final answers) substantially improves accuracy on arithmetic, symbolic, and commonsense reasoning tasks in sufficiently large models. Under the ICL-as-pattern-continuation view, this is intuitive: the prompt is no longer just showing the model an input/output format, it's showing it a *process* — and the model continues that process for the new problem, effectively spending more forward-pass computation (more tokens generated, each conditioning the next) before committing to an answer. Chain-of-thought is thus best understood not as the model "reasoning" in a human sense, but as prompting-induced allocation of additional inference-time computation along a demonstrated trajectory.

Open questions worth sitting with: (1) How far does in-context pattern-continuation generalize beyond surface-level formats to genuinely novel task structures never seen in any form during pretraining — is ICL fundamentally retrieval/interpolation over pretraining data, or something more compositional? (2) Why does ICL capability (and induction-head-driven behavior) reliably strengthen with scale — is this simply "bigger models see more diverse patterns during pretraining," or is there a more fundamental sense in which scale is required for the relevant circuits to form and compose reliably? Neither is fully settled in the literature above.

---

## Implications for agentic-dev

Everything agentic-dev's Chunk 02 (Prompting) recommends is a direct, practical consequence of the mechanisms above — not stylistic preference. Concretely:

- **The system/user split** ("static" role/tone/background/examples/instructions vs. "dynamic" content/output-format/ask) exists because in-context learning is entirely about *what's in the context window right now*. Separating stable task-definition from variable content isn't just tidiness — it changes what the induction-head-style pattern-completion machinery locks onto: stable text in the same position across calls reliably predicts "this is the rule," while the user slot reliably predicts "this is the data to run the rule against."

- **"The delimiters are the single biggest win, because Claude keys on the boundaries."** This is precisely the induction-head mechanism made visible: induction heads work by finding a previous occurrence of the current token/pattern and copying what followed it. Explicit tags like `<instructions>`, `<content>`, `<examples>` give the model unambiguous, repeatable boundary tokens to anchor "which prior span do I continue matching against." Without delimiters, the boundary between "this is an instruction" and "this is content to process" is inferred from natural-language cues alone, which is strictly noisier — the same reason Xie et al.'s Bayesian framing predicts that clearer evidence of the latent "task concept" (here, sharper structural boundaries) yields a more concentrated, more reliable posterior over what to do.

- **Few-shot examples in the `<examples>` system-prompt slot** are a direct application of Brown et al. (2020): a small number of input/output demonstrations, placed in-context with no gradient update, shift the model's completions toward the demonstrated format/reasoning style, exactly as GPT-3's few-shot evaluations showed at scale. The worked zero-shot-vs-few-shot comparison above *is* the agentic-dev `<examples>` slot in miniature — this is why agentic-dev tells you to reach for examples specifically when output format or reasoning style, not just topic, needs to be locked down.

- **"Instructions are ordered steps... numbered if order matters"** and the "reminder of key rule right before the ask" both exploit next-token conditioning directly: a numbered list gives the pattern-completion mechanism an explicit sequential structure to continue (step *n* predicts step *n+1*), and restating the key constraint immediately before generation begins maximizes its influence on the very next tokens produced, since nearby context dominates what a model conditions on most strongly for the immediate continuation.

- **"Imperative, not verbose" and "precise references over compression"** reflect the Bayesian-inference framing directly: natural, grammatical, specific instructions are lower-ambiguity evidence about the intended latent task than either padded prose or keyword-soup, both of which widen the space of "concepts" consistent with the prompt and produce less reliable completions.

- **Chain-of-thought is the theoretical basis for any agentic-dev step that asks Claude to work through instructions in order before producing output** — an ordered `<instructions>` block that walks through analysis before conclusions is functionally a chain-of-thought scaffold, giving the model more conditioned inference-time computation before it commits to a final answer, exactly per Wei et al. (2022).

In short: agentic-dev's prompting chunk is not a style guide layered on top of the model — it is a direct engineering response to how in-context learning actually works mechanistically (induction heads keying on boundaries and repeated patterns) and statistically (sharper evidence narrows the inferred task).

---

## Checklist

- I can explain why in-context learning requires no weight updates — it's conditioning, not training.
- I can state what an induction head does in one sentence (finds a prior occurrence of the current token, predicts what followed it).
- I can name the Bayesian-inference framing for why more/clearer examples improve reliability (narrower posterior over the latent task).
- I can produce a zero-shot vs. few-shot version of the same prompt and explain concretely what changed in the model's conditioning.
- I can connect "delimiters help the model key on boundaries" to a specific mechanistic explanation (induction heads / pattern continuation), not just intuition.
- I can explain chain-of-thought as inference-time compute allocation rather than as the model "reasoning" like a human.

## References

1. Brown, T. B., et al. "Language Models are Few-Shot Learners." NeurIPS 2020. https://arxiv.org/abs/2005.14165
2. Xie, S. M., Raghunathan, A., Liang, P., & Ma, T. "An Explanation of In-context Learning as Implicit Bayesian Inference." ICLR 2022. https://arxiv.org/abs/2111.02080
3. Olsson, C., et al. "In-context Learning and Induction Heads." Anthropic, Transformer Circuits Thread, 2022. https://arxiv.org/abs/2209.11895
4. Wei, J., et al. "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models." NeurIPS 2022. https://arxiv.org/abs/2201.11903
5. agentic-dev, Chunk 02: Prompting. https://jonaprieto.github.io/agentic-dev/chunks/02-prompting/

## Chunk summary

In-context learning lets a fixed set of weights produce very different behavior depending purely on what's placed in the prompt — no training involved — and this is explained at the statistical level as implicit Bayesian inference over a latent task, and at the mechanistic level by induction heads that complete patterns already visible in the current context. Chain-of-thought prompting is the special case of showing the model a reasoning *process* to continue, effectively buying more inference-time computation. Every concrete recommendation in agentic-dev's Chunk 02 — system/user separation, explicit delimiters, few-shot examples, ordered instructions, precise wording — is a direct engineering exploitation of these same mechanisms, not independent stylistic advice.
