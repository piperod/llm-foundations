---
title: "00 · What is a Language Model"
parent: Chunks
nav_order: 1
---

# Chunk 00: What is a Language Model

**Purpose.** Establish a precise working definition of a language model — a learned probability distribution over the next token — as the foundation for every chunk that follows.
**Previously.** Nothing. This is the start of the curriculum.
**Today.** The autoregressive objective: what it is, how generation works as repeated sampling, and why the simplicity of the objective is a poor guide to the capability of the resulting system.

![Output token probabilities (logits) for GPT-2: a score for every token in the 50,257-token vocabulary, showing the model's full probability distribution over the next token.](https://jalammar.github.io/images/gpt2/gpt2-output-scores-2.png)

*Figure 1: After processing the input, the model assigns a score to every token in its vocabulary; the resulting distribution is what generation samples from. Source: "The Illustrated GPT-2," Jay Alammar, 2019 — https://jalammar.github.io/illustrated-gpt2/.*

---

## Curiosity

A language model is a system trained on a very large amount of text to answer one question: given a passage, what is likely to come next? Presented with "The capital of France is", it assigns high probability to "Paris" and negligible probability to unrelated words. To produce longer text, the system selects a likely next word, appends it to the passage, and asks the question again — now predicting what follows "The capital of France is Paris". Repeated many times, this single operation produces paragraphs, essays, and working code. (In practice models operate on *tokens* — units slightly smaller than words — which chunk 01 covers.)

The phone-keyboard autocomplete comparison is accurate as far as the mechanism goes: both systems predict a plausible next word from what came before. The difference is scale, and scale turns out to matter more than the comparison suggests. A phone keyboard is trained on a narrow corpus with a small model; it can finish common phrases. A large language model is trained on a substantial fraction of publicly available text using a network with billions of adjustable parameters. Predicting the next word well *across that entire range* — legal contracts, Python scripts, chemistry papers, dialogue — is only possible for a system that has internalized a great deal of structure: grammar, factual regularities, conventions of code, patterns of argument. The training objective is simple; what a model must represent internally to satisfy it at scale is not.

This distinction — simple objective, rich learned structure — is the correct frame for everything in this curriculum. When later chunks discuss why prompts work, why models hallucinate, or why agents drift, each explanation traces back to the fact that the system is, mechanically, a next-token predictor, and to what that mechanism does and does not guarantee.

## Builder

The operational picture, one level down. At each step of generating a response, the model computes a probability distribution over its entire vocabulary, conditioned on everything currently in context: the system prompt, the conversation history, retrieved documents, tool outputs, and every token of the response generated so far. One token is then selected from that distribution (how it is selected — greedy, temperature-scaled, nucleus sampling — is chunk 07's subject), appended, and the computation repeats.

For the prompt "The best way to reverse a string in Python is", the model produces something like:

```
P(next token | context) =
  "to"     : 0.41
  "using"  : 0.22
  "with"   : 0.14
  "a"      : 0.09
  ...      (remaining probability spread over the rest of the vocabulary)
```

There is no lookup step, no separate reasoning module, and no distinction between "answering" and "writing": the full response, including code blocks and explanations, is produced by this one repeated computation.

Three consequences follow directly, and they are the practical content of this chunk:

1. **The context is the entire input.** The distribution is conditioned on the tokens in context and nothing else. The model has no access to intent behind an ambiguous prompt; ambiguity spreads probability mass across several plausible continuations, and the sampled output lands on one of them. Prompt specificity works by concentrating that mass — a mechanical effect, not a stylistic one.

2. **Generation is path-dependent.** Each token is conditioned on all previously generated tokens, so an early error — a wrong assumption in the first sentence, a misremembered API name — shifts every subsequent distribution toward continuations consistent with that error. Models tend to elaborate on their mistakes rather than correct them, because coherent continuation is precisely what the objective rewards.

3. **Emitted reasoning is functional, not decorative.** Intermediate tokens — a plan, a step-by-step derivation — become part of the conditioning context for every later token. Requesting explicit intermediate steps changes what the final answer is conditioned on. This is the mechanism underlying chain-of-thought prompting (chunk 05) and plan-first workflows.

## Expert

An autoregressive language model defines a distribution over token sequences $$x_1, \dots, x_T$$ via the chain rule:

$$P(x_1, \dots, x_T) = \prod_{t=1}^{T} P(x_t \mid x_1, \dots, x_{t-1})$$

Training fits the conditional $$P(x_t \mid x_{<t}; \theta)$$ by maximum likelihood — equivalently, minimizing the cross-entropy loss

$$\mathcal{L}(\theta) = -\frac{1}{T}\sum_{t=1}^{T} \log P(x_t \mid x_{<t}; \theta)$$

averaged over the training corpus. Reported perplexity is the exponential of this loss, $$\mathrm{PPL} = e^{\mathcal{L}}$$, interpretable as the effective branching factor of the model's uncertainty; a model with perplexity 20 is, on average, as uncertain as a uniform choice among 20 tokens. The framing of language as a prediction problem predates neural methods: Shannon (1951) estimated the entropy of printed English at roughly 1 bit per character using human next-letter prediction, establishing both the objective and its information-theoretic interpretation.

The modern lineage is compact. Bengio et al. (2003) introduced the neural parameterization: learn a distributed vector representation per word jointly with the next-word probability function, addressing the data-sparsity failure of count-based n-gram models. Radford et al. (2018) demonstrated that a Transformer decoder trained on this objective alone, then lightly fine-tuned, transfers broadly to downstream language tasks. Brown et al. (2020) scaled the same recipe to 175B parameters and showed strong zero- and few-shot task performance with no fine-tuning at all — the clearest evidence that the unmodified objective, at sufficient scale, yields task-general competence.

The open question is what that competence consists of. Wei et al. (2022) catalogued *emergent abilities* — capabilities that appear discontinuously with scale rather than improving smoothly — and read them as evidence of qualitative transitions. Schaeffer et al. (2023) countered that most reported discontinuities are artifacts of discontinuous evaluation metrics (exact-match scoring, multiple-choice accuracy), and that the underlying competence typically improves smoothly when measured continuously. The dispute is unresolved, and it matters: whether next-token prediction alone accounts for multi-step reasoning, or whether apparent reasoning is better described as sophisticated interpolation over training data, remains an active research question rather than a settled fact. A defensible position for a practitioner: the objective is fully specified by the equations above, while the function that minimizes it at scale is only partially understood.

---

## Implications for agentic-dev

A prompt, a `CLAUDE.md` file, and a system instruction are all the same object: tokens in the conditioning context $$x_{<t}$$. There is no instruction interpreter separate from the token predictor. This single fact explains several agentic-dev practices:

- **Explicit instructions change behavior reliably** because they shift probability mass toward consistent continuations — the same influence any context tokens exert, applied deliberately.
- **Plan mode and step-first workflows improve multi-step edits** because committed intermediate tokens become conditioning context for subsequent predictions. The plan is not a preamble; it is input to everything generated after it.
- **Instructions are steering, not configuration.** An instruction shifts a distribution; it does not constrain outputs the way a type system or a permission check does. This is why Claude Code layers deterministic guardrails — permission prompts, sandboxing, hooks, tool schemas — on top of prompted behavior for anything safety-relevant, rather than relying on instructions alone.

Later chunks refine this picture (attention in chunk 02, in-context learning in chunk 05, context limits in chunk 06); the conditioning mechanism introduced here is the common foundation.

---

## Exercises

1. **(Curiosity)** Take the sentence "The weather in Bogotá in July is usually" and write down five plausible next words with rough probabilities that sum to 1. Then pick the most likely word, append it, and repeat twice more. Note how each choice constrains the next — this is the autoregressive loop performed by hand.
2. **(Builder)** Write two versions of a prompt asking for a function's documentation: one ambiguous ("tell me about parsing here") and one specific ("document the `parse_date(s: str) -> datetime` function: purpose, parameters, return value, one example"). Explain, in terms of probability mass over continuations, why the second reliably produces the intended format.
3. **(Expert)** A model reports cross-entropy loss of 2.3 nats per token on a held-out corpus. Compute its perplexity, and explain what that number means as a branching factor. Then explain why two models with identical perplexity can differ substantially in downstream task performance — what does the aggregate loss not measure?

## Checklist

After this chunk you should be able to:

- [ ] State what a language model computes at each generation step, and what the computation is conditioned on.
- [ ] Explain why the autocomplete comparison is mechanically accurate but predictively misleading.
- [ ] Describe the autoregressive loop: distribution, selection, append, repeat.
- [ ] Explain path dependence: why an early generation error propagates rather than self-corrects.
- [ ] Write the chain-rule factorization and the cross-entropy training objective, and relate loss to perplexity.
- [ ] Summarize the emergence debate (Wei et al. vs. Schaeffer et al.) and why it is unresolved.

## References

1. Bengio, Y., Ducharme, R., Vincent, P., & Jauvin, C. "A Neural Probabilistic Language Model." *Journal of Machine Learning Research* 3 (2003): 1137–1155. https://www.jmlr.org/papers/v3/bengio03a.html
2. Shannon, C. E. "Prediction and Entropy of Printed English." *Bell System Technical Journal* 30.1 (1951): 50–64. https://doi.org/10.1002/j.1538-7305.1951.tb01366.x
3. Radford, A., Narasimhan, K., Salimans, T., & Sutskever, I. "Improving Language Understanding by Generative Pre-Training." OpenAI technical report (2018). https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf
4. Brown, T. B., et al. "Language Models are Few-Shot Learners." *NeurIPS* (2020). arXiv:2005.14165. https://arxiv.org/abs/2005.14165
5. Wei, J., et al. "Emergent Abilities of Large Language Models." *Transactions on Machine Learning Research* (2022). arXiv:2206.07682. https://arxiv.org/abs/2206.07682
6. Schaeffer, R., Miranda, B., & Koyejo, S. "Are Emergent Abilities of Large Language Models a Mirage?" *NeurIPS* (2023). arXiv:2304.15004. https://arxiv.org/abs/2304.15004

## Chunk summary

A language model is a learned conditional distribution $$P(x_t \mid x_{<t})$$, trained by maximum likelihood over a large corpus and applied autoregressively to generate text. The objective has been essentially unchanged from Bengio et al. (2003) through GPT-3 (Brown et al., 2020); what changed is scale, and with it the structure the model must internalize to keep reducing loss. Whether that structure amounts to reasoning or to something weaker remains genuinely disputed (Wei et al., 2022; Schaeffer et al., 2023). For agentic work the load-bearing consequence is that all instruction — prompts, `CLAUDE.md`, plans — operates through one channel, the conditioning context, which is why prompting works, why it has limits, and why deterministic guardrails exist alongside it.
