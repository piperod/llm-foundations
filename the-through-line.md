---
title: The Through-Line
nav_order: 2
---

# The Through-Line

Read this before the chunks. It is the single idea the whole workshop hangs on.

## You never program a language model. You shape a distribution.

At every step, a language model does exactly one thing: it produces a probability distribution over the next token, then a token is drawn from it. That is the entire mechanism (chunk 00). It follows that **you cannot instruct a language model the way you instruct a program** — there is no rule engine, no branch, no variable to set. There is only the distribution over the next token, and the levers that reshape it.

Every technique in this workshop is one of those levers. Training reshapes the distribution and freezes the result into the weights. A prompt reshapes it at runtime. Examples, temperature, retrieved documents, tool results, and an agent's own prior steps each reshape it further. Naming the lever is the same as naming the chunk:

- **Pretraining and post-training** carve the distribution into the weights (chunks 03, 04).
- **The prompt, examples, and structure** reshape it at runtime without touching the weights (chunk 05).
- **Retrieval and tools** reshape it by adding fresh tokens to the context (chunk 09).
- **Temperature and truncation** reshape it in the final step before a token is drawn (chunk 07).

Once the distribution is the object you are working on, the disparate practices of prompting, context management, RAG, and agent design stop being separate skills. They are all the same activity — moving probability mass — applied at different points in the pipeline.

## The three movements

The chunks are grouped by *where* in the pipeline the reshaping happens.

| Movement | Chunks | Where the reshaping happens |
|---|---|---|
| **I · Carving the distribution** | 00–04 | Training writes the distribution into the weights; you understand this but cannot change it at runtime. |
| **II · Steering the distribution** | 05–07 | Context and decoding reshape a *fixed* model's distribution — the levers you actually control. |
| **III · The distribution in motion** | 08–10 | Sampling repeatedly and feeding the output back reshapes the distribution over time; behaviors emerge and compound. |
| *Applied supplements* | 11–12 | The cost of the tokens you add to reshape it, and the risk when untrusted tokens reshape it against you. |

The two levers you can name at any moment are **the weights** (what training baked in — fixed) and **the context** (what you feed in — yours to control). Since the weights are frozen once you are using the model, essentially all of your leverage is the context. The discipline of using that leverage deliberately — deciding what goes into the window, in what order, and what is kept out — is **context engineering**, and it is what agentic-dev is, in practice.

## One example that shows the whole thesis: sycophancy

Sycophancy — a model telling you what you want to hear rather than what is true — is the clearest case of two reshapings compounding, and it recurs across the workshop.

1. **The weights were reshaped toward approval.** Preference training (chunk 04) optimizes the distribution toward completions human raters prefer. Raters, being human, tend to prefer answers that agree with them, so the trained distribution acquires a standing bias toward agreement — before you type anything.
2. **The context reshapes it further.** When your prompt states an opinion, that opinion is now context (chunk 05). Conditioned on "the user believes X," the already-agreeable distribution concentrates even harder on completions that endorse X.
3. **The two compound.** A baked-in bias (weights) is triggered by a runtime cue (context), and the agreeable completion wins. Neither lever alone fully explains the behavior; the interaction does.

The remedy is a distribution-shaping move as well: reshape the context to remove the cue. Ask for a critique before a verdict, present the question without signalling your preferred answer, or require the model to argue both sides — each is a deliberate edit to what the distribution is conditioned on. That is context engineering applied to a training-side bias, and it is the pattern the whole workshop is training you to see.

---

*Next: the [Chunks](chunks/), grouped by the three movements above.*
