---
title: Chunks
has_children: true
nav_order: 3
---

# Chunks

The workshop hangs on one idea, laid out in [The Through-Line](../the-through-line/): a language model does nothing but produce a distribution over the next token, and every technique here is a lever that reshapes that distribution. The chunks are grouped by *where* in the pipeline the reshaping happens — training carves the distribution, context steers it, and sampling in a loop sets it in motion.

Each chunk covers one mechanism at three levels — Curiosity, Builder, Expert — followed by an "Implications for agentic-dev" section, exercises spanning the three tiers, and references to the primary literature. Read the core sequence (00–10) in order on a first pass; the supplements (11, 12) can be read any time after chunk 09.

### Movement I · Carving the distribution

*How training writes the distribution into the weights — the half you understand but cannot change at runtime.*

| # | Chunk | What it explains |
|---|-------|---|
| 00 | [What is a Language Model](00-what-is-a-language-model) | Next-token prediction — the distribution the whole workshop reshapes |
| 01 | [Tokenization & Embeddings](01-tokenization-embeddings) | The alphabet the distribution is defined over, and its input coordinates |
| 02 | [The Transformer & Attention](02-transformer-attention) | The computation that turns context into a distribution, and its cost |
| 03 | [Pretraining & Scaling Laws](03-pretraining-scaling-laws) | The first and largest force that carves the distribution |
| 04 | [From Base Model to Assistant](04-base-model-to-assistant) | Post-training reshapes it toward preferred answers — the root of sycophancy |

### Movement II · Steering the distribution

*How context and decoding reshape a fixed model's distribution — the levers you actually control.*

| # | Chunk | What it explains |
|---|-------|---|
| 05 | [In-Context Learning & Prompting Theory](05-in-context-learning-prompting) | Reshaping the distribution at runtime with no retraining |
| 06 | [Context Windows & Long-Context Behavior](06-context-windows-long-context) | How much of the context actually pulls on the output |
| 07 | [Decoding & Sampling](07-decoding-sampling) | The final reshaping, just before a token is drawn |

### Movement III · The distribution in motion

*What happens when you sample repeatedly and feed the output back — behaviors emerge and compound.*

| # | Chunk | What it explains |
|---|-------|---|
| 08 | [Hallucination, Calibration & Uncertainty](08-hallucination-calibration) | What the distribution does where training left it under-constrained |
| 09 | [Retrieval-Augmented Generation & Tool Use](09-retrieval-tool-use) | Reshaping the distribution by injecting grounded tokens |
| 10 | [Agents — Putting It Together](10-agents-putting-it-together) | The capstone: a loop that reshapes its own distribution every step |

### Applied supplements

*The cost of the tokens you add to reshape the distribution, and the risk when untrusted tokens reshape it against you.*

| # | Chunk | What it explains |
|---|-------|---|
| 11 | [Tokenomics](11-tokenomics) | The economics of LLM inference — pricing, caching, and cost control |
| 12 | [Security & Key Exfiltration](12-security-key-exfiltration) | The threat model for secrets in agentic workflows, and the defenses that work |

After chunk 10, you're ready for [agentic-dev](https://jonaprieto.github.io/agentic-dev/) itself.
