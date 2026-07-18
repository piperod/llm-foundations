---
title: Chunks
has_children: true
nav_order: 2
---

# Chunks

A core sequence of eleven chunks, in order, plus two applied supplements. Each covers one mechanism at three levels — Curiosity, Builder, Expert — followed by an "Implications for agentic-dev" section connecting the mechanism to a concrete practice in agentic coding tools, exercises spanning the three tiers, and references to the primary literature.

Chunks build on one another; read the core sequence in order on a first pass. The supplements (11, 12) apply the mechanisms to cost and security and can be read any time after chunk 09.

| # | Chunk | What it explains |
|---|-------|---|
| 00 | [What is a Language Model](00-what-is-a-language-model) | Next-token prediction — what an LLM fundamentally is |
| 01 | [Tokenization & Embeddings](01-tokenization-embeddings) | How text becomes tokens, and tokens become vectors |
| 02 | [The Transformer & Attention](02-transformer-attention) | Self-attention, and why context has a compute cost |
| 03 | [Pretraining & Scaling Laws](03-pretraining-scaling-laws) | Why bigger/more data helps predictably, and knowledge cutoffs |
| 04 | [From Base Model to Assistant](04-base-model-to-assistant) | SFT, RLHF, Constitutional AI, DPO — why models refuse/comply |
| 05 | [In-Context Learning & Prompting Theory](05-in-context-learning-prompting) | Why examples and structure in a prompt change behavior |
| 06 | [Context Windows & Long-Context Behavior](06-context-windows-long-context) | Why context has limits, and "lost in the middle" |
| 07 | [Decoding & Sampling](07-decoding-sampling) | Temperature, top-p/top-k, and why output isn't deterministic |
| 08 | [Hallucination, Calibration & Uncertainty](08-hallucination-calibration) | Why models confabulate, and what calibration means |
| 09 | [Retrieval-Augmented Generation & Tool Use](09-retrieval-tool-use) | RAG and tool-calling — the mechanics behind MCP tools |
| 10 | [Agents — Putting It Together](10-agents-putting-it-together) | The capstone: an agent is an LLM in a loop, and every prior mechanism compounds inside it |
| 11 | [Tokenomics](11-tokenomics) | Applied supplement: the economics of LLM inference — pricing, caching, and cost control |
| 12 | [Security & Key Exfiltration](12-security-key-exfiltration) | Applied supplement: the threat model for secrets in agentic workflows, and the defenses that work |

After chunk 10, you're ready for [agentic-dev](https://jonaprieto.github.io/agentic-dev/) itself.
