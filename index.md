---
title: Home
layout: home
nav_order: 1
permalink: /
---

# LLM Foundations

A prerequisite curriculum on how large language models work, written to be read before [agentic-dev](https://jonaprieto.github.io/agentic-dev/), a hands-on workshop on Claude Code. agentic-dev teaches practice — CLAUDE.md files, prompt structure, plan mode, review gates. This curriculum supplies the theory those practices rest on, with primary sources cited throughout.

## The idea

One claim runs through the whole workshop: a language model does nothing but produce a probability distribution over the next token, and every technique — training, prompting, retrieval, tools, agent loops — is a lever that reshapes that distribution. You never program a language model; you shape a distribution. [The Through-Line](the-through-line/) develops this and is the recommended starting point.

Once the distribution is the object you are working on, the disparate practices of an agentic workflow stop being separate skills. Context windows are limited because attention has a quadratic cost. Structured prompts work because examples reshape the runtime distribution. Models turn sycophantic because preference training biased the distribution toward agreement and your stated opinion pulls it further. Understanding the mechanism changes how you work: what you expect, what you verify, and which lever you reach for when something fails.

## How this works: chunks

A core sequence of eleven chunks, numbered 00 through 10, each covering one mechanism, followed by two applied supplements (11 · Tokenomics, 12 · Security & Key Exfiltration) that put the mechanisms to work on cost and security. Every chunk has the same structure:

- **Curiosity** — the concept in plain language, for the reader who wants to understand what is happening; assumes no technical background
- **Builder** — the operational knowledge for someone who works with LLMs daily and needs to reason about cost, reliability, and behavior
- **Expert** — the formal treatment: definitions, equations, primary papers, and open research questions
- **Implications for agentic-dev** — the specific practice each mechanism explains
- **Exercises** — short problems spanning the three tiers

Chunks build on one another; read them in order on a first pass. The final chunk hands off directly into agentic-dev's own material.

## Chunks

The full list is on the [Chunks](chunks/) page.

---

*Companion curriculum to [agentic-dev](https://jonaprieto.github.io/agentic-dev/). Source on [GitHub](https://github.com/piperod/llm-foundations).*
