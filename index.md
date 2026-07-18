---
title: Home
layout: home
nav_order: 1
permalink: /
---

# LLM Foundations

A prerequisite curriculum on how large language models work, written to be read before [agentic-dev](https://jonaprieto.github.io/agentic-dev/), a hands-on workshop on Claude Code. agentic-dev teaches practice — CLAUDE.md files, prompt structure, plan mode, review gates. This curriculum supplies the theory those practices rest on, with primary sources cited throughout.

## The idea

Every practice in an agentic coding workflow is a response to a specific property of the underlying model. Context windows are limited because attention has a quadratic cost. Structured prompts with examples work because of in-context learning. Models refuse or comply consistently because of preference training, not hardcoded rules. Agents drift and compound errors over long sessions because each of these mechanisms reappears, repeatedly, inside a loop.

Understanding the mechanisms changes how you work: what you expect from the tool, what you verify before trusting its output, and which lever you reach for when something fails.

## How this works: chunks

Eleven chunks, numbered 00 through 10, each covering one mechanism. Every chunk has the same structure:

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
