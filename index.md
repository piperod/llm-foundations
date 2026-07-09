---
title: Home
layout: home
nav_order: 1
permalink: /
---

# LLM Foundations

A prerequisite curriculum on how large language models actually work — grounded in cited papers, not vibes. It exists to be read *before* [agentic-dev](https://jonaprieto.github.io/agentic-dev/), a workshop on Claude Code mastery: agentic-dev tells you what to do (write a CLAUDE.md, structure your prompts, use plan mode); this fills in why those things work, at the level of the model itself.

## The idea

Every practice in an agentic coding workflow is downstream of some mechanism in how LLMs are built and run. Context windows have limits because attention has a cost. Prompts with structure and examples work because of in-context learning. Models refuse or comply consistently because of RLHF, not because of a rule someone hardcoded. Agents drift and compound errors over long sessions because every mechanism above reappears, repeatedly, inside a loop.

Knowing this doesn't just satisfy curiosity — it changes what you expect from the tool, what you check before trusting its output, and where the real levers are when something isn't working.

## How this works: chunks

Eleven chunks, numbered 00 through 10, each covering one mechanism. Every chunk has the same shape:

- **Beginner** — plain language, no math, no unexplained jargon
- **Practitioner** — the mental model you need if you use LLMs daily but haven't studied the theory
- **Expert** — the formal treatment, the actual papers, and the open questions
- **Implications for agentic-dev** — the concrete bridge from "here's the mechanism" to "here's the agentic-dev practice it explains"

Read them in order the first time through — later chunks build on earlier ones and hand off directly into agentic-dev's own material at the end.

## Chunks

See the full list on the [Chunks](chunks/) page.

---

*Companion curriculum to [agentic-dev](https://jonaprieto.github.io/agentic-dev/). Source on [GitHub](https://github.com/piperod/llm-foundations).*
