# LLM Foundations — Notes Expansion Plan

Prerequisite curriculum for a repo mimicking [agentic-dev](https://jonaprieto.github.io/agentic-dev/) (Jekyll + Just the Docs, "chunks" as modules), but scoped to LLM fundamentals rather than Claude Code tooling. This is phase 1: expanded notes with verified paper references, insights, and explicit "implications for agentic-dev" bridges. Phase 2 (later) ports this into the actual Jekyll site.

## Structure per chunk

Each chunk file (`notes/NN-slug.md`) follows this shape:

- **Purpose / Previously / Today** header (agentic-dev convention)
- **Beginner** tier — plain language, analogies, zero math, zero jargon
- **Practitioner** tier — mental model, how it shows up day-to-day, what you'd tune/observe
- **Expert** tier — formal treatment, the actual papers, open research questions
- **Implications for agentic-dev** — explicit callout connecting this topic to a real agentic-dev concept/behavior
- **References** — verified papers only (title, authors, venue/year, link)
- **Checklist** + **Chunk summary** footer (agentic-dev convention)

## Chunk list

| # | Slug | Topic | Bridges to agentic-dev via |
|---|------|-------|---|
| 00 | what-is-a-language-model | Next-token prediction, what an LLM actually is | Why "autocomplete" undersells what's happening |
| 01 | tokenization-embeddings | Tokenization, embeddings | Token counts = cost/latency; why weird inputs break |
| 02 | transformer-attention | Transformer architecture, self-attention | Why context has a cost; "attention is all you need" |
| 03 | pretraining-scaling-laws | Pretraining objective, scaling laws | Why bigger/more data helps predictably; knowledge cutoffs |
| 04 | base-model-to-assistant | SFT, RLHF, Constitutional AI, DPO | Why raw models ramble; why models refuse/comply |
| 05 | in-context-learning-prompting | In-context learning theory | Directly explains agentic-dev Chunk 02 (prompting) |
| 06 | context-windows-long-context | Context windows, long-context behavior | Explains CLAUDE.md, context budgeting, subagents |
| 07 | decoding-sampling | Decoding, sampling, temperature | Why identical prompts vary; temp=0 ≠ deterministic |
| 08 | hallucination-calibration | Hallucination, calibration, uncertainty | Explains the "confidence bar" in Chunk 02 |
| 09 | retrieval-tool-use | RAG, tool use, function calling | Mechanics behind MCP tools, function calling |
| 10 | agents-putting-it-together | Multi-step agents, planning, memory | Capstone — hands off into agentic-dev's own chunks |

## Status

- [x] 00 what-is-a-language-model
- [x] 01 tokenization-embeddings
- [x] 02 transformer-attention
- [x] 03 pretraining-scaling-laws
- [x] 04 base-model-to-assistant
- [x] 05 in-context-learning-prompting
- [x] 06 context-windows-long-context
- [x] 07 decoding-sampling
- [x] 08 hallucination-calibration
- [x] 09 retrieval-tool-use
- [x] 10 agents-putting-it-together

All 11 chunks expanded with verified references (WebSearch-confirmed papers only, no fabricated citations), three-tier Beginner/Practitioner/Expert content, and an explicit "Implications for agentic-dev" bridge per chunk. Next phase: scaffold the actual Jekyll + Just the Docs site (mimicking agentic-dev's repo structure) and port this content into it.
