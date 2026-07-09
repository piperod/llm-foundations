# Chunk 09: Retrieval-Augmented Generation & Tool Use

**Purpose.** Explain how giving a model external retrieval or callable tools — instead of relying only on what it memorized during pretraining — changes what it can reliably do, and how models are trained or prompted to invoke tools mid-generation.
**Previously.** Chunk 08 covered hallucination, calibration, and why a model's stated confidence often doesn't track its actual correctness.
**Today.** One major practical mitigation for that problem: retrieval-augmented generation (RAG) and tool use, including ReAct-style reasoning+acting and how tool-calling behavior gets trained or elicited.

![RAG architecture diagram: a query encoder and document index feed a Maximum Inner Product Search retriever, which passes top-K retrieved documents to a seq2seq generator that produces the final prediction.](https://ar5iv.labs.arxiv.org/html/2005.11401/assets/x1.png)

*Figure 1: Overview of the RAG architecture — a pre-trained retriever (query encoder + document index, using MIPS to find top-K documents) is combined with a pre-trained seq2seq generator and fine-tuned end-to-end. Source: "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks," Lewis et al., 2020 — https://ar5iv.labs.arxiv.org/html/2005.11401.*

---

## Beginner

Think about the difference between a closed-book exam and an open-book exam. In a closed-book exam, you answer purely from what you memorized — if you never learned a fact, or you misremember it, there's no way to fix that in the moment. An open-book exam is different: you can pause, flip to the right page, check the actual number or quote, and then write your answer. You're still the one reasoning and writing, but you're no longer trusting your memory alone for the facts.

A plain LLM, answering purely from its training data, is taking a closed-book exam every single time. Everything it "knows" was baked in during training and frozen at some cutoff date. If you ask about something that happened after that cutoff, or something obscure enough that it wasn't well represented in training data, the model has to guess — and as chunk 08 covered, it often guesses fluently and confidently even when wrong.

**Retrieval-augmented generation (RAG)** is the open-book version: before (or while) answering, the system fetches relevant documents — from a search engine, a company's internal files, a database — and hands them to the model as extra context. The model then writes its answer grounded in those retrieved passages instead of purely from memory.

**Tool use** is a broader version of the same idea: instead of just retrieving documents, the model can call any external capability — a calculator, a code interpreter, a web search, a weather API — mid-conversation, see the result, and use that result to keep going. It's like the exam allowing not just a textbook, but a calculator, a lab, and a phone-a-friend.

The common thread: the model stops being asked to be a perfect, self-contained oracle, and starts being asked to be a reasoner that knows when to look something up or hand off a computation to something more reliable — and then incorporate what comes back. This is a big deal because it directly addresses the closed-book weakness from chunk 08: a model that can check a source is a model that can be wrong less often, and wrong more visibly (you can inspect what it retrieved).

## Practitioner

When people say a model "calls a tool," here's the actual mechanics: the developer describes available tools to the model in the prompt or API request (name, description, and a schema for its arguments — usually JSON). During generation, instead of producing only natural-language text, the model can emit a structured output that says, in effect, "call this tool with these arguments." The calling application (not the model) intercepts that structured output, actually executes the tool — hits an API, runs a query, reads a file — and feeds the result back into the model's context as an "observation." The model then continues generating, now with that observation available, and can call another tool, or produce a final answer.

This changes reliability in a concrete way: instead of asking the model to be right the first time about something outside its training data, you're asking it to (a) recognize it needs external information, (b) formulate a query or call precisely enough that a deterministic system can act on it, and (c) synthesize the returned result into an answer. Steps (a)-(c) are individually easier for the model to get right than "recall the correct fact from parametric memory," especially for anything time-sensitive, long-tail, or proprietary (internal company data the model never saw in training).

**Worked example**, in the style of the ReAct pattern (Yao et al., 2022):

```
Question: What is the population of the city where the 2024 Summer Olympics were held, and is that more or less than Boston?

Thought: I should find which city hosted the 2024 Summer Olympics.
Action: search("2024 Summer Olympics host city")
Observation: Paris, France hosted the 2024 Summer Olympics.

Thought: Now I need Paris's population and Boston's population to compare.
Action: search("Paris population")
Observation: Paris city proper population is approximately 2.1 million.

Thought: I need Boston's population too.
Action: search("Boston population")
Observation: Boston's population is approximately 650,000.

Thought: Paris (2.1M) is larger than Boston (650K), so the answer is "more."
Answer: Paris hosted the 2024 Olympics, and its population (~2.1M) is significantly more than Boston's (~650K).
```

Notice the interleaving: **thought** (plan the next step), **action** (a structured tool call), **observation** (the tool's return value inserted into context), repeated until the model has enough grounded information to answer. This is exactly the loop Yao et al. formalized and named ReAct — reasoning traces and actions produced in alternation, each informing the next, rather than reasoning-only (which can't check facts) or acting-only (which has no plan for what to look up next or how to interpret results).

The practical upshot for anyone building on top of LLM APIs: tool use doesn't make the model smarter in the parametric-knowledge sense — it makes wrongness *correctable* and *inspectable* mid-generation, which is a different and often more valuable property than raw capability.

## Expert

**RAG architecture.** Lewis et al. (2020) formalize RAG as a hybrid parametric/non-parametric model: a **retriever** (in the original paper, DPR — Dense Passage Retrieval, a bi-encoder that embeds the query and a large corpus of documents into a shared vector space and retrieves the top-k nearest documents) supplies passages to a **generator** (a seq2seq model, BART in the original paper) that conditions its output on both the input and the retrieved passages. The whole pipeline — retriever and generator — is fine-tuned end-to-end, with the retrieved documents treated as a latent variable marginalized over during training. This is architecturally distinct from **REALM** (Guu et al., 2020), which integrates retrieval into *pretraining* itself: REALM masks tokens and trains the model to retrieve a supporting document that helps predict the masked span, learning a retriever jointly with the language model from the start rather than bolting retrieval onto a model after the fact. Both papers converge on the same insight — separating "what to say" from "where the facts live" lets you update the knowledge store without retraining the model, and lets you inspect exactly which passage supported a given claim.

**Tool use as a trained or elicited behavior.** Two distinct approaches matter here. **Toolformer** (Schick et al., 2023, Meta AI) trains tool use in a self-supervised way: starting from a handful of hand-written examples per API (a calculator, a QA system, search engines, a translator, a calendar), the model generates many candidate insertions of API calls into raw text, actually executes those calls, and keeps only the insertions where the resulting API output measurably reduces the model's loss on predicting the following tokens (versus not calling the tool, or calling it with different arguments). The model is then fine-tuned on this self-filtered dataset of "text with API calls that demonstrably helped." No human labels the individual decisions of *when* to call a tool — the model discovers that itself, filtered by whether the call was useful. **ReAct** (Yao et al., 2022; published at ICLR 2023) is instead a prompting/inference-time pattern rather than a training method: it interleaves free-form reasoning traces ("thoughts") with actions and observations within a single generation stream, using few-shot exemplars to teach the model the thought-action-observation format. Modern production tool use (OpenAI function calling, Anthropic's tool use API) combines both lineages — models are post-trained (SFT/RLHF, per chunk 04) specifically on tool-call transcripts so they reliably emit valid structured calls, while ReAct-style reasoning-then-acting remains the dominant *inference-time control pattern* wrapping those calls.

**The structured-output requirement.** For a tool call to be usable by a calling program, the model can't just say "I'll search for the population of Paris" in prose — it must emit something machine-parseable: a function name and arguments matching a declared JSON schema. This is why production tool-use APIs constrain generation (via fine-tuning toward a specific format, and often grammar-constrained decoding) to reliably produce valid JSON matching the tool's schema, rather than hoping the model's natural-language description of intent can be regex-parsed.

**Open questions.** (1) *Retrieval quality as the new bottleneck* — RAG shifts failure modes rather than eliminating them: if the retriever returns irrelevant or outdated passages, a well-behaved generator will still produce a wrong (though perhaps confidently-grounded-sounding) answer, so retrieval quality, chunking strategy, and corpus freshness become the dominant lever, not model capability. (2) *Multi-tool orchestration reliability* — as the number of available tools grows and tasks require chaining several calls, models must correctly select among near-duplicate tools, recover from failed or malformed calls, and avoid infinite retry loops; there isn't yet a settled, general theory (as opposed to task-specific engineering fixes) for guaranteeing this stays reliable as tool count and task depth scale up.

---

## Implications for agentic-dev

This chunk is the direct mechanistic ancestor of how Claude Code's tools work at all — not an analogy, a lineage.

- **MCP servers are productionized retrieval/tool endpoints.** The Model Context Protocol (Anthropic, announced November 2024) standardizes exactly the "tool" half of Lewis/Schick/Yao's picture: an MCP server exposes a set of callable capabilities (read a file, query a database, search Slack) with declared schemas, and an MCP client (Claude Code) can discover and invoke them. This is the engineering formalization of what Toolformer's API-call insertions and ReAct's "Action:" lines did in a research prototype — except now the tool roster is dynamic, discoverable at runtime, and standardized across many providers instead of being a fixed, hand-coded list.
- **Tool/function definitions in Claude Code's system are the schema requirement made concrete.** Every tool Claude Code exposes (Read, Edit, Bash, Grep, and MCP-provided tools alike) ships with a JSON Schema describing its name, description, and parameters — precisely the structured-output contract Expert-tier discussed. When Claude "decides to call Read on file X," it is emitting a structured tool-call object that matches that schema, the same mechanism Toolformer trained models to produce and that modern tool-use fine-tuning (post-training, per chunk 04) makes reliable.
- **The reasoning-then-acting loop in Claude Code's agent loop is ReAct, running in production.** When Claude Code investigates a bug, its trace of "I should check where this function is defined" (thought) -> Grep call (action) -> matched file contents returned (observation) -> "now let me read that file" (next thought) is structurally identical to the Paris/Boston worked example above. Yao et al.'s core finding — that interleaving explicit reasoning with actions outperforms acting without a plan or reasoning without the ability to check anything — is the reason Claude Code narrates intent before invoking a tool, rather than silently chaining calls.
- **Why external file/search tools change reliability, concretely.** Chunk 08 established that a model's parametric memory about, say, "what does this specific function in this specific repo do" is not knowledge it has — it was never in training data. Giving Claude Code a Read/Grep/Bash toolset doesn't make it smarter about your codebase; it lets it become an open-book reasoner about your codebase, exactly the closed-book/open-book distinction from the Beginner tier, applied to software engineering instead of trivia.

---

## Checklist

- [ ] I can explain the closed-book vs. open-book analogy for parametric memory vs. retrieval/tool use.
- [ ] I can describe the thought -> action -> observation -> answer loop and give a simple worked example.
- [ ] I can name the retriever and generator components of the original RAG architecture (Lewis et al.) and how REALM differs (retrieval integrated into pretraining).
- [ ] I can explain how Toolformer decides which tool calls are worth keeping (self-supervised filtering by loss reduction).
- [ ] I can explain why tool calls must be structured/schema-conformant rather than free natural language.
- [ ] I can connect MCP servers, Claude Code's tool schemas, and its reasoning-then-acting trace back to ReAct and Toolformer specifically, not just "AI can use tools" in the abstract.
- [ ] I can state at least one open question about retrieval quality or multi-tool reliability.

## References

1. Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks (Lewis, Perez, Piktus, Petroni, Karpukhin, Goyal, Küttler, Lewis, Yih, Rocktäschel, Riedel, Kiela, NeurIPS 2020 / arXiv:2005.11401) — https://arxiv.org/abs/2005.11401
2. REALM: Retrieval-Augmented Language Model Pre-Training (Guu, Lee, Tung, Pasupat, Chang, ICML 2020 / arXiv:2002.08909) — https://arxiv.org/abs/2002.08909
3. Toolformer: Language Models Can Teach Themselves to Use Tools (Schick, Dwivedi-Yu, Dessì, Raileanu, Lomeli, Zettlemoyer, Cancedda, Scialom, 2023, Meta AI / arXiv:2302.04761) — https://arxiv.org/abs/2302.04761
4. ReAct: Synergizing Reasoning and Acting in Language Models (Yao, Zhao, Yu, Du, Shafran, Narasimhan, Cao, ICLR 2023 / arXiv:2210.03629) — https://arxiv.org/abs/2210.03629
5. Introducing the Model Context Protocol (Anthropic, November 2024) — https://www.anthropic.com/news/model-context-protocol

## Chunk summary

RAG (Lewis et al.) and REALM (Guu et al.) show that separating "what to say" from "where the facts live" via a retriever-plus-generator architecture reduces reliance on frozen parametric memory; Toolformer (Schick et al.) shows tool-call behavior can be learned self-supervised by keeping only calls that measurably help; ReAct (Yao et al.) shows interleaving explicit reasoning with actions and observations outperforms either alone. Together these four papers are not background theory for Claude Code's tool use — they are its direct mechanistic basis: MCP servers are standardized tool/retrieval endpoints, Claude Code's tool schemas are the structured-output contract these papers require, and its thought-action-observation trace when investigating a codebase is ReAct running in a production coding agent.
