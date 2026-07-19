---
title: "09 · Retrieval-Augmented Generation & Tool Use"
parent: Chunks
nav_order: 10
---

# Chunk 09: Retrieval-Augmented Generation & Tool Use

**Purpose.** Explain how coupling a language model to external retrieval and callable tools changes what it can reliably do, and how the behavior of invoking tools mid-generation is trained or elicited.
**Previously.** Chunk 08 covered hallucination and calibration: a model's parametric knowledge is frozen at training time, and its stated confidence is an unreliable guide to its correctness.
**Today.** One major class of mitigations: retrieval-augmented generation (RAG) as a hybrid of parametric and non-parametric memory, tool use as a trained and elicited behavior, the ReAct control pattern, and the engineering that makes model-emitted tool calls machine-executable.

> **Through-line.** Movement III. Retrieval and tools reshape the distribution by injecting fresh, grounded tokens into the context — external evidence added to the conditioning set.

![RAG architecture diagram: a query encoder and document index feed a Maximum Inner Product Search retriever, which passes top-K retrieved documents to a seq2seq generator that produces the final prediction.](https://ar5iv.labs.arxiv.org/html/2005.11401/assets/x1.png)

*Figure 1: Overview of the RAG architecture — a pre-trained retriever (query encoder + document index, using MIPS to find top-K documents) is combined with a pre-trained seq2seq generator and fine-tuned end-to-end. Source: "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks," Lewis et al., 2020 — https://ar5iv.labs.arxiv.org/html/2005.11401.*

---

## Curiosity

A language model answering from its weights alone is limited in two structural ways. First, its knowledge is frozen: everything it can state was encoded during training, so events after the training cutoff, private data, and the contents of any specific codebase are simply absent. Second, its knowledge is unverifiable in the moment: as chunk 08 established, the model produces fluent output whether or not the underlying fact was well represented in training, and it cannot pause to check. The situation is comparable to a closed-book exam; retrieval converts it into an open-book one, where the facts are looked up rather than recalled.

**Retrieval-augmented generation (RAG)** implements this conversion for text. Before the model answers, a separate retrieval system searches an external corpus — a document collection, a database, a web index — for passages relevant to the query, and those passages are placed in the model's context. The model then generates conditioned on both the question and the retrieved evidence. The division of labor matters: the corpus supplies the facts and can be updated at any time without retraining, while the model supplies the language competence to read the passages and compose an answer. Because the retrieved passages are inspectable, a wrong answer can often be traced to a specific retrieved (or missing) source.

**Tool use** generalizes retrieval from documents to arbitrary external capabilities. A model with tool access can, mid-generation, request that a calculator evaluate an expression, that a search engine run a query, or that a program execute — and then continue generating with the result available in context. Retrieval is the special case where the tool returns passages of text.

Both mechanisms change the model's role. Instead of serving as a self-contained store of facts, it serves as a coordinator: it decides when external information or computation is needed, formulates the request, and integrates what comes back. This is one mechanism behind the reliability of production systems built on models — not the entire solution to hallucination, but a substantial and inspectable part of it.

## Builder

The mechanics of a tool call are worth stating exactly, because the model executes nothing itself. The developer declares the available tools in the API request: each tool has a name, a natural-language description, and a schema (typically JSON Schema) for its arguments. During generation, the model may emit, in place of ordinary text, a structured object naming a tool and supplying arguments. The calling application intercepts that object, executes the corresponding operation — an HTTP request, a database query, a file read — and appends the result to the model's context as an observation. Generation then resumes, and the model may call another tool or produce a final answer. The model's contribution is entirely the decision to call, the choice of tool, and the arguments; execution and result-passing are conventional software.

This decomposition changes where reliability comes from. Rather than requiring the model to recall a fact correctly from parametric memory, the system requires it to (a) recognize that external information is needed, (b) formulate a call precise enough for a deterministic system to act on, and (c) synthesize the returned result. Each of these subtasks is generally easier for a model than accurate recall, particularly for time-sensitive, long-tail, or proprietary information that was never in the training corpus.

ReAct (Yao et al., 2023) names the dominant control pattern for chaining such calls: the model alternates free-form reasoning ("thoughts") with tool invocations ("actions") and their returned results ("observations"), each step conditioning the next. A representative trace:

```
Question: What is the population of the city where the 2024 Summer
Olympics were held, and is that more or less than Boston's?

Thought: I need the 2024 Summer Olympics host city.
Action: search("2024 Summer Olympics host city")
Observation: Paris, France hosted the 2024 Summer Olympics.

Thought: I need populations for Paris and Boston.
Action: search("Paris city population")
Observation: Paris city proper has approximately 2.1 million residents.

Action: search("Boston population")
Observation: Boston has approximately 650,000 residents.

Thought: 2.1 million exceeds 650,000.
Answer: Paris hosted the 2024 Olympics; its population (~2.1M) is
greater than Boston's (~650K).
```

The interleaving is the substance of the pattern. Reasoning-only generation cannot check any intermediate claim; action-only generation has no explicit plan for what to look up next. Yao et al. report that the combination outperforms either mode alone on knowledge-intensive and decision-making benchmarks, and the trace makes failures diagnosable — a malformed query or misread observation is visible in the transcript.

Tool use does not add parametric knowledge. It makes errors correctable and inspectable during generation, which for most engineering purposes is the more valuable property.

## Expert

**The RAG generative model.** Lewis et al. (2020) formalize retrieval-augmented generation as a latent-variable model in which the retrieved passage is marginalized out. In the RAG-Sequence variant, the probability of output $$y$$ given input $$x$$ is approximated by summing over the top-$$k$$ retrieved passages:

$$P(y \mid x) \approx \sum_{z \in \mathrm{top\text{-}k}(P_\eta(\cdot \mid x))} P_\eta(z \mid x)\, P_\theta(y \mid x, z)$$

where $$z$$ ranges over passages from the corpus, $$P_\eta(z \mid x)$$ is the retriever's distribution, and $$P_\theta(y \mid x, z)$$ is a seq2seq generator (BART in the original work) conditioned on the input concatenated with the passage. The retriever is a dense dual-encoder in the style of DPR: queries and passages are embedded into a shared vector space by separate encoders, relevance is scored by inner product, and retrieval is maximum inner-product search (MIPS) over a precomputed passage index. Retriever and generator are trained jointly by maximizing the marginal likelihood, with no supervision on which passage to retrieve — the retriever improves because useful passages raise the likelihood of the correct output. (RAG-Token instead marginalizes per generated token.) REALM (Guu et al., 2020) is the pretraining-time analogue: retrieval is integrated into masked language modeling itself, with the retriever trained from the start to fetch documents that help predict masked spans. Both designs make the knowledge store a swappable index — updatable without retraining, and inspectable per prediction.

**Tool use as a self-supervised behavior.** Toolformer (Schick et al., 2023) demonstrates that tool-calling can be learned without human labels on when to call. Starting from a few demonstrations per API (calculator, question answering, search, translation, calendar), the model samples candidate API-call insertions into ordinary training text, executes them, and applies a filtering criterion: a candidate call is kept only if conditioning on its result reduces the language-modeling loss on the subsequent tokens, relative to not calling or calling with the result withheld. The model is then fine-tuned on the retained, self-annotated text. The criterion is notable for grounding "when is a tool useful" in the pretraining objective itself rather than in human judgment.

**ReAct formalized.** ReAct (Yao et al., 2023) operates at inference time rather than training time. The generation is a trajectory of interleaved thoughts, actions, and observations, where actions are drawn from a fixed tool set, each observation is appended to the context before the next step, and few-shot exemplars establish the format. Thoughts are unconstrained language and serve as conditioning for subsequent action selection — the mechanism is the same context-conditioning covered in chunk 00, applied to action choice.

**From research prototype to production interface.** Contemporary function calling combines both lineages. Models are post-trained (per chunk 04) on tool-call transcripts so that emitting well-formed calls is a reliably reproduced behavior, and inference-time systems additionally apply constrained or grammar-guided decoding so that emitted calls parse against the declared JSON schema by construction rather than by hope. The Model Context Protocol (Anthropic, 2024 — an engineering announcement, not a peer-reviewed result) standardizes the surrounding plumbing: a protocol by which servers advertise tool definitions with schemas, and clients discover and invoke them at runtime. The trained behavior (emitting structured calls) and the engineering (schemas, parsers, protocol) are complementary halves of one contract.

**Open questions.** Two are load-bearing. First, retrieval quality becomes the bottleneck: the marginalization above is only as good as the support of $$P_\eta$$, so irrelevant, stale, or conflicting passages yield fluent, grounded-sounding, wrong answers — RAG relocates failure modes rather than eliminating them, and chunking, index freshness, and query formulation become the dominant levers. Second, multi-tool orchestration lacks a general reliability theory: as tool count and task depth grow, models must select among near-duplicate tools, recover from malformed or failed calls, and terminate retry loops, and current guarantees are task-specific engineering rather than principled bounds.

---

## Implications for agentic-dev

The mechanisms above are the direct engineering basis of Claude Code's operation, not a loose inspiration for it.

- **Tool definitions are the structured-output contract in production.** Every tool Claude Code exposes — Read, Edit, Bash, Grep, and MCP-provided tools alike — carries a JSON schema for its name and parameters. A "decision to read a file" is the emission of a schema-conformant call object, the behavior that tool-use post-training makes reliable and constrained decoding makes parseable.
- **MCP servers are standardized retrieval and tool endpoints.** An MCP server advertising callable capabilities with declared schemas, discoverable at runtime, is the protocol-level generalization of Toolformer's fixed API roster and ReAct's fixed action set.
- **The agent loop is a ReAct trajectory.** A debugging session — stated intent, a Grep call, returned matches appended to context, a follow-up Read — is structurally the thought/action/observation trace of the Builder example, running against a codebase instead of a search index.
- **Grounding mitigates hallucination, with a propagation caveat.** File contents and command output in context substitute retrieved evidence for parametric guesswork about a codebase (chunk 08 covers why the parametric guess fails). The caveat from the Expert tier applies directly: a stale file read or a misleading grep result is a retrieval error, and the generator will elaborate on it as confidently as on a correct one.

---

## Exercises

1. **(Curiosity)** List three questions a model cannot reliably answer from parametric memory alone (consider training cutoffs, private data, and long-tail facts), and state for each whether retrieval, a non-retrieval tool, or both would address it.
2. **(Builder)** Write a plausible ReAct trace, in the thought/action/observation format of the worked example, for the task "find out why `parse_config()` throws a `KeyError` in this repository," using a tool set of `grep(pattern)`, `read(path)`, and `run(command)`. The trace should reach a diagnosis in three to five actions, and each thought should justify the next action from the preceding observation.
3. **(Expert)** Assume a generator $$P_\theta(y \mid x, z)$$ that answers perfectly whenever the passage $$z$$ contains the answer. Identify two distinct ways the RAG-Sequence marginalization can still assign high probability to a wrong answer: (a) the correct passage is outside the top-$$k$$ support of $$P_\eta$$, and (b) retrieved passages conflict and the retriever's weights $$P_\eta(z \mid x)$$ favor the wrong one. For each, state which component you would modify and why increasing $$k$$ is not a complete fix for either.

## Checklist

After this chunk you should be able to:

- [ ] State the two structural limits of parametric memory (staleness, unverifiability) and how retrieval and tool use each address them.
- [ ] Describe the tool-call cycle precisely: declared schema, model-emitted call object, application-side execution, observation appended to context.
- [ ] Write the RAG-Sequence marginalization and identify the roles of $$P_\eta$$ and $$P_\theta$$, including how the retriever is trained without retrieval labels.
- [ ] Contrast RAG (retrieval attached at fine-tuning) with REALM (retrieval integrated into pretraining).
- [ ] State Toolformer's self-supervised filtering criterion in terms of language-modeling loss on subsequent tokens.
- [ ] Characterize ReAct as an inference-time trajectory of thoughts, actions, and observations, and explain what the interleaving adds over either mode alone.
- [ ] Connect Claude Code's tool schemas, MCP servers, and agent loop to the specific mechanisms above, and state the two open questions (retrieval quality as bottleneck, multi-tool orchestration reliability).

## References

1. Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., Küttler, H., Lewis, M., Yih, W., Rocktäschel, T., Riedel, S., & Kiela, D. "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks." *NeurIPS* (2020). arXiv:2005.11401. https://arxiv.org/abs/2005.11401
2. Guu, K., Lee, K., Tung, Z., Pasupat, P., & Chang, M. "REALM: Retrieval-Augmented Language Model Pre-Training." *ICML* (2020). arXiv:2002.08909. https://arxiv.org/abs/2002.08909
3. Schick, T., Dwivedi-Yu, J., Dessì, R., Raileanu, R., Lomeli, M., Zettlemoyer, L., Cancedda, N., & Scialom, T. "Toolformer: Language Models Can Teach Themselves to Use Tools." *NeurIPS* (2023). arXiv:2302.04761. https://arxiv.org/abs/2302.04761
4. Yao, S., Zhao, J., Yu, D., Du, N., Shafran, I., Narasimhan, K., & Cao, Y. "ReAct: Synergizing Reasoning and Acting in Language Models." *ICLR* (2023). arXiv:2210.03629. https://arxiv.org/abs/2210.03629
5. Anthropic. "Introducing the Model Context Protocol." Engineering announcement (2024). https://www.anthropic.com/news/model-context-protocol

## Chunk summary

Retrieval-augmented generation treats the retrieved passage as a latent variable: a dense dual-encoder retriever $$P_\eta$$ scores passages by inner product, and the generator's output is marginalized over the top-$$k$$, with both components trained jointly (Lewis et al., 2020); REALM integrates the same mechanism into pretraining (Guu et al., 2020). Tool use extends retrieval to arbitrary external capabilities: Toolformer shows the behavior is learnable self-supervised, keeping only calls whose results reduce subsequent language-modeling loss (Schick et al., 2023), while ReAct supplies the inference-time control pattern of interleaved thoughts, actions, and observations (Yao et al., 2023). Production function calling combines post-trained call emission with schema-constrained decoding, and MCP standardizes tool discovery and invocation. The mechanisms mitigate the parametric-memory failures of chunk 08, with two open costs: retrieval quality becomes the bottleneck, and multi-tool orchestration lacks general reliability guarantees.
