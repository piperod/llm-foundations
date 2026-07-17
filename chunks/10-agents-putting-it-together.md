---
title: "10 · Agents — Putting It Together"
parent: Chunks
nav_order: 11
---

# Chunk 10: Agents — Putting It Together

**Purpose.** Synthesize the curriculum: an agent is not a new kind of model but a frozen LLM invoked repeatedly inside a loop, and every mechanism from chunks 00–09 reappears inside that loop, now compounding over many steps instead of acting once.
**Previously.** Chunk 09 covered retrieval-augmented generation and tool use — mechanisms for reading information and taking actions outside the model's weights.
**Today.** The agent loop (act, observe, update context, repeat), planning and memory architectures, self-correction without weight updates, evaluation of multi-step trajectories, and the three failure modes — error compounding, context rot, and goal drift — that appear only when many LLM calls are chained. This is the capstone chunk: it introduces few new mechanisms and instead shows how the prior ones interact.

![Anthropic's "Autonomous agent" diagram: an LLM calls tools in a loop based on environmental feedback, with a human able to provide input, continuing until the task is complete.](https://www-cdn.anthropic.com/images/4zrzovbb/website/58d9f10c985c4eb5d53798dea315f7bb5ab6249e-2401x1000.png)

*Figure 1: Anthropic's canonical "autonomous agent" diagram — an LLM in a loop, calling tools based on feedback from its environment until the task is complete. Source: "Building Effective AI Agents," Erik Schluntz and Barry Zhang, Anthropic, Dec 2024 — https://www.anthropic.com/research/building-effective-agents.*

---

## Beginner

Every prior chunk treated a single exchange: an input goes in, a response comes out, and the interaction ends. An **agent** is a language model used repeatedly. The system examines the current state of a task, selects one small action — running a search, editing a file, executing a command — performs it, observes the result, and selects the next action from the updated state, continuing until the task is complete or abandoned. The loop is the new element; the model inside it is the same next-token predictor introduced in chunk 00. The arrangement resembles a new employee working through an unfamiliar project one task at a time, checking each result before choosing the next.

Three failure modes distinguish this arrangement from single-exchange use, and none of them is visible when watching any one step in isolation. First, **errors compound**: a mistaken conclusion recorded at an early step is treated as established fact by every later step, so downstream mistakes appear as natural continuations of a plan that was already quietly broken. Second, **the working record degrades**: the transcript of actions and observations grows with every step, and the model's ability to identify which parts still matter declines as relevant information is buried under accumulated intermediate results. Third, **the goal drifts**: over many steps, the working objective can shift away from the original request without any single step constituting a visible wrong turn.

Each tendency traces to mechanisms established earlier in the curriculum. Generation proceeds one token at a time with no ability to revise earlier commitments; memory is limited to a finite context window; and the model's confidence in its own output is not a reliable indicator of correctness. Agents remain highly useful despite all three tendencies. Using them well requires managing these mechanisms deliberately rather than trusting the loop to correct itself.

## Practitioner

The operational difference between chat and agents is that a single-turn error is a one-time cost while a multi-step error is a compounding one. In chat, a hallucination or misreading surfaces immediately and the user corrects it in the next turn. In an agent loop, the same error is written into the context, conditioned on by every subsequent step, and can steer the trajectory for twenty more steps before anyone notices — because each individual step, viewed alone, is a locally reasonable continuation of what precedes it.

The three failure modes, operationally:

- **Error compounding.** A wrong assumption at step 3 does not disappear; it becomes conditioning context for every later step. If step 3 records "the config file is at `src/config.json`" and the path is wrong, every subsequent read or write inherits the error, and the model may generate plausible-sounding justifications for the wrong path — hallucination (chunk 08) interacting with the loop structure.
- **Context rot.** Every tool call, file read, and intermediate result is appended to the context, and signal is progressively diluted by accumulated noise. Chroma's 2025 report ("Context Rot: How Increasing Input Tokens Impacts LLM Performance") tested 18 frontier models and found that performance is not flat across the context window: it degrades unevenly as input grows, well within advertised context limits, and degrades faster when relevant information competes with large volumes of irrelevant text. This is related to the lost-in-the-middle effect from chunk 06, with the aggravating factor that agent loops are precisely what generates long, noisy contexts.
- **Goal drift.** Over a long session, the working objective shifts as the agent responds to the most recent tool outputs rather than re-grounding in the original instruction. No individual step breaks; the sum of many small directional nudges does.

A composite, illustrative trace of a typical compounding failure:

```
Step 1:  Task: "Fix the failing test in payment_test.py."
Step 4:  Agent misdiagnoses the failure as a timezone bug (it is a rounding bug).
Step 7:  Agent edits timezone-handling code; the test still fails.
Step 9:  Conditioned on a context now full of timezone edits, the agent concludes
         the fix is incomplete and broadens the change to unrelated date-handling code.
Step 14: The test still fails; two unrelated files have been modified on the basis
         of the wrong premise recorded at step 4.
```

No step is individually implausible. The failure resides in the accumulation, which is why per-step review of such a trace often finds nothing to object to.

Five safeguards have consistent practical value. **Checkpoints**: pause for human or automated confirmation of progress rather than running twenty steps unsupervised. **Explicit re-grounding**: periodically restate the original goal into context rather than assuming it survived fifteen steps of tool output. **Narrow, verifiable steps**: prefer steps whose outcomes are checkable — a test passing, a diff reviewable — over steps that are merely plausible. **Context pruning**: drop or summarize stale tool output instead of letting the transcript grow without bound. **Fresh sub-contexts**: run a bounded sub-task in a clean context rather than in one ever-growing session (the subagent pattern, discussed below).

## Expert

Formally, an agent is a policy

$$\pi(a_t \mid c_t)$$

where $$c_t$$ is the accumulated context at step $$t$$ — system prompt, task statement, prior actions, prior observations — and $$a_t$$ is the next action: a tool call, a reasoning step, or a final answer. After the action executes, the environment (a tool, a file system, a user) returns an observation $$o_t$$, and the context updates by concatenation:

$$c_{t+1} = c_t \oplus a_t \oplus o_t$$

Two properties of this formalization carry most of the weight. The policy is a **frozen** pretrained LLM: no parameter changes during the episode. Consequently, **all state lives in $$c_t$$**: everything the agent "knows" about the task — plan, progress, mistakes — exists as tokens in the context and nowhere else. Interleaving reasoning traces with actions in this loop is the ReAct pattern (Yao et al., 2022, covered in chunk 09). The loop therefore re-applies, at every step, each mechanism from earlier chunks: autoregressive generation with no revision of emitted tokens (chunk 00), finite context with position- and length-dependent attention degradation (chunks 02, 06), stochastic decoding unless deliberately constrained (chunk 07), and a hallucination and miscalibration base rate that does not vanish because the model is "reasoning" (chunk 08).

**Reliability compounds multiplicatively.** Under the idealizing assumption that each step succeeds independently with probability $$p$$, an $$n$$-step trajectory succeeds with probability

$$P(\text{trajectory}) = p^{\,n}, \qquad \text{e.g. } 0.95^{20} \approx 0.358.$$

A per-step reliability that would be considered excellent in single-turn evaluation yields roughly one clean 20-step trajectory in three. The independence assumption, moreover, fails in the unfavorable direction: an erroneous action or observation is appended to $$c_t$$ and conditions every subsequent step, so step failures are positively correlated — one error lowers the success probability of everything downstream, and the context-rot findings above describe the same degradation at the level of context quality rather than discrete errors. Two consequences for evaluation follow. Per-step accuracy is not a sufficient statistic for trajectory success, so agent benchmarks must score end states of full trajectories — correctness, step count, error recovery, token and latency cost — rather than sampled steps. And because a single trajectory is one draw from a high-variance distribution, credible reliability estimates require many rollouts per task, which is expensive and remains methodologically unsettled. Xi et al.'s survey covers the evaluation landscape alongside agent construction and applications, and is a reasonable single entry point to the wider literature.

**Planning and memory architectures.** Sumers et al. (2023) organize agent design using a cognitive-science vocabulary — working memory, episodic memory, semantic memory, procedural memory — as an alternative to treating the context as an undifferentiated scratchpad; the taxonomy's value is that it names which structure a given engineering intervention modifies. Park et al. (2023) provide the reference architecture for memory beyond a single context window: agents maintain a *memory stream* of timestamped natural-language observations, periodically synthesize higher-level *reflections* from clusters of related memories, and retrieve memories at decision time using a score combining recency, importance, and relevance. This is retrieval-augmented generation (chunk 09) applied to the agent's own history rather than to external documents, and it exists because $$c_t$$ is finite and cannot hold everything an agent has done.

**Self-correction without weight updates.** Shinn et al.'s Reflexion (2023) implements reinforcement entirely in context: after a failed attempt, the agent generates a natural-language self-reflection on the failure, stores it in an episodic buffer, and includes it in the context of the next attempt. In the terms of the formalization above, this is exact: the policy $$\pi$$ is immutable, so the only channel through which experience can improve behavior is $$c_t$$ — in-context learning (chunk 05) doing the work that gradient updates do in classical reinforcement learning.

Two questions remain open. First, how trajectory reliability should be evaluated at scale: multiplicative composition plus correlated errors means that headline per-step metrics systematically overstate multi-step reliability, and no standard methodology yet corrects for this. Second, whether long-horizon coherence is fundamentally limited by a fixed context window combined with strictly left-to-right generation and no mechanism for revising earlier commitments, or whether it is an engineering problem addressable with better memory architectures. Neither question has a settled answer as of this writing.

---

## Implications for agentic-dev

Each mechanism in this curriculum corresponds to a specific agentic-dev practice; this mapping is the point of the capstone.

- **Tokenization and cost (chunk 01):** every loop step re-sends the accumulated context, so token cost compounds per session — the reason model choice and prompt length are managed deliberately rather than ignored.
- **Attention and context limits (chunks 02, 06):** `CLAUDE.md` is re-attended at every step of every loop, which is why it is kept short and high-signal.
- **Alignment-shaped behavior (chunk 04):** an unsupervised agent proceeds confidently rather than pausing to ask, which is why plan mode and human checkpoints insert verification before the loop is trusted to continue.
- **In-context learning (chunk 05):** agent instructions steer behavior through the same conditioning mechanism as any prompt, which is what makes instruction files effective at all.
- **Context window management (chunk 06):** subagents give a bounded sub-task a fresh, narrow context — a direct countermeasure to context rot and drift.
- **Decoding non-determinism (chunk 07):** identical prompts can produce different trajectories run to run, which is why gates and CI carry more evidential weight than a single successful run.
- **Hallucination and calibration (chunk 08):** an agent's stated confidence that a step succeeded is not evidence that it did, which is why verification passes and review steps are structural rather than optional.
- **Tool use (chunk 09):** every action in the loop is a tool call, and hooks are checkpoints wired into the action–observation boundary.

---

## Exercises

1. **(Beginner)** Trace the agent loop by hand for the task "find and fix a broken link on a website": write five action–observation steps. Then mark one step where a wrong conclusion would propagate to later steps, one point where accumulated output could bury the original goal, and one step where the objective could drift without any visible error.
2. **(Practitioner)** Rewrite the vague agent task "clean up the authentication code" as a sequence of four narrow steps, each with an objectively checkable completion criterion (a passing test, a reviewable diff, a lint report). For each rewritten step, state which ambiguity in the original phrasing it eliminates.
3. **(Expert)** Compute the trajectory success probability $$p^n$$ for $$(p, n) \in \{(0.99, 10), (0.95, 10), (0.95, 20), (0.90, 20)\}$$. Then, for the $$p = 0.95$$, $$n = 20$$ case, suppose a single checkpoint placed after step $$k$$ detects any accumulated error and allows retrying only the failed segment: determine which $$k$$ most improves the expected outcome, and explain how positively correlated step failures (rather than independent ones) change the value of the checkpoint.

## Checklist

After this chunk you should be able to:

- [ ] Describe an agent as a frozen LLM policy $$\pi(a_t \mid c_t)$$ in a loop of action, observation, and context concatenation, with all state in the context.
- [ ] Explain why errors compound across an agent trajectory but surface immediately in single-turn chat.
- [ ] Distinguish context rot from the lost-in-the-middle effect and state why agent loops aggravate it.
- [ ] Compute trajectory success probability from per-step reliability under independence, and explain why independence fails in the unfavorable direction.
- [ ] Name the five safeguards — checkpoints, re-grounding, narrow verifiable steps, context pruning, fresh sub-contexts — and match each to the failure mode it addresses.
- [ ] Describe the memory-stream architecture (Park et al.), Reflexion (Shinn et al.), and locate each within the Sumers et al. memory taxonomy.
- [ ] Map each prior chunk's mechanism to a specific agentic-dev practice.

## References

1. Park, J. S., O'Brien, J. C., Cai, C. J., Morris, M. R., Liang, P., & Bernstein, M. S. "Generative Agents: Interactive Simulacra of Human Behavior." *UIST* (2023). arXiv:2304.03442. https://arxiv.org/abs/2304.03442
2. Shinn, N., Cassano, F., Gopinath, A., Narasimhan, K., & Yao, S. "Reflexion: Language Agents with Verbal Reinforcement Learning." *NeurIPS* (2023). arXiv:2303.11366. https://arxiv.org/abs/2303.11366
3. Schluntz, E., & Zhang, B. "Building Effective AI Agents." Anthropic engineering blog post, not peer-reviewed (2024). https://www.anthropic.com/research/building-effective-agents
4. Sumers, T. R., Yao, S., Narasimhan, K., & Griffiths, T. L. "Cognitive Architectures for Language Agents." arXiv preprint (2023). arXiv:2309.02427. https://arxiv.org/abs/2309.02427
5. Xi, Z., et al. "The Rise and Potential of Large Language Model Based Agents: A Survey." *Science China Information Sciences* 68(2) (2025); arXiv:2309.07864 (2023). https://arxiv.org/abs/2309.07864
6. Hong, K., Troynikov, A., & Huber, J. "Context Rot: How Increasing Input Tokens Impacts LLM Performance." Chroma technical research report, not peer-reviewed (2025). https://www.trychroma.com/research/context-rot

## Chunk summary

An agent is a frozen LLM applied as a policy $$\pi(a_t \mid c_t)$$ inside a loop whose only mutable state is the context, updated by concatenation at every step. Because the loop re-applies every mechanism this curriculum has covered — autoregressive generation, finite attention over a growing context, stochastic decoding, nonzero hallucination rates — reliability composes multiplicatively ($$p^n$$ under independence, worse under the correlated errors that real loops produce), and agents fail through accumulation: compounding errors, context rot, and goal drift rather than any single visibly wrong step. This chunk closes the 11-chunk LLM Foundations curriculum, which has run from next-token prediction (chunk 00) through the agentic loop; the intent throughout has been to make the mechanisms visible before introducing tooling built on them. The agentic-dev material that follows — `CLAUDE.md`, prompting structure, plan mode, review passes, hooks, subagents, gates and CI — is the practitioner's toolkit for managing exactly these mechanisms, and is the appropriate next step.
