# Chunk 07: Decoding & Sampling

**Purpose.** Explain how a language model turns a probability distribution over "next token" into an actual chosen token — and why this single step is the root cause of non-deterministic LLM output.
**Previously.** Chunk 06 covered context windows and how models behave as input grows toward and past their context limit.
**Today.** We cover greedy decoding, temperature, top-k, and top-p (nucleus) sampling — the mechanics of picking a token — and why "the same prompt" can produce different answers.

---

## Beginner

Every time a language model generates text, it doesn't produce a word directly. It produces a giant list of every possible next word (technically "token," a word or word-piece), each with a probability attached — like "the" at 40%, "a" at 25%, "an" at 8%, and thousands of other words with tiny slivers of probability. The interesting question is: given that list, how does the model pick just one?

There are two basic strategies, and they feel very different.

The first is **greedy decoding**: always pick the single most likely word. This sounds like the obvious "correct" choice, but in practice it produces oddly flat, repetitive text — imagine a person who, when unsure what to say, always defaults to the blandest possible phrase. Greedy text tends to loop ("I love it. I love it. I love it.") because the safest next word after a safe word is often itself.

The second is **sampling**: treat the probability list like a weighted lottery. A word with 40% probability wins 40% of the time, a word with 5% probability wins 5% of the time, but it's not guaranteed — you might occasionally draw a rare word. This introduces variety and feels more natural, but drawn too loosely, it can also occasionally pick something bizarre from far down the list.

**Temperature** is a dial that controls how "flat" or "peaked" that lottery is. Low temperature makes the model more conservative, favoring the already-likely words even more strongly (closer to greedy). High temperature flattens the odds, giving unlikely words a better shot, which makes output more surprising and sometimes less coherent.

**Top-k** and **top-p** are ways of trimming the lottery before you draw from it, so you're never drawing from the truly absurd, near-zero-probability tail. Top-k keeps only the k most likely words on the ticket. Top-p (also called nucleus sampling) keeps however many words are needed to cover a target percentage of total probability — say, the fewest words that together add up to 90%.

The practical upshot: because word selection involves a randomized draw, asking a chatbot the exact same question twice can yield different wording, different structure, or even a different answer — not because the model is broken, but because that's how the lottery works.

## Practitioner

When you call an LLM API, you're almost always given at least two levers over this process: `temperature` and `top_p` (some APIs also expose `top_k`). Understanding what they actually do — mathematically, not just vibes — matters for building anything that depends on consistent output.

**Temperature** rescales the logits (the raw, pre-probability scores) before the softmax turns them into probabilities. Concretely, each logit is divided by the temperature value before the softmax exponential is applied. Temperature near 0 sharpens the distribution until it collapses onto the single highest-scoring token (equivalent to greedy decoding). Temperature of 1 leaves the distribution as the model originally computed it. Temperature above 1 flattens the distribution, pulling probability mass toward the tail and increasing the odds of picking a less likely token.

**Top-p / top-k** are separate, complementary controls that operate on the shape of the candidate pool, not the sharpness of the weighting. Top-k restricts sampling to a fixed-size shortlist (e.g., the 40 most likely tokens) regardless of how the model's confidence is distributed. Top-p restricts the pool dynamically — take the smallest set of top tokens whose cumulative probability crosses the threshold p, which adapts pool size to how confident the model is at each step (a very confident step might have a nucleus of 2 tokens; an uncertain one might need 50).

A useful mental model for combining these:

| Setting | Behavior | When to use |
|---|---|---|
| temperature=0 (or near 0), no top-p change | Deterministic-ish, always picks highest-probability token | Extracting structured data, classification, code generation, tool-call arguments — anything where you want the "best" answer, not variety |
| temperature=0.7, top_p=0.9 | Balanced — some variety, still coherent | General chat, drafting, summarization |
| temperature=1.0+, top_p=0.95+ | High variety, more surprising, occasional incoherence | Brainstorming, creative writing, generating diverse samples for later filtering/voting |

The counterintuitive part practitioners hit early: **`temperature=0` does not guarantee you'll get the same output every time you call the API**, even for the identical prompt. This is not a sampling issue at all — it's a side effect of how inference servers batch requests. Modern serving stacks (like vLLM) group multiple simultaneous requests together for efficiency, and floating-point arithmetic on GPUs is not associative — summing the same numbers in a different order (because the batch composition changed) produces microscopically different results. Those tiny differences can occasionally flip which token has the (technically) highest logit, cascading into a different generated sequence. This has been demonstrated concretely by Thinking Machines Lab's engineering work on "batch invariance" in inference kernels (see References).

Practical implication: if you need output stability, set temperature to 0 (or very low) to remove sampling-driven randomness, but don't build systems that assume byte-for-byte identical output across calls — build in tolerance, retries, or validation instead.

## Expert

Formally, at each generation step the model outputs a vector of logits `z` over the vocabulary. These are converted to a probability distribution via softmax: `p_i = exp(z_i / T) / Σ_j exp(z_j / T)`, where `T` is the temperature. As `T → 0`, the distribution converges to a one-hot vector on `argmax(z)` — greedy decoding. As `T → ∞`, the distribution approaches uniform over the vocabulary.

**Decoding strategies** fall into two families:

- *Maximization-based* — greedy search (take argmax at each step) and beam search (maintain the top-b partial sequences by cumulative log-probability, expanding and pruning at each step). Beam search was developed for speech recognition and machine translation, where it reliably outperforms greedy decoding on tasks with a roughly correct, low-entropy target output.
- *Stochastic* — sample from the (possibly reshaped) distribution: temperature sampling, top-k sampling, and top-p (nucleus) sampling.

Holtzman et al. (2020), "The Curious Case of Neural Text Degeneration," is the key diagnostic and prescriptive paper here. They show that maximization-based decoding (greedy and beam search) on open-ended generation tasks produces text that is measurably less diverse and more repetitive than human text along multiple distributional statistics, and that this repetition is often self-reinforcing — once a model repeats a phrase, the probability of repeating it again increases. Their proposed fix, **nucleus (top-p) sampling**, truncates the distribution's unreliable low-probability tail dynamically: at each step, sample only from the smallest set of tokens whose cumulative probability mass exceeds threshold p, then renormalize and sample within that set. This adapts the candidate pool size to the model's local confidence, unlike fixed-k truncation.

**Top-k sampling** predates and is complementary to nucleus sampling; it was introduced for open-ended generation by Fan, Lewis, and Dauphin (2018), "Hierarchical Neural Story Generation," who used it to improve story generation quality over pure sampling or beam search by restricting each draw to the k highest-probability tokens before renormalizing. Its limitation, which nucleus sampling addresses, is that a fixed k is too restrictive when the distribution is flat (many plausible continuations) and too permissive when the distribution is peaked (few plausible continuations).

**Why greedy/beam degenerate**: language models are trained to maximize likelihood of human text under teacher forcing, but human text is not itself the argmax path through the model's own distribution at each step — human writing contains locally "suboptimal" (lower-probability) word choices that maximization-based decoding will never select, and forcing pure maximization pushes the output off the manifold of typical human text, into repetitive high-confidence loops.

Two open questions worth flagging: (1) Beyond sampling randomness, what other sources of run-to-run non-determinism exist in production inference (mixed-precision arithmetic, non-associative floating-point summation order under dynamic batching, MoE routing decisions), and how much of "non-determinism" attributed to temperature is actually infrastructure-level? (2) How should the exploration/quality tradeoff in decoding be tuned per-task — is there a principled way to set temperature/top-p per use case rather than by convention/trial-and-error?

---

## Implications for agentic-dev

This chunk explains a failure mode every agentic-dev user eventually hits: **running the identical Claude Code prompt twice and getting different results.** The proximate cause is sampling — at temperature > 0, each generation step draws from a probability distribution rather than always taking the top token, so the exact sequence of tokens (and therefore tool calls, file edits, or reasoning steps) can legitimately differ between runs.

Setting `temperature=0` (or as close to it as an API allows) collapses the sampling step to argmax and removes this source of variance — but, as covered above, it does **not** fully guarantee determinism. Production inference servers batch requests dynamically for throughput, and floating-point summation is not associative, so the exact same prompt can be numerically routed through a slightly different computation depending on what else is running on the GPU at that moment. That can flip a near-tied logit and change the selected token, even at temperature 0. This is a solvable but not universally solved problem — see Thinking Machines Lab's "batch invariance" engineering work below for how far current mitigation goes.

Concretely, this should shape how you build and debug agent loops:

- **Don't build tool-call parsing that assumes byte-identical output.** If your agent loop parses a model's output to extract a function call, JSON blob, or file diff, treat minor phrasing/formatting variance as expected, not a bug — use structured output modes (JSON mode, function-calling schemas) rather than regexing free text, since those reduce (but don't eliminate) the surface area sampling can perturb.
- **Lower temperature for steps where you need reliability, not creativity** — tool-call argument generation, code edits, structured extraction — and reserve higher temperature for steps like brainstorming or drafting where variety is a feature.
- **Don't treat a single failed run as proof of a broken prompt or broken tool.** Because output is sampled, one bad run can be noise; before concluding your agent's instructions are wrong, check whether the behavior reproduces across several runs.
- **Set reproducibility expectations correctly in tests/evals for agents.** If you're writing automated tests against LLM output, "exact string match" tests are inherently flaky by construction — even at temperature 0 — so prefer semantic/structural assertions (does the tool call have the right shape? does the file compile?) over exact text comparison.

---

## Checklist

- [ ] I can explain the difference between greedy decoding and sampling in plain language.
- [ ] I know what the temperature parameter mathematically does to the softmax distribution.
- [ ] I can distinguish top-k (fixed-size candidate pool) from top-p/nucleus (dynamic, probability-mass-based candidate pool).
- [ ] I can explain why greedy/beam search tend to produce repetitive text (Holtzman et al., 2020).
- [ ] I understand why temperature=0 reduces but does not fully guarantee deterministic output.
- [ ] I know to avoid exact-string-match assumptions when parsing LLM tool calls in an agent loop.

## References

1. "The Curious Case of Neural Text Degeneration" (Ari Holtzman, Jan Buys, Li Du, Maxwell Forbes, Yejin Choi; ICLR 2020) — https://arxiv.org/abs/1904.09751
2. "Hierarchical Neural Story Generation" (Angela Fan, Mike Lewis, Yann Dauphin; ACL 2018) — https://arxiv.org/abs/1805.04833
3. "Defeating Nondeterminism in LLM Inference" (Thinking Machines Lab, 2025) — https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/

## Chunk summary

Decoding is the last-mile step where a probability distribution becomes an actual token: greedy decoding always takes the top choice and tends toward bland, repetitive output, while temperature, top-k, and top-p reshape or truncate the distribution to control how much randomness enters that choice. This randomness is the direct reason identical prompts can yield different completions, and it's why agentic-dev workflows should treat temperature=0 as "much more consistent," not "guaranteed identical" — a distinction rooted in how GPU batching and floating-point arithmetic interact with inference, not just in sampling theory.
