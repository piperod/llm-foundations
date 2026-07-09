# Chunk 03: Pretraining & Scaling Laws

**Purpose.** Explain what "pretraining" actually is (predicting the next token over enormous text corpora) and why bigger models trained on more data reliably get better, in a way that's been measured and can be predicted in advance.
**Previously.** Chunk 02 covered the transformer and self-attention — the architecture that processes a sequence of tokens.
**Today.** How that architecture gets trained at internet scale, and what "scaling laws" say about the relationship between model size, data size, compute, and resulting capability.

![Three log-log panels showing test loss decreasing smoothly as a power law with compute, dataset size, and number of parameters, from Kaplan et al. 2020.](https://ar5iv.labs.arxiv.org/html/2001.08361/assets/x1.png)

*Figure 1: Test loss falls as a smooth power law with each of compute, dataset size, and model parameters, holding across seven orders of magnitude. Source: "Scaling Laws for Neural Language Models," Kaplan, McCandlish, Henighan, et al. (OpenAI, 2020) — https://ar5iv.labs.arxiv.org/html/2001.08361.*

---

## Beginner

Imagine the world's most extreme fill-in-the-blank game. You take a huge pile of text — a big chunk of the public internet, books, code, articles — and you hide the last word of every sentence, over and over, billions of times. The model's only job during training is: guess the next word. Then you show it the real word, it adjusts itself slightly to be less wrong next time, and you repeat this an almost unimaginable number of times. That's it. That's pretraining. There's no separate "understanding module" — the ability to write code, hold a conversation, or explain a concept all emerges from getting extremely good at this one game, played at enormous scale.

Here's the surprising part, discovered by researchers who trained hundreds of models of different sizes: as you make the model bigger, feed it more text, and give it more computing power, its skill at this guessing game improves in a smooth, predictable way — not randomly, not with diminishing returns that appear out of nowhere, but following a curve you can plot on a graph and extrapolate. This predictability is why AI labs can decide "we're going to spend $X and get roughly this much improvement" before they even start training. It's less like a gamble and more like an engineering budget.

This also explains two things you've probably noticed as a chatbot user. First, why there are different "sizes" of the same model family (a small fast one, a big powerful one) — size is a dial, and moving it changes cost and capability in predictable ways. Second, why every model has a "knowledge cutoff" date — it only knows about the world up to whenever its training text was collected. Ask it about something that happened after that date, and it genuinely cannot know, the same way you wouldn't know about a news story from after you stopped reading the news.

## Practitioner

The mental model to hold: a pretrained model's raw capability is mostly a function of three dials — parameters (size), data (tokens seen), and compute (size × data × a constant, roughly). Two landmark studies shaped how labs turn those dials.

Kaplan et al.'s "Scaling Laws for Neural Language Models" (2020) trained over 200 models and found that test loss falls as a smooth power law in each of model size, dataset size, and compute, holding across seven orders of magnitude. Their prescription, which shaped GPT-3's design, favored making models very large relative to the data they saw. Hoffmann et al.'s 2022 "Chinchilla" paper (Training Compute-Optimal Large Language Models) revisited this with more careful experiments and found the earlier recipe under-used data: for a fixed compute budget, you get a better model by training a smaller model on much more data. Their headline result: Chinchilla, a 70B-parameter model trained on 1.4 trillion tokens, outperformed Gopher, a 280B model trained on only ~300B tokens, using the same compute budget. The practical upshot for anyone choosing between model tiers today: "bigger" and "better-trained" are not the same axis, and a well-trained smaller model can beat a poorly-trained larger one.

This is why model families ship in tiers (small/fast, medium, large/frontier) rather than one size. Each tier sits at a different point on the capability-vs-cost-vs-latency curve, and — per the scaling laws above — that placement is not arbitrary; it reflects a deliberate compute-allocation decision made at training time.

A concrete worked comparison, using patterns typical of how providers describe their own tiers (illustrative, not tied to one vendor's exact numbers):

| Tier | Relative size | Typical use | Cost/latency | Knowledge cutoff behavior |
|---|---|---|---|---|
| Small/fast (e.g. Haiku-class) | Smallest | Simple, high-volume, mechanical tasks | Cheapest, fastest | Same cutoff family as siblings, but may have been trained/released earlier or with a lighter refresh |
| Mid (e.g. Sonnet-class) | Medium | Default daily driver, most coding/reasoning tasks | Balanced | Refreshed periodically; cutoff moves forward with each retrain |
| Large/frontier (e.g. Opus-class) | Largest | Hard reasoning, architecture, ambiguous problems | Most expensive, slowest | Same mechanism — cutoff is whenever its pretraining corpus was collected |

A "knowledge cutoff" in practice means: the model has zero training signal about events, libraries, APIs, or people that only became text after the corpus was assembled. It isn't "fuzzy" about recent events — it's structurally blind to them, the same way it's blind to a private codebase it was never shown. This is also why retrieval/tools (chunk 09) exist: they patch this structural gap without retraining.

## Expert

Formally, pretraining minimizes the autoregressive cross-entropy loss: for a sequence of tokens x₁...x_T, the model minimizes −Σ log P(x_t | x_<t; θ) over θ, averaged over an enormous corpus. What Kaplan et al. (2020) showed is that this loss L, as a function of non-embedding parameter count N, dataset size D, and compute C, follows approximate power laws — L(N) ∝ N^−α_N, L(D) ∝ D^−α_D, and combined a joint form — fit across models spanning roughly seven orders of magnitude of compute. Their compute-optimal prescription allocated a large share of additional compute to model size over data, which informed GPT-3's design (Brown et al., 2020, "Language Models are Few-Shot Learners," NeurIPS 2020) — a 175B-parameter model whose few-shot behavior itself became evidence that scaling produces qualitatively new capabilities, not just lower loss.

Hoffmann et al. (2022) re-derived the compute-optimal frontier with a cleaner experimental design (IsoFLOP curves across ~400 models) and found N and D should scale roughly equally with compute — contradicting Kaplan's original allocation, which had systematically under-weighted data. This is the "Kaplan vs. Chinchilla" debate: it's not a disagreement about whether scaling laws hold, but about the optimal split of a fixed compute budget between parameters and tokens. Chinchilla (70B params, 1.4T tokens) outperforming Gopher (280B params, ~300B tokens) at matched compute was the empirical case closer. Most frontier labs have since trained past "Chinchilla-optimal" on the token axis deliberately — because inference cost (which scales with parameters, paid every query) matters as much as training cost (paid once), so it's often worth over-training a smaller model on more tokens than the training-compute-optimal point would suggest.

Two open questions worth flagging honestly rather than asserting resolved. First, the data wall: multiple analyses (see e.g. discussions of data-constrained pretraining, arXiv:2606.16246, and broader commentary on public text exhaustion) raise the concern that high-quality public text may not scale indefinitely with compute, pushing labs toward synthetic data, multi-epoch reuse, or new modalities — whether these substitutes preserve the same scaling exponents is unsettled. Second, emergence: Wei et al. (2022), "Emergent Abilities of Large Language Models" (TMLR), catalogued capabilities (e.g. multi-step arithmetic, certain few-shot tasks) that appear sharply above a scale threshold rather than improving smoothly, in apparent tension with the smooth loss curves scaling laws predict. Whether this reflects genuine discontinuity in capability or an artifact of discontinuous evaluation metrics (a critique raised by later work) remains actively debated, and this note does not take a side.

---

## Implications for agentic-dev

Three concrete, causal links from this chunk to agentic-dev practice:

1. **Knowledge cutoffs are a direct consequence of pretraining mechanics, not a limitation someone forgot to fix.** A model only has gradient signal from tokens that existed in its training corpus at collection time. Anything after that — a new library API, a recent CVE, this week's news — literally never appeared in any training example. This is precisely why agentic-dev workflows lean on tools, web search, and up-to-date docs (chunk 09) instead of trusting the model's unaided memory for anything time-sensitive.

2. **Choosing a model tier via Claude Code's `/model` command is choosing a point on the scaling-law cost/capability curve, not a cosmetic setting.** Switching to a smaller/faster tier (e.g. a Haiku-class model) for mechanical, high-volume work and reserving a larger tier (e.g. an Opus-class model) for hard architectural reasoning mirrors exactly the compute-allocation tradeoff Kaplan and Hoffmann quantified: more parameters and more training compute buy predictable capability gains, at a predictable cost and latency price. That tradeoff is why "start with the default, escalate to the bigger model only when it struggles" is sound practice, not caution for its own sake.

3. **"Just make it bigger" has been a reliable lever historically because the scaling laws are empirically smooth over many orders of magnitude** — which is exactly why labs can commit huge budgets to a training run in advance and expect a specific loss reduction. But the Chinchilla correction shows the lever has structure (size and data must scale together) and the data-wall/emergence open questions in the Expert section above are the reasons nobody should assume the lever is infinite or that gains will keep arriving on the same schedule.

---

## Checklist

- [ ] I can explain pretraining as next-token prediction over a large corpus, with no separate "understanding" step.
- [ ] I can state, in plain terms, why a knowledge cutoff exists and why it's not a bug.
- [ ] I can name the three "dials" scaling laws relate: model size, data size, compute.
- [ ] I can explain the difference between the Kaplan (2020) and Chinchilla (2022) compute-optimal recipes in one sentence.
- [ ] I can justify why switching model tiers via `/model` is a capability/cost tradeoff grounded in scaling laws, not just a speed setting.
- [ ] I can name at least one open question (data wall or emergence) without overstating it as settled.

## References

1. Kaplan, J., McCandlish, S., Henighan, T., Brown, T. B., Chess, B., Child, R., Gray, S., Radford, A., Wu, J., & Amodei, D. "Scaling Laws for Neural Language Models" (2020, OpenAI). https://arxiv.org/abs/2001.08361
2. Hoffmann, J., Borgeaud, S., Mensch, A., et al. "Training Compute-Optimal Large Language Models" (2022, DeepMind — "Chinchilla"). https://arxiv.org/abs/2203.15556
3. Brown, T. B., Mann, B., Ryder, N., et al. "Language Models are Few-Shot Learners" (NeurIPS 2020, OpenAI). https://arxiv.org/abs/2005.14165
4. Wei, J., Tay, Y., Bommasani, R., et al. "Emergent Abilities of Large Language Models" (TMLR, 2022). https://arxiv.org/abs/2206.07682

## Chunk summary

Pretraining is next-token prediction at massive scale, and scaling laws (Kaplan et al., 2020; refined by Hoffmann et al.'s Chinchilla work, 2022) show that loss falls predictably as model size, data, and compute grow — which is why labs can budget training runs in advance and why GPT-3-scale results (Brown et al., 2020) were, in a real sense, anticipated rather than accidental. For agentic-dev, this predictability is exactly why model tiers exist and why picking one via `/model` is a real, quantifiable capability/cost tradeoff, and why every model's knowledge cutoff is a hard structural boundary rather than a fuzzy limitation — with open questions (data exhaustion, emergence per Wei et al., 2022) meaning the "just scale it up" lever, while historically reliable, isn't guaranteed to keep behaving the same way forever.
