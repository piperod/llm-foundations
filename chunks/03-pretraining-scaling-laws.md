---
title: "03 · Pretraining & Scaling Laws"
parent: Chunks
nav_order: 4
---

# Chunk 03: Pretraining & Scaling Laws

**Purpose.** Describe pretraining — fitting the next-token objective from chunk 00 to a web-scale corpus — and the scaling laws that make the outcome of a training run predictable as a function of model size, data, and compute.
**Previously.** Chunk 02 covered the transformer and self-attention — the architecture that processes a sequence of tokens.
**Today.** How that architecture is trained at internet scale; the Kaplan and Chinchilla scaling results; the compute-optimal allocation of parameters versus tokens; and what these results imply for model tiers and knowledge cutoffs.

> **Through-line.** Movement I. Pretraining is the first and largest force that carves the distribution, writing it into the weights from raw text; scaling laws say how sharply.

![Three log-log panels showing test loss decreasing smoothly as a power law with compute, dataset size, and number of parameters, from Kaplan et al. 2020.](https://ar5iv.labs.arxiv.org/html/2001.08361/assets/x1.png)

*Figure 1: Test loss falls as a smooth power law with each of compute, dataset size, and model parameters, holding across seven orders of magnitude. Source: "Scaling Laws for Neural Language Models," Kaplan, McCandlish, Henighan, et al. (OpenAI, 2020) — https://ar5iv.labs.arxiv.org/html/2001.08361.*

---

## Curiosity

Pretraining is the first and largest stage of building a language model. The model defined in chunk 00 — a learned distribution over the next token — is fit to a corpus assembled from a substantial fraction of publicly available text: web pages, books, articles, and source code. Training proceeds by prediction and correction. At every position in every document, the model predicts the next token; the prediction is compared with the actual token; and the model's parameters are adjusted slightly to make that error smaller. Repeated across trillions of tokens, this single procedure is the origin of essentially all of a base model's capability. There is no separate stage in which the model is taught grammar, facts, or programming; whatever it must represent to predict well, it acquires as a side effect of the objective.

The central empirical finding about this process is its predictability. Researchers who trained many models of varying sizes found that performance on the prediction task improves smoothly as three quantities grow: the number of model parameters, the number of training tokens, and the total computation spent. The improvement follows curves regular enough to extrapolate, which lets a laboratory estimate in advance roughly how capable a model a given budget will produce. In this one respect a large training run resembles a civil-engineering project more than a scientific experiment: cost and outcome are estimated before construction begins.

Two familiar product facts follow directly. First, model families ship in tiers — small, fast, and inexpensive through large, slow, and capable — because size is a controllable input whose effects on capability and cost are quantified rather than guessed. Second, every model has a knowledge cutoff, because the training corpus is collected at a specific point in time; text that did not exist at collection contributes nothing to the model's parameters, so events, libraries, and people that appeared afterward are absent from it entirely.

## Builder

Two studies define the operational picture of how labs allocate a training budget.

Kaplan et al. (2020) trained over 200 models and established that test loss falls as a smooth power law in each of model size, dataset size, and compute, holding across roughly seven orders of magnitude. Their compute-optimal prescription favored spending additional compute predominantly on model size, and it shaped GPT-3: 175 billion parameters trained on approximately 300 billion tokens. Hoffmann et al. (2022) revisited the question with a cleaner experimental design and reached a different allocation: for a fixed compute budget, a better model results from training a smaller network on substantially more data. Their demonstration, Chinchilla — 70 billion parameters trained on 1.4 trillion tokens — outperformed Gopher, a 280-billion-parameter model trained on roughly 300 billion tokens, at matched compute. The disagreement is not about whether scaling laws hold; it is about how a fixed budget should be split between parameters and tokens, and the Chinchilla result showed the earlier recipe had systematically under-used data.

The operational consequence is that parameter count is not a capability ranking. A model's quality depends jointly on its size, its training-token budget, its data quality, and its post-training — so a newer, smaller model frequently outperforms an older, larger one that was trained under a less efficient allocation. When comparing models across vendors or generations, headline parameter counts (where they are even disclosed) are among the least informative numbers available; benchmark performance at a given cost and latency is the relevant comparison.

This is also why model families ship in tiers rather than one size. Each tier occupies a deliberate point on the capability/cost/latency frontier, chosen at training time as a compute-allocation decision of exactly the kind Kaplan and Hoffmann quantified. Selecting a tier for a task is selecting a point on that frontier.

A knowledge cutoff, operationally, means the model received zero gradient signal from anything that became text after the corpus was assembled. The model is not vaguely uncertain about recent events; it is structurally blind to them, in the same way it is blind to a private codebase it was never shown. Retrieval and tool use (chunk 09) exist to patch this gap at inference time without retraining.

> **Try it.** The [FineWeb report](https://huggingface.co/spaces/HuggingFaceFW/blogpost-fineweb-v1) documents a real open pretraining corpus — roughly 15 trillion tokens extracted from web crawls — including the deduplication and quality-filtering pipeline that produced it. Skim the filtering sections and note how much of the raw crawl is discarded, and on what criteria. The corpus a model is trained on is a sequence of engineering decisions, and the knowledge cutoff discussed above is simply the date this pipeline last ran.

## Expert

Pretraining minimizes the autoregressive cross-entropy objective defined in chunk 00, averaged over a corpus on the order of $$10^{11}$$–$$10^{13}$$ tokens. Kaplan et al. (2020) measured how the resulting loss $$L$$ depends on non-embedding parameter count $$N$$ and dataset size $$D$$, finding approximate power laws — $$L(N) \propto N^{-0.076}$$ when data is not the bottleneck, and $$L(D) \propto D^{-0.095}$$ when model size is not — together with a joint form combining the two. Extrapolating along their fitted compute frontier assigned most of each increment of compute to $$N$$ rather than $$D$$, a prescription reflected in GPT-3's configuration (Brown et al., 2020): 175B parameters, ~300B tokens.

Hoffmann et al. (2022) re-derived the compute-optimal frontier using IsoFLOP curves over approximately 400 models and proposed the parametric loss

$$L(N, D) = E + \frac{A}{N^{\alpha}} + \frac{B}{D^{\beta}}$$

with fitted values $$E \approx 1.69$$, $$A \approx 406.4$$, $$B \approx 410.7$$, $$\alpha \approx 0.34$$, $$\beta \approx 0.28$$. Here $$E$$ is the irreducible loss of the data distribution, and the two remaining terms are the penalties for finite model size and finite data. Because $$\alpha \approx \beta$$, minimizing $$L(N, D)$$ subject to a compute constraint $$C \approx 6ND$$ yields $$N_{\mathrm{opt}} \propto C^{a}$$ and $$D_{\mathrm{opt}} \propto C^{b}$$ with $$a \approx b \approx 0.5$$: parameters and tokens should scale roughly equally with compute. The resulting rule of thumb — approximately 20 training tokens per parameter at the compute-optimal point — became the standard reference. By this account GPT-3 was substantially over-parameterized (175B parameters would call for trillions of tokens, not ~300B), and Chinchilla's 70B/1.4T configuration beating Gopher's 280B/~300B at matched compute is the empirical demonstration.

Compute-optimality in the Chinchilla sense minimizes loss per unit of *training* compute, which is not the deployment-relevant quantity. Training cost is paid once; inference cost scales with $$N$$ and is paid on every query. For a model that will serve high volumes, total cost is dominated by inference, and the optimum shifts toward smaller models trained well past the Chinchilla-optimal token count. This inference-cost argument is one mechanism behind the current practice of deliberately over-training compact models far beyond 20 tokens per parameter.

Two questions remain genuinely open. First, the data constraint: if high-quality public text grows more slowly than compute budgets, the $$D$$ axis eventually binds. Muennighoff et al. (2023) studied this regime directly and found that repeating the corpus for up to roughly four epochs yields losses close to those from an equal quantity of unique data, with returns diminishing rapidly toward zero on further repetition; whether multi-epoch reuse, synthetic data, or new modalities preserve the fitted scaling exponents beyond that regime is unsettled. Second, emergence: Wei et al. (2022) catalogued capabilities that appear sharply above a scale threshold rather than improving smoothly, in apparent tension with the smooth loss curves the scaling laws describe. The counter-position — that most reported discontinuities are artifacts of discontinuous evaluation metrics — is summarized in chunk 00, and the dispute is unresolved. Scaling laws predict *loss* with considerable reliability; the mapping from loss to specific downstream capabilities is the weaker link in the extrapolation.

---

## Implications for agentic-dev

1. **Selecting a tier via Claude Code's `/model` command is selecting a point on the scaling-law cost/capability frontier, not a cosmetic setting.** A smaller, faster tier for mechanical high-volume work and a larger tier for architectural reasoning reproduces, at the user level, the compute-allocation tradeoff Kaplan and Hoffmann quantified at the lab level. The practice of starting with the default tier and escalating only when it struggles is grounded in that frontier: additional capability has a measured, nonzero price in cost and latency.

2. **Knowledge cutoffs are a structural property of corpus collection, not an omission.** A model has gradient signal only from tokens that existed when its corpus was assembled; a new library API, a recent CVE, or this week's release notes never appeared in any training example. Agentic workflows therefore route anything time-sensitive through tools, web search, and current documentation (chunk 09) rather than the model's unaided memory. The conditioning mechanism by which retrieved text influences output is the one established in chunk 00.

3. **Scaling has been a reliable lever historically, and the lever has structure.** Smooth power laws over many orders of magnitude are why labs can commit large budgets in advance; the Chinchilla correction shows the returns depend on the parameter/token split, and the data-constraint and emergence questions above are the standing reasons not to assume gains arrive indefinitely on the same schedule.

---

## Exercises

1. **(Curiosity)** A colleague asks a model about a Python library released three months after the model's knowledge cutoff, receives a confident but wrong answer, and concludes the model is "lying." Explain, in terms of how the pretraining corpus is collected, why the model cannot know about the library, and why its confident tone is unrelated to whether it was trained on the topic.
2. **(Builder)** A teammate insists on using an older 175B-parameter model over a newer 70B-parameter one because "more parameters means smarter." Write a three-to-four-sentence reply using the Chinchilla-versus-Gopher result: state why parameter count alone does not rank capability, and name at least two other training-time factors that determine model quality.
3. **(Expert)** Assume a training budget of $$C = 10^{23}$$ FLOPs, the approximation $$C \approx 6ND$$, and the Chinchilla-optimal ratio $$D \approx 20N$$. Solve for the compute-optimal $$N$$ and $$D$$. Then suppose the model will serve billions of queries: explain why a deployment-aware lab might instead train a model half that size on correspondingly more tokens, and identify which cost the Chinchilla analysis does not account for.

## Checklist

After this chunk you should be able to:

- [ ] Describe pretraining as fitting the next-token objective to a web-scale corpus, with no separate stage for facts or skills.
- [ ] State why a knowledge cutoff exists and why the model is structurally blind, rather than vaguely uncertain, past it.
- [ ] Name the three quantities scaling laws relate — parameters, tokens, compute — and state that loss falls as a power law in each.
- [ ] Contrast the Kaplan (2020) and Hoffmann (2022) compute-optimal allocations, and state the ~20 tokens-per-parameter heuristic with the Chinchilla-versus-Gopher evidence.
- [ ] Write the Chinchilla parametric loss $$L(N,D) = E + A/N^{\alpha} + B/D^{\beta}$$ and derive why $$\alpha \approx \beta$$ implies equal scaling of $$N$$ and $$D$$.
- [ ] Explain the inference-cost argument for over-training smaller models past the training-compute-optimal point.
- [ ] Summarize the data-constraint (Muennighoff et al., 2023) and emergence (Wei et al., 2022) questions without overstating either as settled.

## References

1. Kaplan, J., McCandlish, S., Henighan, T., et al. "Scaling Laws for Neural Language Models." arXiv preprint (2020). arXiv:2001.08361. https://arxiv.org/abs/2001.08361
2. Hoffmann, J., Borgeaud, S., Mensch, A., et al. "Training Compute-Optimal Large Language Models." *NeurIPS* (2022). arXiv:2203.15556. https://arxiv.org/abs/2203.15556
3. Brown, T. B., et al. "Language Models are Few-Shot Learners." *NeurIPS* (2020). arXiv:2005.14165. https://arxiv.org/abs/2005.14165
4. Wei, J., et al. "Emergent Abilities of Large Language Models." *Transactions on Machine Learning Research* (2022). arXiv:2206.07682. https://arxiv.org/abs/2206.07682
5. Muennighoff, N., et al. "Scaling Data-Constrained Language Models." *NeurIPS* (2023). arXiv:2305.16264. https://arxiv.org/abs/2305.16264

## Chunk summary

Pretraining fits the next-token objective of chunk 00 to a corpus of trillions of tokens, and scaling laws make the result predictable: loss falls as a power law in parameters, data, and compute (Kaplan et al., 2020), with the compute-optimal split between parameters and tokens corrected by Hoffmann et al. (2022) — roughly equal scaling, about 20 tokens per parameter, demonstrated by Chinchilla (70B/1.4T) outperforming Gopher (280B/~300B) at matched compute. Deployment economics push past that optimum, since inference cost scales with model size and is paid per query. For agentic work, the load-bearing consequences are that model tiers are quantified points on a cost/capability frontier, selectable via `/model`, and that knowledge cutoffs are a structural property of corpus collection, which retrieval and tools exist to patch. Data constraints (Muennighoff et al., 2023) and the emergence debate (Wei et al., 2022) remain open questions about how far the extrapolation extends.
