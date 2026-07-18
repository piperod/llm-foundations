---
title: "04 · From Base Model to Assistant"
parent: Chunks
nav_order: 5
---

# Chunk 04: From Base Model to Assistant

**Purpose.** Explain the post-training pipeline — supervised fine-tuning, RLHF, Constitutional AI, and DPO — that converts a raw next-token predictor into an assistant with instruction-following behavior, refusal behavior, and a consistent voice.
**Previously.** Chunk 03 covered pretraining and scaling laws: how a base model learns to predict text from a large raw corpus.
**Today.** How demonstration data, preference feedback, and preference-optimization objectives reshape the base model's output distribution into assistant behavior, and what that reshaping does and does not change.

![Diagram of the three-step InstructGPT pipeline: (1) collect demonstration data and train a supervised policy via SFT, (2) collect comparison data and train a reward model, (3) optimize the policy against the reward model using PPO.](https://ar5iv.labs.arxiv.org/html/2203.02155/assets/x2.png)

*Figure 1: The three-step RLHF pipeline used to turn GPT-3 into InstructGPT — SFT on demonstrations, reward model training on ranked comparisons, then PPO to optimize the policy against the reward model. Source: Ouyang et al., "Training Language Models to Follow Instructions with Human Feedback," OpenAI, 2022 — https://ar5iv.labs.arxiv.org/html/2203.02155 (Figure 2).*

---

## Beginner

A base model is the direct product of pretraining (chunk 03): a next-token predictor with no notion of a user, a request, or a role. Given the prompt "What is the capital of France?", a base model does not necessarily answer. It may continue with "What is the capital of Germany?" — a plausible continuation if the prompt resembles a quiz worksheet — because its only learned behavior is to extend text the way its training corpus would. Answering is one continuation among many, and nothing in the pretraining objective privileges it. A base model is comparable to an actor who has memorized every script ever written but has been assigned no character; post-training assigns the role.

**Post-training** (also called **alignment training**) is the set of additional stages that convert this predictor into an assistant. Two stages carry most of the effect.

The first is **supervised fine-tuning (SFT)**. Human labelers write example exchanges — a prompt paired with a high-quality response — and the model is trained further, with the same next-token objective from chunk 00, on this curated data. SFT establishes the format: prompts are requests, and the model's part in the exchange is to respond to them directly.

The second is **preference learning**. Humans, or in some pipelines AI systems following written principles, compare pairs of candidate responses and record which is better. The model is then optimized to produce responses of the preferred kind — clearer, safer, better calibrated. **RLHF** (reinforcement learning from human feedback) is the original technique for LLMs; **Constitutional AI** substitutes principle-guided AI feedback for much of the human labeling; **DPO** reaches a similar endpoint with substantially less machinery. All three receive formal treatment in the Expert tier.

The combined result is an assistant model: a system that answers rather than continues, declines certain requests, expresses uncertainty, and maintains a recognizable voice. These behaviors originate in post-training. Pretraining supplies the knowledge; post-training determines how that knowledge is deployed in conversation.

## Practitioner

Interacting with a base-model endpoint makes the effect of post-training concrete. Given an instruction, a base model may complete the instruction as though it were part of a longer document, generate both sides of a dialogue, or drift into adjacent text. The model has capability without conversational judgment: the knowledge required to answer is present, but nothing in its training selects "respond to the request" over other statistically plausible continuations.

Post-training, for the most part, does not add factual knowledge. It reshapes the *distribution over behaviors* the model expresses: which persona to adopt (helpful assistant rather than arbitrary document author), which response shape to default to (address the request directly, then elaborate), and where the boundaries lie (decline, comply, hedge, or ask a clarifying question, depending on stakes and ambiguity).

**Worked example.** Prompt: "How do I pick a lock?"

- A **base model**, having encountered this string in forums, fiction, and locksmith manuals, may continue with any of them: a story fragment, a technical breakdown, a forum reply, or a further question — whichever continuation is statistically likely, with no representation of who is asking or why.
- An **assistant model** has been shaped, through SFT examples and preference comparisons, to treat this as a request with dual-use potential. It typically produces a measured, legitimate answer — explaining lock mechanisms in a locksmithing or lockout context — rather than either stonewalling or supplying a context-free procedure, because raters and feedback rules rewarded exactly that calibrated middle course over both extremes.

The same process explains why an assistant has a consistent style at all. Demonstration data and preference comparisons are not random with respect to tone: they consistently reward a particular register — direct, hedging under uncertainty, non-sycophantic. Consistency in the training signal produces consistency in output style. It also explains why assistants from different labs differ in character despite comparable base-model capability: the preferences encoded in their SFT data and reward signals differ.

Two operational consequences follow. First, when a model declines a task it is demonstrably capable of, the refusal is a post-training decision rather than a capability gap; the underlying knowledge is intact. Second, many jailbreaks operate by constructing prompts that shift the model back toward base-model-like completion behavior, working around the assistant persona rather than through it. Both observations follow from the same fact: assistant behavior is a trained disposition layered over the base distribution, not a filter applied after generation.

> **Try it.** Browse the [UltraChat dataset](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k), an open corpus of the kind used for supervised fine-tuning. Inspect a few records: each is a multi-turn conversation in exactly the format an assistant is expected to produce. The consistent voice, the willingness to answer questions rather than continue them, and the turn-taking structure of every assistant model trace back to training data of this shape — the demonstration phase of the pipeline formalized in the Expert tier below.

## Expert

**Supervised fine-tuning.** The pretrained model is fine-tuned with the standard next-token cross-entropy loss on curated (prompt, response) pairs written or selected by labelers. SFT teaches format and style but has two structural limits: it scales only as fast as humans can write good demonstrations, and it cannot easily encode comparative judgments — "this response is better than that one" has no natural expression as a single demonstration. Preference learning addresses both.

**Reward modeling.** Christiano et al. (2017) introduced the recipe of learning a reward function from human pairwise preferences and optimizing a policy against it; Ouyang et al. (2022) adapted it to instruction-following LLMs as the InstructGPT pipeline (Figure 1). Labelers rank sampled model outputs, and a reward model $$r_\phi$$ is trained on these comparisons under the Bradley–Terry model, which posits that the probability of preferring completion $$y_w$$ over $$y_l$$ for prompt $$x$$ is

$$P(y_w \succ y_l \mid x) = \sigma\big(r_\phi(x, y_w) - r_\phi(x, y_l)\big)$$

where $$\sigma$$ is the logistic function. Maximizing the likelihood of the observed preferences fits a scalar reward whose differences predict human choices.

**Policy optimization.** The policy $$\pi_\theta$$, initialized from the SFT model, is then trained to maximize the KL-regularized objective

$$\max_{\pi_\theta} \; \mathbb{E}_{x \sim \mathcal{D},\, y \sim \pi_\theta(\cdot \mid x)}\big[r_\phi(x, y)\big] - \beta\, \mathbb{D}_{\mathrm{KL}}\big(\pi_\theta(\cdot \mid x) \,\|\, \pi_{\mathrm{ref}}(\cdot \mid x)\big)$$

The first term pushes the policy toward completions the learned reward model scores highly. The KL term penalizes divergence from the reference policy $$\pi_{\mathrm{ref}}$$ — the frozen SFT model — and $$\beta$$ sets the exchange rate between reward and divergence. The penalty exists because $$r_\phi$$ is a proxy trained on a finite comparison set: an unconstrained policy can achieve high proxy reward with outputs far from the distribution on which the reward model is reliable. In the InstructGPT recipe this objective is optimized with PPO (proximal policy optimization). Ouyang et al. report that human raters preferred a 1.3B-parameter InstructGPT model over the 175B-parameter base GPT-3, which indicates that alignment quality and raw scale are separable axes.

**Constitutional AI.** Bai et al. (2022) replace much of the human labeling for harmlessness with AI feedback governed by an explicit list of written principles. The method has two phases: a supervised phase in which the model critiques and revises its own outputs against the principles and is fine-tuned on the revisions, and an RL phase (RLAIF) in which a preference model trained on AI-generated comparisons supplies the reward signal. The values steering the model become auditable — they are written down — and the human-labeling bottleneck shrinks, at the cost of depending on the model's ability to critique its own outputs faithfully.

**Direct preference optimization.** Rafailov et al. (2023) observe that the KL-regularized objective above has a closed-form optimal policy: $$\pi^*(y \mid x) \propto \pi_{\mathrm{ref}}(y \mid x)\, e^{r(x,y)/\beta}$$. Inverting this expresses the reward as a function of the policy itself; substituting into the Bradley–Terry model eliminates $$r_\phi$$ entirely and yields a classification-style loss over preference pairs $$(x, y_w, y_l)$$:

$$\mathcal{L}_{\mathrm{DPO}} = -\mathbb{E}\left[\log \sigma\!\left(\beta \log \frac{\pi_\theta(y_w \mid x)}{\pi_{\mathrm{ref}}(y_w \mid x)} - \beta \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\mathrm{ref}}(y_l \mid x)}\right)\right]$$

No explicit reward model is trained and no RL rollouts are run; the policy is fit directly on preference data with supervised gradients. DPO is simpler and more stable to run than PPO-based RLHF, though whether it matches RLHF's ceiling on harder alignment tasks remains debated.

**Open questions.** First, reward hacking: $$r_\phi$$ is an imperfect proxy for human preference, and optimizing hard against it — with PPO or DPO — can exploit the proxy's weaknesses rather than improve genuine quality, an instance of Goodhart's law. KL regularization and iterative re-labeling mitigate this failure mode; they do not eliminate it. Second, capability versus surface behavior: it remains an open empirical question, discussed since Ouyang et al., how much preference optimization improves a model's underlying judgment as opposed to reshaping refusal patterns and style while leaving capabilities — and their failure modes — largely intact.

---

## Implications for agentic-dev

Several behaviors that agentic workflows depend on are post-training artifacts rather than architectural properties, and this changes how they should be reasoned about.

- **Instruction-following itself is installed, not innate.** A well-specified task prompt reliably elicits an on-task response because SFT data consistently paired instruction-shaped prompts with direct answers and preference optimization reinforced that pairing. Against a base model, the same prompt carries no such guarantee.
- **Refusals and permission-asking are trained dispositions.** Claude's tendency to pause and confirm before destructive actions was shaped by demonstration data and preference comparisons that rewarded confirming over silently proceeding in high-stakes situations. Because it is a learned disposition rather than a separate rule layer, it can be invoked, adjusted, and reasoned about in natural-language prompting — the conditioning mechanism of chunk 00 applies to it as to any other behavior.
- **Tone-context prompting works with the trained prior, not against it.** Agentic-dev's chunk 02 prescribes tone rules — evidence-proportional confidence, naming ambiguity rather than smoothing it over. These instructions are effective because post-training already concentrated the output distribution around a coherent default stance; a prompt shifts that stance rather than constructing one from nothing. A base model offers no stable persona for such instructions to steer.
- **Voice persists across a session for the same reason.** Preference optimization rewards consistency of register across outputs, which is why tone established once in a system prompt tends to hold across many turns instead of requiring re-establishment.

---

## Exercises

1. **(Beginner)** For the prompt "What are three good exercises for lower-back pain?", write one plausible base-model continuation that is *not* an answer (for example, a continuation treating the prompt as part of a listicle or FAQ page) and one plausible assistant response. Identify which post-training stage — SFT or preference learning — is most responsible for the difference.
2. **(Practitioner)** Take the same prompt and predict how the two models diverge in *shape*, not just content: response length, whether the model addresses the reader, whether it hedges or adds safety caveats. For each divergence, state whether demonstration data (format) or preference comparisons (calibration and tone) is the more likely source, and why the distinction matters when diagnosing unwanted model behavior.
3. **(Expert)** Consider the KL-regularized objective as $$\beta \to 0$$. Characterize the resulting optimization problem, and explain the failure sequence: why the policy migrates off the distribution where $$r_\phi$$ was trained, why the reward model's scores become unreliable exactly there, and what the resulting outputs look like (degenerate, repetitive, or adversarial completions with high proxy reward). State briefly what the opposite limit $$\beta \to \infty$$ recovers, and why useful values of $$\beta$$ must therefore be interior.

## Checklist

After this chunk you should be able to:

- [ ] Explain why a base model completes a prompt instead of answering it, in terms of the pretraining objective.
- [ ] Describe the three stages of the InstructGPT pipeline: SFT, reward-model training, PPO.
- [ ] Write the KL-regularized RLHF objective and state the function of each term, including the role of $$\beta$$.
- [ ] State the Bradley–Terry model and its role in fitting a reward model to pairwise preferences.
- [ ] Explain how DPO uses the closed-form optimal policy to eliminate the explicit reward model, and write its loss.
- [ ] Describe what Constitutional AI substitutes for human harmlessness labels, and the tradeoff involved.
- [ ] Connect refusals, permission-asking, and consistent tone to post-training rather than to hardcoded filters.
- [ ] Articulate the reward-hacking concern and the capability-versus-surface-behavior question as open problems.

## References

1. Ouyang, L., Wu, J., Jiang, X., Almeida, D., Wainwright, C., et al. "Training Language Models to Follow Instructions with Human Feedback." *NeurIPS* (2022). arXiv:2203.02155. https://arxiv.org/abs/2203.02155
2. Christiano, P. F., Leike, J., Brown, T. B., Martic, M., Legg, S., & Amodei, D. "Deep Reinforcement Learning from Human Preferences." *NeurIPS* (2017). arXiv:1706.03741. https://arxiv.org/abs/1706.03741
3. Bai, Y., Kadavath, S., Kundu, S., Askell, A., et al. "Constitutional AI: Harmlessness from AI Feedback." Anthropic technical report (2022). arXiv:2212.08073. https://arxiv.org/abs/2212.08073
4. Rafailov, R., Sharma, A., Mitchell, E., Ermon, S., Manning, C. D., & Finn, C. "Direct Preference Optimization: Your Language Model is Secretly a Reward Model." *NeurIPS* (2023). arXiv:2305.18290. https://arxiv.org/abs/2305.18290

## Chunk summary

A base model predicts plausible continuations and nothing more; post-training converts it into an assistant. SFT on demonstrations installs the request–response format; preference learning — RLHF with a Bradley–Terry reward model and KL-regularized PPO (Christiano et al., 2017; Ouyang et al., 2022), Constitutional AI's principle-guided AI feedback (Bai et al., 2022), or DPO's reward-model-free reformulation (Rafailov et al., 2023) — shapes calibration, refusal behavior, and tone. The KL term against the SFT reference is load-bearing: it keeps optimization on the distribution where the learned reward is trustworthy. For agentic work, refusals, permission-asking, and stable voice are trained dispositions, which is why they respond to the same conditioning mechanism (chunk 00) as every other prompted behavior, and why tone instructions steer an existing prior rather than build one.
