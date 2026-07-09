# Chunk 08: Hallucination, Calibration & Uncertainty

**Purpose.** Explain why LLMs generate confident-sounding but false content ("hallucination"), what it means for a model to be well-calibrated, and what research shows about getting models to express uncertainty honestly.
**Previously.** Chunk 07 covered decoding and sampling — the mechanics of how a model turns probabilities into the actual words you see, token by token.
**Today.** Why the words that decoding produces can be fluent, well-formed, and simply wrong — and what's known about detecting or reducing that.

---

## Beginner

Imagine a friend who is extremely well-read, speaks fluently on almost any topic, and never says "um" or hesitates — but occasionally, when asked something they don't actually know, they'll answer anyway, smoothly and confidently, making up a plausible-sounding detail rather than admitting they're not sure. Not because they're lying on purpose — they don't *notice* they're doing it. That's a close analogy for **hallucination** in language models: the model generates text that sounds authoritative and fits the shape of a correct answer, but is factually wrong, invented, or unsupported.

This happens because of how these models work. Chunk 00-03 covered this: an LLM's core skill is predicting a plausible next word given everything before it. "Plausible-sounding" and "true" are different targets. Most of the time they line up — the most statistically likely continuation of "the capital of France is" really is "Paris." But when a model is asked something obscure, something outside its training data, or something requiring precise facts (a court case citation, a person's birth date, a library function's exact name), the model still produces *a* plausible continuation — because generating fluent text is what it was built and trained to do. It doesn't have a built-in alarm bell that goes off when it's guessing versus when it's certain.

This is why hallucinations are so deceptive: the wrong answer reads exactly like the right one. There's no italicized "I'm not sure about this part" baked into the model's writing style by default. A confidently stated fake citation and a confidently stated real citation look identical on the page.

**Calibration** is the technical name for the fix people want: a model whose stated confidence actually tracks its accuracy. If a well-calibrated model says "I'm 90% sure," it should be right about 90% of the time it says that. Research (covered below) shows models are sometimes surprisingly good at this internally — but getting them to *say* an honest confidence level out loud, in a way you can trust, is a separate and harder problem. That gap — between what a model "knows" internally and what it chooses to tell you — is the core theme of this chunk.

## Practitioner

Here's the mental model worth internalizing: a language model's fluency is not evidence of its accuracy. These are produced by the same mechanism (next-token prediction trained on enormous text) but they are not the same variable. A model that has never seen the answer to your question will still produce an answer with the same sentence rhythm, the same confident register, the same absence of hedging words, as a model that has seen the answer a thousand times. Confidence in *tone* and confidence in *fact* are decoupled by default. This is the single most important thing to internalize as a daily user of LLM output: fluency is a property of the writing, not a certificate of truth.

Why doesn't the model just sound less sure when it's guessing? Partly because of how it's trained and evaluated (more on this in Expert), and partly because most training data — human writing on the internet — rarely hedges either, except in specific genres (academic caveats, forum "I think" posts). The model learned to write like people write, and people mostly write as if they know what they're talking about.

**Practical signals and strategies that help (not eliminate):**

- **Specificity is where hallucination concentrates.** Names, dates, citations, exact API signatures, quotes, and numbers are the highest-risk content — these are the places where "plausible" and "true" most often diverge. Vague, general claims are comparatively low-risk.
- **Ask for sourcing or ask the model to abstain.** Explicitly instructing "cite where this comes from" or "say 'insufficient evidence' if you're not sure" changes the model's behavior, because it's now optimizing against a different implicit target than "produce a complete-sounding answer." This doesn't make the model *know* more, but it changes what it's willing to claim.
- **Cross-check anything load-bearing.** If a fact will drive a decision (a legal citation, a security claim, a library's actual behavior), verify it against a real source or by execution — don't take fluency as verification.
- **Ask the same question multiple ways.** If a model gives inconsistent answers to rephrased versions of the same factual question, that inconsistency itself is a signal of low underlying confidence, even though each individual answer sounded equally sure.

**Worked example.** Ask an LLM: "What was the exact score and date of the first NBA game LeBron James played for the Miami Heat?" A model may answer fluently with a specific date, opponent, and score — all stated in the same tone it would use for "What is 2+2?" If the model hasn't reliably retained that exact box score, it may invent a plausible-sounding one (a real opponent, a believable score, a date in the right season) rather than saying "I recall he debuted for Miami in the 2010-11 season opener but I'm not confident on the exact score." The failure isn't that the model is bad at basketball trivia in general — it's that specific numeric facts are exactly the content type most likely to be confabulated, and the fluent delivery gives no warning sign. The practical fix here is simply: don't trust a specific number or date from an LLM without checking it, no matter how sure it sounds.

## Expert

**Sources of hallucination**, per the taxonomy in Ji et al. (2023, ACM Computing Surveys), split broadly into *intrinsic* hallucination (output contradicts the source/input it was given) and *extrinsic* hallucination (output makes claims the source can't verify one way or the other — which is the common case in open-ended factual generation with no fixed "source" at all). Contributing mechanisms include: gaps or contradictions in training data (the fact simply wasn't seen, or was seen inconsistently); exposure bias from teacher-forced training (the model is trained on ground-truth prefixes but at inference time conditions on its own possibly-erroneous prior tokens, so small errors compound); and the pressure of a well-formed-output objective — next-token prediction rewards fluent, high-likelihood continuations, not verified ones.

A complementary and more formal account comes from Kalai, Nachum, Vempala & Zhang, "Why Language Models Hallucinate" (OpenAI, 2025, arXiv:2509.04664). They frame hallucination as a statistical inevitability of the pretraining objective: if a model cannot perfectly separate true statements from plausible-but-false ones in its training distribution, cross-entropy minimization alone will produce some confident errors — this holds even with error-free training data, because rare or singleton facts are inherently hard to distinguish from equally-shaped invented ones. They further argue that post-training evaluation is complicit: benchmarks that score binary correctness give zero credit for "I don't know" and partial credit for a lucky guess, which trains models (and the humans grading them) to prefer confident guessing over calibrated abstention — the same incentive structure as a multiple-choice exam that doesn't penalize guessing.

**Calibration**, formally, means the predicted confidence of a claim matches the empirical accuracy of claims made at that confidence level: among all answers a model gives at "80% confidence," roughly 80% should be correct. This is measurable independent of *how* the confidence is produced. Kadavath et al. (2022, Anthropic, arXiv:2207.05221) found that large language models are often well-calibrated on multiple-choice and true/false formats when confidence is read from the model's internal token probabilities, and that models can be trained to predict "P(IK)" — the probability the model itself will answer a given question correctly — without reference to any specific candidate answer, suggesting models carry real internal signal about the boundaries of their own knowledge. However, this calibration is generally strongest under in-distribution, well-specified formats and degrades under distribution shift or open-ended generation.

The harder, separate finding is that **internal calibration does not automatically transfer to verbalized confidence**. A model can carry accurate internal signal about how likely its answer is correct (recoverable via token probabilities or a probe like P(True)/P(IK)) while still stating a confidence *in words* that is poorly calibrated or systematically overconfident — because verbalizing "90% sure" is itself a generated text choice, subject to the same fluency-over-accuracy pressures as any other output. Lin, Hilton & Evans, "Teaching Models to Express Their Uncertainty in Words" (2022, arXiv:2205.14334), showed this gap is not unbridgeable: a GPT-3 model fine-tuned on their CalibratedMath task suite could learn to emit verbalized confidence levels ("90% confidence," "high confidence") that were reasonably well-calibrated against actual accuracy, generalized moderately under distribution shift, and tracked the model's own uncertainty rather than merely imitating human hedging phrases in the training data. This demonstrates verbalized calibration is trainable, though it was shown on a narrow benchmark rather than open-domain factual QA.

TruthfulQA (Lin, Hilton & Evans, ACL 2022, arXiv:2109.07958) probes a related but distinct failure: models don't only guess wrong on obscure facts, they can actively reproduce *popular human misconceptions* learned from training text (folk beliefs, urban legends, misquotes) with high confidence — and the paper found this got *worse*, not better, with model scale on their benchmark, because larger models better learned the human-generated training distribution, misconceptions included.

**Open questions:** (1) Is hallucination reducible via scale and better training alone, or is it a structural consequence of the pretraining objective (per Kalai et al.) that requires changing evaluation incentives, not just adding data or parameters? (2) How should a deployed assistant resolve the tension between "always be helpful" and "say I don't know" — an assistant that abstains too readily is useless, one that never abstains is untrustworthy, and the right threshold is context-dependent (a casual trivia question versus a medical dosage question) in a way current models don't yet reliably detect on their own.

---

## Implications for agentic-dev

Everything above is the mechanistic reason agentic-dev's prompting guidance insists on setting an explicit **confidence bar** and instructing the model to say "insufficient evidence" rather than guess. That instruction isn't stylistic advice — it's a direct countermeasure to a specific, documented failure mode: fluent generation and verified generation are decoupled by default (this chunk's core finding), and left unprompted, a model defaults to producing a complete-sounding answer because that's what its training data and evaluation incentives reward (Kalai et al., 2025). Telling the model explicitly what confidence threshold to require, and explicitly permitting — even demanding — abstention, pushes it toward the calibrated-verbalization behavior that Lin, Hilton & Evans (2022) showed is achievable but not automatic. Without that instruction, the model has no signal that abstaining is an acceptable, rewarded output in this conversation.

This is also the direct justification for why agentic coding workflows can't stop at "ask the model and trust the answer." If verbalized confidence is unreliable even when internal calibration (Kadavath et al., 2022) is decent, then the only robust way to know whether generated code, a cited fact, or a claimed test result is actually correct is to check it against ground truth outside the model: running the actual test suite, executing the code, doing a human review pass, or inserting a checkpoint where a person confirms a claim before it's acted on. A model saying "this should work" or "I'm confident this is correct" carries much weaker evidentiary weight than the token-probability studies might suggest, precisely because that stated confidence is generated text, not a measurement. Verification steps exist to supply the ground-truth signal the model's own fluency cannot.

---

## Checklist

- [ ] I can explain, using the confident-friend analogy, why fluent output isn't evidence of accurate output.
- [ ] I can distinguish intrinsic vs. extrinsic hallucination (Ji et al., 2023).
- [ ] I can state what "calibration" means formally (stated confidence matches empirical accuracy).
- [ ] I can explain the gap between a model's internal confidence signal (Kadavath et al., 2022) and its verbalized confidence, and why the latter can be worse.
- [ ] I can name at least one concrete prompting strategy (confidence bar, explicit abstention instruction, ask for sourcing) that reduces hallucination.
- [ ] I can explain why TruthfulQA found some misconceptions got worse, not better, with model scale.
- [ ] I can connect this chunk to why agentic-dev requires verification steps (tests, review, human checkpoints) rather than trusting model output at face value.

## References

1. Survey of Hallucination in Natural Language Generation (Ji, Lee, Frieske, Yu, Su, et al., 2023, ACM Computing Surveys 55(12), Article 248) — https://dl.acm.org/doi/10.1145/3571730
2. Language Models (Mostly) Know What They Know (Kadavath, Conerly, Askell, Henighan, et al., 2022, Anthropic / arXiv:2207.05221) — https://arxiv.org/abs/2207.05221
3. Teaching Models to Express Their Uncertainty in Words (Lin, Hilton, Evans, 2022, arXiv:2205.14334) — https://arxiv.org/abs/2205.14334
4. TruthfulQA: Measuring How Models Mimic Human Falsehoods (Lin, Hilton, Evans, ACL 2022 / arXiv:2109.07958) — https://arxiv.org/abs/2109.07958
5. Why Language Models Hallucinate (Kalai, Nachum, Vempala, Zhang, 2025, OpenAI / arXiv:2509.04664) — https://arxiv.org/abs/2509.04664

## Chunk summary

Hallucination happens because fluent, well-formed generation and factually verified generation are different targets that the same training process conflates by default; models often carry decent internal signal about their own uncertainty (Kadavath et al.) but don't reliably surface it in words unless specifically trained or prompted to (Lin, Hilton & Evans), and current training/evaluation incentives actively reward confident guessing over honest abstention (Kalai et al., 2025). Agentic-dev's confidence-bar and "insufficient evidence" prompting rules are a direct countermeasure to this gap, and it's exactly why agentic coding workflows lean on tests, review passes, and human checkpoints rather than accepting a model's stated confidence as proof of correctness.
