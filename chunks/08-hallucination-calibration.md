---
title: "08 · Hallucination, Calibration & Uncertainty"
parent: Chunks
nav_order: 9
---

# Chunk 08: Hallucination, Calibration & Uncertainty

**Purpose.** Explain why language models generate confident-sounding but false content, define calibration as the property that stated confidence should track empirical accuracy, and survey what is known about eliciting honest uncertainty from models.
**Previously.** Chunk 07 covered decoding and sampling — the mechanics by which a probability distribution over tokens becomes emitted text.
**Today.** Why the text that decoding produces can be fluent, well-formed, and false; how that failure is measured; and which interventions reduce it.

![Calibration curves for language models of various sizes on BIG Bench multiple-choice tasks, with a dashed diagonal line indicating perfect calibration (stated confidence matches actual accuracy).](https://ar5iv.labs.arxiv.org/html/2207.05221/assets/x5.png)

*Figure 1: Calibration curves for various model sizes on multiple-choice tasks from BIG Bench, with a dashed line marking perfect calibration. Source: "Language Models (Mostly) Know What They Know," Kadavath et al., Anthropic, 2022 (Figure 4, left panel) — https://ar5iv.labs.arxiv.org/html/2207.05221.*

---

## Beginner

A language model can produce text that is fluent, specific, well-structured, and false. This failure mode is called **hallucination**: output that has the shape of a correct answer — the right register, the right level of detail, plausible names and numbers — without being anchored to fact. The behavior resembles a well-read acquaintance who, when asked something outside their knowledge, supplies an invented detail as smoothly as a remembered one; the fabrication is not deliberate, and nothing in the delivery marks it.

The mechanism follows from the training objective established in chunk 00. The model is trained to produce plausible continuations of text, and plausibility and truth are distinct targets that usually coincide. The most probable continuation of "The capital of France is" really is "Paris". But when a prompt requires a fact the model has not reliably retained — an obscure court citation, an exact date, a library function's precise signature — the model still produces *a* continuation of the correct shape, because producing well-formed continuations is the only operation it performs. **Fluency and accuracy are decoupled**: the confidence of the prose is a property of the writing, not a measurement of the underlying knowledge. A fabricated citation and a genuine one are typographically identical, and no hedging appears by default to distinguish them.

**Calibration** names the property that would mitigate this. A model is well-calibrated when its stated confidence tracks its accuracy: among answers offered at 90% confidence, roughly 90% are correct. Research reviewed below shows that models often carry usable *internal* uncertainty signal, while getting that signal expressed honestly in output text is a separate and harder problem. The gap between what a model represents internally and what it states is the organizing theme of this chunk.

## Practitioner

Given that fluency carries no evidential weight, the practical questions are where hallucination risk concentrates and which interventions measurably shift model behavior.

Risk concentrates in **specific, low-redundancy claims**. Proper names, dates, numeric values, direct quotations, citations, and exact API signatures are the content types where plausible and true most often diverge, because a general claim ("Python lists are mutable") is supported by thousands of training instances while a specific one (a particular box score, a docket number, a function's keyword arguments in one library version) may appear once or never. The model's continuation machinery fills such gaps with values of the right *type* — a real opponent, a believable score, a date in the correct season — which is precisely what makes the errors hard to spot.

The absence of hedging has two sources. First, training data: most human writing presents claims as fact, and hedged registers (academic caveats, forum qualifications) are a minority genre, so imitating the corpus means writing with unmarked confidence. Second, evaluation incentives, developed in the Expert tier: standard benchmarks reward a confident guess over an abstention.

Interventions that reduce — not eliminate — the failure:

- **Explicit abstention instructions.** Directing the model to answer only above a confidence threshold and otherwise state "insufficient evidence" changes the output distribution: the model does not know more, but it claims less. Requests for sourcing operate similarly, by making unverifiable claims costlier to emit.
- **Consistency probing.** Rephrasing a factual question several ways and comparing answers exposes low underlying confidence: a model that returns three different dates for the same event is signaling uncertainty that no single answer displayed.
- **Verification of load-bearing claims.** Any fact that will drive a decision — a legal citation, a security property, a library's actual behavior — should be checked against a primary source or by execution. This is the only intervention with a guarantee attached.

A compact illustration: asked for the exact date and score of a specific player's debut with a new team, a model may state all three values in the same register it would use for elementary arithmetic. If the fact was rare in training, the values are likely well-typed inventions. The defect is not weak domain knowledge in general; it is that precise low-frequency values are exactly where the plausibility objective and the truth diverge, and the surface text provides no warning.

## Expert

**Taxonomy and mechanisms.** Ji et al. (2023) divide hallucination into *intrinsic* — output that contradicts the provided source or input — and *extrinsic* — output making claims the source cannot verify either way, which is the dominant case in open-ended generation where no fixed source exists. Contributing mechanisms include gaps and inconsistencies in training data; exposure bias from teacher forcing, in which a model trained on ground-truth prefixes must condition at inference on its own possibly erroneous prior tokens, compounding small errors (the path dependence of chunk 00); and the likelihood objective itself, which rewards fluent high-probability continuations rather than verified ones.

**Calibration, formally.** A model is calibrated if

$$P(\text{correct} \mid \text{confidence} = c) = c \quad \text{for all } c \in [0,1].$$

The standard aggregate measure is **Expected Calibration Error (ECE)**: predictions are binned by confidence, and ECE is the weighted average gap between each bin's mean confidence and its empirical accuracy,

$$\mathrm{ECE} = \sum_{m=1}^{M} \frac{|B_m|}{n} \,\bigl|\, \mathrm{acc}(B_m) - \mathrm{conf}(B_m) \,\bigr|.$$

Calibration is measurable regardless of how the confidence is produced — from token probabilities, a trained probe, or verbalized text — which permits the comparisons below.

**Internal signal.** Kadavath et al. (2022) report three findings. First, large models' token-level probabilities are strongly calibrated on multiple-choice and true/false formats, with calibration improving with scale. Second, in the *P(True)* method, the model is shown its own sampled answer and asked whether it is true; the probability assigned to "True" serves as a usable confidence estimate for the answer's correctness. Third, a trained *P(IK)* head predicts the probability that the model will answer a given question correctly, without reference to any candidate answer — evidence that models carry genuine internal signal about the boundaries of their own knowledge. All three degrade under distribution shift and in open-ended generation, where the well-specified answer format that supports calibration is absent.

**Verbalized confidence.** Internal calibration does not transfer automatically to stated confidence, because verbalizing "90% sure" is itself a generation step subject to the same plausibility pressures as any other output. Lin, Hilton & Evans (2022) showed the gap is trainable: a GPT-3 model fine-tuned on their CalibratedMath suite learned to emit verbalized confidence levels that were reasonably calibrated against actual accuracy, generalized moderately under distribution shift, and tracked the model's own uncertainty rather than imitating hedging phrases in the training data. The demonstration is on a narrow arithmetic benchmark, not open-domain factual QA.

**Learned misconceptions.** TruthfulQA (Lin, Hilton & Evans, ACL 2022) probes a distinct failure: models do not only err on obscure facts but confidently reproduce *popular human misconceptions* — folk beliefs, urban legends, misquotes — absorbed from training text. On this benchmark truthfulness *decreased* with model scale, because larger models fit the human-generated training distribution more faithfully, misconceptions included.

**A statistical account.** Kalai, Nachum, Vempala & Zhang (2025) argue hallucination is a predictable consequence of the pretraining objective, via a reduction from generation to classification. Consider the binary task of distinguishing true statements from plausible false ones of the same form. A generator that cannot solve this classification problem must, when minimizing cross-entropy, assign probability mass to the plausible false statements, so its generative error rate is lower-bounded by a quantity tied to the classification difficulty — even when the training data itself is error-free. *Singleton facts*, those appearing exactly once in training, make the classification problem statistically hard, so the singleton rate sets a floor on hallucination for that class of facts. A complementary incentive argument addresses post-training: benchmarks scored on binary correctness assign zero credit to "I don't know" and positive expected value to a guess, so optimizing against them systematically favors confident guessing over calibrated abstention.

**Open questions.** (1) Whether hallucination is reducible through scale and better training data alone, or is structural in the sense of Kalai et al. — in which case the evaluation incentives, not just the models, must change. (2) How a deployed assistant should set its abstention threshold: an assistant that abstains too readily is useless, one that never abstains is untrustworthy, and the appropriate threshold varies with stakes (casual trivia versus medical dosage) in a way current models do not reliably detect unaided.

---

## Implications for agentic-dev

The agentic-dev prompting guidance — set an explicit **confidence bar**, and instruct the model to state "insufficient evidence" rather than guess — is a direct countermeasure to the documented incentive structure. Left unprompted, a model defaults to a complete-sounding answer because that is what its training data and evaluation regime reward (Kalai et al., 2025); an explicit abstention instruction supplies the signal, absent by default, that declining to answer is an accepted output. This pushes toward the calibrated-verbalization behavior Lin, Hilton & Evans (2022) showed is achievable but not automatic.

The same findings justify why agentic workflows treat stated confidence as weak evidence. Even where internal calibration is good (Kadavath et al., 2022), the sentence "I am confident this is correct" is generated text, not a measurement. Verification — running the test suite, executing the code, a review pass, a human checkpoint before a claim is acted on — is the only channel that supplies ground truth from outside the model, and it is the load-bearing step in any workflow where correctness matters.

---

## Exercises

1. **(Beginner)** Ask a model one common factual question and one obscure one (for example, a well-known capital city versus the exact publication date of a minor technical report). Compare the two answers for tone, hedging, and specificity, and record whether anything in the surface text distinguishes the answer likely to be correct from the one likely to be invented.
2. **(Practitioner)** Take a model-generated answer about a software library that contains (a) a general design claim, (b) a function name, (c) a keyword-argument list with default values, and (d) a version number with a release date. Rank the four claim types by hallucination risk, and justify the ranking in terms of training-data redundancy and the specificity of each claim.
3. **(Expert)** Design a small evaluation that distinguishes a calibrated model from an overconfident one. Specify the question set, how confidence is elicited (verbalized percentage, P(True), or token probability), the binning scheme, and the plot: a reliability diagram of empirical accuracy against stated confidence, with the diagonal as the calibrated reference. State what curve an overconfident model produces (accuracy below the diagonal at high confidence), and report ECE as the summary statistic.

## Checklist

After this chunk you should be able to:

- [ ] Explain why fluent output is not evidence of accurate output, in terms of the training objective.
- [ ] Distinguish intrinsic from extrinsic hallucination (Ji et al., 2023) and name three contributing mechanisms.
- [ ] State the formal definition of calibration and describe how ECE aggregates it.
- [ ] Summarize the Kadavath et al. (2022) findings — token-probability calibration, P(True), P(IK) — and the conditions under which they degrade.
- [ ] Explain why internal calibration does not automatically transfer to verbalized confidence, and what Lin, Hilton & Evans (2022) demonstrated about closing the gap.
- [ ] Explain TruthfulQA's finding that misconception errors worsened with scale.
- [ ] Sketch the Kalai et al. (2025) reduction from generation to classification, the singleton-fact floor, and the benchmark-incentive argument.
- [ ] Connect these results to abstention prompting and to verification as the ground-truth channel in agentic workflows.

## References

1. Ji, Z., Lee, N., Frieske, R., Yu, T., Su, D., et al. "Survey of Hallucination in Natural Language Generation." *ACM Computing Surveys* 55.12, Article 248 (2023). https://dl.acm.org/doi/10.1145/3571730
2. Kadavath, S., Conerly, T., Askell, A., Henighan, T., et al. "Language Models (Mostly) Know What They Know." Anthropic (2022). arXiv:2207.05221. https://arxiv.org/abs/2207.05221
3. Lin, S., Hilton, J., & Evans, O. "Teaching Models to Express Their Uncertainty in Words." (2022). arXiv:2205.14334. https://arxiv.org/abs/2205.14334
4. Lin, S., Hilton, J., & Evans, O. "TruthfulQA: Measuring How Models Mimic Human Falsehoods." *ACL* (2022). arXiv:2109.07958. https://arxiv.org/abs/2109.07958
5. Kalai, A. T., Nachum, O., Vempala, S. S., & Zhang, E. "Why Language Models Hallucinate." OpenAI (2025). arXiv:2509.04664. https://arxiv.org/abs/2509.04664

## Chunk summary

Hallucination is fluent generation without factual grounding, and it follows from a training objective that targets plausibility rather than truth. Its taxonomy is intrinsic versus extrinsic (Ji et al., 2023); its measurement is calibration, formalized as $$P(\text{correct} \mid \text{confidence}=c)=c$$ and aggregated by ECE. Models carry substantial internal uncertainty signal — calibrated token probabilities, P(True), P(IK) — that degrades off-distribution (Kadavath et al., 2022) and does not automatically surface in verbalized confidence, though fine-tuning can partially close that gap (Lin, Hilton & Evans, 2022). Scale worsens misconception errors (TruthfulQA), and Kalai et al. (2025) argue both a statistical floor on hallucination and an evaluation regime that rewards guessing over abstention. Agentic-dev's confidence-bar prompting counters the incentive problem; verification outside the model supplies the ground truth that stated confidence cannot.
