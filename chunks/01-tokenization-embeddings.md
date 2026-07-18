---
title: "01 · Tokenization & Embeddings"
parent: Chunks
nav_order: 2
---

# Chunk 01: Tokenization & Embeddings

**Purpose.** Specify the two learned components that stand between raw text and the model proper: the tokenizer, which maps strings to units from a fixed vocabulary, and the embedding matrix, which maps those units to the vectors all subsequent computation operates on.
**Previously.** Chunk 00 defined a language model as a learned conditional distribution $$P(x_t \mid x_{<t})$$ over the next token, applied autoregressively, with the conditioning context as the sole channel of influence.
**Today.** What a token is and why it is not a word; how subword algorithms (BPE, SentencePiece) construct vocabularies; the embedding matrix and its relation to earlier static embeddings; and the operational consequences — token-denominated cost, context budgets, and tokenization-induced failure modes.

![Color-coded embedding vectors for the words queen, woman, girl, boy, man, king, queen, and water, each shown as a row of 50 colored cells representing vector dimension values.](https://jalammar.github.io/images/word2vec/queen-woman-girl-embeddings.png)

*Figure 1: Each word is represented as a vector of numbers (color-coded by value), and words with related meanings share similar patterns across dimensions. Source: "The Illustrated Word2vec," Jay Alammar, 2019 — https://jalammar.github.io/illustrated-word2vec/.*

---

## Beginner

A language model does not operate on words. Before any prediction occurs, input text is segmented into **tokens**: units drawn from a fixed inventory decided before training. A token may be a whole common word ("the", "cat"), a fragment of a longer one ("token" + "ization"), a punctuation mark, or a single character. As a rough guide, one token corresponds to about three quarters of an English word, though the ratio varies considerably with language and content.

The inventory is built from subwords because the two obvious alternatives fail in opposite directions. A vocabulary of whole words is unbounded — names, typos, inflections, technical coinages, and every other language would each demand an entry — so any fixed word list leaves an unbounded remainder it cannot represent. A vocabulary of individual characters covers everything but makes every sequence long and discards the recurring structure of language: roots, prefixes, suffixes. Subword tokenization is the compromise: a fixed vocabulary of some tens of thousands of pieces, chosen so that frequent strings become single tokens while anything else can still be spelled out by composition, in the way "un", "believ", and "able" assemble "unbelievable" even if that exact word never appeared when the vocabulary was built.

Tokens are discrete symbols; the model's arithmetic requires numbers. Each token is therefore mapped to an **embedding**: a vector of several hundred to several thousand numbers, learned during training. Tokens that occur in similar contexts acquire numerically similar vectors — the model is never given definitions, only co-occurrence at very large scale. An embedding functions like a coordinate on a map, except with hundreds of dimensions rather than two: proximity in the space corresponds to relatedness in usage. From this point onward the model computes exclusively over these vectors; the text itself is an interface for humans.

Two everyday consequences follow. Services priced "per token" are not priced per word. And unusual spellings, rare names, and non-Latin scripts often segment in unexpected ways — which is one mechanism behind familiar failures such as miscounting the letters in an uncommon word.

## Practitioner

Tokens are the unit of every quantity a practitioner manages: training data volume, context-window capacity, and billing are all denominated in tokens. Word counts correlate with token counts but diverge systematically, and the divergences concentrate precisely where cost and reliability matter.

For ordinary English prose, the working heuristic is roughly four characters, or three quarters of a word, per token. The heuristic degrades quickly outside that regime. The string `"tokenization"` may be a single token if it was frequent enough in the tokenizer's training corpus to earn a vocabulary entry, or it may segment as `token` + `ization`, or as `tok` + `en` + `ization` — the outcome is a property of the specific tokenizer, not of the string. Code identifiers, rare technical terms, emoji, and text in non-Latin scripts all fragment into more and smaller pieces than everyday English. Illustrative magnitudes (exact counts vary by tokenizer):

| Text | Approx. tokens | Note |
|---|---|---|
| `"The cat sat on the mat."` | ~7 | Common short words, near 1 token/word |
| `"antidisestablishmentarianism"` | ~6–8 | One rare word, several subword pieces |
| `"def calculate_user_score(user_id):"` | ~10–12 | Code: punctuation and identifier fragments each cost tokens |
| A paragraph of Thai or Hindi | often 2–4× the tokens of equivalent English prose | Scripts underrepresented in tokenizer training corpora |

Three operational rules follow. First, the context window is a token budget: pasted logs, deeply nested JSON, minified files, and non-English content exhaust it faster than plain prose of the same visible length. Second, API pricing is quoted per token — typically per million, with input and output priced separately — so verbose prompts and verbose outputs are both billed quantities, and estimating cost from word count systematically undercounts for code and non-English text. Third, when a model behaves anomalously on one specific string — refusing to repeat it, or producing uncharacteristically confused output — the segmentation of that string is a legitimate first hypothesis; the Expert tier states the mechanism precisely. For any decision that depends on an exact count, the provider's token-counting tool is authoritative; heuristics are for orders of magnitude.

> **Try it.** Open the [Tiktokenizer](https://tiktokenizer.vercel.app/) and paste `hello world`, `Helloworld`, `HELLO WORLD`, and ` hello world` (leading space). Each variant segments differently: case, spacing, and punctuation all produce distinct tokens, which is why token counts diverge from word counts and why superficially similar strings can behave differently in a model. Then paste a rare identifier from your own codebase and a sentence in a non-English language, and compare their token counts against English prose of the same visible length. For embeddings, open the [TensorFlow Embedding Projector](https://projector.tensorflow.org/), search for a word, and inspect its nearest neighbors in the projected space — the geometric structure described in this chunk, computed from real data.

## Expert

A tokenizer is a deterministic map from strings to sequences over a finite vocabulary $$V$$, with $$V$$ learned from a reference corpus before model training begins. Three subword construction algorithms dominate practice.

**Byte-Pair Encoding (BPE)**, adapted for neural machine translation by Sennrich, Haddow, and Birch (2016), trains as follows: (1) initialize $$V$$ with all individual characters — or all 256 bytes in byte-level variants, which guarantees every string is representable; (2) count occurrences of every adjacent symbol pair in the corpus; (3) merge the most frequent pair into a single new symbol and add it to $$V$$; (4) repeat from step 2 until $$|V|$$ reaches a target size. Segmentation at inference replays the learned merges in order. A worked example on the corpus {"low", "lower", "lowest"}, each occurring once and initially segmented into characters:

```
l o w      l o w e r      l o w e s t
```

Pair counts: (l,o) = 3, (o,w) = 3, (w,e) = 2, all others ≤ 1. The first merge (ties broken by a fixed arbitrary rule) produces the symbol `lo`; the second, (lo,w) with count 3, produces `low`; the third, (low,e) with count 2, produces `lowe`:

```
low        lowe r         lowe s t
```

After three merges the vocabulary is {l, o, w, e, r, s, t, lo, low, lowe}, and "lower" segments as `lowe` + `r`. Production vocabularies extend this loop for tens of thousands of merges: GPT-2's byte-level BPE vocabulary contains 50,257 tokens, and modern tokenizers typically range from roughly 32K to over 100K entries. Because the base vocabulary covers all characters or bytes, any string has a valid segmentation and no genuine "unknown token" arises.

**WordPiece**, used in BERT, follows the same greedy-merge structure but selects the merge that maximizes training-corpus likelihood rather than raw pair frequency. **Unigram-LM tokenization** inverts the direction: it starts from a large candidate vocabulary and iteratively prunes entries to maximize corpus likelihood under a unigram model, admitting multiple probabilistic segmentations of one string. **SentencePiece** (Kudo & Richardson, 2018) is an implementation layer supporting both BPE and unigram modes; its consequential design decision is treating input as a raw Unicode or byte stream rather than pre-split whitespace-delimited words, which removes the assumption that "word" is a meaningful unit — necessary for languages without whitespace boundaries such as Japanese, Thai, and Chinese.

The interface between tokens and the network is the embedding matrix

$$E \in \mathbb{R}^{|V| \times d},$$

where $$d$$ is the model dimension. A token with ID $$i$$ contributes row $$E_i$$ as its initial representation (equivalently, a one-hot vector multiplied by $$E$$). The rows of $$E$$ are ordinary parameters, trained jointly with the rest of the network by gradient descent on the next-token objective of chunk 00. This joint training distinguishes them from the static embeddings of the preceding era: **word2vec** (Mikolov et al., 2013) learned one fixed vector per word type from local context windows via the skip-gram and CBOW objectives, and **GloVe** (Pennington et al., 2014) factorized global co-occurrence statistics. Both assign a single vector per type — "bank" the institution and "bank" the riverbank share one — and are best read as historical precursors that established the foundational result: useful semantic structure is learnable from distributional statistics alone, including approximate vector-arithmetic regularities such as $$\mathrm{vec}(\text{king}) - \mathrm{vec}(\text{man}) + \mathrm{vec}(\text{woman})$$ lying near $$\mathrm{vec}(\text{queen})$$. In a transformer, the static row $$E_i$$ is only the entry point; attention layers (chunk 02) transform it into a context-dependent representation.

Two consequences of learned vocabularies merit explicit statement. First, **token inflation for underrepresented languages**: $$V$$ is optimized to compress the tokenizer's training corpus, and that corpus typically over-represents English, so text in less-represented languages segments into more tokens per unit of meaning. The effect directly inflates both API cost and context consumption for the same semantic content, a documented equity concern with no settled remedy short of rebalancing vocabulary training corpora or removing the tokenizer entirely — byte-level, tokenization-free modeling is an active research direction that trades the fixed vocabulary for longer sequences and higher compute. Second, **undertrained tokens**: membership in $$V$$ does not guarantee training signal. A string can enter the vocabulary through the merge statistics of the tokenizer corpus yet occur rarely in the model's training data, leaving its row of $$E$$ scarcely updated from initialization. Rumbelow and Watkins (2023) documented the extreme case: "glitch tokens" such as " SolidGoldMagikarp" that elicit erratic, hard-to-explain behavior when placed in a prompt. The mechanism generalizes — rare segmentations correspond to thin training signal, which surfaces as unreliable behavior on specific strings.

---

## Implications for agentic-dev

Chunk 00 established that everything influencing generation enters through one channel, the conditioning context $$x_{<t}$$. This chunk adds that the channel is metered. Three consequences for working in Claude Code:

- **Context budgeting is token budgeting.** When Claude Code reads a file, a log, or a directory listing into context, capacity is consumed by token count, not by lines or visible characters. Punctuation-dense code, nested JSON, minified files, and non-English content cost more tokens per visible character than prose — the reason an apparently short file can dominate the budget, and a criterion for deciding what to load into context versus what to reference by path.
- **Billing is per token, input and output separately.** System prompts, pasted context, file contents read by tools, and generated output are each metered quantities. Accurate cost estimation uses a token counter, not a word count.
- **String-specific anomalies have a tokenization hypothesis.** When behavior degrades on one particular identifier, an unusual Unicode sequence, or a string pasted from a log, mis-tokenization or an undertrained token is a real, documented cause (Rumbelow & Watkins, 2023) and belongs among the first hypotheses checked, ahead of vaguer explanations.

---

## Exercises

1. **(Beginner)** Select a sentence containing at least one proper noun and one uncommon word (for example, "Ximena refactored the anachronistic parser"). Predict the token count and where the splits fall, then verify with any public tokenizer-visualization tool. Explain why the token count exceeds the word count.
2. **(Practitioner)** A 1,500-word English document is sent as input, and the model returns a 300-word summary. Using the prose heuristic of roughly three quarters of a word per token, estimate the input and output token counts, then compute the total cost at hypothetical prices of $3 per million input tokens and $15 per million output tokens. State two properties of the document (language, format, or content) that would make the estimate an undercount, and roughly by what factor.
3. **(Expert)** Take the corpus {"hug" × 10, "pug" × 5, "hugs" × 5}, initialized as characters. Compute the pair frequencies, perform the first three BPE merges by hand, and write out the resulting vocabulary. Then segment the unseen word "pugs" using your merges, and explain why BPE has no unknown-token failure mode when the base vocabulary covers all characters.
4. **(Beginner)** Suppose an embedding space had three interpretable dimensions — *medical*, *formal register*, *person* — and the following vectors: "surgeon" = [1.0, 0.8, 1.0], "scalpel" = [1.0, 0.5, 0.0], "chat" = [0.0, 0.1, 0.0]. Propose plausible vectors for "physician" and "hospital", and identify a word pair whose difference vector would approximate the *medical* direction. Then state the caveat that makes this exercise an idealization: real learned embedding dimensions are not individually interpretable — semantic structure is distributed across directions in the space, recoverable only in aggregate (as the nearest-neighbor structure in the Try-it activity shows).

## Checklist

After this chunk you should be able to:

- [ ] Define a token, and explain why vocabularies are built from subwords rather than whole words or characters.
- [ ] State the BPE training loop precisely and execute its first merges by hand on a toy corpus.
- [ ] Describe the embedding matrix $$E \in \mathbb{R}^{|V| \times d}$$, how a token ID selects a row, and why the rows are trained jointly with the model.
- [ ] Distinguish jointly trained embeddings from static word2vec/GloVe vectors, and state what the static methods established.
- [ ] Explain why code and underrepresented languages consume more tokens per unit of content, and the cost and context-budget consequences.
- [ ] Explain the undertrained-token ("glitch token") mechanism and apply it as a diagnostic hypothesis for string-specific model failures.

## References

1. Sennrich, R., Haddow, B., & Birch, A. "Neural Machine Translation of Rare Words with Subword Units." *ACL* (2016): 1715–1725. https://aclanthology.org/P16-1162/
2. Kudo, T., & Richardson, J. "SentencePiece: A Simple and Language Independent Subword Tokenizer and Detokenizer for Neural Text Processing." *EMNLP: System Demonstrations* (2018): 66–71. https://aclanthology.org/D18-2012/
3. Mikolov, T., Chen, K., Corrado, G., & Dean, J. "Efficient Estimation of Word Representations in Vector Space." arXiv:1301.3781 (2013). https://arxiv.org/abs/1301.3781
4. Pennington, J., Socher, R., & Manning, C. D. "GloVe: Global Vectors for Word Representation." *EMNLP* (2014): 1532–1543. https://aclanthology.org/D14-1162/
5. Rumbelow, J., & Watkins, M. "SolidGoldMagikarp (plus, prompt generation)." *LessWrong / AI Alignment Forum* (2023). https://www.lesswrong.com/posts/aPeJE8bSo6rAFoLqg/solidgoldmagikarp-plus-prompt-generation

## Chunk summary

Text enters a language model through two learned components: a subword tokenizer that maps strings to IDs from a fixed vocabulary — BPE and its relatives, typically 32K to 100K+ entries, 50,257 for GPT-2 — and an embedding matrix $$E \in \mathbb{R}^{|V| \times d}$$ whose rows, trained jointly with the model, are the vectors all later computation operates over. Static word2vec and GloVe embeddings are the historical precursors that established semantic structure as learnable geometry from distributional statistics alone. Operationally, tokens denominate both context windows and billing; code and underrepresented languages inflate token counts per unit of content; and vocabulary entries with thin training signal are one documented source of erratic, string-specific behavior (Rumbelow & Watkins, 2023). All three facts surface daily in agentic development, where every file read and every response generated is a metered draw on the conditioning context of chunk 00.
