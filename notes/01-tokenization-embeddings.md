# Chunk 01: Tokenization & Embeddings

**Purpose.** Understand how raw text is converted into the discrete units ("tokens") a language model actually predicts, and how those tokens become the numeric vectors ("embeddings") the model computes over.
**Previously.** Chunk 00 established that a language model's core job is next-token prediction — a probability distribution over "what comes next."
**Today.** We open up what "token" actually means, why it isn't a word, how subword tokenization (BPE, SentencePiece) builds a vocabulary, and how tokens are mapped to vectors before any computation happens.

![Color-coded embedding vectors for the words queen, woman, girl, boy, man, king, queen, and water, each shown as a row of 50 colored cells representing vector dimension values.](https://jalammar.github.io/images/word2vec/queen-woman-girl-embeddings.png)

*Figure 1: Each word is represented as a vector of numbers (color-coded by value), and words with related meanings share similar patterns across dimensions. Source: "The Illustrated Word2vec," Jay Alammar, 2019 — https://jalammar.github.io/illustrated-word2vec/.*

---

## Beginner

When you type a message to a chatbot, the model doesn't see your sentence the way you do — as words separated by spaces. Before it can do anything, your text gets chopped into small pieces called **tokens**. A token might be a whole common word ("the", "cat"), a chunk of a longer word ("tokeniz" + "ation"), or even a single character or punctuation mark. Roughly speaking, one token is about 3/4 of an English word on average, though this varies a lot by language and text type.

Why not just use whole words? Two reasons. First, there are far too many possible words — new names, typos, made-up words, technical jargon, and every language on earth — to give each one its own slot in a fixed list. Second, using individual letters would make sequences enormous and would throw away the useful "chunkiness" of language (word roots, common suffixes). Subword tokens are the compromise: a manageable-size vocabulary (tens of thousands of entries) that can still spell out anything by combining pieces, the way "un-", "believe", and "-able" combine to build "unbelievable" even if the model has never seen that exact word before.

Once text is split into tokens, each token is converted into a list of numbers called a **vector** (also called an **embedding**). Think of it like a coordinate on a map, except instead of 2 dimensions (latitude/longitude) it might have hundreds or thousands of dimensions. Words with similar meanings or uses end up with vectors that are numerically close together — the model learns this arrangement during training, not by being told definitions, but by seeing which words tend to appear in similar contexts millions of times. This vector is the actual input the model's math operates on; the text itself is just how humans read it in and out.

This matters for two everyday reasons: services that charge "per token" aren't charging per word, and text with unusual spelling, rare names, or non-English scripts often gets split in surprising ways — which is why chatbots sometimes stumble on things like reversing an unusual word or counting letters in it.

## Practitioner

If you use LLM APIs or chat tools regularly, the single most useful mental shift is: **stop thinking in words, start thinking in tokens.** Tokens are the unit the model was trained on, the unit it counts against context limits, and the unit providers bill by. Words and tokens correlate but diverge constantly, and the divergence is where surprises happen.

A rough, commonly cited heuristic for English is about 4 characters or roughly 3/4 of a word per token — but this is only a heuristic, and it breaks down fast outside typical English prose. Consider the string `"tokenization"`. Depending on the tokenizer's vocabulary, this might come back as a single token (if "tokenization" is common enough to have earned its own vocabulary entry) or split into pieces like `token` + `ization`, or even `tok` + `en` + `ization`. Rare technical terms, code identifiers (`camelCaseVariableNames`), non-English text, and emoji tend to fragment into more, smaller tokens than everyday English prose — sometimes 2-3x more tokens per "word" than you'd expect. This is exactly why the same sentence in English versus, say, Thai or Japanese can consume dramatically different token counts even though it conveys the same information — a real and well-documented fairness/cost concern for non-English users.

A practical worked comparison (illustrative, based on typical BPE-style tokenizer behavior — exact counts vary by tokenizer):

| Text | Approx. tokens | Note |
|---|---|---|
| `"The cat sat on the mat."` | ~7 | Common short words, near 1 token/word |
| `"antidisestablishmentarianism"` | ~6-8 | Rare long word, splits into several subword pieces |
| `"def calculate_user_score(user_id):"` | ~10-12 | Code: punctuation and identifiers each cost tokens |
| A paragraph of Thai or Hindi script | often 2-4x the tokens of an equivalent English paragraph | Non-Latin scripts historically under-served by tokenizers trained mostly on English/Latin text |

What you should actually track day to day: (1) your context window is a token budget, not a word or page budget — long code files, pasted logs, and non-English content eat it faster than plain English prose of the same apparent length; (2) API costs are quoted per-token (usually per million tokens, input and output priced differently), so verbose prompts and verbose outputs both cost money; (3) if a model behaves strangely on a specific string — refuses to repeat it, hallucinates, or gets uncharacteristically confused — a real, documented cause is that the string tokenizes into something the model rarely or never saw associated with meaning during training (see the "glitch token" discussion in Expert, below). When in doubt, use a tokenizer's official counting tool rather than estimating from word count.

## Expert

Formally, tokenization is the process of mapping a raw string to a sequence of items from a fixed, finite vocabulary V, learned ahead of time from a training corpus. Three subword algorithms dominate practice:

- **Byte-Pair Encoding (BPE)**, adapted for NLP by Sennrich, Haddow, and Birch (2016), starts from a base vocabulary of characters (or bytes) and greedily merges the most frequent adjacent pair repeatedly, building up a vocabulary of increasingly long subword units. This gives open-vocabulary coverage: any string can be represented as some sequence of known subwords, down to individual bytes/characters if necessary, so there is no true "unknown token" problem.
- **WordPiece**, used in BERT, is similar to BPE but chooses merges that maximize training-data likelihood rather than raw frequency.
- **Unigram language model tokenization** (Kudo, 2018, the basis of one mode in SentencePiece) starts from a large candidate vocabulary and iteratively prunes subwords to maximize corpus likelihood under a unigram model, and can produce multiple valid segmentations of the same string probabilistically.

**SentencePiece** (Kudo & Richardson, 2018) is not itself a distinct algorithm so much as an implementation layer: it treats input as a raw stream of Unicode (or bytes) rather than pre-tokenized whitespace-separated words, which makes it language-agnostic — critical for languages without whitespace word boundaries (Japanese, Thai, Chinese). This is why most modern multilingual LLMs use SentencePiece-style or byte-level BPE pipelines rather than assuming "words" as a starting unit at all.

Once tokenized, each token ID indexes into an **embedding matrix** — a learned lookup table of shape (vocabulary size x model dimension) — producing the initial vector representation the transformer computes over. Critically, in modern LLMs this embedding table is trained jointly with the rest of the model via the next-token objective, not pre-trained separately. This differs from the earlier "static embedding" era: **word2vec** (Mikolov et al., 2013) learned one fixed vector per word from local context windows (skip-gram / CBOW), and **GloVe** (Pennington, Socher & Manning, 2014) learned vectors from global co-occurrence statistics via matrix factorization. Both produced one vector per word type, with no sensitivity to context ("bank" the riverbank and "bank" the financial institution shared a vector). Contextual embeddings inside a transformer, by contrast, start from a token's static embedding but transform it through attention layers into a context-dependent representation — the vector for a token changes depending on surrounding tokens. Static word2vec/GloVe embeddings remain useful pedagogically and in lightweight retrieval applications precisely because they isolate the "meaning as geometry" idea from the contextual machinery covered in later chunks.

Open questions worth sitting with: (1) **Tokenizer-induced fairness** — because vocabularies are learned from corpora skewed toward high-resource languages (disproportionately English), tokenizing other languages produces more tokens per unit of meaning, directly inflating cost and consuming context budget disproportionately for non-English users — a documented, still-unresolved problem. (2) **Tokenization-free / byte-level modeling** is an active research direction (e.g., byte-level and patch-based models) attempting to remove the fixed-vocabulary tokenizer entirely, trading a more elegant, universal input representation for higher sequence lengths and compute cost. Neither problem has a settled answer.

---

## Implications for agentic-dev

Two very concrete consequences follow directly from everything above, and both show up constantly when using an agentic coding tool like Claude Code:

- **Context window budgeting is token budgeting, not word or line budgeting.** When Claude Code reads a large file, a long log, or an entire directory into context, what fills up (and eventually forces truncation or summarization) is the token count of that content — not its word count or line count. Code with lots of punctuation, deeply nested JSON, minified files, or non-English comments/strings consumes tokens faster per visible character than plain English prose does, because of exactly the subword-splitting behavior described above. Knowing this changes how you decide what to paste in versus point to, how aggressively to summarize before feeding content back to the model, and why "this file looks short but blew the context" happens.
- **API cost is metered per token, input and output separately**, so verbose system prompts, large pasted context, and long generated responses are all literally billed line items — not abstractions. Estimating cost by "how much text is this" using word count will systematically mislead you for code, logs, and non-English content; using an actual tokenizer/token-counting tool gives an accurate number.
- **Unusual strings cause unusual model behavior, and tokenization is a real, documented root cause.** The "glitch token" phenomenon — most famously the token for the string "SolidGoldMagikarp," documented by Rumbelow and Watkins in their 2023 "SolidGoldMagikarp (plus, prompt generation)" writeup — showed that tokens which exist in a tokenizer's vocabulary but were rare or absent in the actual training data can produce erratic, hard-to-explain model output when they appear in a prompt. Practically, this means when Claude Code (or any LLM tool) behaves strangely on a specific rare identifier, unusual Unicode, an obscure library name, or a weird string pasted from a log, tokenization is a legitimate first hypothesis to check — not just "the model is being weird." Rare tokens correspond to rare or thin training signal, which shows up as unreliable behavior.

---

## Checklist

- [ ] I can explain why a "token" is not the same thing as a "word."
- [ ] I know that context windows and API costs are measured in tokens, not words or characters.
- [ ] I can name at least one subword tokenization algorithm (BPE, WordPiece, or unigram/SentencePiece) and roughly how it builds its vocabulary.
- [ ] I understand that an embedding is a learned vector, and that tokens with similar usage end up with similar vectors.
- [ ] I know that non-English text and code typically consume more tokens per unit of content than plain English prose.
- [ ] I can explain, at a high level, why an unusual/rare string might cause an LLM to behave oddly (tokenizer-training mismatch).

## References

1. Sennrich, R., Haddow, B., & Birch, A. (2016). *Neural Machine Translation of Rare Words with Subword Units.* Proceedings of ACL 2016 (Volume 1: Long Papers), pp. 1715-1725. https://aclanthology.org/P16-1162/
2. Kudo, T., & Richardson, J. (2018). *SentencePiece: A simple and language independent subword tokenizer and detokenizer for Neural Text Processing.* Proceedings of EMNLP 2018: System Demonstrations, pp. 66-71. https://aclanthology.org/D18-2012/
3. Mikolov, T., Chen, K., Corrado, G., & Dean, J. (2013). *Efficient Estimation of Word Representations in Vector Space.* arXiv:1301.3781. https://arxiv.org/abs/1301.3781
4. Pennington, J., Socher, R., & Manning, C. D. (2014). *GloVe: Global Vectors for Word Representation.* Proceedings of EMNLP 2014, pp. 1532-1543. https://aclanthology.org/D14-1162/
5. Rumbelow, J., & Watkins, M. (2023). *SolidGoldMagikarp (plus, prompt generation).* LessWrong / AI Alignment Forum. https://www.lesswrong.com/posts/aPeJE8bSo6rAFoLqg/solidgoldmagikarp-plus-prompt-generation

## Chunk summary

Text becomes tokens via learned subword algorithms (BPE, WordPiece, unigram/SentencePiece) because a fixed-size vocabulary of word-pieces balances coverage against sequence length; tokens then become vectors via a learned embedding lookup, the numeric substrate all later transformer computation runs on. For anyone operating an agentic coding tool, this isn't academic: context windows and API bills are denominated in tokens, code and non-English text burn through that budget faster than plain English prose of similar length, and rare or out-of-distribution strings can trigger genuinely strange model behavior traceable to thin or mismatched tokenizer training data.
