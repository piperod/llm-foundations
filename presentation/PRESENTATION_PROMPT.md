# Presentation generation prompt — "LLM Foundations"

A complete, self-contained brief for generating the workshop slide deck, plus original
from-scratch figure prompts. Hand **Part A** to a deck-generation agent or designer; hand
each item in **Part B** to a diagram/SVG-capable model (or a designer) to produce the
figures referenced by the slides. Everything here is original — no copyrighted paper or
blog figure is reproduced.

---

# PART A — Master deck prompt

## A1. Role and objective

> You are a senior technical presentation designer. Produce a slide deck for **"LLM
> Foundations,"** a prerequisite workshop on how large language models work, meant to be
> taught before the *agentic-dev* Claude Code workshop. The deck teaches, in one sitting,
> how an LLM produces text and how every practical technique is a way of *reshaping the
> model's next-token probability distribution*. Optimize for a live-taught workshop: each
> slide is a talking point, not a document. Favor one idea per slide, large legible type,
> and a strong recurring visual motif. Target 45–55 slides. Output 16:9 (13.33in × 7.5in).

## A2. Audience and the tier system

The workshop serves three reader levels, named **Curiosity** (no background), **Builder**
(uses LLMs daily), and **Expert** (research-literate). In a live deck you cannot triple
every slide, so handle tiers this way:

- Teach the **main slide at the Builder level** — concrete, operational, one worked example.
- Add a small **corner badge** on each content slide showing which tier the *depth control*
  is set to, and include an optional **"Go deeper (Expert)"** slide immediately after dense
  chunks (attention, scaling laws, alignment, calibration) carrying the equation and the
  citation. These deeper slides are visually marked so a facilitator can skip them for a
  general audience.
- Curiosity framing lives in the **one-line plain-language statement** at the top of each
  chunk slide (the "what this really means" line).

## A3. The central theme — enforce it on every slide

The deck has one spine: **a language model only ever produces a probability distribution
over the next token, and every technique is a lever that reshapes that distribution.**
Enforce it mechanically:

- The **probability-bar histogram** (a row of vertical bars of unequal height, one bar
  highlighted) is the deck's signature motif. It appears on the title, on every section
  divider, and as the literal subject of several figures. It is the visual shorthand for
  "the distribution."
- Group the chunks into **three movements**, each a reshaping stage:
  - **Movement I · Carving the distribution** — training writes it into the weights (chunks 00–04).
  - **Movement II · Steering the distribution** — context and decoding reshape a fixed model (05–07).
  - **Movement III · The distribution in motion** — sampling in a loop; behaviors emerge (08–10).
  - **Applied supplements** — cost (11) and security (12) of the tokens you feed it.
- Every chunk slide carries a one-line **"Distribution move:"** caption naming how that
  chunk reshapes the distribution (supplied per-slide in A5).
- **Sycophancy is the flagship synthesis example** and gets its own slide near the end:
  training biased the distribution toward agreement (a Movement I reshaping), the user's
  stated opinion pulls it further (a Movement II reshaping), and the two compound.

## A4. Design system

**Palette** (carry these hex values exactly; blue reads as "signal/probability," amber as
"the chosen token"):

| Role | Hex | Use |
|---|---|---|
| Primary blue | `2563EB` | bars, key arrows, highlighted elements |
| Deep navy | `1E3A8A` | title + section-divider backgrounds, headers |
| Amber accent | `D97706` | the single highlighted/"chosen" element, callouts |
| Slate ink | `1F2937` | body text on light |
| Muted slate | `64748B` | captions, secondary labels, axes |
| Light ground | `F8FAFC` | content-slide background |
| Hairline | `E5E7EB` | dividers, grid lines, card borders |

**Contrast/sandwich:** dark navy backgrounds for the title, the three section dividers, and
the closing slide; light `F8FAFC` for all content slides.

**Typography** (safe, ships with Office, renders true-to-width): headers **Cambria** (or a
geometric sans such as Calibri Bold at 36–44pt); body **Calibri** or **Arial** at 16–20pt;
captions 11–13pt muted; equations in the deep-dive slides set larger (28–32pt) with generous
whitespace. Never use accent underlines beneath titles.

**Motif and layout grammar:**

- One distinctive element, repeated: the **probability-bar histogram**, small in the top-
  right of section dividers and large as a figure where relevant.
- **Tier badge:** a small pill in the top-right corner of content slides — `CURIOSITY` /
  `BUILDER` / `EXPERT` — filled navy with white text.
- **Two-column default** for chunk slides: left column = text (plain-language line, 2–3
  bullets, "Distribution move" caption, Try-it/exercise chip); right column = the figure.
- **Callout chips:** rounded rectangles, amber border, for "Try it" (live tool) and
  "Exercise." No decorative color bars or edge stripes.
- Every content slide has a figure or diagram; no text-only content slides.

**Slide templates** (define and reuse):

1. **Title** — dark; deck name, subtitle, the large bar-motif, a faint "distribution curve" backdrop.
2. **Section divider** — dark; movement number + name, one-sentence description, small bar motif.
3. **Chunk slide** — light; two-column (text left, figure right), tier badge, distribution-move caption.
4. **Deep-dive slide** — light; centered equation + one annotation + citation footer; marked `EXPERT`.
5. **Interlude / concept slide** — light; a single large diagram (through-line map, sycophancy, agent loop) with minimal text.
6. **Closing** — dark; recap of the spine, hand-off to agentic-dev, resources/QR.

## A5. Deck outline (full slide list)

Produce these slides in order. `[Fig Bn]` references a figure prompt in Part B. The
"Distribution move" text is the mandatory caption for that slide.

**Opening**
1. **Title** — "LLM Foundations — how language models actually work." Subtitle: "A prerequisite for agentic development." `[Fig B1 — bar motif / cover]`.
2. **How to read this** — the three tiers (Curiosity / Builder / Expert) as three badges with one line each; note that Expert deep-dive slides are optional. `[Fig B2 — three-tier badge strip]`.
3. **The one idea** — "You never program a language model. You shape a distribution." State the thesis. `[Fig B3 — the two knobs: weights vs. context reshaping one distribution]`.
4. **The three movements** — the map of the whole deck. `[Fig B4 — three-movements pipeline map]`.

**Movement I · Carving the distribution (training)** — divider slide `[Fig B5 — movement I badge]`.
5. **Chunk 00 — What is a language model.** Plain line: "It predicts the next token, over and over." Distribution move: *defines the object everything else reshapes.* `[Fig B6 — the next-token loop]`.
6. *(Deep dive, Expert)* Chunk 00 — the autoregressive objective + perplexity. `[Fig B7 — chain-rule / cross-entropy panel]`.
7. **Chunk 01 — Tokenization & embeddings.** Plain line: "Text becomes tokens; tokens become vectors." Distribution move: *fixes the alphabet the distribution is defined over.* `[Fig B8 — text→tokens→IDs→vectors pipeline]`.
8. **Chunk 02 — The transformer & attention.** Plain line: "Each token weighs every other token." Distribution move: *the computation that turns a context into a distribution.* `[Fig B9 — attention as a weighting over prior tokens]`.
9. *(Deep dive, Expert)* Chunk 02 — scaled dot-product attention + the O(n²) cost. `[Fig B10 — attention equation + cost curve]`.
10. **Chunk 03 — Pretraining & scaling laws.** Plain line: "More data and compute lower the loss, predictably." Distribution move: *the first and largest force that carves the distribution.* `[Fig B11 — scaling-law log-log curve]`.
11. **Chunk 04 — From base model to assistant.** Plain line: "Preference training reshapes it toward answers people approve of." Distribution move: *bends the distribution toward preferred completions — the seed of sycophancy.* `[Fig B12 — base→SFT→preference→assistant, with before/after distributions]`.

**Movement II · Steering the distribution (context)** — divider `[Fig B5 variant — movement II badge]`.
12. **Chunk 05 — In-context learning & prompting.** Plain line: "Examples in the prompt change behavior with no retraining." Distribution move: *reshapes the distribution at runtime.* `[Fig B13 — few-shot reshaping / pattern completion]`.
13. **Chunk 06 — Context windows & long context.** Plain line: "The model uses the start and end of a long context best." Distribution move: *decides how much of your context actually pulls on the output.* `[Fig B14 — context window + lost-in-the-middle U-curve]`.
14. **Chunk 07 — Decoding & sampling.** Plain line: "Temperature and top-p decide how the token is picked." Distribution move: *the final reshaping, just before a token is drawn.* `[Fig B15 — temperature reshaping the bar histogram at T=0.5/1/2]`.

**Movement III · The distribution in motion (emergence)** — divider `[Fig B5 variant — movement III badge]`.
15. **Chunk 08 — Hallucination & calibration.** Plain line: "Fluent and true are different targets." Distribution move: *what the distribution does where training left it under-constrained.* `[Fig B16 — calibration plot: confidence vs. accuracy]`.
16. **Chunk 09 — Retrieval & tool use.** Plain line: "Give it a way to look things up and act." Distribution move: *injects fresh, grounded tokens into the context.* `[Fig B17 — retrieval/tool loop]`.
17. **Chunk 10 — Agents.** Plain line: "An LLM in a loop, feeding its own output back." Distribution move: *reshapes its own distribution every step; errors compound.* `[Fig B18 — agent loop with compounding annotation]`.

**Applied supplements**
18. **Chunk 11 — Tokenomics.** Plain line: "Every token you add is billed, every turn." Distribution move: *prices the context lever.* `[Fig B19 — cost stack / caching economics]`.
19. **Chunk 12 — Security & key exfiltration.** Plain line: "Untrusted text in the context is a lever pulled against you." Distribution move: *when an attacker reshapes your distribution.* `[Fig B20 — lethal trifecta, original three-circle design]`.

**Synthesis & close**
20. **Sycophancy — the whole thesis in one behavior.** Two reshapings compound. `[Fig B21 — sycophancy: two reshapings stacking]`.
21. **Recap — the distribution and its levers.** One diagram summarizing training / context / decoding / loop as four levers on one distribution. `[Fig B22 — levers recap]`.
22. **Where this goes: agentic-dev.** Hand-off; context engineering is operating the lever you control. Resources + repo/QR. `[Fig B1 variant — dark closing]`.

For chunks that warrant two content slides in a longer session (02, 04, 08, 10), split the
worked example onto a second slide using the same template; keep the figure on the first.

## A6. Per-slide content rules & speaker notes

- **Headline:** a claim, not a topic ("Each token weighs every other token," not "Attention").
- **Body:** at most 3 bullets, ≤ 12 words each; lead with the plain-language line.
- **Distribution-move caption:** the mandatory italic line at the bottom-left of every chunk slide, from A5.
- **Callout chip:** include the chunk's real "Try it" tool where one exists (Tiktokenizer for 01, BertViz for 02, FineWeb for 03, UltraChat for 04, a token counter for 11, gitleaks for 12) or one exercise otherwise.
- **Speaker notes (`addNotes`, plain text):** 60–120 words per content slide — the facilitator's script: the intuition, the one example to say aloud, and the transition sentence to the next slide. On deep-dive slides, notes explain the equation term by term.
- **Citations:** deep-dive slides carry a muted footer with the primary source (author, year); do not clutter content slides with references.

## A7. Global constraints & QA

- **Figures are generated from scratch** per Part B; do not embed third-party paper/blog images.
- **Accessibility:** body text ≥ 16pt; maintain ≥ 4.5:1 contrast; never encode meaning by color alone (pair color with a label or shape).
- **Consistency:** identical margins (≥ 0.5in), the same two-column grid, the same tier-badge position on every content slide.
- **No AI-slide tells:** no underline accents beneath titles, no full-width header/footer color bars, no edge stripes, no cream backgrounds.
- **QA pass:** check every slide for text overflow, figure-label legibility at projection size, consistent movement colors, and that every content slide has both a figure and a distribution-move caption.

---

# PART B — Figure generation prompts (from scratch)

All figures share one house style so the deck looks like a single set. **Prepend the
following preamble to every figure prompt below**, then append the figure-specific body.
These are written for a vector/SVG-capable generator or a designer; label-heavy technical
diagrams should be produced as vector art (raster image models render text poorly). Where a
figure could double as atmospheric cover art, an `[ILLUSTRATION]` note is given.

## B0. Shared house-style preamble (prepend to every figure)

> Produce a clean, minimal **technical diagram as vector art**, 16:9-friendly, on a
> transparent or `F8FAFC` background. Use only this palette: primary blue `#2563EB`, deep
> navy `#1E3A8A`, amber accent `#D97706` (reserved for the single most important / "chosen"
> element), slate ink `#1F2937` for text, muted slate `#64748B` for secondary labels and
> axes, hairline `#E5E7EB` for grids and borders. Sans-serif labels (Inter/Calibri/Arial),
> 14–18px for primary labels, 11–12px italic for annotations. Rounded rectangles (6px
> radius) for nodes; 2px arrows with a single consistent arrowhead. Generous whitespace,
> no drop shadows, no gradients, no decorative stripes, no clip-art. The visual signature of
> the whole set is a **probability-bar histogram**: a row of vertical bars of unequal height
> representing a distribution over next tokens, with exactly one bar in amber (the chosen
> token) and the rest in blue. Reuse that motif wherever a "distribution" is shown.

## B1. Cover / title motif — `[ILLUSTRATION or DIAGRAM]`
Body: A large probability-bar histogram of ~12 blue bars of varying height spanning the lower
third, one bar amber and slightly taller, each bar faintly labeled with a token fragment
("the", "cat", "mat", "▁to", "using", …). Behind it, a very faint smooth bell-like curve
tracing the tops of the bars, in hairline gray, suggesting the underlying distribution. Dark
navy background, ample negative space in the upper two-thirds for the title text (leave it
empty — title is added in the deck). Mood: precise, quiet, confident. Deliver a dark version
(navy bg) for the title/closing and a light version (F8FAFC) for reuse.

## B2. Three-tier badge strip — `[DIAGRAM]`
Body: Three equal pill-shaped badges in a horizontal row, each navy-filled with white
uppercase label — CURIOSITY, BUILDER, EXPERT — and one muted sub-line beneath each: "plain
language," "operational, daily use," "formal + equations + papers." A thin amber underline
connects them left-to-right suggesting increasing depth (a single 2px line, not a stripe
under a title). Keep it schematic and small.

## B3. The two knobs — `[DIAGRAM]` (signature conceptual figure)
Body: Center: one probability-bar histogram labeled "distribution over the next token." Two
large labeled dials/knobs feed into it with arrows. Left knob, navy: "WEIGHTS — set by
training, frozen at runtime." Right knob, blue: "CONTEXT — what you feed in, yours to
control." Under the right knob add a small amber tag: "← your leverage." The two arrows
converge on the histogram, visually "shaping" it. Caption line beneath (muted italic): "Only
two things set the distribution." Make the composition symmetric and calm; this figure must
read instantly.

## B4. Three-movements pipeline map — `[DIAGRAM]`
Body: A left-to-right pipeline of three labeled stages, each a rounded panel, with a small
probability-bar histogram inside each showing the distribution getting progressively
"sharper" (Movement I broad, II narrower with one bar rising, III with the amber bar clearly
dominant). Stage labels: "I · Carving (training → weights)", "II · Steering (context +
decoding)", "III · In motion (sampling in a loop)". A fourth, smaller detached panel below,
in muted style: "Applied: cost + security." Arrows connect I→II→III. Put small chunk-number
ranges under each (00–04, 05–07, 08–10). This is the deck's structural map.

## B5. Movement section badges — `[DIAGRAM]` (three variants)
Body: A tall dark-navy divider graphic. Large Roman numeral (I / II / III) in blue, movement
name beside it, and in the corner the signature histogram in the state matching that movement
(I: flat/broad bars; II: one bar rising with an arrow labeled "context"; III: the amber bar
dominant with a small circular "loop" arrow around the histogram). Deliver three variants
sharing identical layout so they read as a series.

## B6. The next-token loop — `[DIAGRAM]`
Body: A horizontal flow: (1) a rounded panel "context: ▁The ▁cat ▁sat ▁on ▁the" → (2) a box
"model" → (3) the signature probability-bar histogram with candidates "mat 0.42 / rug 0.31 /
floor 0.20 / roof 0.06" and the amber "mat" bar tallest → (4) a chip "append ▁mat" → a curved
arrow looping back to the context panel labeled "repeat." Emphasize the loop. This is the
single most important beginner figure; keep it uncluttered.

## B7. Autoregressive objective panel — `[DIAGRAM, Expert]`
Body: A centered, elegantly typeset math panel (as vector text, not an image of an equation):
top line, the chain-rule factorization P(x₁…x_T) = ∏ P(xₜ | x₍₁…ₜ₋₁₎); middle, the cross-
entropy loss L = −(1/T)Σ log P(xₜ | x_<t); bottom, "perplexity = e^L = effective branching
factor," with a tiny 3-bar histogram whose caption reads "PPL 20 ≈ uniform over 20 tokens."
Muted footer: "Shannon 1951; Bengio et al. 2003." Lots of whitespace; this is a "breathe"
slide.

## B8. Text → tokens → IDs → vectors — `[DIAGRAM]`
Body: Four-stage left-to-right pipeline. (1) neutral panel: the raw string "unbelievable". (2)
three adjacent subword chips: "un" | "believ" | "able". (3) three ID chips beneath them: 403 /
8821 / 227. (4) three little vertical "vector glyphs" (each a stack of 4–5 short colored cells
of varying length) — one per token — labeled "embedding". Top axis labels the four stages:
"text → subword tokens → token IDs → vectors". Note visually that step (2) is where a rare
word fragments into pieces.

## B9. Attention as weighting — `[DIAGRAM]`
Body: A single sentence laid out as a row of token chips: "The trophy didn't fit in the
suitcase because it was too big." Draw curved arrows from the chip "it" back to "trophy" and
"suitcase," thick/amber for strong attention, thin/gray for weak arrows to function words.
Beside it, a small n×n heat grid (5×5) of blue cells at varying opacity captioned "attention
weights (one head)." Annotation: "each token's meaning is rebuilt as a weighted mix of the
others."

## B10. Attention equation + cost — `[DIAGRAM, Expert]`
Body: Left: the typeset formula Attention(Q,K,V) = softmax(QKᵀ / √d_k) · V, with tiny
callouts on Q ("query"), K ("key"), V ("value") and "÷√d_k keeps softmax un-saturated."
Right: a small log-log-ish curve of "attention work vs. context length" rising steeply,
annotated "≈ n²: 10× longer ⇒ 100× the pairwise comparisons." Muted footer: "Vaswani et al.
2017." Keep the two halves balanced.

## B11. Scaling-law curve — `[DIAGRAM]`
Body: A schematic log-log line chart. X-axis (muted): "compute (log)"; Y-axis: "test loss
(log)". One smooth blue power-law curve descending from upper-left, steep then flattening but
never hitting a floor, labeled on the curve "loss falls predictably." Add a faint amber dashed
vertical marker labeled "compute-optimal split (≈ scale N and D together)". Corner annotation
in muted italic: "Schematic, not measured data — Kaplan 2020; Hoffmann 2022." No numeric
ticks (it is illustrative).

## B12. Base → assistant, with distribution reshaping — `[DIAGRAM]`
Body: A four-node left-to-right pipeline — "base model" → "SFT (demonstrations)" → "preference
training (RLHF / CAI / DPO)" → "assistant" — each an evenly sized rounded panel with a one-line
sub-label. Above the first and last nodes, draw two small probability-bar histograms over the
same candidate set: the base one broad and flat; the assistant one concentrated on the
"helpful, approved" bar (amber). A thin annotation between them: "post-training reshapes the
distribution — including a bias toward answers people approve of." Plant the sycophancy seed
visually here.

## B13. In-context learning — `[DIAGRAM]`
Body: Two stacked mini-prompts. Top ("zero-shot"): "Extract company & sentiment: …" → a
histogram with probability spread across several output-format bars (no clear winner). Bottom
("few-shot"): the same task preceded by two worked "Input → Output" example chips → a histogram
now concentrated on the correct format bar (amber). Caption: "examples reshape the runtime
distribution — no weights change." Optionally add a tiny inset showing the induction pattern
[A][B]…[A]→[B] with a copy arrow.

## B14. Context window + lost-in-the-middle — `[DIAGRAM]`
Body: Left: a long vertical "context" bar filled with faint token lines, with one distinctive
amber "fact" line placed once near the middle. Right: a U-shaped curve, X-axis "position of
the fact (start → middle → end)", Y-axis "recall accuracy", high at both ends and dipping in
the middle; the dip aligned with the amber fact's position. Annotation: "the model uses the
start and end of a long context best." Muted footer: "Liu et al. 2023."

## B15. Temperature reshaping — `[DIAGRAM]` (motif centerpiece)
Body: Three probability-bar histograms side by side over the same five candidate tokens,
labeled T = 0.5, T = 1.0, T = 2.0. Left (T=0.5): very peaked, one tall amber bar, others tiny.
Middle (T=1.0): the model's natural spread. Right (T=2.0): flattened, bars nearly equal. Above,
the formula P(xᵢ) = exp(zᵢ/T) / Σ exp(zⱼ/T). Add a dashed "top-p cutoff" line on one histogram
graying out the low-probability tail. Caption: "decoding is the last reshaping before a token
is drawn." This figure is the deck's clearest embodiment of the theme — make it the visual
peak.

## B16. Calibration plot — `[DIAGRAM]`
Body: A square chart. X-axis "stated confidence," Y-axis "actual accuracy," both 0→100%. A
dashed amber diagonal labeled "perfect calibration." A solid blue curve sitting below the
diagonal labeled "typical: overconfident." Small annotation: "fluent ≠ true; confidence in
tone ≠ confidence in fact." Muted footer: "schematic; Kadavath et al. 2022." Keep it spare.

## B17. Retrieval / tool loop — `[DIAGRAM]`
Body: A cycle: "query" → "retriever / tool" (with a small stacked-documents glyph beside it
labeled "index / tools") → "results" → "model reads query + results" → "answer". A dashed
feedback arrow from "model" back to "retriever" labeled "may call again (thought → action →
observation)". Amber-highlight the injected "results" node with a tag "fresh tokens added to
context." Muted footer: "Lewis et al. 2020; Yao et al. 2023."

## B18. Agent loop with compounding — `[DIAGRAM]`
Body: A four-node clockwise cycle — "plan" → "act (tool call)" → "observe" → "update context"
→ back to "plan". Around the "update context" node, show the context panel visibly growing
(three stacked layers, the newest amber). Outside the loop, a muted callout box with an arrow
to "update context": "context grows every turn → error compounding, context rot, drift." Small
inline note: "reliability ≈ pⁿ (0.95²⁰ ≈ 0.36)". Muted footer: "Anthropic, Building Effective
Agents, 2024."

## B19. Tokenomics / caching — `[DIAGRAM]`
Body: Left: a stacked-bar "cost per request" broken into segments — "system prompt (cached)",
"conversation (re-sent each turn)", "output (priced higher)" — with the output segment amber
and taller per token. Right: two bars comparing "no cache" vs "prompt cache" for a repeated
prefix, the cached one much shorter, annotated "cache reads ≈ 0.1× input." Caption: "every
token you add is billed, and re-billed each turn." Note in muted text: "illustrative; verify
current prices."

## B20. Lethal trifecta (original design) — `[DIAGRAM]`
Body: Three overlapping circles (a clean Venn), each labeled: "access to private data",
"exposure to untrusted content", "ability to send data out". The central intersection is
amber and labeled "exfiltration risk." Around the edges, three small "cut here" scissor marks
on the three overlaps with a caption: "remove any one capability and the attack chain breaks."
Keep it diagrammatic and calm, not alarmist. Muted footer: "framing after S. Willison, 2025."
(Original composition — do not copy any existing diagram.)

## B21. Sycophancy — two reshapings compound — `[DIAGRAM]` (synthesis figure)
Body: Top row, one histogram over answers to a debatable question, roughly even — labeled
"base distribution." Arrow down labeled "① preference training (weights)" → a second histogram
now tilted toward the "agreeable" answer (amber leaning). Arrow down labeled "② your stated
opinion (context)" → a third histogram sharply concentrated on the agreeable answer. Side note:
"a baked-in bias, triggered by a runtime cue." Bottom, a small green "remedy" chip: "reshape
the context: ask for critique first / hide your preference." This figure must make the whole
thesis legible at a glance.

## B22. Levers recap — `[DIAGRAM]`
Body: Center: one large probability-bar histogram ("the distribution"). Four labeled arrows
point at it from around the frame, each a lever the deck covered: "training (weights)",
"prompt + examples (context)", "retrieval + tools (context)", "temperature + top-p (decoding)".
The amber-highlighted lever is "context," tagged "the one you control." Caption: "one object,
four levers — that is the whole workshop." Mirror the composition of B3 to bookend the deck.

---

## Build notes

- If generating with `pptxgenjs` or a template: set `LAYOUT_WIDE` (13.33×7.5), safe fonts,
  and follow the templates in A4. Keep charts native where a real chart is wanted (B11, B16
  can be true line charts); the bar-histogram motif can be a native bar chart with one series
  colored amber.
- Produce figures at ≥ 2× the placed size (or as SVG) so labels stay crisp when projected.
- Keep the movement color usage consistent between the section dividers (B5) and the pipeline
  map (B4) so the three movements are recognizable by position and label, never by color alone.
