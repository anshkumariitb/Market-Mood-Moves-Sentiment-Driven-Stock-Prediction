# Market Mood & Moves — Complete Study Guide
### Sentiment-Driven Stock Prediction (FinBERT + LSTM) | From Zero to Interview-Ready

---

## How to Use This Guide

This guide assumes you know **nothing** about NLP or machine learning. It builds the project up layer by layer, in the same order an interviewer would probe it. Each part ends with the interview questions that part prepares you for.

**Suggested 4-day plan:**

| Day | Study | Outcome |
|-----|-------|---------|
| Day 1 | Parts 1–2 (Big Idea + NLP/BERT) | You can explain WHY sentiment matters and HOW a machine reads text |
| Day 2 | Part 3 (FinBERT) | You can defend the core model choice — the most-asked question |
| Day 3 | Part 4 (Time Series + LSTM) | You can explain the prediction side |
| Day 4 | Parts 5–7 (Pipeline, Evaluation, Interview Q&A) | You can walk through the full project + answer results questions |

Your three resume lines, and where each is defended:

1. *"Developed a 5-stage framework linking financial news sentiment to stock price moves via FinBERT scores & LSTM models"* → Parts 1, 3, 4, 5
2. *"Architected the data pipeline — NewsAPI & Yahoo Finance ingestion with NLTK/spaCy based tokenization & lemmatization"* → Part 5
3. *"Defined risk-aware strategy evaluation using walk-forward backtesting — Sharpe, Sortino, CAGR & max drawdown metrics"* → Part 6

---

# PART 1: The Big Idea (Why This Project Exists)

## 1.1 The problem with price-only prediction

Traditional stock prediction uses only **quantitative data**: past prices, trading volumes, returns. But markets are moved by humans, and humans react to **news**. When a company announces record profits, the price jumps *before* that jump shows up in any historical price chart. Price data is a **lagging** record of what already happened; news is a **leading** signal of what's about to happen.

**The analogy to remember:** predicting stocks from prices alone is like driving while looking only in the rear-view mirror. News sentiment is the windshield.

## 1.2 Behavioral finance in one paragraph

Classical finance theory (the Efficient Market Hypothesis) says prices instantly reflect all information, so nothing is predictable. **Behavioral finance** counters: real investors are emotional — they panic-sell on bad headlines, FOMO-buy on hype, overreact and underreact. These emotional patterns leave footprints. If you could **measure the mood of the market**, you'd have information that prices haven't fully absorbed yet.

## 1.3 So what does the project actually do?

It builds a system with two halves:

- **Half 1 (Language):** Read financial news automatically and convert each headline/article into a *sentiment score* — a number saying how positive, negative, or neutral the news is. This is done with **FinBERT**.
- **Half 2 (Time):** Feed those sentiment scores *together with* price history into a model that understands sequences over time — an **LSTM** — to model how sentiment shifts relate to future price movement.

Then everything is judged not by "was the prediction right?" alone, but by **finance-grade evaluation**: would a strategy based on these signals survive real market risk?

**One-line pitch (memorize this):** *"I designed a framework that turns financial news into numerical sentiment using FinBERT, aligns it with price history, models the combined sequence with an LSTM, and evaluates the resulting signals with risk-adjusted metrics like Sharpe and max drawdown."*

### Interview questions Part 1 prepares you for
- "Walk me through your project." → the one-line pitch, then expand half by half.
- "Why do you think news adds anything beyond price data?" → lagging vs leading indicators, behavioral finance.
- "Isn't the market efficient? Doesn't price already reflect news?" → EMH vs behavioral finance; sentiment captures the *speed and psychology* of absorption, which is imperfect in the short term.

---

# PART 2: How a Machine Reads Text (NLP Basics → BERT)

## 2.1 Words must become numbers

Computers can't process "profits surged" directly — text must become vectors (lists of numbers). The evolution of how this is done IS the story of modern NLP:

**Stage 1 — Static embeddings (Word2Vec, GloVe):** Every word gets ONE fixed vector, learned from huge text corpora, so that similar words get similar vectors. Problem: **polysemy** — one word, many meanings.

**The "bank" example (use this in interviews):**
- "I deposited money at the **bank**." (financial)
- "They sat on the river **bank**." (geographical)

Word2Vec gives *both* sentences the SAME vector for "bank" — a mushy average of all meanings. In finance, where words like *short*, *bond*, *interest*, *margin* all have double meanings, this is fatal.

**Stage 2 — Contextual embeddings (BERT):** The vector for a word is computed **fresh from its surrounding sentence**. "Bank" near "deposited money" gets a finance-flavored vector; "bank" near "river" gets a geography-flavored one. Formally: the embedding of word *wᵢ* is a function of ALL words in the sentence, not a lookup.

## 2.2 BERT in plain language

**BERT** = **B**idirectional **E**ncoder **R**epresentations from **T**ransformers. Three ideas:

**(a) Transformer encoder + self-attention.** Self-attention lets every word "look at" every other word in the sentence and decide which ones matter for its meaning. Mechanically, each word produces three vectors:
- **Q (Query):** what am I looking for?
- **K (Key):** what do I offer?
- **V (Value):** my actual information.

Attention score = how well my Query matches your Key; the output is a weighted mix of Values. Formula (know it exists, don't memorize deeply): `Attention(Q,K,V) = softmax(QKᵀ/√dₖ)V`. The √dₖ scaling just stops the numbers from getting too large before the softmax.

**Analogy:** a meeting room where every word raises a question (Q), every other word holds up a sign of what it knows (K), and information (V) flows from the most relevant sign-holders.

**(b) Bidirectional.** Older models read left-to-right only. BERT reads both directions at once — the meaning of "bank" is informed by words before AND after it.

**(c) Input construction.** BERT's input = **Token embeddings + Segment embeddings + Position embeddings**, added together:
- **Token (WordPiece):** words are split into subwords — `investment → invest + ##ment`, `playing → play + ##ing`. Why it matters: rare financial jargon never becomes "unknown word"; it's assembled from known pieces.
- **Position:** transformers process all words simultaneously (not one by one), so they'd otherwise have no idea of word ORDER. Position embeddings inject "you are word #3."
- **Segment:** marks which sentence (A or B) a token belongs to, for two-sentence tasks.

**(d) GELU activation.** BERT uses GELU instead of ReLU — a smoother curve, which gives more stable gradients in very deep networks. One sentence is all an interviewer wants here.

## 2.3 How BERT is trained (pre-training objectives)

**Masked Language Modeling (MLM):** Take a sentence, hide **15%** of its tokens, ask the model to guess them. Of the selected tokens: **80%** become [MASK], **10%** become a random word, **10%** stay unchanged (so the model can't just learn "predict something weird only where I see [MASK]"). This forces deep contextual understanding — like learning a language by doing millions of fill-in-the-blank exercises.

**Next Sentence Prediction (NSP):** Given sentences A and B, predict: does B actually follow A? Teaches sentence-level relationships.

### Interview questions Part 2 prepares you for
- "Why not just use Word2Vec? It's much cheaper." → polysemy / the bank example / finance jargon ambiguity.
- "Explain self-attention like I'm not an ML person." → the meeting-room analogy, Q/K/V.
- "What is WordPiece and why does it help with financial text?" → subwords, no out-of-vocabulary failures on jargon.
- "How is BERT trained?" → MLM (15%, 80/10/10) + NSP.
- "Why does BERT need positional embeddings?" → parallel processing loses word order; positions restore it.

---

# PART 3: FinBERT — The Heart of the Project

## 3.1 The domain shift problem (THE most-asked question)

Plain BERT is pre-trained on Wikipedia and books — general English. Financial English is a different dialect:

| Word | Everyday meaning | Financial meaning |
|------|-----------------|-------------------|
| **Liability** | something bad about you | a routine balance-sheet item (neutral) |
| **Loss** | always negative | a normal reported metric (often neutral) |
| **Depression** | mental health condition | economic downturn |
| **Short** | opposite of tall | betting a stock will fall |

A general model reads "the company reported a liability of ₹200 Cr" as *negative sentiment* when a finance expert would call it *neutral information*. This mismatch is **domain shift** — and it is the entire justification for FinBERT.

## 3.2 FinBERT's 3-stage training pipeline (memorize the stages)

1. **Stage 1:** Start from BERT pre-trained on general English.
2. **Stage 2:** Continue pre-training on a large **financial corpus** (TRC2-Financial — Reuters financial news) so the model absorbs finance vocabulary and usage.
3. **Stage 3:** **Supervised fine-tuning** on the **Financial PhraseBank** dataset — thousands of sentences from financial news, hand-labeled by finance experts as **Positive / Negative / Neutral**. The classification is read off the special [CLS] token.

**Output:** for any input sentence, FinBERT produces **three probabilities** — P(positive), P(negative), P(neutral) — summing to 1. E.g. "Q3 profits beat estimates" → (0.91, 0.03, 0.06). This is your "3-class FinBERT sentiment" resume phrase.

## 3.3 The three challenges (great interview differentiators)

**Challenge 1 — Catastrophic forgetting (as a FEATURE).** Normally, when a neural net trains on new data it "forgets" older learning — considered a bug. But for FinBERT, partially forgetting *everyday* word meanings while learning *financial* ones is exactly what you want: the model should stop thinking "depression = sadness" and start thinking "depression = economic downturn." **Key insight to say out loud:** a phenomenon that's usually a weakness becomes an advantage when your goal is domain specialization.

**Challenge 2 — The 512-token limit.** BERT-family models accept at most **512 tokens** (~350–400 words) per pass. Financial documents like annual reports/10-Ks run to tens of thousands of words. Naive fix: **truncation** (keep first 512 tokens) — but the bad news is often buried on page 40. Better fix: **hierarchical chunking** — split the document into 512-token chunks, run FinBERT on each, then **aggregate** the chunk scores (usually by averaging). Mention that newer models (Longformer, BigBird) natively handle long text, but chunking is the practical standard with FinBERT.

**Challenge 3 — Slanted Triangular Learning Rate (STLR).** During fine-tuning, the learning rate first **rises linearly** for the first ~10% of steps (warm-up: gentle start so the pretrained weights aren't wrecked by early large updates), then **decays linearly** (fine adjustments as the model settles). The asymmetric rise-fast/fall-slow triangle balances quick adaptation with stable convergence.

### Interview questions Part 3 prepares you for
- "Why FinBERT instead of BERT?" → domain shift + the liability/loss table. (This WILL be asked.)
- "How was FinBERT trained?" → 3 stages, TRC2-Financial, Financial PhraseBank.
- "What does FinBERT output exactly?" → 3-class probability distribution per sentence.
- "How would you handle a 50-page annual report given the 512-token limit?" → chunk → score → aggregate.
- "What's catastrophic forgetting?" → definition + why it HELPS here (this answer impresses).

---

# PART 4: Modeling Time — From Sequences to LSTM

## 4.1 Why normal ML fails on stock data

Most ML assumes data points are **IID** — independent and identically distributed (like separate coin flips). Stock prices violate this completely: today's price depends on yesterday's, momentum persists, shocks echo for weeks. Price data is a **dependent sequence**, so we need models built for sequences.

Formally, a price series x₁, x₂, …, x_T has joint probability decomposed by the chain rule:
`P(x₁,…,x_T) = ∏ₜ P(xₜ | x₁,…,xₜ₋₁)` — each value depends on ALL history. Conditioning on everything is impractical, so models approximate.

## 4.2 The ladder of sequence models (tell it as a story)

**Rung 1 — Markov / Autoregressive models:** assume today depends only on the last τ days (a fixed window). Cheap, but a small window forgets important distant events (last year's scandal), and a big window explodes in complexity. Finance has **long-range dependencies**, so fixed windows fail.

**Rung 2 — RNN (Recurrent Neural Network):** instead of a fixed window, keep a running **hidden state** hₜ — a compressed summary of everything seen so far, updated at each step: `hₜ = φ(Wₓxₜ + W_h hₜ₋₁ + b)`. Elegant, but training breaks: gradients flow backward through time by repeated multiplication with the same weight matrix — if its values are < 1, gradients shrink exponentially → the **vanishing gradient problem** → the network effectively can't remember far-back events. In finance, where an old regime change still matters, that's fatal.

**Rung 3 — LSTM (Long Short-Term Memory):** fixes vanishing gradients with an explicit **memory cell** (long-term memory Cₜ) separate from the hidden state (short-term hₜ), managed by three **gates** — each a learned valve between 0 and 1:

| Gate | Formula shape | Plain meaning | Financial interpretation |
|------|--------------|---------------|--------------------------|
| **Forget gate** fₜ | σ(W_f[xₜ,hₜ₋₁]) | how much old memory to keep | discard an outdated market regime after a crash |
| **Input gate** iₜ | σ(W_i[xₜ,hₜ₋₁]) | how much new info to store | absorb a sudden sentiment shock |
| **Output gate** oₜ | σ(W_o[xₜ,hₜ₋₁]) | how much memory to reveal now | expose only what's relevant to today's prediction |

Memory update: `Cₜ = fₜ ⊙ Cₜ₋₁ + iₜ ⊙ Ĉₜ` — old memory scaled by forget gate, plus new candidate scaled by input gate. Because memory flows through **addition** (not repeated multiplication), gradients survive over long horizons.

**The analogy to remember:** the LSTM cell state is the market's **diary**; the forget gate is an eraser, the input gate is a pen, the output gate decides which diary page to read aloud today.

## 4.3 Multimodal inputs — where the two halves meet

At each time step, the LSTM's input vector combines **both worlds**: returns, volatility, technical indicators (price world) + aggregated FinBERT sentiment scores (news world). Sentiment acts as a **leading** indicator (expectations, psychology); price features are **lagging** (realized outcomes). Modeling them together is the project's core thesis — that's what "multimodal" means on your resume.

### Interview questions Part 4 prepares you for
- "Why LSTM and not linear regression / plain RNN?" → IID violation → sequences → vanishing gradients → gates. Tell the ladder story.
- "Explain the LSTM gates simply." → diary analogy + the finance column of the table.
- "What exactly goes into the model at each time step?" → multimodal vector: returns, volatility, indicators, sentiment scores.
- "What is the vanishing gradient problem?" → repeated multiplication by <1 values shrinks gradients; LSTM's additive memory path fixes it.

---

# PART 5: The Pipeline & Tools (Your Resume Line 2)

## 5.1 The 5 stages (memorize this flow)

```
[1] Data Collection ──► [2] Sentiment Analysis ──► [3] Feature Engineering ──► [4] Time-Series Modelling ──► [5] Training & Evaluation
     news + prices           FinBERT scores            align & combine              LSTM                        metrics & checks
```

**Stage 1 — Data Collection & Preprocessing.** Two streams: (a) financial news headlines/articles, fetched programmatically via **NewsAPI** (a service that returns news articles for a query like a company name, as structured JSON); (b) historical stock prices via **Yahoo Finance / yfinance** (a free Python library that downloads open-high-low-close prices, volumes, for any ticker). Clean both: remove duplicates, handle missing days, timestamp everything.

**Stage 2 — Sentiment Analysis.** Push each cleaned news text through FinBERT → get (positive, negative, neutral) probabilities per sentence/headline.

**Stage 3 — Feature Engineering.** The trickiest practical step: **temporal alignment** — news published Tuesday 9 PM must map to Wednesday's trading day, not Tuesday's (markets were closed). Then aggregate (e.g., average all of a day's headline sentiments into one daily score) and combine with price-derived features (daily returns, rolling volatility, technical indicators).

**Stage 4 — Time-Series Modelling.** Feed the combined (multimodal) daily feature vectors, in sequence, to the LSTM.

**Stage 5 — Training & Evaluation.** Train, then evaluate with the finance metrics of Part 6.

## 5.2 Text preprocessing: tokenization & lemmatization (NLTK / spaCy)

- **Tokenization** = splitting raw text into units (words/subwords). "Profits surged 20%" → ["Profits", "surged", "20", "%"].
- **Lemmatization** = reducing words to their dictionary base form: *surged → surge*, *better → good*, *companies → company*. Smarter than crude stemming (which would chop "surged" to "surg").
- **NLTK** and **spaCy** are the two standard Python NLP libraries that do this (plus stopword removal — dropping "the", "of", "is").
- **Nuance worth saying in an interview:** heavy preprocessing matters most for classical NLP; BERT-family models have their OWN tokenizer (WordPiece) and want fairly raw text — so in this pipeline, NLTK/spaCy cleaning is used for exploratory analysis and data quality, while FinBERT applies its own tokenization internally. Knowing this distinction signals real understanding.

## 5.3 The tool list in one breath

Python, pandas (dataframes), NumPy, yfinance (prices), NewsAPI (news), NLTK/spaCy (text cleaning), HuggingFace Transformers (FinBERT), PyTorch (deep learning backend), Jupyter notebooks (experimentation), matplotlib/seaborn (plots).

### Interview questions Part 5 prepares you for
- "Walk me through your pipeline stage by stage." → the 5-box diagram.
- "How did you get your data?" → NewsAPI for news, yfinance for prices.
- "What's the hardest part of combining news with prices?" → temporal alignment (after-hours news → next trading day) + aggregation of multiple headlines per day.
- "Tokenization vs lemmatization?" → definitions + examples above.
- "Doesn't BERT do its own tokenization?" → yes, WordPiece — the nuance in 5.2.

---

# PART 6: Risk-Aware Evaluation (Your Resume Line 3)

## 6.1 The core principle (say this early in any evaluation answer)

**"Predictive accuracy alone is not sufficient in finance."** A model can be right 60% of the time and still lose money if its wrong days are catastrophic. So evaluation must measure *risk-adjusted* performance, not just correctness.

## 6.2 The four metrics — definition, formula, intuition

**(1) Sharpe Ratio** = (Rₚ − R_f) / σₚ — portfolio return minus risk-free rate, divided by volatility.
*Intuition:* how much excess return per unit of risk taken. Sharpe of 1+ is decent, 2+ is very good.
*Limitation (bonus point):* it penalizes upside volatility too — getting rich too fast counts as "risk."

**(2) Sortino Ratio** — same as Sharpe but divides by **downside deviation only** (volatility of negative returns).
*Intuition:* punishes only the bad kind of volatility. The fix for Sharpe's limitation.

**(3) CAGR** = (V_final/V_initial)^(1/n) − 1 — compound annual growth rate over n years.
*Intuition:* the "smoothed" yearly growth rate; lets you compare strategies run over different time spans fairly.

**(4) Maximum Drawdown** = max peak-to-trough fall in portfolio value.
*Intuition:* the worst pain an investor would have endured. A strategy that returns 15%/yr but once fell 60% would have been abandoned by any real human mid-way. This measures survivability, both financial and psychological.

## 6.3 Walk-forward backtesting & lookahead bias

**Backtesting** = simulating a strategy on historical data. The cardinal sin is **lookahead bias / information leakage** — accidentally letting the model "see the future" (e.g., training on 2023 data and testing on 2022, or normalizing with statistics computed over the full period).

**Walk-forward validation** fixes this: train on window [Jan–Jun] → test on [Jul] → slide forward → train on [Feb–Jul] → test on [Aug] → repeat. The model only ever predicts periods it has never seen, exactly mirroring real deployment. **Why not regular cross-validation?** Random shuffling would mix future data into training — invalid for time series.

## 6.4 Real-world frictions (mentioning these = instant credibility)

- **Transaction costs:** brokerage, exchange fees, taxes — small per trade, deadly at high frequency.
- **Slippage:** you decide to buy at ₹100 but your order fills at ₹100.30 because the market moved.
- **Regime stability:** a good strategy must be checked across bull, bear, and sideways markets; sentiment features help detect regime shifts early because mood turns before prices do.

### Interview questions Part 6 prepares you for
- "How did you evaluate the model?" → core principle first, then the 4 metrics.
- "Sharpe vs Sortino?" → downside deviation distinction.
- "What is walk-forward validation and why not normal cross-validation?" → time ordering, lookahead bias.
- "What would make a backtest misleading?" → leakage, ignored costs, single-regime testing, overfitting.

---

# PART 7: The Interview Playbook

## 7.1 Your 60-second project pitch (rehearse until smooth)

> "Markets move on news before that news is reflected in price history, so I designed a framework that quantifies news sentiment and combines it with market data. It has five stages: I architected data collection using NewsAPI for financial news and Yahoo Finance for prices; the text is cleaned with NLTK and spaCy; each news item is scored by FinBERT — a BERT model domain-adapted for finance — into positive, negative and neutral probabilities; those sentiment scores are temporally aligned with returns, volatility and technical indicators to form multimodal features; and an LSTM models the combined sequence, since price data violates IID assumptions and needs long-memory modeling. Finally, I defined a risk-aware evaluation methodology — walk-forward backtesting with Sharpe, Sortino, CAGR and maximum drawdown — because in finance, accuracy without risk control is meaningless."

## 7.2 The dangerous questions — and honest answers that still impress

**Q: "What accuracy / Sharpe ratio did you achieve?"**
A: "The project's scope was the framework design and evaluation methodology — the report deliberately separates conceptual design from empirical runs, which are the next phase. What I can tell you is exactly HOW I'd judge the results: walk-forward out-of-sample performance, Sharpe above 1 after transaction-cost assumptions, and drawdown stability across regimes — because a backtest number without those controls is meaningless." *(Redirects from a number you don't have to judgment you do have — interviewers respect this.)*

**Q: "Did you write all the code yourself?"**
A: Answer truthfully about your contribution. A safe honest frame: "It was a mentored program — the environment and resource scaffolding were provided, and my work was the design of the pipeline and the analysis in the reports." Never claim code you can't open and explain.

**Q: "Why not just count positive/negative words instead of a huge model?"**
A: Dictionary methods fail on context: "the company *beat* expectations" (positive) vs "shares got *beaten* down" (negative) — same root word. And finance-specific meanings (liability, loss) need domain adaptation. That's the Word2Vec → BERT → FinBERT ladder.

**Q: "Would this actually make money?"**
A: Honest answer: short-horizon sentiment signals decay fast and get arbitraged; the framework is best understood as an information edge that must survive costs, slippage and regime shifts — which is exactly why the evaluation design emphasizes those. "It's a research framework, not a money machine, and I evaluate it like one."

**Q: "What would you improve next?"**
A: Pick two: run the full empirical backtest; try Longformer for full-document sentiment; add social media (Reddit/X) as a second sentiment source; test attention-based time-series models (Transformers) against the LSTM; expand to more tickers and longer history.

## 7.3 Rapid-fire flashcards (cover the right column, quiz yourself)

| Question | Answer in one line |
|---|---|
| Why sentiment at all? | News is leading, price is lagging; investors are emotional (behavioral finance) |
| Why not Word2Vec? | Static vectors can't handle polysemy — "bank" problem |
| What makes BERT special? | Contextual, bidirectional embeddings via self-attention |
| MLM in numbers? | Mask 15% of tokens: 80% [MASK], 10% random, 10% unchanged |
| Why FinBERT? | Domain shift: liability/loss are neutral in finance, negative in general English |
| FinBERT's 3 training stages? | General BERT → financial corpus pre-training (TRC2) → fine-tune on Financial PhraseBank |
| FinBERT output? | P(positive), P(negative), P(neutral) per sentence |
| 512-token problem? | Chunk long docs into 512-token pieces, score each, aggregate (average) |
| Catastrophic forgetting? | Losing old knowledge when training on new data — helpful here for domain focus |
| Why LSTM over RNN? | RNNs suffer vanishing gradients; LSTM's gated additive memory preserves long-term info |
| The 3 LSTM gates? | Forget (erase), Input (write), Output (read) — the market-diary analogy |
| Multimodal features? | Sentiment scores + returns + volatility + technical indicators per time step |
| Walk-forward validation? | Train on past window, test on next, slide forward — prevents lookahead bias |
| Sharpe vs Sortino? | Sharpe penalizes all volatility; Sortino only downside deviation |
| Max drawdown? | Largest peak-to-trough loss — measures worst-case pain/survivability |
| Hardest pipeline step? | Temporal alignment of after-hours news to next trading day + daily aggregation |

## 7.4 Glossary (every term on your resume lines)

- **FinBERT** — BERT further trained on financial text and fine-tuned for financial sentiment classification.
- **LSTM** — recurrent neural network variant with gated memory, built for long sequences.
- **NewsAPI** — web service returning news articles as structured data for programmatic queries.
- **yfinance** — Python library that downloads Yahoo Finance market data (prices, volumes).
- **NLTK / spaCy** — standard Python NLP libraries (tokenization, lemmatization, stopwords).
- **Tokenization** — splitting text into words/subwords.
- **Lemmatization** — reducing words to dictionary base form (surged → surge).
- **Multimodal** — combining more than one data type (text sentiment + numerical prices).
- **Walk-forward backtesting** — sequential train/test evaluation that respects time order.
- **Sharpe / Sortino / CAGR / Max Drawdown** — risk-adjusted return, downside-adjusted return, annualized growth, worst historical loss.
- **Lookahead bias** — accidentally using future information in training; invalidates results.
- **Domain shift** — vocabulary/meaning mismatch between general and specialized text.

---

## Final Checklist Before the Interview

1. ☐ Can you deliver the 60-second pitch without reading?
2. ☐ Can you explain why FinBERT beats BERT with the liability/loss example?
3. ☐ Can you tell the RNN → vanishing gradient → LSTM gates story?
4. ☐ Can you draw the 5-stage pipeline on paper from memory?
5. ☐ Can you define all 4 evaluation metrics and say why walk-forward matters?
6. ☐ Have you prepared the honest "what were your results?" answer (7.2)?
7. ☐ Have you skimmed your own two reports once more the night before?

If all 7 are checked, you can defend every word on those three resume lines. Good luck!
