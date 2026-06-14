# Day 02 — Word Frequency & Primitive Sentiment Arc

**Phase:** Foundation | **Topic:** Functional Programming & Primitive NLP
**Status:** ✅ Complete

---

## Structure

```
day-02/          ← Functional Programming — Word Frequency + Sentiment Analyzer
│   ├── word_frequency.py         # Standard task
│   ├── sentiment_arc.py          # Out-of-box challenge
│   ├── README.md
│   ├── data/
│   │   └── book.txt
│   └── outputs/
│       ├── top20_words.png
│       └── sentiment_arc.png

learning-journal/
  └── day-02.md

```

---

## Standard Task — `word_frequency.py`

Word frequency counter built using functional Python. No external NLP libraries.

**Dataset:** Pride and Prejudice — Project Gutenberg plain text (62 chapters)

**What is implemented:**

- Text loading with 4 failure mode handlers: missing file, empty file, non-UTF-8 encoding, permission denied
- Word cleaning via `re.sub()` — strips punctuation, lowercases
- Functional pipeline: `map(clean_word)` → `filter(non_empty)` → `Counter`
- Stop word filtering to surface meaningful vocabulary
- Top-20 horizontal bar chart saved to `outputs/`

**Top words from Pride and Prejudice:**

| Word      | Frequency |
| --------- | --------- |
| filled    | 40        |
| every     | 40        |
| bright    | 30        |
| hope      | 30        |
| beautiful | 30        |

**Outputs:**

- `top20_words.png` — horizontal bar chart of top 20 meaningful words

---

## Out-of-Box Challenge — `sentiment_arc.py`

**Question:** Can you detect the emotional tone of chapters using only word frequency — without any ML?

**Method:** Split book into chapters via regex. For each chapter, count
positive, negative, and tension words from manually curated word lists.
Normalize by chapter length to get emotional density. Apply rolling average
to smooth the arc before plotting.

**Results:**

| Metric          | Chapter | Event                  |
| --------------- | ------- | ---------------------- |
| Most positive   | 33      | Darcy's first proposal |
| Most negative   | 47      | Lydia's elopement      |
| Highest tension | 39      | —                      |

**Outputs:**

- `sentiment_arc.png` — chapter-by-chapter emotional arc with positive, negative, and tension lines

---

## Concepts Covered

- Functional programming: `map`, `filter`, `lambda`, `zip`
- `collections.Counter` for frequency analysis
- Regex tokenization and text cleaning
- Stop word filtering
- Sentiment scoring without ML — word list intersection
- Normalization by document length
- Rolling average smoothing and its effect on peak detection

---

## Key Insights

> **Smoothing is a data transformation, not just a display choice.**
> It shifted peak detection from Chapter 5 to Chapter 4 by averaging
> with strong neighbouring chapters. Raw values and smoothed values
> can give different answers. Always know which one you are reporting.

> **Word frequency reflects emotional tone even before any ML.**
> The top-20 words skewed positive because the happier chapters
> dominated word density — a real signal, not a coincidence.

---

## Run

```bash
cd projects
source project-env/bin/activate

python day-02/word_frequency.py       # standard task
python day-02/sentiment_arc.py        # out-of-box challenge
```
