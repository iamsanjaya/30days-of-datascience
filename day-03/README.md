# Day 03 — NumPy Distance Metrics from Scratch

> **Phase 1 — Foundation** | 30-Day DS/ML Internship Roadmap
> **Priority:** CRITICAL

---

## What I Built

Two scripts implementing core distance and similarity metrics from scratch using only NumPy — no sklearn, no scipy.

| Script                   | Description                                                       |
| ------------------------ | ----------------------------------------------------------------- |
| `numpy_distances.py`     | Standard task — 4 distance/similarity operations + visualizations |
| `distance_perception.py` | Out-of-box challenge — "Distance is Perception"                   |

---

## Standard Task — `numpy_distances.py`

### Implementations (zero loops, fully vectorized)

**1. Euclidean Distance**

```
√(Σ (aᵢ - bᵢ)²)
```

Measures straight-line distance in n-dimensional space. Sensitive to magnitude — large values dominate.

**2. Cosine Similarity**

```
(a · b) / (‖a‖ · ‖b‖)
```

Measures the angle between two vectors, ignoring magnitude. Range: `[-1, 1]`.

**3. Row-wise Normalization**

Scales each row to unit L2 norm (norm = 1). Equivalent to projecting onto the unit hypersphere. Used before cosine similarity comparisons.

**4. Correlation Matrix (Pearson)**

Manual implementation via:

1. Center each column (subtract mean)
2. Compute covariance matrix via dot product
3. Divide by outer product of standard deviations

### Verification Results

```
Euclidean Distance:  5.1962   ✓ matches np.linalg.norm
Cosine Similarity:   0.9746
Row norms after normalization: [1. 1. 1.]  ✓ all unit
Max absolute difference (manual vs np.corrcoef): 1.11e-16  ✓ floating point noise
```

### Output

![Distance Metrics](plots/distance_metrics.png)

---

## Out-of-Box Challenge — "Distance is Perception"

> _Find two examples where Euclidean and Cosine give OPPOSITE answers about closeness._

### Example 1: Euclidean-Close, Cosine-Far

**Scenario:** Movie rating platform. Two users with opposite tastes.

| User | Action | Comedy | Preference   |
| ---- | ------ | ------ | ------------ |
| A    | 1      | 8      | Loves comedy |
| B    | 8      | 1      | Loves action |
| C    | 2      | 9      | Similar to A |

```
A vs B → Euclidean: 9.899  | Cosine: 0.2462  ← OPPOSITE tastes
A vs C → Euclidean: 1.414  | Cosine: 0.9956  ← SAME tastes
```

**The problem:** Euclidean would group A and B together. Cosine correctly separates them. A recommender using Euclidean would serve Action movies to a Comedy lover.

---

### Example 2: Euclidean-Far, Cosine-Close

**Scenario:** A harsh critic vs a generous rater — same preferences, different scales.

| User | Ratings         | Style                 |
| ---- | --------------- | --------------------- |
| D    | [2, 3, 1, 2, 3] | Harsh critic          |
| E    | [8, 9, 7, 8, 9] | Generous rater        |
| F    | [2, 1, 3, 2, 1] | Different preferences |

```
D vs E → Euclidean: 13.416 (far)  | Cosine: 0.9721 (same taste) ✓
D vs F → Euclidean:  3.464 (close)| Cosine: 0.7506 (diff taste) ✗
```

**The problem:** Euclidean groups D+F together because they use the same scale. Cosine groups D+E together because they have identical preferences. Euclidean would recommend the wrong movies.

---

### Why This Matters for ML — 10-Line Explanation

1. Distance metrics are not neutral — they embed assumptions about what "similar" means. Choosing the wrong one is a silent bug.
2. Euclidean distance measures **absolute position**. It answers: _how much?_
3. Cosine similarity measures **angular alignment**. It answers: _which direction?_
4. When **magnitude** matters (house size, transaction amount), use Euclidean.
5. When **direction** matters (user taste, document topic, word meaning), use Cosine.
6. A movie recommender should ignore whether a user rates 1–3 or 1–10. Cosine does. Euclidean doesn't.
7. A fraud detector cares about absolute transaction amounts. Euclidean (or scaled equivalent) is correct there.
8. KNN and K-Means default to Euclidean. Changing the metric is often more impactful than changing the model.
9. Word embeddings (Word2Vec, BERT) always use cosine similarity — two synonyms may have different L2 norms but near-identical directions.
10. **Rule of thumb:** if normalization would help, use cosine. If absolute scale matters, use Euclidean.

### Output

![Distance Perception](plots/distance_perception.png)

---

## Folder Structure

```
day-03/
├── numpy_distances.py        # Standard task
├── distance_perception.py         # Out-of-box challenge
├── README.md
└── outputs/
    ├── distance_metrics.png
    └── distance_perception.png
```

---

## Milestone Checklist

- [x] All 4 operations implemented with zero loops
- [x] Results verified against NumPy/SciPy (max diff: 1.11e-16)
- [x] Distance paradox examples are mathematically correct
- [x] Written explanation is clear to a non-technical person
- [x] Visualizations saved to `plots/`
- [x] GitHub commit: `day-03: numpy distance metrics from scratch`

---

## Key Takeaway

> Most DS candidates can _run_ distance metrics. Almost none can explain _when each one lies_ — and build examples that prove it.
