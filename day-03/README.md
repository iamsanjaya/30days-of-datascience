# Day 03 — NumPy Distance Metrics from Scratch

**Phase:** Foundation | **Topic:** NumPy — Arrays, Broadcasting, Vectorization
**Status:** ✅ Complete

---

## Structure

```
day-03/
├── numpy_distances.py        # standard task
├── distance_perception.py    # out-of-box challenge
├── README.md
└── outputs/
    ├── distance_metrics.png
    └── distance_perception.png

learning-journal/
  └── day-03.md
```

---

## Standard Task — `numpy_distances.py`

Four core distance and similarity operations implemented from scratch using only NumPy. No sklearn, no scipy.

**What is implemented:**

- Euclidean distance: `√(Σ (aᵢ - bᵢ)²)`
- Cosine similarity: `(a · b) / (‖a‖ · ‖b‖)`
- Row-wise L2 normalization: scales each row to unit norm
- Pearson correlation matrix via centered dot product

All four operations are fully vectorized — zero loops.

**Verification Results:**

| Operation                     | Result     | Verified Against    |
| ----------------------------- | ---------- | ------------------- |
| Euclidean distance            | 5.1962     | `np.linalg.norm` ✅ |
| Row norms after normalization | [1. 1. 1.] | all unit ✅         |
| Max diff — correlation matrix | 1.11e-16   | `np.corrcoef` ✅    |

**Outputs:**

- `outputs/distance_metrics.png` — 3-panel plot: Euclidean vs Cosine scatter, row norms before/after normalization, correlation heatmap

---

## Out-of-Box Challenge — `distance_perception.py`

**Question:** Find two examples where Euclidean and Cosine give OPPOSITE answers about closeness.

### Example 1: Euclidean-Close, Cosine-Far

**Scenario:** Movie rating platform — two users with opposite tastes.

| User | Action | Comedy | Preference   |
| ---- | ------ | ------ | ------------ |
| A    | 1      | 8      | Loves comedy |
| B    | 8      | 1      | Loves action |
| C    | 2      | 9      | Similar to A |

```
A vs B → Euclidean: 9.899  | Cosine: 0.2462  ← OPPOSITE tastes
A vs C → Euclidean: 1.414  | Cosine: 0.9956  ← SAME tastes
```

**The problem:** Euclidean groups A and B together. Cosine correctly separates them.
A recommender using Euclidean would serve Action movies to a Comedy lover.

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

**The problem:** Euclidean groups D+F together because they use the same scale.
Cosine groups D+E together because they have identical preferences.

**Outputs:**

- `outputs/distance_perception.png` — 2-panel plot: 2D vector space (Example 1), bar comparison (Example 2)

---

## Concepts Covered

- Euclidean distance as absolute position in n-dimensional space
- Cosine similarity as angular alignment — magnitude-invariant
- Row-wise L2 normalization and projection onto the unit hypersphere
- Pearson correlation via centered covariance — Bessel's correction (`n-1`)
- Broadcasting and `np.outer` for fully vectorized matrix operations
- When magnitude is information vs when magnitude is noise
- Why word embeddings (Word2Vec, BERT) use cosine, not Euclidean

---

## Key Insights

> **Euclidean measures how far. Cosine measures which direction.**
> Choosing the wrong metric is a silent bug — the model trains without error
> but recommends the wrong items, clusters the wrong users, or retrieves
> the wrong documents.

> **Rule of thumb:** if normalization would improve your results,
> you likely need cosine. If absolute scale carries meaning, use Euclidean.

---

## Run

```bash
cd day-03
source ../.venv/bin/activate

python numpy_distances.py       # standard task
python distance_perception.py   # out-of-box challenge
```
