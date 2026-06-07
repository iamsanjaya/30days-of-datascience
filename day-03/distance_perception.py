# ============================================================
# DAY 3 — Out-of-Box Challenge: "Distance is Perception"
# ============================================================
# Two concrete examples where Euclidean and Cosine metrics
# give OPPOSITE answers about closeness.
#
# Then: a 10-line written explanation of why this matters
# for ML — specifically recommendation systems.
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

np.random.seed(42)


#  Helper Functions
def euclidean(a, b):
    return np.sqrt(np.sum((a - b) ** 2))


def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


# ================================================================
# EXAMPLE 1: Euclidean-CLOSE but Cosine-FAR
# ================================================================
# Scenario: Two users on a movie rating platform.
# Both gave very similar ABSOLUTE ratings, but their DIRECTIONS differ.
#
# User A: rated Action=1, Comedy=8  → loves comedy, dislikes action
# User B: rated Action=8, Comedy=1  → loves action, dislikes comedy
#
# Euclidean distance is small because |1-8| + |8-1| is the only spread
# over a 2D space — but their TASTES are opposite directions.
# ────────────────────────────────────────────────────────────────

user_A1 = np.array([1.0, 8.0])  # (Action=1, Comedy=8)
user_B1 = np.array([8.0, 1.0])  # (Action=8, Comedy=1)

# For contrast: a user who is truly "close" in direction to A
user_C1 = np.array([2.0, 9.0])  # (Action=2, Comedy=9) — same taste profile as A

ed_AB1 = euclidean(user_A1, user_B1)
cs_AB1 = cosine_sim(user_A1, user_B1)
ed_AC1 = euclidean(user_A1, user_C1)
cs_AC1 = cosine_sim(user_A1, user_C1)

print("=" * 60)
print("EXAMPLE 1: Euclidean-CLOSE, Cosine-FAR")
print("=" * 60)
print(f"User A (likes comedy): {user_A1}")
print(f"User B (likes action): {user_B1}")
print(f"User C (very similar to A): {user_C1}\n")
print(f"A vs B  → Euclidean: {ed_AB1:.3f}  |  Cosine: {cs_AB1:.4f}")
print(f"A vs C  → Euclidean: {ed_AC1:.3f}  |  Cosine: {cs_AC1:.4f}")
print()
print("  Euclidean says A and B are just as 'close' as A and C.")
print("   Cosine correctly identifies B as OPPOSITE in taste to A.")
print("   Euclidean would give A and B the SAME recommendations. WRONG.\n")


# ================================================================
# EXAMPLE 2: Euclidean-FAR but Cosine-CLOSE
# ================================================================
# Scenario: Two users who rate everything consistently,
# but one is a "harsh critic" and the other is "generous".
#
# User D: rated all movies 1–3 (harsh)   → [2, 3, 1, 2, 3]
# User E: rated all movies 7–9 (generous)→ [8, 9, 7, 8, 9]
#
# Euclidean distance is HUGE (absolute scale differs massively).
# But their PREFERENCES are identical — same rank order, same direction.
# Cosine similarity is ~1 (pointing in the same direction).
# ────────────────────────────────────────────────────────────────

user_D = np.array([2.0, 3.0, 1.0, 2.0, 3.0])  # harsh critic
user_E = np.array([8.0, 9.0, 7.0, 8.0, 9.0])  # generous rater

# A truly different user for contrast
user_F = np.array([2.0, 1.0, 3.0, 2.0, 1.0])  # different preference order

ed_DE = euclidean(user_D, user_E)
cs_DE = cosine_sim(user_D, user_E)
ed_DF = euclidean(user_D, user_F)
cs_DF = cosine_sim(user_D, user_F)

print("=" * 60)
print("EXAMPLE 2: Euclidean-FAR, Cosine-CLOSE")
print("=" * 60)
print(f"User D (harsh, same prefs as E): {user_D}")
print(f"User E (generous, same prefs as D): {user_E}")
print(f"User F (different preferences): {user_F}\n")
print(f"D vs E  → Euclidean: {ed_DE:.3f}  |  Cosine: {cs_DE:.4f}")
print(f"D vs F  → Euclidean: {ed_DF:.3f}  |  Cosine: {cs_DF:.4f}")
print()
print("  Euclidean says D and F are similar (small distance).")
print("   Cosine correctly identifies D and E as having identical tastes.")
print("   → Euclidean would recommend the WRONG user's movies to D.\n")


# ================================================================
# WHY THIS MATTERS FOR ML: 10-LINE EXPLANATION
# ================================================================
explanation = """
WHY THIS MATTERS FOR MACHINE LEARNING
══════════════════════════════════════════════════════════════

1.  Distance metrics are not neutral — they embed assumptions about
    what "similar" means. Choosing the wrong one is a silent bug.

2.  Euclidean distance measures absolute position in feature space.
    It treats "how much" as the primary question.

3.  Cosine similarity measures angular alignment — the direction
    of a vector. It treats "which way" as the primary question.

4.  When MAGNITUDE encodes important information (e.g. house size in
    square feet matters absolutely), use Euclidean.

5.  When DIRECTION encodes the meaning (e.g. user taste profiles,
    document topics, word embeddings), use Cosine.

6.  In a movie recommendation system, whether a user rates on a
    1-3 scale or 1-10 scale is irrelevant to their preferences.
    Cosine eliminates this scale bias. Euclidean does not.

7.  In a fraud detection system, the absolute transaction amount
    matters enormously. Euclidean distance (or scaled equivalent)
    is appropriate there.

8.  KNN, K-Means, and most distance-based models default to Euclidean.
    Changing the metric is often more impactful than changing the model.

9.  This is why word embeddings (Word2Vec, BERT) are always compared
    using cosine similarity, not Euclidean distance. Two synonyms
    might have very different L2 norms but near-identical directions.

10. Rule of thumb: if you would expect normalization to improve results,
    you likely need cosine. If absolute scale matters, use Euclidean.
"""
print(explanation)


# ================================================================
# VISUALIZATION
# ================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle(
    "Day 3 — Distance is Perception: When Metrics Disagree",
    fontsize=13,
    fontweight="bold",
)

# ── Plot 1: Example 1 in 2D vector space ──
ax = axes[0]
origin = np.array([0, 0])


def draw_vector(ax, vec, color, label, lw=2.5):
    ax.annotate(
        "", xy=vec, xytext=origin, arrowprops=dict(arrowstyle="->", color=color, lw=lw)
    )
    ax.text(
        vec[0] + 0.2, vec[1] + 0.2, label, color=color, fontsize=11, fontweight="bold"
    )


draw_vector(ax, user_A1, "steelblue", "A\n(Comedy lover)")
draw_vector(ax, user_B1, "tomato", "B\n(Action lover)")
draw_vector(ax, user_C1, "seagreen", "C\n(Like A)")

ax.set_xlim(-1, 11)
ax.set_ylim(-1, 11)
ax.set_xlabel("Action Rating")
ax.set_ylabel("Comedy Rating")
ax.set_title(
    "Example 1: Euclidean-Close, Cosine-Far\n"
    f"A↔B: Euclidean={ed_AB1:.1f}, Cosine={cs_AB1:.3f}\n"
    f"A↔C: Euclidean={ed_AC1:.1f}, Cosine={cs_AC1:.3f}",
    fontsize=10,
)
ax.grid(True, alpha=0.3)
ax.axhline(0, color="gray", lw=0.8)
ax.axvline(0, color="gray", lw=0.8)

# Add annotation: angle between A and B
theta = np.linspace(0, np.arccos(cs_AB1), 50)
r = 2.0
ax.plot(
    r * np.cos(theta) * user_A1[0] / np.linalg.norm(user_A1),
    r * np.sin(theta) * user_A1[1] / np.linalg.norm(user_A1),
    color="gray",
    lw=1.2,
    linestyle="--",
    alpha=0.7,
)

note_ab = (
    f"A vs B\nEuclidean: {ed_AB1:.1f} ← looks close\n"
    f"Cosine: {cs_AB1:.3f} ← but tastes are OPPOSITE"
)
note_ac = (
    f"A vs C\nEuclidean: {ed_AC1:.1f} ← slightly further\n"
    f"Cosine: {cs_AC1:.3f} ← but same taste!"
)
ax.text(
    0.5,
    10.0,
    note_ab,
    fontsize=7.5,
    color="tomato",
    bbox=dict(facecolor="white", alpha=0.8, boxstyle="round,pad=0.3"),
)
ax.text(
    5.5,
    3.5,
    note_ac,
    fontsize=7.5,
    color="seagreen",
    bbox=dict(facecolor="white", alpha=0.8, boxstyle="round,pad=0.3"),
)

# ── Plot 2: Example 2 as bar comparison ──
ax2 = axes[1]
movies = ["Film 1", "Film 2", "Film 3", "Film 4", "Film 5"]
x = np.arange(len(movies))
width = 0.25

bars_D = ax2.bar(
    x - width,
    user_D,
    width,
    label=f"User D (harsh)\nMean={user_D.mean():.1f}",
    color="steelblue",
    alpha=0.8,
)
bars_E = ax2.bar(
    x,
    user_E,
    width,
    label=f"User E (generous)\nMean={user_E.mean():.1f}",
    color="seagreen",
    alpha=0.8,
)
bars_F = ax2.bar(
    x + width,
    user_F,
    width,
    label=f"User F (diff. prefs)\nMean={user_F.mean():.1f}",
    color="tomato",
    alpha=0.8,
)

ax2.set_xticks(x)
ax2.set_xticklabels(movies)
ax2.set_ylabel("Rating")
ax2.set_ylim(0, 11)
ax2.set_title(
    "Example 2: Euclidean-Far, Cosine-Close\n"
    f"D↔E: Euclidean={ed_DE:.1f} (far), Cosine={cs_DE:.4f} (same taste)\n"
    f"D↔F: Euclidean={ed_DF:.1f} (close), Cosine={cs_DF:.4f} (diff taste)",
    fontsize=10,
)
ax2.legend(fontsize=8)
ax2.grid(axis="y", alpha=0.3)

summary_text = (
    "Euclidean groups D+F together (similar scale).\n"
    "Cosine groups D+E together (same preferences).\n"
    "→ For recommendations, COSINE wins here."
)
ax2.text(
    0.02,
    0.02,
    summary_text,
    transform=ax2.transAxes,
    fontsize=8,
    color="black",
    bbox=dict(facecolor="lightyellow", alpha=0.9, boxstyle="round,pad=0.4"),
)

plt.tight_layout()

output_path = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(output_path, exist_ok=True)
plt.savefig(
    os.path.join(output_path, "distance_perception.png"),
    dpi=150,
    bbox_inches="tight",
)
print("Out-of-box output saved: outputs/distance_perception.png")
