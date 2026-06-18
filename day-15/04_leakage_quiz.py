# 04_leakage_quiz.py — Day 15: Out-of-Box Challenge
# "What Leaks?"
#
# 5 self-contained snippets — some leaky, some clean.
# All use Ames Housing data (no external downloads).
# Each snippet: verdict printed, exact leak location annotated, fix shown.
#
# Leak taxonomy:
#   TYPE-A: Scaler/imputer fit on full data before split
#   TYPE-B: Target encoding computed on full data before split
#   TYPE-C: Future information encoded in a feature
#   TYPE-D: Test rows contaminate cross-validation preprocessing
#   CLEAN : No leakage — correct pattern

# %%
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score

from config import DATA_PATH, TARGET, DROP_COLS, NUMERIC_FEATURES, OUTPUT_DIR

rng = np.random.default_rng(42)
os.makedirs(OUTPUT_DIR, exist_ok=True)
# ── Load Ames Housing (numeric cols only for leakage demos) ───────────────────
df_full = pd.read_csv(DATA_PATH).drop(columns=DROP_COLS)
X_all = df_full[NUMERIC_FEATURES].copy()
y_all = df_full[TARGET].copy()

# Fill numeric NaNs with column median so snippets focus on leakage, not imputation
X_all_clean = X_all.fillna(X_all.median(numeric_only=True))

# ─────────────────────────────────────────────────────────────────────────────
# SNIPPET 1 — TYPE-A LEAK: Scaler fit on entire X before split
# ─────────────────────────────────────────────────────────────────────────────
# %%
print("=" * 60)
print("SNIPPET 1 — Verdict: LEAKY  (TYPE-A)")
print("=" * 60)

# ── LEAKY version ──
scaler_leak = StandardScaler()
X_scaled_leak = scaler_leak.fit_transform(X_all_clean)  # ← LEAK line
X_tr_l, X_te_l, y_tr_l, y_te_l = train_test_split(
    X_scaled_leak, y_all, test_size=0.2, random_state=42
)
score_leak = Ridge().fit(X_tr_l, y_tr_l).score(X_te_l, y_te_l)

# ── CLEAN version ──
X_tr_c, X_te_c, y_tr_c, y_te_c = train_test_split(
    X_all_clean, y_all, test_size=0.2, random_state=42
)
scaler_clean = StandardScaler()
X_tr_scaled = scaler_clean.fit_transform(X_tr_c)  # fit on train only
X_te_scaled = scaler_clean.transform(X_te_c)  # transform test
score_clean = Ridge().fit(X_tr_scaled, y_tr_c).score(X_te_scaled, y_te_c)

print("Leak location : scaler.fit_transform(X_all_clean) before split")
print(
    "Why it leaks  : scaler learns mean/std from test rows → test is not truly unseen"
)
print(f"Leaky R²  : {score_leak:.6f}")
print(f"Clean R²  : {score_clean:.6f}")
print(f"Inflation : {score_leak - score_clean:+.6f}")
print("Fix       : split FIRST, then fit scaler on X_train only\n")

# ─────────────────────────────────────────────────────────────────────────────
# SNIPPET 2 — CLEAN: Full sklearn Pipeline — split-safe by construction
# ─────────────────────────────────────────────────────────────────────────────
# %%
print("=" * 60)
print("SNIPPET 2 — Verdict: CLEAN")
print("=" * 60)

X_tr2, X_te2, y_tr2, y_te2 = train_test_split(
    X_all_clean, y_all, test_size=0.2, random_state=42
)

pipe2 = Pipeline(
    [
        ("scaler", StandardScaler()),
        ("model", Ridge()),
    ]
)
pipe2.fit(X_tr2, y_tr2)  # scaler sees only X_tr2
score_pipe = pipe2.score(X_te2, y_te2)

print("Why it's clean: Pipeline.fit() fits each step on X_tr2 only.")
print("                transform() is then applied to X_te2 without re-fitting.")
print(f"Pipeline R²   : {score_pipe:.6f}")
print("Fix       : N/A — this is the correct pattern\n")

# ─────────────────────────────────────────────────────────────────────────────
# SNIPPET 3 — TYPE-B LEAK: Target encoding computed on full data before split
# ─────────────────────────────────────────────────────────────────────────────
# %%
print("=" * 60)
print("SNIPPET 3 — Verdict: LEAKY  (TYPE-B)")
print("=" * 60)

df3 = df_full[["Neighborhood", TARGET]].copy()  # type: ignore[index]

# ── LEAKY: compute neighborhood mean on full df3 before split ──
target_map_leak = df3.groupby("Neighborhood")[TARGET].mean()  # ← LEAK line
df3["nbhd_encoded"] = df3["Neighborhood"].map(target_map_leak)

X3 = df3[["nbhd_encoded"]].fillna(df3["nbhd_encoded"].median())
y3 = df3[TARGET]
X3_tr, X3_te, y3_tr, y3_te = train_test_split(X3, y3, test_size=0.2, random_state=42)
score_te_leak = Ridge().fit(X3_tr, y3_tr).score(X3_te, y3_te)

# ── CLEAN: compute neighborhood mean on train rows only ──
X3_tr_raw, X3_te_raw, y3_tr2, y3_te2 = train_test_split(
    df3[["Neighborhood"]], y3, test_size=0.2, random_state=42
)
train_map = X3_tr_raw.join(y3_tr2).groupby("Neighborhood")[TARGET].mean()
X3_tr_enc = (
    X3_tr_raw["Neighborhood"]
    .map(train_map)
    .fillna(train_map.mean())
    .values.reshape(-1, 1)
)
X3_te_enc = (
    X3_te_raw["Neighborhood"]
    .map(train_map)
    .fillna(train_map.mean())
    .values.reshape(-1, 1)
)
score_te_clean = Ridge().fit(X3_tr_enc, y3_tr2).score(X3_te_enc, y3_te2)

print("Leak location : groupby('Neighborhood')[TARGET].mean() on full df3")
print("Why it leaks  : test-row sale prices influence the neighborhood encoding")
print("                the model indirectly 'knows' test targets during training")
print(f"Leaky R²  : {score_te_leak:.6f}")
print(f"Clean R²  : {score_te_clean:.6f}")
print(f"Inflation : {score_te_leak - score_te_clean:+.6f}")
print("Fix       : compute target encoding on training fold rows only\n")

# ─────────────────────────────────────────────────────────────────────────────
# SNIPPET 4 — TYPE-C LEAK: Future sale information used as feature
# ─────────────────────────────────────────────────────────────────────────────
# %%
print("=" * 60)
print("SNIPPET 4 — Verdict: LEAKY  (TYPE-C)")
print("=" * 60)

# Ames has Yr Sold + Mo Sold — simulate a time-ordered prediction scenario.
# Suppose you're predicting SalePrice for month T using neighbourhood
# average price. If you include the NEXT month's avg price as a feature,
# that is a future-information leak.
df4 = df_full[["Yr Sold", "Mo Sold", "Neighborhood", TARGET]].copy()  # type: ignore[index]
df4 = df4.sort_values(["Yr Sold", "Mo Sold"]).reset_index(drop=True)

# ── LEAKY: shift(-1) → next month's mean price ──
df4["next_month_nbhd_avg"] = df4.groupby("Neighborhood")[TARGET].transform(
    lambda s: s.shift(-1)
)  # ← LEAK line

print("Leak location : groupby('Neighborhood')[TARGET].transform(shift(-1))")
print("Why it leaks  : for a house sold in month T, the feature contains")
print("                the sale price of the NEXT house sold in that neighborhood")
print("                — information unavailable at actual prediction time")
print("Fix           : use shift(+1) (previous sale) or a rolling lag window\n")

# ─────────────────────────────────────────────────────────────────────────────
# SNIPPET 5 — TYPE-D LEAK: Preprocessing outside pipeline inside CV loop
# ─────────────────────────────────────────────────────────────────────────────
# %%
print("=" * 60)
print("SNIPPET 5 — Verdict: LEAKY  (TYPE-D)")
print("=" * 60)

# ── LEAKY: scale entire X_all_clean BEFORE passing to cross_val_score ──
scaler5 = StandardScaler()
X_prescaled = scaler5.fit_transform(X_all_clean)  # ← LEAK line

# cross_val_score now operates on already-scaled data whose mean/std came
# from ALL rows including each fold's validation set.
scores_leak5 = cross_val_score(Ridge(), X_prescaled, y_all, cv=5, scoring="r2")

# ── CLEAN: pass raw X inside a Pipeline to cross_val_score ──
pipe5 = Pipeline([("scaler", StandardScaler()), ("model", Ridge())])
scores_clean5 = cross_val_score(pipe5, X_all_clean, y_all, cv=5, scoring="r2")

print(
    "Leak location : StandardScaler().fit_transform(X_all_clean) before cross_val_score"
)
print("Why it leaks  : each fold's validation rows contributed to the scaler's")
print("                mean/std → validation is not truly unseen per fold")
print(f"Leaky CV R²  : {scores_leak5.mean():.6f}  ± {scores_leak5.std():.6f}")
print(f"Clean  CV R² : {scores_clean5.mean():.6f}  ± {scores_clean5.std():.6f}")
print(f"Inflation    : {scores_leak5.mean() - scores_clean5.mean():+.6f}")
print("Fix       : pass a Pipeline object to cross_val_score — never pre-scale\n")

# ── Summary ───────────────────────────────────────────────────────────────────
# %%
print("=" * 60)
print("QUIZ SUMMARY")
print("=" * 60)
rows = [
    ("1", "LEAKY", "TYPE-A", "Scaler fit on full X before train/test split"),
    ("2", "CLEAN", "—", "Pipeline: each step fit on X_train only"),
    ("3", "LEAKY", "TYPE-B", "Target encoding uses test-row prices"),
    ("4", "LEAKY", "TYPE-C", "shift(-1) encodes next sale — future info"),
    ("5", "LEAKY", "TYPE-D", "Pre-scaled X passed to cross_val_score"),
]
for num, verdict, leak_type, note in rows:
    print(f"  Snippet {num}: {verdict:<6}  {leak_type:<8}  {note}")

# ── Plot: Leakage inflation summary ──────────────────────────────────────────
# %%
snippets = [
    "S1\nTYPE-A\nScaler\nbefore split",
    "S2\nCLEAN\nPipeline",
    "S3\nTYPE-B\nTarget\nencoding",
    "S4\nTYPE-C\nFuture\nshift",
    "S5\nTYPE-D\nPre-scale\nin CV",
]
inflations = [
    score_leak - score_clean,
    0.0,
    score_te_leak - score_te_clean,
    0.0,  # TYPE-C — no R² measured (time-series construction)
    scores_leak5.mean() - scores_clean5.mean(),
]
colors = ["#d9534f" if v > 0 else "#6aaa64" for v in inflations]

fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.bar(snippets, inflations, color=colors)
ax.axhline(0, color="black", linewidth=0.8)
ax.set_ylabel("R² Inflation (leaky − clean)")
ax.set_title(
    "Data Leakage Quiz — Measured R² Inflation per Snippet\n"
    "(red = leaky · green = clean · TYPE-C not quantified)"
)
for bar, val in zip(bars, inflations):
    if val != 0.0:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.0002,
            f"{val:+.4f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "04_leakage_inflation.png"), dpi=150)
plt.close()
print("\nSaved: outputs/04_leakage_inflation.png")

print("\n04_leakage_quiz.py complete.")
