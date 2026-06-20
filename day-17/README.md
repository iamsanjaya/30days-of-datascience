# Day 17 — Model Card + Robustness Stress Test

**Subject model:** Day 15's Ames Housing XGBoost pipeline, rebuilt exactly as
it stands (raw Ames features only — Day 14's engineered features were never
folded in; see `MODEL_CARD.md` §5 for why that matters).

## Why this differs slightly from the roadmap's literal wording

The roadmap says "take your best model from Days 8–16." The actual best
_validated_ artifact is Day 15's pipeline as it was coded — not a merged
"Day 14 features + Day 15 tuning" version, since that combination was never
fit. Rebuilding it exactly as it stands keeps the model card honest about
what was actually built, rather than describing a model that doesn't exist.
Day 15 never persisted its winning grid-search hyperparameters, so
`01_rebuild_best_pipeline.py` re-runs the same 12-combo grid to rediscover
them before freezing the pipeline.

## File Structure

```
day-17/
├── 01_rebuild_best_pipeline.py      # Rediscovers Day 15's grid-search params (never persisted),
│                                    # freezes pipeline + test split to models/
├── 02_distribution_shift_test.py    # Old vs. new homes within the held-out test set (1960 split)
├── 03_noise_injection_test.py      # 20% Gaussian noise on numeric test features
├── 04_label_corruption_test.py     # 10% of test labels shuffled
├── 05_robustness_aggregator.py     # Combines 02-04 into one table, drafts ROBUSTNESS_REPORT.md
├── 06_visualize_results.py         # Saves robustness_comparison.png + predicted_vs_actual_by_era.png
├── config.py                       # Paths, rediscovered grid, exact stress-test parameters
├── README.md                       # Day-17 summary, run order, results
├── MODEL_CARD.md                   # Six-section model card (problem, data, architecture, metrics+CI, failure modes, ethics)
├── ROBUSTNESS_REPORT.md            # Auto-generated results table + hand-written 5-sentence reflection
├── models/                         # .joblib artifacts (gitignored)
│   ├── best_pipeline.joblib
│   └── test_split.joblib
└── outputs/                       # JSON results + plots (gitignored)
    ├── baseline_metrics.json
    ├── best_params.json
    ├── distribution_shift_results.json
    ├── label_corruption_results.json
    ├── noise_injection_results.json
    └── plots/
        ├── predicted_vs_actual_by_era.png
        └── robustness_comparison.png

learning-journal/
└── day-17.md

```

## Run order

```bash
# from the repo root
python day-17/01_rebuild_best_pipeline.py     # rediscovers best params, freezes pipeline + test split
python day-17/02_distribution_shift_test.py   # old vs. new homes within the held-out test set
python day-17/03_noise_injection_test.py      # 20% Gaussian noise on numeric test features
python day-17/04_label_corruption_test.py     # 10% of test labels shuffled
python day-17/05_robustness_aggregator.py     # builds the results table in ROBUSTNESS_REPORT.md
python day-17/06_visualize_results.py         # saves plots to outputs/plots/
```

Each stress test script depends on `models/best_pipeline.joblib` and
`models/test_split.joblib` from script 01 — run in order.

`outputs/` holds JSON metrics/results; `models/` holds the `.joblib`
artifacts — kept separate so binaries don't clutter the readable results.

## Files

| File                            | Purpose                                                                                                                         |
| ------------------------------- | ------------------------------------------------------------------------------------------------------------------------------- |
| `config.py`                     | Paths, rediscovered grid, exact stress-test parameters (20% noise, 10% label corruption, 1960 build-era split)                  |
| `01_rebuild_best_pipeline.py`   | Cross-day import of Day 15's `pipeline_factory.py` via config-swap; re-runs grid search; persists pipeline, test split, metrics |
| `02_distribution_shift_test.py` | Old-vs-new-home slice comparison within the held-out test set                                                                   |
| `03_noise_injection_test.py`    | Feature-noise stress test                                                                                                       |
| `04_label_corruption_test.py`   | Label-corruption stress test                                                                                                    |
| `05_robustness_aggregator.py`   | Combines 02–04 into one table, drafts `ROBUSTNESS_REPORT.md`                                                                    |
| `06_visualize_results.py`       | Saves `outputs/plots/robustness_comparison.png` and `outputs/plots/predicted_vs_actual_by_era.png`                              |
| `MODEL_CARD.md`                 | Six-section model card (problem, data, architecture, metrics+CI, failure modes, ethics)                                         |
| `ROBUSTNESS_REPORT.md`          | Auto-generated results table + hand-written 5-sentence reflection                                                               |

## Results (actual run, 2026)

- **Best params found:** `learning_rate=0.1, max_depth=3, n_estimators=300`
- **Baseline test RMSE:** $25,121 (95% CI: $19,966–$30,435), RMSLE 0.109, R² 0.921, n=586
- **Distribution shift:** old homes (pre-1960) RMSE $18,204 / RMSLE 0.134 vs.
  new homes (1960+) RMSE $27,543 / RMSLE 0.096 — RMSE and RMSLE disagree on
  which slice is "better" (see `MODEL_CARD.md` §5 for why)
- **20% feature noise:** RMSE degraded +25.1%
- **10% label corruption:** RMSE degraded +110.8% — by far the most damaging
  stress test, ~4x worse than feature noise
