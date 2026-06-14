# Day 07 — Visualization Storytelling

**Dataset:** NYC TLC Green Taxi — Jan/Feb 2023 (`cleaned.parquet` from Day 06)
**Theme:** Every chart is an argument. Every title is a conclusion.

---

## File Structure

```
day-07/
├── config.py          # paths, titles, plot style — single source of truth
├── 01_charts.py       # 6-chart EDA visual report
├── 02_deception.py    # honest vs misleading chart pair (out-of-box challenge)
├── outputs/           # saved PNGs — gitignored
│   ├── 01_hourly_demand.png
│   ├── 02_fare_vs_distance.png
│   ├── 03_payment_tip.png
│   ├── 04_tip_by_dow.png
│   ├── 05_vendor_peak.png
│   ├── 06_fare_distribution.png
│   └── 07_deception_pair.png
└── README.md

learning-journal/
  └── day-07.md
```

---

## Standard Task — 6-Chart EDA Report

Each chart answers one specific question. Title = conclusion.

| Chart | Question                                           | Conclusion                                                   |
| ----- | -------------------------------------------------- | ------------------------------------------------------------ |
| 01    | Where is demand concentrated through the day?      | Demand peaks at 8 AM and 6 PM — midday and night are quiet   |
| 02    | Do longer trips earn proportionally more per mile? | Fare grows with distance, but short trips earn more per mile |
| 03    | How has payment type shifted Jan → Feb?            | Card payments dominate — cash usage fell further in February |
| 04    | Are tips a weekend phenomenon?                     | Weekend tips are higher and more variable than weekday tips  |
| 05    | How does vendor volume split at peak vs off-peak?  | Vendor 2 handles more peak-hour volume than Vendor 1         |
| 06    | What does the full fare distribution look like?    | Most fares cluster under $20 — high-fare outliers are rare   |

---

## Out-of-Box Challenge — Lie With Charts (Ethically)

**Target:** Chart 01 (hourly demand).

**The deception:** Truncate the Y-axis to start near the data minimum (~92% of the lowest value). A modest ~15% rush-hour variation visually reads as a dramatic surge.

**Why this matters:** The truncated Y-axis is the most frequently deployed deception in corporate dashboards, media charts, and political reporting. It is rarely accidental — it amplifies small differences into apparent crises or victories. Recognizing it takes seconds once you know to check where the Y-axis starts.

**Honest version:** Y-axis starts at zero. Same data. Accurate impression.

See: `outputs/07_deception_pair.png`

---

## How to Run

```bash
# From repo root
python day-07/01_charts.py
python day-07/02_deception.py
```

Charts are saved to `day-07/outputs/` at 300 DPI.

---

## Key Design Decisions

- **`config.py` as single source of truth:** Every path, title string, and style setting lives here. Changing a title in one place updates all charts. No magic strings buried inside functions.
- **Y-axis at zero by default:** `ax.set_ylim(0)` is explicit in every time-series and bar chart. Not relying on Matplotlib's default which can silently truncate.
- **Sampling for scatter plots:** 8,000-point sample with `alpha=0.15` and `rasterized=True` prevents overplotting without losing distribution shape.
- **Tip chart filtered to card payments:** TLC only records tip amounts for card transactions — cash tips are untracked. Showing all payment types would understate tips deceptively.
- **`save_chart()` helper:** Consistent DPI, `bbox_inches="tight"`, and `plt.close()` in one place. No forgotten `plt.show()` calls in production scripts.
