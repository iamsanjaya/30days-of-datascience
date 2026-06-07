# Day 05 — Pandas EDA: Titanic Dataset

## Objective

Full exploratory data analysis on the Titanic dataset answering 5 business questions with data and visualizations, plus an out-of-box finding that goes beyond standard Titanic analyses.

---

## Dataset

- **Source:** [Kaggle — Titanic: Machine Learning from Disaster](https://www.kaggle.com/competitions/titanic)
- **File:** `data/titanic_uncleaned.csv`
- **Shape:** 891 rows × 12 columns
- **Key missing values:** Cabin (77.1%), Age (19.9%), Embarked (0.2%)

---

## Questions Answered

### Q1 — Which attributes correlate most strongly with survival?

Sex is the strongest predictor (r = 0.54), nearly twice as strong as Pclass (r = -0.34). Fare correlates positively (r = 0.26), reflecting its overlap with class. Age and SibSp are weak predictors in isolation.

### Q2 — Do deck/cabin patterns differ from Pclass survival?

Decks B, D, and E all outperform overall 1st class survival (75% vs 63%). Physical location on the ship — proximity to lifeboats — added survival advantage beyond what ticket price alone explains. Notable: 77% of Cabin data is missing, and that missingness itself correlates with lower survival (unknown cabin → likely 3rd class).

### Q3 — Did family size help or hurt survival odds?

Non-linear relationship. Solo travelers survived at 30.4%, small families (2–4) at 57.9%, and large families (5+) at only 16.1%. Family size 4 was the sweet spot at 72.4%. Large families likely couldn't coordinate evacuation fast enough.

### Q4 — Was there a "golden age" range with higher survival rates?

Children (0–12) had the highest survival at 58%. Seniors (61+) had the lowest at 22.7%. The Age × Pclass cross-tab reveals the starkest finding: 2nd class children survived at 100%, while 3rd class adults over 36 survived at just 8.6%.

### Q5 — Simple survival score formula

A heuristic score built from findings:

| Condition          | Points |
| ------------------ | ------ |
| Female             | +3     |
| 1st class          | +2     |
| 2nd class          | +1     |
| Child (age ≤ 12)   | +2     |
| Small family (2–4) | +1     |
| Large family (5+)  | -1     |

| Score Tier         | Survival Rate |
| ------------------ | ------------- |
| Very Low Risk (6+) | 93.4%         |
| Low Risk (4–5)     | 77.9%         |
| Medium Risk (2–3)  | 39.4%         |
| High Risk (0–1)    | 11.5%         |

---

## Out-of-Box Finding — The Hidden Social Hierarchy in Passenger Names

Most Titanic analyses use Sex, Age, and Pclass. This analysis extracts **name titles** (Mr., Mrs., Miss., Master., etc.) from the Name column — a field standard tutorials ignore entirely.

| Title                    | Survival Rate | Median Age |
| ------------------------ | ------------- | ---------- |
| Mrs                      | 79.4%         | 35         |
| Miss                     | 70.3%         | 21         |
| Master                   | 57.5%         | 3.5        |
| Rare (nobility/officers) | 34.8%         | 48.5       |
| Mr                       | 15.7%         | 30         |

**Key insight:** `Master` — the Victorian honorific for young boys — survived at nearly Mrs-level rates despite being male. A 7-year-old boy in 3rd class had dramatically better odds than his adult male companions because his title signaled "child" to evacuation crew. Meanwhile, rare titles signaling nobility (Countess, Sir, Colonel) provided _no survival advantage_ over standard 1st class passengers — extreme social rank offered no extra protection in the chaos of evacuation.

**Why this matters for ML:** Title is a single field that simultaneously encodes gender, approximate age, marital status, and social rank. Any model that ignores it is leaving meaningful signal on the table.

---

## Files

```
day-05/
├── data/
│   └── titanic_uncleaned.csv   # raw Kaggle data (gitignored)
├── eda_titanic.py               # full EDA script
└── README.md
```

---

## Key Pandas Concepts Used

- `.info()`, `.describe()`, `.isnull().sum()` — baseline EDA checklist
- `.str[0]`, `.str.extract()` — string operations on Series
- `pd.cut()` — binning continuous variables into labeled groups
- `.groupby().agg()` — multi-metric aggregation
- `.corr()` — correlation matrix
- feature engineering: `FamilySize`, `Deck`, `TitleGroup`, `score`
