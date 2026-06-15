# 🧠 30 Days of Data Science & Machine Learning

> A 30-day intensive self-training program — from Python fundamentals to deployed ML models.
> Built in public. One project per day. No days off.

---

## 🎯 Mission

Position myself as a strong DS/ML intern candidate in 30 days by building
real projects, not just watching tutorials.

**Commitment:** 6–8 hours/day of deep, focused work.
**Philosophy:** Build things. Break things. Understand why. Repeat.

---

## 🛠️ Tech Stack

    Core Python       — NumPy, Pandas, Matplotlib, Seaborn, Scikit-learn
    Classical ML      — Linear/Logistic Regression, Trees, XGBoost, LightGBM
    Deep Learning     — TensorFlow/Keras, CNNs, Transfer Learning, HuggingFace
    MLOps             — sklearn Pipelines, Optuna, MLflow, Gradio/Streamlit
    Supporting        — SQL, Git + GitHub, Jupyter/VS Code

---

## 📅 Progress

| Day   | Topic                                   | Project                                    | Status |
| ----- | --------------------------------------- | ------------------------------------------ | ------ |
| 01    | Python OOP & Data Structures            | Student Grade System                       | ✅     |
| 02    | Functional Programming & Error Handling | Word Frequency + Sentiment Analyzer        | ✅     |
| 03    | NumPy Mastery                           | Distance Metrics from Scratch              | ✅     |
| 04    | NumPy Linear Algebra                    | Gradient Descent from Scratch              | ✅     |
| 05    | Pandas EDA                              | Titanic Deep Dive                          | ✅     |
| 06    | Pandas Advanced                         | Multi-table Merge + Missingness Analysis   | ✅     |
| 07    | Visualization                           | Data Storytelling + Chart Deception Study  | ✅     |
| 08    | Linear Regression                       | From Math to Code + Assumption Diagnostics | ✅     |
| 09    | Logistic Regression                     | Threshold Cost Analysis                    | ✅     |
| 10    | Decision Trees & Random Forest          | Path Analysis + Comparison                 | ✅     |
| 11    | XGBoost & Evaluation                    | Metric Selection Philosophy                | ✅     |
| 12    | Class Imbalance                         | Degradation Analysis                       | ✅     |
| 13    | Clustering & Dimensionality Reduction   | Customer Segmentation                      | ⏳     |
| 14    | Feature Engineering                     | Anti-Feature Ablation Study                | ⏳     |
| 15    | sklearn Pipelines                       | Data Leakage Quiz                          | ⏳     |
| 16    | Hyperparameter Tuning                   | Optuna + MLflow                            | ⏳     |
| 17    | End-to-End ML Review                    | Model Card + Stress Test                   | ⏳     |
| 18    | Neural Networks from Scratch            | Weight Visualization Experiment            | ⏳     |
| 19    | Keras Deep Dive                         | Architecture Search + Pruning              | ⏳     |
| 20    | Training Dynamics                       | Regularization + LR Range Test             | ⏳     |
| 21    | CNNs — Transfer Learning                | ResNet + Grad-CAM                          | ⏳     |
| 22    | CNNs — Data Efficiency                  | Niche Domain Classifier                    | ⏳     |
| 23    | NLP Fundamentals                        | BoW to BERT + Failure Forensics            | ⏳     |
| 24-25 | Capstone #1                             | Structured Data ML Project                 | ⏳     |
| 26-27 | Capstone #2                             | Deep Learning / NLP + Live Demo            | ⏳     |
| 28    | SQL for Data Science                    | HackerRank + Pandas-to-SQL                 | ⏳     |
| 29    | Interview Prep                          | Think Out Loud + Original Questions        | ⏳     |
| 30    | Launch Day                              | Applications + Cold Emails                 | ⏳     |

---

## 📓 Learning Journal

Daily reflections are tracked in [`/learning-journal/`](./learning-journal/)

Each entry covers:

- What I built
- What surprised me
- What I don't fully understand yet

---

## 📁 Repository Structure

    30days-of-datascience/
    ├── day-01/          ← Python OOP — Student Grade System
    │   ├── student.py                # Student class with OOP
    │   ├── grader.py                 # CLI grader + leaderboard
    │   ├── students.csv              # Sample student data
    │   └── README.md
    ├── day-02/          ← Functional Programming — Word Frequency + Sentiment Analyzer
    │   ├── word_frequency.py         # Standard task
    │   ├── sentiment_arc.py          # Out-of-box challenge
    │   ├── README.md
    │   ├── data/
    │   │   └── book.txt
    │   └── outputs/
    │       ├── top20_words.png
    │       └── sentiment_arc.png
    ├── day-03/          ← NumPy — Distance Metrics from Scratch
    │   ├── numpy_distances.py        # Standard task
    │   ├── distance_perception.py    # Out-of-box challenge
    │   ├── README.md
    │   └── outputs/
    │       ├── distance_metrics.png
    │       └── distance_perception.png
    ├── day-04/           ← NumPy — Linear Algebra & Gradient Descent
    │   ├── gradient_descent.py       # standard task
    │   ├── divergence_threshold.py   # out-of-box challenge
    │   ├── README.md
    │   └── outputs/
    │       ├── loss_curves.png
    │       ├── predicted_vs_true.png
    │       ├── residuals.png
    │       ├── divergence_threshold.png
    │       └── divergence_geometry.png
    ├── day-05/             ← Pandas EDA: Titanic Dataset
    │   ├── eda_titanic.py          # Main EDA script
    │   ├── README.md
    │   ├── data/
    │   │   └── titanic_uncleaned.csv
    │   └── outputs/
    │       ├── oob_title_analysis.csv
    │       ├── q1_survival_correlations.png
    │       ├── q2_deck_vs_pclass_survival.png
    │       ├── q3_family_size_survival.png
    │       ├── q4_age_survival.png
    │       └── q5_survival_score.png
    ├── day-06/             ← Pandas Advanced — NYC TLC Multi-Table Merge & Missingness Analysis
    │   ├── 01_tlc_merge.py          # Standard task  (run this first)
    │   ├── 02_missingness.py        # Out-of-box challenge (run this second)
    │   ├── README.md
    │   ├── data/
    │   │   ├── raw/
    │   │   │   ├── green_tripdata_2023-01.parquet      (source — 68K rows)
    │   │   │   └── yellow_tripdata_2023-01.parquet     (source — 3M rows)
    │   │   └── processed/
    │   │       └── tlc_combined_clean_jan2023.parquet  (generated by 01_tlc_merge.py)
    │   └── outputs/
    │        ├── q1_trips_by_hour.png
    │        ├── q2_median_fare.png
    │        ├── q3_trip_distance.png
    │        ├── q4_tip_by_payment.png
    │        ├── q5_speed_by_hour.png
    │        └── missingness_heatmap.png
    ├── day-07/             ← Visualization - Data Storytelling + Chart Deception Study
    │   ├── config.py                   # paths, constants, plot style - single
    │   ├── 01_charts.py                # the 6 - chart EDA story
    │   ├── 02_deception.py             # honest vs misleading chart pair
    │   ├── README.md
    │   └── outputs/
    │       ├── 01_hourly_demand.png
    │       ├── 02_fare_vs_distance.png
    │       ├── 03_payment_tip.png
    │       ├── 04_tip_by_dow.png
    │       ├── 05_vendor_peak.png
    │       ├── 06_fare_distribution.png
    │       └── 07_deception_pair.png
    ├── day-08/             ← Linear Regression - From Math to Code + Assumption Diagnostics
    │   ├── config.py                                # constants: paths, hyperparameters
    │   ├── 01_linear_regression_three_ways.py       # normal equation, gradient descent, sklearn — verified identical
    │   ├── 02_assumption_violations.py              # diagnostic toolkit: 4 OLS assumption violations with plots
    │   ├── README.md
    │   ├── data/
    │   │   └── housing.csv
    │   └── outputs/
    │       ├── 01_three_implementations.png
    │       └── 02_assumption_violations.png
    ├── day-09/         ← Logistic Regression + Threshold Cost Analysis
    │   ├── config.py                   # constants: paths, costs, random state
    │   ├── 01_train_logistic.py        # train model, ROC, confusion matrix, precision-recall
    │   ├── 02_threshold_analysis.py    # threshold sweep: precision, recall, F1 tradeoff
    │   ├── 03_cost_framework.py        # cost-weighted optimal threshold selection
    │   ├── decision_document.md        # non-technical threshold decision doc for hospital setting
    │   ├── README.md
    │   ├── data/
    │   │   └── heart.csv
    │   └── outputs/
    │       ├── 01_diagnostic_plots.png
    │       ├── 02_threshold_analysis.png
    │       └── 03_cost_framework.png
    ├── day-10/         ← Decision Trees & Random Forest — Telco Customer Churn
    │   ├── config.py                       # constants: paths, hyperparameters, column groups, random state
    │   ├── 01_train_decision_tree.py       # load, clean, encode, train DT + RF, CV comparison, feature importances
    │   ├── 02_path_tracing.py              # decision_path tracing, same-leaf analysis, probability calibration plots
    │   ├── 03_random_forest_comparison.py  # OOB error vs n_estimators, feature importance stability across seeds
    │   ├── README.md
    │   ├── data/
    │   │   └── WA_Fn-UseC_-Telco-Customer_Churn.csv.xls
    │   └── outputs/
    │       ├── decision_tree_structure.png
    │       ├── dt_vs_rf_cv_comparison.png
    │       ├── dt_vs_rf_feature_importance.png
    │       ├── dt_vs_rf_probability_distribution.png
    │       ├── leaf_probability_distribution.png
    │       ├── rf_importance_stability.png
    │       └── rf_oob_error_vs_n_estimators.png
    ├── day-11/         ← XGBoost & Evaluation - Metric Selection Philosophy
    │   ├── config.py                       # shared constants: paths, random state, model params, colours
    │   ├── 01_xgboost_titanic.py           # XGBoost vs Random Forest — Stratified 5-Fold CV + diagnostic plots
    │   ├── 02_metric_comparison.py         # threshold tuning per metric — rescue scenario ethics analysis
    │   ├── README.md
    │   ├── data/
    │   │   ├── raw/
    │   │   │   └── titanic_uncleaned.csv   # original Kaggle Titanic — untouched
    │   │   └── processed/
    │   │       └── titanic_cleaned.csv     # imputed, encoded, leakage columns dropped
    │   └── outputs/
    │       ├── confusion_matrix.png
    │       ├── cv_scores.png
    │       ├── metric_comparison.png
    │       ├── precision_recall_curve.png
    │       ├── roc_curve.png
    │       └── threshold_analysis.png
    ├── day-12/         ← Class Imbalance — Strategy Comparison & Degradation Study
    │   ├── config.py                           # shared constants: paths, random state, colours
    │   ├── 01_imbalance_strategy_comparison.py # SMOTE · Undersampling · class_weight · Baseline — 5-Fold CV
    │   ├── 02_outofbox_challenge.py            # manufactured imbalance 1:2 → 1:100 — degradation + gap heatmap
    │   ├── README.md
    │   ├── data/
    │   │   └── breastcancer.csv                # original dataset — untouched
    │   └── outputs/
    │       ├── strategy_comparison.png
    │       ├── degradation_curves.png
    │       └── metric_gap_heatmap.png
    └── learning-journal/
        ├── day-01.md
        ├── day-02.md
        ├── day-03.md
        ├── day-04.md
        ├── day-05.md
        ├── day-06.md
        ├── day-07.md
        ├── day-08.md
        ├── day-09.md
        ├── day-10.md
        ├── day-11.md
        └── day-12.md

---

## 👤 About

**Sanjaya Chaudhary** — aspiring Data Scientist from Kathmandu, Nepal.
Building in public. Learning by doing.

[![GitHub](https://img.shields.io/badge/GitHub-iamsanjaya-black?logo=github)](https://github.com/iamsanjaya)
