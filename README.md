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
| 13    | Clustering & Dimensionality Reduction   | Customer Segmentation                      | ✅     |
| 14    | Feature Engineering                     | Anti-Feature Ablation Study                | ✅     |
| 15    | sklearn Pipelines                       | Data Leakage Quiz                          | ✅     |
| 16    | Hyperparameter Tuning                   | Optuna + MLflow                            | ✅     |
| 17    | End-to-End ML Review                    | Model Card + Stress Test                   | ✅     |
| 18    | Neural Networks from Scratch            | Weight Visualization Experiment            | ✅     |
| 19    | Keras Deep Dive                         | Architecture Search + Pruning              | ✅     |
| 20    | Training Dynamics                       | Regularization + LR Range Test             | ✅     |
| 21    | CNNs — Transfer Learning                | ResNet + Grad-CAM                          | ✅     |
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
    │
    ├── day-02/          ← Functional Programming — Word Frequency + Sentiment Analyzer
    │   ├── word_frequency.py         # Standard task
    │   ├── sentiment_arc.py          # Out-of-box challenge
    │   ├── README.md
    │   ├── data/
    │   │   └── book.txt
    │   └── outputs/
    │       ├── top20_words.png
    │       └── sentiment_arc.png
    │
    ├── day-03/          ← NumPy — Distance Metrics from Scratch
    │   ├── numpy_distances.py        # Standard task
    │   ├── distance_perception.py    # Out-of-box challenge
    │   ├── README.md
    │   └── outputs/
    │       ├── distance_metrics.png
    │       └── distance_perception.png
    │
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
    │
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
    │
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
    │
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
    │
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
    │
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
    │
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
    │
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
    │
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
    │
    ├── day-13/         ← clustering subjectivity study + customer profiling
    │   ├── config.py                       # all paths, constants, hyperparameters
    │   ├── 01_preprocess.py                # load, encode Gender, StandardScaler
    │   ├── 02_kmeans_selection.py          # elbow method + silhouette score → pick K
    │   ├── 03_cluster_profiles.py          # fit final K-Means, profile each cluster
    │   ├── 04_dimensionality_reduction.py  # PCA scree + t-SNE cluster scatter
    │   ├── 05_subjectivity.py              # K=3 vs K=5 vs K=8 — clustering as a lens
    │   ├── data/
    │   │    ├── raw/
    │   │    │   └── Mall_Customers.csv
    │   │    └── processed/
    │   │        ├── X_scaled.csv
    │   │        ├── df_with_encoded.csv
    │   │        ├── df_clustered.csv
    │   │        └── reduced_coords.csv
    │   ├── models/
    │   │   └── scaler.joblib
    │   │
    │   └── outputs/
    │        ├── plots/
    │        │   ├── k_selection.png
    │        │   ├── cluster_scatter.png
    │        │   ├── pca_visualization.png
    │        │   ├── tsne_visualization.png
    │        │   └── subjectivity_comparison.png
    │        │
    │        └── reports/
    │             ├── cluster_profiles.csv
    │             └── clustering_subjectivity_insight.txt
    │
    ├── day-14/         ← feature engineering + anti-feature ablation study
    │   ├── config.py                       # paths, CV settings, XGB params, thresholds
    │   ├── 01_feature_engineering.py       # baseline vs engineered comparison (CV RMSLE)
    │   ├── 02_anti_feature_ablation.py     # permutation importance + greedy ablation
    │   ├── README.md
    │   ├── utils/
    │   │   ├── feature_builder.py          # engineered feature generation
    │   │   ├── encoding.py                 # label encoding utilities
    │   │   └── evaluation.py               # CV RMSLE + reporting utilities
    │   │
    │   ├── data/
    │   │   ├── raw/
    │   │   │   └── AmesHousing.csv
    │   │   └── processed/
    │   │       ├── engineered.csv
    │   │       ├── ablation_log.csv
    │   │       └── feature_importance.csv
    │   └── outputs/
    │       └── plots/
    │           ├── ablation_curve.png
    │           └── feature_importance.png
    │
    ├── day-15/          ← sklearn pipeline + grid search + leakage quiz on ames housing
    │   ├── config.py              # All constants, paths, feature lists, hyperparams
    │   ├── pipeline_factory.py    # build_preprocessor() + build_pipeline() — reusable
    │   ├── 01_load_explore.py     # Load, dtype audit, missing-value audit, train/test split
    │   ├── 02_pipeline.py         # Full pipeline + 5-fold CV + test set evaluation
    │   ├── 03_grid_search.py      # GridSearchCV over pipeline (12 combos × 5 folds = 60 fits)
    │   ├── 04_leakage_quiz.py     # 5-snippet leakage quiz — 3 leaky, 2 clean
    │   └── data/
    │   │   └── AmesHousing.csv
    │   └── outputs/                # plots saved here
    │       ├── 01_missing_value audit.png
    │       ├── 01_target_distribution.png
    │       ├── 02_cv_fold_rmse.png
    │       ├── 02_feature_importance.png
    │       ├── 03_default \_vs_tuned.png
    │       ├── 03_gridsearch_heatmap_lr01.png
    │       ├── 03_gridsearh_heatmap_lr005.png
    │       └── 04_leakage_inflation.png
    │
    ├── day-16/         ← Hyperparameter Tuning: Optuna + MLflow
    │   ├── config.py               # All constants, paths, search space (single source of truth)
    │   ├── data_loader.py          # Load Ames Housing, clean, one-hot encode, split
    │   ├── optuna_tuner.py         # 55-trial Optuna study with MLflow child run logging
    │   ├── random_vs_bayesian.py   # Out-of-box: random vs Bayesian trials, exploration labels
    │   ├── plots.py                # All visualization logic (no plot calls elsewhere)
    │   ├── main.py                 # Orchestrator: runs all steps in order
    │   ├── README.md
    │   ├── data/                   # gitignored
    │   │   └── train.csv           # Ames Housing (from Kaggle)
    │   └── outputs/                # gitignored
    │       ├── optimization_history.png
    │       ├── param_importance.png
    │       ├── parallel_coordinates.png
    │       └── random_vs_bayesian.png
    │
    ├── day-17/         ← Model Card + Robustness Stress Test (Ames Housing)
    │   ├── 01_rebuild_best_pipeline.py     # Rediscovers Day 15's grid-search params (never persisted),
    │   │                                   # freezes pipeline + test split to models/
    │   ├── 02_distribution_shift_test.py   # Old vs. new homes within the held-out test set (1960 split)
    │   ├── 03_noise_injection_test.py      # 20% Gaussian noise on numeric test features
    │   ├── 04_label_corruption_test.py     # 10% of test labels shuffled
    │   ├── 05_robustness_aggregator.py     # Combines 02-04 into one table, drafts ROBUSTNESS_REPORT.md
    │   ├── 06_visualize_results.py         # Saves robustness_comparison.png + predicted_vs_actual_by_era.png
    │   ├── config.py                       # Paths, rediscovered grid, exact stress-test parameters
    │   ├── README.md                       # Day-17 summary, run order, results
    │   ├── MODEL_CARD.md                   # Six-section model card (problem, data, architecture, metrics+CI, failure modes, ethics)
    │   ├── ROBUSTNESS_REPORT.md            # Auto-generated results table + hand-written 5-sentence reflection
    │   ├── models/                         # .joblib artifacts (gitignored)
    │   │   ├── best_pipeline.joblib
    │   │   └── test_split.joblib
    │   └── outputs/                       # JSON results + plots (gitignored)
    │       ├── baseline_metrics.json
    │       ├── best_params.json
    │       ├── distribution_shift_results.json
    │       ├── label_corruption_results.json
    │       ├── noise_injection_results.json
    │       └── plots/
    │           ├── predicted_vs_actual_by_era.png
    │           └── robustness_comparison.png
    │
    ├── day-18/         ← Neural Networks from Scratch + Weight Visualization Experiment
    │   ├── 01_numpy_nn_scratch.py                  # 2-layer NN (784→128 ReLU→10 softmax) built from scratch in
    │   │                                           # NumPy: forward pass, backprop, mini-batch GD on MNIST
    │   ├── 02_keras_nn_replica.py                  # Identical architecture in Keras (plain SGD, same lr/epochs)
    │   │                                           # to confirm the from-scratch math matches a framework
    │   ├── 03_weight_visualization_experiment.py   # Trains a 2nd model on shuffled labels (real images,
    │   │                                          # scrambled targets); visualizes hidden-layer weights from
    │   │                                          # both models reshaped to 28x28 (real vs. memorized)
    │   ├── 04_linear_softmax_templates.py         # Bonus: no-hidden-layer softmax model (784→10 direct).
    │   │                                          # Output weights reshape into recognizable digit templates —
    │   │                                         # contrasts with 03's uninterpretable MLP hidden-layer weights
    │   ├── config.py                             # Paths, architecture sizes, hyperparameters for all 4 scripts
    │   ├── README.md                             # Day-18 summary, run order, dataset source, results
    │   ├── data/
    │   │   └── raw/
    │   │       └── train.csv                      # Kaggle Digit Recognizer (real MNIST, CSV format)
    │   ├── models/                                # gitignored — trained weights/params, consumed across scripts
    │   │   ├── numpy_nn_params.npz                # from 01_numpy_nn_scratch.py
    │   │   ├── keras_nn_replica.keras             # from 02_keras_nn_replica.py , loaded by 03 for the "real labels" comparison
    │   │   └── keras_nn_random_label.keras        # from 03_weight_visualization_experiment.py
    │   │
    │   ├── outputs/                               # tracked — figures + metrics meant for review
    │   │   ├── metrics/
    │   │   │   ├── numpy_nn_history.csv          # per-epoch loss/test-acc from 01
    │   │   │   └── keras_nn_history.csv           # per-epoch loss/acc/val from 02
    │   │   └── plots/
    │   │       ├── weights_real_labels.png       # 03 — hidden-layer weights, real-label model
    │   │       └── weights_shuffled_labels.png   # 03 — hidden-layer weights, shuffled-label model
    │   │
    │   └── utils/
    │       ├── __init__.py
    │       ├── data_loader.py                  # load_mnist_csv(), one_hot_encode()
    │       ├── preprocess.py                   # prepare_data() — shared train/test prep for 03 and 04
    │       └── nn_utils.py                     # forward/backward pass, activations, loss — used only by 01
    │
    ├── day-19/         ← Keras Deep Dive: Architecture Decisions & the Lottery Ticket Hypothesis
    │   ├── 01_preprocess_data.py              # Clean Adult dataset, encode, scale, split
    │   ├── 02_architecture_search.py          # Train 75 architecture configs, log results
    │   ├── 03_train_best_architecture.py      # Train + save best config, save initial weights
    │   ├── 04_lottery_ticket_pruning.py       # Lottery ticket + random pruning experiments
    │   ├── 05_compare_results.py              # Generate comparison visualizations
    │   ├── README.md                          # Project overview and results
    │   ├── config.py                          # Centralized paths and hyperparameters
    │   ├── data
    │   │   ├── processed
    │   │   │   └── splits.npz                 # Preprocessed train/val/test splits
    │   │   └── raw
    │   │       └── adult.csv                  # Original Adult Census Income dataset
    │   ├── models
    │   │   ├── best_model.keras               # Best-performing trained network
    │   │   ├── best_model_initial_weights.npz # Original initialization for lottery-ticket rewinding
    │   │   └── preprocessor.joblib            # Saved preprocessing pipeline
    │   ├── outputs
    │   │   ├── architecture_search
    │   │   │   ├── 3d_architecture_grid.png   # 3D architecture search visualization
    │   │   │   ├── depth_width_heatmaps.png   # Depth-width performance heatmaps
    │   │   │   └── grid_results.csv           # Results from all 75 architectures
    │   │   ├── best_model
    │   │   │   ├── best_config.json           # Selected architecture configuration
    │   │   │   ├── test_metrics.json          # Final test performance metrics
    │   │   │   └── training_history.csv       # Epoch-by-epoch training history
    │   │   └── pruning
    │   │       ├── comparison.csv             # Full vs lottery-ticket vs random-pruned metrics
    │   │       └── pruning_comparison.png     # Pruning experiment visualization
    │   └── utils
    │       ├── __init__.py
    │       └── architecture.py                # Shared MLP model builder
    │
    ├── day-20/         ← Training Dynamics: Diagnosing an Overfit CNN
    │   ├── config.py                       # paths, hyperparameters, regularization configs
    │   ├── 01_data_preparation.py          # verify download, class balance, overfit subset check
    │   ├── 02_overfit_baseline.py          # train the deliberately oversized, unregularized CNN
    │   ├── 03_regularization_comparison.py # Dropout / L2 / BatchNorm / Augmentation / EarlyStop / combined
    │   ├── 04_lr_range_test.py             # out-of-box: LR Range Test
    │   ├── utils/
    │   │   ├── **init**.py
    │   │   ├── data.py                     # PNG loading, reproducible stratified subsetting
    │   │   ├── architecture.py             # overfit baseline + toggleable regularized CNN
    │   │   ├── training.py                 # compile/fit helpers, history I/O, LR range test callback
    │   │   └── visualization.py            # training-curve grids, LR range test plot
    │   ├── data/raw/cifar10/               # gitignored — place the Kaggle download here
    │   ├── outputs/                        # gitignored — plots + JSON histories land here
    │   │   ├── class_balance.png
    │   │   ├── lr_range_test.png
    │   │   ├── overfit_baseline.png
    │   │   ├── regularization_comparison.png
    │   │   └── histories/                  # per-config training history, saved as JSON
    │   ├── models/                         # gitignored — saved .keras model files
    │   │   └── overfit_baseline.keras
    │   └── diagnostic_report.md            # filled in with actual results
    │
    ├──day-21/         ← CNNs: Transfer Learning (ResNet50 / Xception) + Grad-CAM
    │   ├── config.py                       # all hyperparameters, paths, constants, backbone selection
    │   ├── 01*prepare_data.py              # scan raw dir, validate files, build stratified split
    │   ├── 02_train_frozen.py              # Phase 1: train head with frozen backbone
    │   ├── 03_finetune.py                  # Phase 2: unfreeze last 10 layers, fine-tune
    │   ├── 04_compare_results.py           # evaluate both models on held-out test set
    │   ├── 05_gradcam_analysis.py          # Grad-CAM on correct vs misclassified samples
    │   ├── 06_compare_backbones.py         # ResNet50 vs Xception side-by-side (after running both)
    │   ├── utils/
    │   │   ├── data.py                     # scanning, validation, splitting, tf.data pipelines
    │   │   ├── architecture.py             # backbone + head builder, layer unfreezing
    │   │   ├── training.py                 # compile/train/callbacks, history save/load
    │   │   ├── gradcam.py                  # Grad-CAM heatmap computation + overlay
    │   │   └── visualization.py            # all plotting (curves, comparisons, grids)
    │   ├── data/raw/PetImages/             # <- place Kaggle dataset here (gitignored)
    │   ├── outputs/                        # generated at runtime (gitignored)
    │   │   ├── splits/                     # train.csv, val.csv, test.csv (shared across backbones)
    │   │   ├── data_quality/               # corrupt_files_removed.csv
    │   │   ├── curves/<backbone>/          # training curve PNGs, per backbone
    │   │   ├── comparison/<backbone>/      # frozen vs fine-tuned metrics + plot, per backbone
    │   │   ├── comparison/                 # backbone_comparison.png (from 06)
    │   └── gradcam/<backbone>/             # Grad-CAM grids + analysis log CSV, per backbone
    │   ├── models/                         # generated at runtime (gitignored)
    │   │   ├── frozen_model_<backbone>.keras
    │   │   └── finetuned_model_<backbone>.keras
    │   └── README.md
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
        ├── day-12.md
        ├── day-13.md
        ├── day-14.md
        ├── day-15.md
        ├── day-16.md
        ├── day-17.md
        ├── day-18.md
        ├── day-19.md
        ├── day-20.md
        └── day-21.md

---

## 👤 About

**Sanjaya Chaudhary** — aspiring Data Scientist from Kathmandu, Nepal.
Building in public. Learning by doing.

[![GitHub](https://img.shields.io/badge/GitHub-iamsanjaya-black?logo=github)](https://github.com/iamsanjaya)
