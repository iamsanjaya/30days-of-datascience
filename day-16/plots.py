# plots.py — Day 16
# Production-safe, Pylance-clean, Optuna-safe plotting layer

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import optuna
import pandas as pd
import seaborn as sns
from optuna.importance import get_param_importances
from optuna.trial import TrialState
from matplotlib import colormaps

from config import FIGURE_DPI, OUTPUTS_DIR, PLOT_PALETTE, PLOT_STYLE


def _setup() -> None:
    sns.set_theme(style=PLOT_STYLE, palette=PLOT_PALETTE)


def _completed_trials(study: optuna.Study) -> list[optuna.trial.FrozenTrial]:
    return study.get_trials(
        deepcopy=False,
        states=[TrialState.COMPLETE],
    )


def _safe_float(value: float | None) -> float:
    if value is None:
        raise ValueError("Encountered None trial value in completed trial set.")
    return float(value)


def plot_optimization_history(study: optuna.Study) -> None:
    _setup()

    trials = _completed_trials(study)

    trial_numbers = np.fromiter((t.number for t in trials), dtype=np.int32)

    rmse_values = np.fromiter(
        (_safe_float(t.value) for t in trials),
        dtype=np.float64,
    )

    best_so_far = np.minimum.accumulate(rmse_values)

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.scatter(
        trial_numbers,
        rmse_values,
        alpha=0.45,
        s=22,
        color="steelblue",
        label="Trial RMSE",
    )

    ax.plot(
        trial_numbers,
        best_so_far,
        color="crimson",
        linewidth=2,
        label="Best so far",
    )

    ax.set_xlabel("Trial Number")
    ax.set_ylabel("CV RMSE (USD)")
    ax.set_title("Optuna Optimization History — Ames Housing XGBoost (55 Trials)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.legend()

    fig.tight_layout()

    out = OUTPUTS_DIR / "optimization_history.png"
    fig.savefig(out, dpi=FIGURE_DPI)
    plt.close(fig)

    print(f"[plots] Saved → {out}")


def plot_param_importance(study: optuna.Study) -> None:
    _setup()

    importances = get_param_importances(study)
    params = list(importances.keys())
    values = list(importances.values())

    fig, ax = plt.subplots(figsize=(8, 5))

    bars = ax.barh(
        params[::-1],
        values[::-1],
        color=sns.color_palette(PLOT_PALETTE, len(params)),
    )

    ax.set_xlabel("Relative Importance (fANOVA)")
    ax.set_title(
        "XGBoost Hyperparameter Importance — fANOVA\n"
        "(which params moved RMSE the most)"
    )

    ax.bar_label(bars, fmt="%.3f", padding=3)

    fig.tight_layout()

    out = OUTPUTS_DIR / "param_importance.png"
    fig.savefig(out, dpi=FIGURE_DPI)
    plt.close(fig)

    print(f"[plots] Saved → {out}")


def plot_parallel_coordinates(study: optuna.Study, top_n: int = 6) -> None:
    _setup()

    importances = get_param_importances(study)
    top_params = list(importances.keys())[:top_n]

    trials = _completed_trials(study)

    records: list[dict] = []

    for t in trials:
        if t.value is None:
            continue

        row = {p: t.params.get(p) for p in top_params}
        row["rmse"] = float(t.value)
        records.append(row)

    df = pd.DataFrame.from_records(records).dropna()

    df_norm = df.copy()

    for col in top_params:
        col_min = df[col].min()
        col_max = df[col].max()
        df_norm[col] = (df[col] - col_min) / (col_max - col_min + 1e-9)

    rmse = df["rmse"].to_numpy(dtype=np.float64)
    rmse_norm = (rmse - rmse.min()) / (rmse.max() - rmse.min() + 1e-9)

    cmap = colormaps["RdYlGn_r"]
    colors = cmap(rmse_norm)

    fig, ax = plt.subplots(figsize=(12, 5))

    x_pos = list(range(len(top_params)))

    for i, (_, row) in enumerate(df_norm.iterrows()):
        ax.plot(
            x_pos,
            [row[p] for p in top_params],
            color=colors[i],
            alpha=0.4,
            linewidth=0.8,
        )

    ax.set_xticks(x_pos)
    ax.set_xticklabels(top_params, rotation=20, ha="right")
    ax.set_yticks([0, 0.5, 1])
    ax.set_yticklabels(["Min", "Mid", "Max"])

    ax.set_title(
        "Parallel Coordinates — Top Hyperparameters\n"
        "(Green = low RMSE | Red = high RMSE)"
    )

    fig.tight_layout()

    out = OUTPUTS_DIR / "parallel_coordinates.png"
    fig.savefig(out, dpi=FIGURE_DPI)
    plt.close(fig)

    print(f"[plots] Saved → {out}")


def plot_random_vs_bayesian(
    random_df: pd.DataFrame,
    bayesian_df: pd.DataFrame,
) -> None:
    _setup()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]

    ax.plot(
        random_df["trial"],
        random_df["best_so_far"],
        marker="o",
        linewidth=2,
        color="steelblue",
        label="Random Search",
    )

    ax.plot(
        bayesian_df["trial"],
        bayesian_df["best_so_far"],
        marker="s",
        linewidth=2,
        color="crimson",
        label="Bayesian (TPE)",
    )

    ax.set_xlabel("Trial Number")
    ax.set_ylabel("Best RMSE so far (USD)")
    ax.set_title("Random vs Bayesian: Running Best RMSE")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.legend()

    ax2 = axes[1]

    colors = bayesian_df["mode"].map(
        {"exploration": "steelblue", "exploitation": "seagreen"}
    )

    ax2.bar(
        bayesian_df["trial"],
        bayesian_df["rmse"],
        color=colors,
        edgecolor="white",
    )

    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor="steelblue", label="Exploration"),
        Patch(facecolor="seagreen", label="Exploitation"),
    ]

    ax2.legend(handles=legend_elements)

    ax2.set_xlabel("Trial Number")
    ax2.set_ylabel("CV RMSE (USD)")
    ax2.set_title("Bayesian Trials: Exploration vs Exploitation")
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))

    fig.tight_layout()

    out = OUTPUTS_DIR / "random_vs_bayesian.png"
    fig.savefig(out, dpi=FIGURE_DPI)
    plt.close(fig)

    print(f"[plots] Saved → {out}")
