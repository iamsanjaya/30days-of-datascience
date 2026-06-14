# %%
"""
Out-of-Box Challenge: Lie With Charts (Ethically).
Takes Chart 1 (hourly demand) and produces two versions side by side:
  LEFT  — misleading: truncated Y-axis starting at ~92% of minimum, making
          a modest rush-hour bump look like a dramatic calculated surge.
  RIGHT — honest: Y-axis starts at zero, same data, accurate visual impression.

Deception caption is embedded in the figure itself and written to a text file.

Chart saved to: day-07/outputs/07_deception_pair.png
"""

# %% Imports
import matplotlib.figure
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd

import config  # applies theme + exposes paths and titles

# Data loading + preparation

# %%
df = pd.read_parquet(config.DATA_PATH)

if not pd.api.types.is_datetime64_any_dtype(df["pickup_datetime"]):
    df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"])

# pickup_hour already exists from Day 06 feature engineering
hourly = df.groupby("pickup_hour").size().reset_index(name="trip_count")


# Deception parameters — calculated from real data so the lie is reproducible

true_min = int(hourly["trip_count"].min())
true_max = int(hourly["trip_count"].max())
y_lie_min = int(true_min * 0.92)  # truncate to 92% of true minimum
y_lie_max = int(true_max * 1.02)

apparent_pct = round((true_max - true_min) / (true_max - y_lie_min) * 100)
actual_pct = round((true_max - true_min) / true_min * 100)

DECEPTION_CAPTION = (
    f"DECEPTION USED: The Y-axis starts at {y_lie_min:,} instead of 0, "
    f"making a {actual_pct}% variation appear to be a ~{apparent_pct}% surge. "
    "This truncated-axis trick is the most common form of visual deception in "
    "business dashboards and media charts — it amplifies noise into apparent signal."
)


# Build the side-by-side figure


# %%
def build_deception_pair(hourly: pd.DataFrame) -> matplotlib.figure.Figure:
    fig, (ax_lie, ax_honest) = plt.subplots(1, 2, figsize=(14, 5.5), sharey=False)
    fig.suptitle(
        "Chart Deception Study — Same Data, Different Impression",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )

    color = "steelblue"
    x = hourly["pickup_hour"]
    y = hourly["trip_count"]

    # --- LEFT: misleading ---

    ax_lie.plot(x, y, color=color, linewidth=2.0, marker="o", markersize=4)
    ax_lie.fill_between(x, y, alpha=0.15, color=color)
    ax_lie.set_ylim(y_lie_min, y_lie_max)  # ← the lie is here
    ax_lie.set_xticks(range(0, 24, 2))
    ax_lie.set_xlabel("Hour of Day")
    ax_lie.set_ylabel("Number of Trips")
    ax_lie.set_title(config.TITLE_MISLEADING, color="crimson")
    ax_lie.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
    ax_lie.annotate(
        f"Y-axis starts at {y_lie_min:,}",
        xy=(0.03, 0.04),
        xycoords="axes fraction",
        fontsize=8.5,
        color="crimson",
        bbox=dict(
            boxstyle="round,pad=0.3", facecolor="lightyellow", edgecolor="crimson"
        ),
    )

    # --- RIGHT: honest ---

    ax_honest.plot(x, y, color=color, linewidth=2.0, marker="o", markersize=4)
    ax_honest.fill_between(x, y, alpha=0.15, color=color)
    ax_honest.set_ylim(0)  # ← honest baseline
    ax_honest.set_xticks(range(0, 24, 2))
    ax_honest.set_xlabel("Hour of Day")
    ax_honest.set_ylabel("Number of Trips")
    ax_honest.set_title(config.TITLE_HONEST, color="darkgreen")
    ax_honest.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda v, _: f"{int(v):,}")
    )
    ax_honest.annotate(
        "Y-axis starts at 0",
        xy=(0.03, 0.04),
        xycoords="axes fraction",
        fontsize=8.5,
        color="darkgreen",
        bbox=dict(
            boxstyle="round,pad=0.3", facecolor="honeydew", edgecolor="darkgreen"
        ),
    )

    # FIX: Calculate layout positions first so the absolute text is placed accurately
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.22)  # leaves exact space below for caption

    # Caption below both charts safely bound
    fig.text(
        0.5,
        0.04,
        DECEPTION_CAPTION,
        ha="center",
        va="bottom",
        fontsize=9,
        color="#333333",
        wrap=True,
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#fff8e1", edgecolor="#ccaa00"),
    )

    return fig


# Save caption as standalone text (for README / journal reference)


def save_caption_text() -> None:
    path = config.OUTPUT_DIR / "07_deception_caption.txt"
    path.write_text(DECEPTION_CAPTION + "\n")
    print(f"Caption saved: {path.name}")


# Run

# %%
if __name__ == "__main__":
    print("Generating deception study chart...")
    print(f"True variation: {actual_pct}%")
    print(f"Apparent variation: {apparent_pct}% (truncated axis)")

    fig = build_deception_pair(hourly)

    # Save visual asset
    out = config.OUTPUT_DIR / "07_deception_pair.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out.name}")

    print("\nDeception study complete.")
