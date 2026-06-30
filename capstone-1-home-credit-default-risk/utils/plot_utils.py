"""
Shared plotting utilities. Never plt.show() — figures are always saved to
disk so they exist as artifacts in outputs/, not just transient windows.
"""

from pathlib import Path

import matplotlib.figure


def save_and_close(
    fig: matplotlib.figure.Figure, output_dir: Path, filename: str
) -> Path:
    """Save a figure at publication quality (300 DPI) and close it."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename
    fig.savefig(path, dpi=300, bbox_inches="tight")
    import matplotlib.pyplot as plt

    plt.close(fig)
    print(f"[save_and_close] Saved {path}")
    return path
