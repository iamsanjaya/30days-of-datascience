# %% Imports
import re
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os

# %% Manually Curated word lists - Cleaned of duplicates
POSITIVE_WORDS = {
    "happy",
    "joy",
    "love",
    "wonderful",
    "great",
    "excellent",
    "beautiful",
    "hope",
    "laugh",
    "smile",
    "delight",
    "pleasure",
    "kind",
    "warm",
    "successful",
    "triumph",
    "peace",
    "brave",
    "glory",
    "friend",
    "gentle",
    "grateful",
    "bright",
    "free",
    "safe",
    "dream",
    "life",
    "sweet",
    "wonder",
    "thankful",
}

NEGATIVE_WORDS = {
    "sad",
    "death",
    "die",
    "dark",
    "fear",
    "pain",
    "hate",
    "evil",
    "misery",
    "cry",
    "sorrow",
    "suffer",
    "terrible",
    "awful",
    "angry",
    "despair",
    "lonely",
    "lost",
    "cruel",
    "cold",
    "dead",
    "blood",
    "grief",
    "wretched",
    "poor",
    "shame",
    "wicked",
    "weep",
    "horror",
    "dreadful",
    "bitter",
}

TENSION_WORDS = {
    "suddenly",
    "danger",
    "escape",
    "trap",
    "chase",
    "fight",
    "attack",
    "run",
    "quickly",
    "desperate",
    "urgent",
    "risk",
    "threat",
    "hurry",
    "warning",
    "alarm",
    "shock",
    "strike",
    "rush",
    "silent",
    "alone",
    "unknown",
    "strange",
    "mystery",
    "secret",
    "hide",
    "watch",
    "wait",
}


# %% Chapter splitting
def split_into_chapters(text: str) -> list[str]:
    """
    Split text on chapter headings.
    Returns a list of chapter strings.
    """
    # Pattern: "CHAPTER" or "Chapter" followed by a number or Roman numeral
    pattern = r"\n\s*(?:CHAPTER|Chapter)\s+[IVXLCDM\d]+"
    parts = re.split(pattern, text)

    # First split is usually the preamble/header - skip if very short
    chapters = [p.strip() for p in parts if len(p.strip()) > 200]
    return chapters


# %% Score a single chapter
def score_chapter(chapter_text: str) -> dict:
    """
    Return normalized sentiment scores for one chapter.
    Normalized = score / total_words (so length doesn't dominate).
    """
    words = re.findall(r"\b[a-z]+\b", chapter_text.lower())
    total = len(words)

    if total == 0:
        return {"positive": 0.0, "negative": 0.0, "tension": 0.0, "total_words": 0}

    # POSITIVE_WORDS, NEGATIVE_WORDS, and TENSION_WORDS are already sets, yielding O(1) lookups
    pos_score = sum(1 for w in words if w in POSITIVE_WORDS) / total
    neg_score = sum(1 for w in words if w in NEGATIVE_WORDS) / total
    ten_score = sum(1 for w in words if w in TENSION_WORDS) / total

    return {
        "positive": pos_score,
        "negative": neg_score,
        "tension": ten_score,
        "total_words": total,
    }


# %% Smooth the arc using true vectorization
def smooth(values: list[float], window: int = 3) -> list[float]:
    """Vectorized rolling average using NumPy convolution"""
    if len(values) < window:
        return values  # Cannot smooth if dataset is smaller than window size

    box = np.ones(window) / window
    # 'same' mode ensures the output array matches the input length
    return list(np.convolve(values, box, mode="same"))


# %% Main analysis
def analyze_book(filepath: str) -> None:
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()
    except FileNotFoundError:
        print(f"[ERROR] File not found: {filepath}")
        return
    except UnicodeDecodeError:
        with open(filepath, "r", encoding="latin-1") as f:
            text = f.read()

    chapters = split_into_chapters(text)
    print(f"[INFO] Detected {len(chapters)} chapters")

    if len(chapters) < 3:
        print("[WARNING] Too few chapters detected. Check chapter header format.")
        print("[HINT] Try adjusting the regex pattern in split_into_chapters()")
        return

    scores = [score_chapter(ch) for ch in chapters]

    chapter_nums = list(range(1, len(scores) + 1))
    pos = smooth([s["positive"] for s in scores])
    neg = smooth([s["negative"] for s in scores])
    ten = smooth([s["tension"] for s in scores])

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(14, 6))

    ax.plot(chapter_nums, pos, color="#2ecc71", linewidth=2.5, label="Positive")
    ax.plot(chapter_nums, neg, color="#e74c3c", linewidth=2.5, label="Negative")
    ax.plot(
        chapter_nums, ten, color="#f39c12", linewidth=2, linestyle="--", label="Tension"
    )

    ax.fill_between(chapter_nums, pos, alpha=0.1, color="#2ecc71")
    ax.fill_between(chapter_nums, neg, alpha=0.1, color="#e74c3c")

    ax.set_xlabel("Chapter", fontsize=12)
    ax.set_ylabel("Normalized Word Frequency", fontsize=12)
    ax.set_title(
        "Emotional Arc — Chapter-by-Chapter Sentiment (Zero ML)",
        fontsize=14,
        fontweight="bold",
    )
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs("day-02/outputs", exist_ok=True)
    plt.savefig("day-02/outputs/sentiment_arc.png", dpi=150)
    print("[OK] Sentiment arc saved to outputs/sentiment_arc.png")
    plt.close()

    # Print a summary
    print("\nChapter Emotional Peaks:")
    raw_pos = [s["positive"] for s in scores]
    raw_neg = [s["negative"] for s in scores]
    raw_ten = [s["tension"] for s in scores]

    print(f"  Most positive chapter : {chapter_nums[raw_pos.index(max(raw_pos))]}")
    print(f"  Most negative chapter : {chapter_nums[raw_neg.index(max(raw_neg))]}")
    print(f"  Highest tension chapter: {chapter_nums[raw_ten.index(max(raw_ten))]}")


if __name__ == "__main__":
    analyze_book("day-02/data/book.txt")
