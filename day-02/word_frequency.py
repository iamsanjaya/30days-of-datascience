# %% Imports
from collections import Counter
import matplotlib
import re
import sys

matplotlib.use("Agg")  # needed if running without a display
import matplotlib.pyplot as plt


# %% Core function - clean a single word
def clean_word(word: str) -> str:
    """Lowercase and strip punctuation from a word."""
    return re.sub(r"[^a-z0-9]", "", word.lower())


# %% Load and process file - handle all failure modes
def load_words(filepath: str) -> list[str]:
    """
    Load words from a text file.

    Handles: missing file, empty file, non-UTF-8 encoding, and permission issues.
    Returns a list of cleaned words.
    """
    raw_text = ""

    try:
        # Attempt to read with UTF-8
        with open(filepath, "r", encoding="utf-8") as f:
            raw_text = f.read()
    except FileNotFoundError:
        print(f"[ERROR] File not found: {filepath}")
        sys.exit(1)
    except PermissionError:
        print(f"[ERROR] Permission denied: {filepath}")
        sys.exit(1)
    except UnicodeDecodeError:
        # Fall back to latin-1 if UTF-8 fails
        print("[WARNING] UTF-8 decode failed. Retrying with latin-1...")
        try:
            with open(filepath, "r", encoding="latin-1") as f:
                raw_text = f.read()
        except Exception as e:
            print(f"[ERROR] Could not read file: {e}")
            sys.exit(1)

    # Check for empty file
    if not raw_text.strip():
        print(f"[ERROR] File is empty: {filepath}")
        sys.exit(1)

    # Tokenize: split on whitespace and clean each token
    raw_words = raw_text.split()
    cleaned_words = [clean_word(w) for w in raw_words if clean_word(w)]

    return cleaned_words


# %% Stop words - common words that add no meaning
STOP_WORDS = {
    "a",
    "all",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "been",
    "by",
    "could",
    "did",
    "do",
    "for",
    "from",
    "had",
    "has",
    "have",
    "he",
    "her",
    "his",
    "i",
    "if",
    "is",
    "it",
    "its",
    "my",
    "no",
    "not",
    "of",
    "on",
    "or",
    "out",
    "said",
    "she",
    "should",
    "so",
    "than",
    "that",
    "the",
    "their",
    "then",
    "they",
    "this",
    "to",
    "up",
    "was",
    "we",
    "were",
    "what",
    "when",
    "which",
    "who",
    "will",
    "with",
    "would",
    "you",
    "your",
}


def get_top_words(words: list[str], n: int = 20) -> list[tuple[str, int]]:
    """Filter stop words and return top-N most frequent words longer than 2 characters."""
    meaningful = [w for w in words if w not in STOP_WORDS and len(w) > 2]
    counter = Counter(meaningful)
    return counter.most_common(n)


# %% Visualization
def plot_top_words(
    top_words: list[tuple[str, int]], title: str, output_path: str
) -> None:
    """Plot a horizontal bar chart of the top words."""
    words, counts = zip(*top_words)  # unpack list of tuples into two lists

    fig, ax = plt.subplots(figsize=(10, 7))
    # Reverse the order so the highest frequency appears at the top of the chart
    bars = ax.barh(words[::-1], counts[::-1], color="steelblue", edgecolor="white")

    ax.set_xlabel("Frequency", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.bar_label(bars, padding=3, fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"[OK] Chart saved to {output_path}")
    plt.close()


# %% Main
def main():
    filepath = "day-02/data/book.txt"
    words = load_words(filepath)

    print(f"[INFO] Total words loaded: {len(words):,}")

    top20 = get_top_words(words, n=20)

    print("\nTop 20 words:")
    for word, count in top20:
        print(f"{word:<20} {count:>6}")

    plot_top_words(
        top20, "Top 20 Most Frequent Words", "day-02/outputs/top20_words.png"
    )


if __name__ == "__main__":
    main()
