import math
import numpy as np
import re
import pandas as pd
import seaborn as sns

from collections import Counter
from matplotlib import pyplot as plt

from sdm.core.config import Config
from sdm.core.text import load_texts, preprocess_text, text_to_tokens


def find_contexts(text, char, size=1):
    """Find all instances of char with surrounding context"""
    return [
        text[max(0, i - size) : min(len(text), i + size + 1)]
        for i, c in enumerate(text)
        if c == char
    ]


def calculate_char_entropies(text):
    """Calculate the entropy of each character position in a string."""
    if not text:
        return {}

    # Get unique characters in the text
    unique_chars = set(text)

    # Calculate entropy for each position
    char_entropies = {}
    for char in unique_chars:
        # For each position, calculate information content -log2(p(char))
        # Count occurrences of this character in the text
        char_count = text.count(char)
        probability = char_count / len(text)
        # Information content = -log2(probability)
        information_content = -math.log2(probability)
        char_entropies[char] = information_content

    return char_entropies


def plot_token_frequencies(wc, n_bins=50):
    some_wc = wc.most_common()[:25] + wc.most_common()[-25:]
    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))

    # Word frequency bar chart
    df = pd.DataFrame(some_wc, columns=["Word", "Count"])
    sns.barplot(
        x="Count",
        y="Word",
        hue="Word",
        data=df,
        palette="Blues_d",
        legend=False,
        ax=ax1,
    )
    ax1.set_title("Top word frequencies")

    # Histogram of word frequencies
    counts = list(wc.values())
    bins = np.logspace(np.log10(0.9), np.log10(max(counts) + 0.1), num=n_bins)
    ax2.hist(counts, bins=bins, alpha=0.7, color="maroon")
    ax2.set_xscale("log")
    ax2.set_xlabel("Word frequency (log scale)")
    ax2.set_ylabel("Count of words")
    ax2.set_title("Histogram of word frequencies")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("./plots/voc_word_frequencies.png")
    plt.show()


if __name__ == "__main__":
    # Get config
    cfg = Config.from_yaml()

    # Load data
    train_texts, _ = load_texts(cfg)
    text = train_texts[1]

    # Basic cleanup:
    # - Replace all consecutive whitespace with a single space
    text = re.sub(r"\s+", " ", text)

    cc = Counter(text)
    alphabet = cc.keys()  # set(string_of_chars)
    n_alphabet = len(alphabet)

    # Clean text
    clean_text = preprocess_text(text)

    # Compute character entropy
    entropies = calculate_char_entropies(clean_text)
    sorted_entropies = sorted(entropies.items(), key=lambda item: item[1])
    # print(sorted_entropies)
    # Use the lowest entropy character as separator
    # For natural language that's most likely the space symbol
    sep = sorted_entropies[0][0]

    tokens = text_to_tokens(clean_text, sep=sep)
    tokens_by_length = sorted(set(tokens), key=len)

    # Word frequency analysis
    wc = Counter(tokens)
    plot_token_frequencies(wc)
