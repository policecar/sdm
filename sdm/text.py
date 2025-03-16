import re
import string


def preprocess_text(text, max_tokens=None):
    """
    Preprocess text by removing punctuation except commas and periods,
    which are treated as separate words.

    Args:
       text: Input text

    Returns:
       List of words from the preprocessed text
    """
    # Convert to lowercase
    text = text.lower()

    # # Add spaces around commas and periods
    # text = text.replace(",", " , ")
    # text = text.replace(".", " . ")
    # text = text.replace("?", " , ")

    # text = text.replace("<|endoftext|>", "")  # for Tiny Stories

    # # Remove other punctuation
    # nonessential_punctuation = (
    #     string.punctuation.replace(",", "")
    #     .replace(".", "")
    #     .replace("?", "")
    #     .replace("'", "")
    # )
    # text = re.sub(f"[{re.escape(nonessential_punctuation)}]", " ", text)

    text = re.sub(f"[{string.punctuation}]", " ", text)

    # Replace multiple spaces with a single space
    text = re.sub(r"\s+", " ", text)

    # Split into words
    words = text.strip().split()

    if max_tokens:
        return words[:max_tokens]
    return words
