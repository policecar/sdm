import glob
import os
import re

from datasets import load_dataset
from typing import List

from sdm.logger import log
from sdm.core.config import Config


def to_ascii(text, source_alphabet=None):
    # Character mapping dictionary
    char_mapping = {
        # Lowercase mappings
        "à": "a",
        "á": "a",
        "ä": "ae",
        "â": "a",
        "ā": "a",
        "À": "a",
        "æ": "a",
        "Æ": "a",
        "ç": "c",
        "Ç": "c",
        "é": "e",
        "è": "e",
        "ê": "e",
        "ë": "e",
        "È": "e",
        "É": "e",
        "î": "i",
        "ñ": "n",
        "ô": "o",
        "ö": "oe",
        "ò": "o",
        "ó": "o",
        "œ": "o",
        "ù": "u",
        "ú": "u",
        "û": "u",
        "ü": "ue",
        "Ü": "u",
        "ſ": "s",
        "…": "...",
        "—": "-",
        "’": "'",
        "‘": "'",
        "“": '"',
        "”": '"',
        "\ufeff": "",  # BOM marker
        # "†": "",
        # "•": "",
        # "°": "",
        # "": "",
        # "£": "",
        # "✠": "",
        # "™": "",
        # "'̸": "",
    }

    # Add identity mappings for ASCII letters, numbers, and common symbols
    for c in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,!?-:;\"'()[]\/&#$%*+=@_|":
        char_mapping[c] = c

    # If no source alphabet is provided, use all characters in the text
    if source_alphabet is None:
        source_alphabet = set(text)

    # Filter out multi-character keys that cause the error
    valid_mapping = {k: v for k, v in char_mapping.items() if len(k) == 1}

    # Create translation table with valid mappings
    trans_table = str.maketrans(valid_mapping)

    # For characters to delete, create a string of those characters
    delete_chars = "".join(
        char for char in source_alphabet if char not in valid_mapping and len(char) == 1
    )

    # Apply translation and handle characters to delete
    result = text.translate(trans_table)

    # Remove characters that should be deleted
    if delete_chars:
        result = "".join(char for char in result if char not in delete_chars)

    return result


def preprocess_text(text) -> str:
    """
    Preprocess text:
       - lowercase
       - replace whitespace with a single space
       - map characters to ascii
    """
    # Convert to lowercase
    text = text.lower()

    # Replace all consecutive whitespace with a single space
    text = re.sub(r"\s+", " ", text)

    # Map to ascii alphabet
    text = to_ascii(text)

    # # Replace punctuation with a single space
    # text = re.sub(f"[{string.punctuation}]", " ", text)

    # text = text.replace("<|endoftext|>", "")  # for Tiny Stories

    return text


def text_to_tokens(text, sep=" ", max_tokens=None) -> List[str]:
    """Split text into tokens, keeping separator with preceding token"""

    # # Split into words
    # tokens = text.strip().split()

    # Pattern matches everything up to and including separator
    pattern = f"[^{re.escape(sep)}]*{re.escape(sep)}|[^{re.escape(sep)}]+"
    tokens = re.findall(pattern, text.strip())
    return tokens[:max_tokens] if max_tokens else tokens


def load_texts(cfg: Config) -> tuple[list[str], list[str]]:
    """ """
    log.info("Loading texts from data_dir")
    train_files = glob.glob(os.path.join(cfg.data.data_dir, "*.txt"))
    train_text, test_text = [], []
    for i, file_path in enumerate(train_files):
        with open(file_path, "r", encoding="utf-8") as file:
            log.info(f"Reading {file_path}")
            text = file.read()
        if i < len(train_files) - 1:  # every file but the last
            train_text.append(text)
        else:
            test_text.append(text)

    return train_text, test_text


def load_traintest_dataset(cfg: Config) -> tuple[list[str], list[str]]:
    log.info(f"Loading dataset {cfg.data.dataset_name}")
    dataset = load_dataset(cfg.data.dataset_name)
    train_text = dataset["train"]["text"]  # type: ignore
    test_text = dataset["test"]["text"]  # type: ignore

    return train_text, test_text
