"""
Temporal Memory Example - Text Learning
Translated from Peter Overmann's Mathematica implementation

This script demonstrates the Temporal Memory algorithm on text data,
showing how the system learns to predict the next word in a sequence.
"""

import logging
import glob
import os
import random
import sys

from time import time
from typing import List

from datasets import load_dataset

from sdm.enc import TokenEncoder
from sdm.io_utils import save_sdr, load_sdr
from sdm.text import preprocess_text
from sdm.ttm import TriadicTemporalMemory

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# ANSI color codes
RED = "\033[31m"
GREEN = "\033[32m"
RESET = "\033[0m"


def generate_sequence(
    tm, encoder: TokenEncoder, tokens: List[str], length: int
) -> List[str]:
    """
    Generate a sequence of tokens using temporal memory

    Args:
        tm:             Trained TemporalMemory instance
        encoder:        Encoder for converting between tokens and vectors
        tokens:         Initial tokens to prime the memory (minimum 2)
        length:         Number of new tokens to generate

    Returns:
        List of generated tokens
    """
    tm.reset()  # reset state

    output = tokens.copy()

    # Process all but the last token to set up the context
    for token in tokens[:-1]:
        tm.process(encoder.encode(token))

    # Process the last token and get first prediction
    prediction = tm.process(encoder.encode(tokens[-1]))

    # Generate sequence
    for _ in range(length):
        next_token = encoder.decode(prediction)
        output.append(next_token)
        # Use prediction as next input
        prediction = tm.process(prediction)

    return output


def tm_learn(tm, encoder: TokenEncoder, tokens: List[str], iter: int = 1):
    """
    Train temporal memory on a sequence of tokens with multiple iterations

    Args:
        memory:         TemporalMemory instance
        encoder:        Encoder for converting tokens to SDRs
        tokens:         List of tokens to train on
        iterations:     Number of times to process the entire sequence
    """
    # Determine if we should log timing based on sequence length
    timeit = len(tokens) > 10000

    for iteration in range(iter):
        # Only start timing for large sequences
        t0 = time() if timeit else 0.0
        if timeit:
            log.info(f"Training iteration {iteration + 1}/{iter}")

        tm.reset()  # Reset state but maintain learned associations
        for token in tokens:
            vector = encoder.encode(token)
            tm.process(vector)

        if timeit:
            elapsed_ms = int((time() - t0) * 1000)
            log.info(f"Iteration {iteration + 1} completed in {elapsed_ms} ms")


def eval_tm(tm, encoder: TokenEncoder, tokens: List[str]):
    """
    Test the Temporal Memory algorithm on a list of tokens

    Args:
        tokens: List of tokens (words) to process
        iterations: Number of iterations to run
    """
    tm.reset()

    # Process first token (can't predict this one)
    if tokens:
        pred_sdr = tm.process(encoder.encode(tokens[0]))

    # Process remaining tokens and compare predictions
    correct_count = 0
    color_tokens = []

    for i in range(1, len(tokens)):
        # Get the prediction from the previous step
        predicted_token = encoder.decode(pred_sdr)

        # Process the current token
        current_token = tokens[i]
        pred_sdr = tm.process(encoder.encode(current_token))

        # Check if prediction was correct
        if predicted_token == current_token:
            correct_count += 1
            color_tokens.append(f"{GREEN}{current_token}{RESET}")
        else:
            color_tokens.append(f"{RED}{current_token}{RESET}")

    log.info(" ".join(color_tokens))

    # Calculate and log accuracy for this iteration
    accuracy = correct_count / (len(tokens) - 1) * 100
    log.info(f"Accuracy: {accuracy:.2f}% ({correct_count}/{len(tokens) - 1})")


def interactive_demo(tm, encoder: TokenEncoder, seq_len=50):
    """
    Interactive prediction loop
    """

    print("\n" + "=" * 50)
    print("Interactive Prediction Mode")
    print("Enter your starting text")
    print("Type 'x' or 'q' to exit or quit")

    while True:
        print("\n" + "-" * 50)
        print("Enter your starting text:")
        user_input = input("> ").strip()

        # Check for exit command
        if user_input.lower() in ["x", "q"]:
            print("Exiting interactive mode.")
            break

        if user_input:
            input_tokens = preprocess_text(user_input)

        sequence = generate_sequence(tm, encoder, input_tokens, seq_len)
        print(" ".join(sequence), end=" ")
        print()
        print("\n" + "=" * 50)


if __name__ == "__main__":
    # CONFIG

    training = True
    hf_dataset = False

    data_dir = "data"
    input_path = f"{data_dir}/pg4300.txt"
    dataset_name = "fhswf/TinyStoriesV2_cleaned"

    mem_path = "exp/tttm"
    model_path = "exp/tttm_50000_4.h5"

    n = 1000
    p = 5
    max_tokens = 50000  # None
    iter = 4

    # MODEL

    tm = TriadicTemporalMemory(n, p)
    enc = TokenEncoder(n, p)

    # DATA

    # Load data either from a huggingface dataset or from the local data_dir

    if hf_dataset:
        log.info(f"Loading dataset {dataset_name}")
        dataset = load_dataset(dataset_name)
        train_text = dataset["train"]["text"]  # type: ignore
        test_text = dataset["test"]["text"]  # type: ignore
    else:
        log.info("Loading texts from data_dir")
        train_files = glob.glob(os.path.join(data_dir, "*.txt"))
        train_text, test_text = [], []
        for i, input_path in enumerate(train_files):
            with open(input_path, "r", encoding="utf-8") as file:
                text = file.read()
            if i < len(train_files) - 1:  # every file but the last
                train_text.append(text)
            else:
                test_text.append(text)

    # TODO: don't assign random vectors to tokens but check if there's
    # a similar token, and if so, use a vector nearby
    # TODO advanced: use sequitur or some other learned grammar to determine tokens

    # LEARNING

    if training:
        log.info("Learning has begun...")

        for i, text in enumerate(train_text):
            tokens = preprocess_text(text, max_tokens=max_tokens)
            total_tokens = 0

            tm_learn(tm, enc, tokens, iter=iter)
            total_tokens += len(tokens)

            if i % 1000 == 0:
                log.info(f"At training sample # {i:6d}")

            if i % 5000 == 0:
                rand_idx = random.randrange(len(test_text))
                test_tokens = preprocess_text(test_text[rand_idx])
                eval_tm(tm, enc, test_tokens)

                start_seq = ["one", "day", "the", "pink", "rabbit"]
                seq = generate_sequence(tm, enc, start_seq, length=23)
                log.info(" ".join(seq))

                if hf_dataset:
                    save_sdr(tm, enc, f"{mem_path}_.h5")

        # PERSISTENCE

        # Save final model

        tm_path = f"{mem_path}_{total_tokens}_{iter}.h5"
        save_sdr(tm, enc, tm_path)

    else:  # no learning, load model from disk
        tm, enc = load_sdr(model_path, TriadicTemporalMemory)
        # TODO: write test validating that tm and enc survive saving and loading unscathed

    # EVALUATION

    tokens = []
    max_eval_tokens = min(10000, len(tokens))
    for text in test_text:
        tokens.extend(preprocess_text(text, max_tokens=max_eval_tokens))

    log.info("Evaluation...")
    eval_tm(tm, enc, tokens)

    # Generate a sequence starting with "the pink rabbit"
    start_seq = ["the", "pink", "rabbit"]
    seq = generate_sequence(tm, enc, start_seq, length=23)
    log.info(" ".join(seq))

    # breakpoint()

    # PLAYGROUND

    interactive_demo(tm, enc)
