#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LLM Probability Metrics

This module provides functions to extract probability distribution metrics
(e.g., maximum logit, logsumexp, entropy, cross entropy) from a causal language model.
It uses the Hugging Face Transformers library.

Example:
    To run a demo:
        python llm_probability_metrics.py
"""

import numpy as np
import torch
import scipy.special
from transformers import AutoTokenizer, AutoModelForCausalLM


def get_continuation_logits(
        text: str,
        tokenizer,
        model,
        detach: bool = True,
        squeeze: bool = True,
        return_token_ids: bool = True,
):
    """
    Tokenizes the input text and returns the model's predicted logits.

    Args:
        text (str): The input text to process.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer.
        model (transformers.PreTrainedModel): The language model.
        detach (bool): If True, detach the logits from the computation graph.
        squeeze (bool): If True, remove the batch dimension.
        return_token_ids (bool): If True, also return the token ids.

    Returns:
        If return_token_ids is True, returns a tuple (logits, token_ids),
        otherwise just the logits.
    """
    token_ids = tokenizer(text, return_tensors="pt")["input_ids"]
    logits = model(token_ids)["logits"]
    if detach:
        logits = logits.detach()
    if squeeze:
        logits = logits.squeeze(0)
        token_ids = token_ids.squeeze(0)
    if return_token_ids:
        return logits, token_ids
    return logits


def decode_individual_tokens(token_ids, tokenizer):
    """
    Decodes a sequence of token IDs into their corresponding string representations.

    Args:
        token_ids (iterable or torch.Tensor): The token IDs to decode.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer.

    Returns:
        list[str]: A list of decoded tokens.
    """
    # Ensure each id is an integer before decoding
    return [tokenizer.decode(int(id_)) for id_ in token_ids]


def calculate_distribution_statistics(
        text: str,
        tokenizer,
        model,
):
    """
    Calculates various probability distribution metrics for the provided text.

    Metrics computed per token include:
        - max_logit: Maximum logit value.
        - logsumexp: LogSumExp of the logits.
        - extension_entropy: Entropy of the predicted distribution.
        - log_prob_of_actual: Log probability of the actual next token.

    Args:
        text (str): The input text to analyze.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer.
        model (transformers.PreTrainedModel): The language model.

    Returns:
        dict: A dictionary containing the computed metrics.
    """
    logits, token_ids = get_continuation_logits(
        text, tokenizer, model, squeeze=True, return_token_ids=True
    )
    # Ensure tensors are on CPU and convert to numpy
    logits = logits.cpu().numpy()
    token_ids = token_ids.cpu().numpy()

    # Compute metrics across the vocabulary (axis=1)
    max_logit = np.max(logits, axis=1)
    logsumexp_vals = scipy.special.logsumexp(logits, axis=1)
    log_prob = logits - logsumexp_vals[:, np.newaxis]
    prob = np.exp(log_prob)
    extension_entropy = -np.sum(prob * log_prob, axis=1)

    # Identify the EOS (end-of-sequence) token id from the tokenizer
    eos_token_id = tokenizer(tokenizer.special_tokens_map["eos_token"])["input_ids"][0]
    # Create a shifted version of token_ids to represent the "actual" next token
    actual_next = np.hstack([token_ids[1:], eos_token_id])
    log_prob_of_actual = log_prob[np.arange(len(logits)), actual_next]

    return {
        "max_logit": max_logit,
        "logsumexp": logsumexp_vals,
        "extension_entropy": extension_entropy,
        "log_prob_of_actual": log_prob_of_actual,
    }


def main():
    """
    Demonstrates the use of the module functions.

    Loads a pretrained model and tokenizer (defaulting to distilgpt2),
    processes an example sentence, and prints out the computed metrics.
    """
    # Define model checkpoints (you can change these as needed)
    tokenizer_checkpoint = "distilgpt2"
    model_checkpoint = tokenizer_checkpoint

    # Load the tokenizer and model from Hugging Face
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_checkpoint, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(model_checkpoint)

    # Example sentence
    sentence = "The cat sat on the mat"

    # Prepend beginning-of-sequence token if available
    if "bos_token" in tokenizer.special_tokens_map and tokenizer.special_tokens_map["bos_token"]:
        input_text = tokenizer.special_tokens_map["bos_token"] + sentence
    else:
        input_text = sentence

    print("Input text:", input_text)

    # Get the logits and token ids for the input text
    logits, token_ids = get_continuation_logits(input_text, tokenizer, model)
    print("Logits shape:", logits.shape)
    print("Token IDs shape:", token_ids.shape)

    # Decode the input tokens for inspection
    initial_tokens = decode_individual_tokens(token_ids, tokenizer)
    print("Initial tokens:", initial_tokens)

    # Greedy decoding: choose the highest logit at each position
    greedy_token_ids = np.argmax(logits.cpu().numpy(), axis=1)
    greedy_tokens = decode_individual_tokens(greedy_token_ids, tokenizer)
    print("Greedy decoded tokens:", greedy_tokens)

    # Calculate and display distribution statistics
    stats = calculate_distribution_statistics(input_text, tokenizer, model)
    print("\nDistribution statistics:")
    for key, value in stats.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
