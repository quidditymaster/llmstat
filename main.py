#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LLM Probability Metrics

This module provides functions to extract probability distribution metrics
(e.g., maximum logit, logsumexp, entropy, cross entropy) from a causal language model.
It uses the Hugging Face Transformers library.
"""

from llmstat import *

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
