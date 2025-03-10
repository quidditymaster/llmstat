
__version__ = "0.0.1"

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
    Note that not all tokens will necessarily have a valid corresponding string (e.g. it is a single byte from a multiple byte UTF8 encoded symbol). 
    In such cases the behavior will be dictated by the particular tokenizer but the resulting string will not in general match the text which would result in the given token id sequence. 

    Args:
        token_ids (iterable or torch.Tensor): The token IDs to decode.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer.

    Returns:
        list[str]: A list of decoded tokens.
    """
    # Ensure each id is an integer before decoding
    return [tokenizer.decode(int(id_)) for id_ in token_ids]

def count_logp_bins(
    logp_values, #array shaped, N x V
    logp_thresholds, #array of thresholds, must be sorted
):
    """
    returns occurrence counts for each row of the logp_values array row by row. 
    The first column of the output represents the number of values less than the smallest value of logp_thresholds and the last column represents values greater than the greatest value of the logp_thresholds.
    The logp_thresholds array must be in sorted order for this function to work properly. 
    """
    N = len(logp_values)
    n_bins = len(logp_thresholds) + 1
    bin_counts = np.zeros((N, n_bins))
    for i in range(N):
        bin_counts[i] = np.bincount(
            np.searchsorted(logp_thresholds, logp_values[i]), 
            minlength=n_bins
        )
    return bin_counts

def calculate_distribution_statistics(
        text: str,
        tokenizer,
        model,
):
    """
    Calculates various probability distribution metrics for the provided text.

    Metrics computed per token include:
        - max_logit: Maximum predicted logit value over all possible tokens.
        - logsumexp: LogSumExp of the logits. This is the logarithm of the normalization constant of the predicted distribution.
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
