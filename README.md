# LLM Probability Metrics

This repository contains a Python module that extracts probability distribution metrics from the output logits of a causal language model. It leverages the [Hugging Face Transformers](https://github.com/huggingface/transformers) library and supports analysis using popular models such as `distilgpt2`.

## Features

- **Tokenization & Inference:** Easily tokenize input text and generate output logits using a pretrained causal language model.
- **Probability Distribution Metrics:** Compute a variety of metrics from model output:
  - **Maximum Logit:** The highest logit value per token.
  - **LogSumExp:** The log-sum-exp value across the logits.
  - **Extension Entropy:** Entropy of the predicted distribution.
  - **Log Probability of Actual Token:** The log probability corresponding to the true continuation.
- **Greedy Decoding:** Decode tokens by selecting the highest logit at each time step.
- **Example Script:** A `main()` function demonstrates usage with a sample sentence.

## Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/your-username/llm_probability_metrics.git
   cd llm_probability_metrics
