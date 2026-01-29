# Reproducibility study of FACTER
This repository is part of a reproducibility study of FACTER:

**FACTER: Fairness-Aware Conformal Thresholding and Prompt Engineering for Enabling Fair LLM-Based Recommender Systems**
*by Arya Fayyazi · Mehdi Kamal · Massoud Pedram*

Compared to the original implementation, this version:
- Refactors the code into a more modular and flexible pipeline
- Adds support for Djinni Recruitment Dataset

**Original paper**: https://icml.cc/virtual/2025/poster/44576

**Original code**: https://github.com/AryaFayyazi/FACTER/blob/Final_Version

## Installation

Install all required dependencies:

```bash
pip install -r Requirements.txt
```

## Acccess to LLMs

The repository is using [Transformers](https://pypi.org/project/transformers/) and there are many model checkpoints on the Hugging Face Hub that can be accessed with it. By default [Mistral - 7B - Instruct - v0.1](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1) is configured, this can be accessed without permission on Hugging Face Hub, but a login is still needed:

- `pip install -U "huggingface_hub"`
- `hf auth login`

For the following 2 models permission needs to be requested in order to be used:
- [Llama 2](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf)
- [Llama 3](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct)

## Mini Test (Debug Mode)

This is recommended to run a mini test using the default configuration to verify that the setup works before running full experiments:

```bash
python main.py --train_size 3 --max_iterations 3
```

## Basic Usage

Run the main pipeline with a selected LLM and dataset(s):

```bash
python main.py --llm_backbone mistralai/Mistral-7B-Instruct-v0.1 --datasets_used ml-1m amazon
```

## Configurable Arguments

The following values are supported for `--datasets_used`:

- `ml-1m` – MovieLens 1M dataset (default)

- `amazon` – Amazon Movies & TV reviews dataset + Amazon Movies & TV metadata dataset

- `hiring` – Djinni Recruitment Dataset

You can pass one or multiple datasets:

`--datasets_used ml-1m`

`--datasets_used ml-1m amazon`

You can also override many default settings directly from the command line, including:

- `--train_size` (debugging / fast runs)
- `--random_seed`
- `--max_iterations`

- `--alpha`

- `--lambda_fairness`

- `--n_reference`

- `--base_similarity`

- `--quantile_decay`

Run with `--help` to see all available options:

```bash
python main.py --help
```

## Reproducibility steps

### LLaMA-3
- `python main.py --llm_backbone meta-llama/Meta-Llama-3-8B-Instruct --datasets_used ml-1m`
- `python main.py --llm_backbone meta-llama/Meta-Llama-3-8B-Instruct --datasets_used amazon`

### LLaMA-2
*Note: results yield Valid@10 = 0.*
- `python main.py --llm_backbone meta-llama/Llama-2-7b-chat-hf --datasets_used ml-1m`

### Mistral
*Note: the improved implementation is required, as the base implementation results in Valid@10 = 0.*
- `python main.py --llm_backbone mistralai/Mistral-7B-Instruct-v0.1 --datasets_used ml-1m --improved`
- `python main.py --llm_backbone mistralai/Mistral-7B-Instruct-v0.1 --datasets_used amazon --improved`

## Carbon Emissions Tracking

To visualize energy usage and carbon emissions, install **CodeCarbon**:
```bash
pip install codecarbon[carbonboard]
```
Start the CarbonBoard dashboard locally:
```bash
carbonboard --filepath="emissions.csv" --port=3333
```

Then open your browser and navigate to http://localhost:3333