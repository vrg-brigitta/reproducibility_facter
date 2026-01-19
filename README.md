# fact_group_8
This repository is a replication of FACTER:

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