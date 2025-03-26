# AI Agentic Engineer

This project extends [The AI Scientist](https://github.com/SakanaAI/AI-Scientist) to create The AI Agentic Engineer that performs automatic testing of agentic framework using [Mixture-of-Agents](https://github.com/togethercomputer/MoA) approach and produce a short report.

Here are some example reports generated by **The AI Agentic Engineer** 📝:

1. [Enhancing Code Generation with a Mixture-of-Agents Framework and Structured Prompting](https://drive.google.com/file/d/18KiBk6UvHKQB0zuAlPqVaOtvgtLsRBb1/view?usp=drive_link)
2. [Optimizing Regression Performance with a Mixture-of-Agents Approach](https://drive.google.com/file/d/1RawCCvQFMxAmdyf35Wv2-_WzoapmIO5q/view?usp=sharing)
3. [Mixture-of-Agents Framework for Enhanced Ecommerce Recommendations](https://drive.google.com/file/d/1Qre1WaO0cWjQlsEk2ioYIFcIeEpj7MaU/view?usp=sharing)

## Introduction

We provide three templates, covering the following tasks: **Code Generation**, **Sequential Opmtimization**, and **Recommendation**. 
These tasks were selected based on their high relevance to Japanese industries.
These templates enable The AI Agentic Engineer to conduct experiments in these areas.

## Requirements

This project was tested on Linux with GPU, but it should also run with CPU, as we use training-free agentic approach.

### Installation

```bash
conda create -n ai_agentic_enginner python=3.11
conda activate ai_agentic_enginner
# Install pdflatex
sudo apt-get install texlive-full

# Install PyPI requirements
pip install -r requirements.txt
```

**Note:** Installing `texlive-full` can take a long time. You may need to [hold Enter](https://askubuntu.com/questions/956006/pregenerating-context-markiv-format-this-may-take-some-time-takes-forever) during the installation.


## Installation of Custom HumanEval: Hand-Written Evaluation Set

We use the custom HumanEval dataset to measure execution time of unit tests.

First, install this repository and set it up:
```
$ cd data
$ git clone https://github.com/openai/human-eval
$ pip install -e human-eval
$ cp human-eval/* . -r
$ tree
```

Make sure it runs samples for evaluation.

```python
$ sed -i 's/#                         exec/                        exec/g' human-eval/human_eval/execution.py
$ evaluate_functional_correctness data/example_samples.jsonl --problem_file=data/example_problem.jsonl
```
This sample execution should print like this:

```commandline
Reading samples...
6it [00:00, 5002.15it/s]
Running test suites...
100%|█████████████████████████████████████████████| 6/6 [00:03<00:00,  1.98it/s]
Writing results to data/example_samples.jsonl_results.jsonl...
100%|██████████████████████████████████████████| 6/6 [00:00<00:00, 28728.11it/s]
{'pass@1': 0.4999999999999999}


```

Finally, copy the custom `evaluation.py` under `human_eval`: 

```
$ cp evaluation.py /human-eval/human_eval
```

### Supported Models and API Keys

The supported models are the same as The AI Scientist. 
Please refer their [repository](https://github.com/SakanaAI/AI-Scientist) if you want to use any particular model.

This project uses the OpenAI API and the [Together API](https://api.together.xyz/).

```bash
export OPENAI_API_KEY="YOUR KEY HERE"
export TOGETHER_API_KEY="YOUR KEY HERE"
```

#### OpenAlex API (Literature Search Alternative)

We used OpenAlex API as an alternative.
OpenAlex does not require API key.

```bash
pip install pyalex
export OPENALEX_MAIL_ADDRESS="YOUR EMAIL ADDRESS"
```

And specify `--engine openalex` when you execute the AI Scientist code.

Note that this is experimental for those who do not have a Semantic Scholar API Key.

## Setting Up the Templates

This section provides instructions for setting up each of the three templates used in our report. Before running The AI Agentic Engineer experiments, please ensure you have completed the setup steps for the templates you are interested in.

### MoA for Code Generation Template

**Description:** This template tests the performance of the MoA model applied on code generation tasks. It asks MoA to generate a solution code given a programming problem. The evaluated pass rates of unit tests for each problem are presented in a table of the report.

**Setup Steps:**

1. **Create baseline runs and copy the results to `MoA_code`:**

   ```bash
   cd templates/SingleLLM_code
   python experiment.py --out_dir run_0
   cp -R run_0 MoA_code/run_0/
   ```

### MoA for Sequential Optimization Template

**Description:** This template applies MoA on optimization tasks. It asks MoA to generate weight for a regression model given the previous MSE performance history. The evaluated MSEs across 30 iterations are presented in a table and a plot. 

**Setup Steps:**

1. **Create baseline runs and copy the results to `MoA_optim`:**

   ```bash
   cd templates/SingleLLM_optim
   python experiment.py --out_dir run_0
   cp -R run_0 MoA_optim/run_0/
   ```

### MoA for Recommendation Template

**Description:** This template applies MoA on next-item recommendation tasks. It asks MoA to generate a top-10 ranked list of item IDs given a list of item IDs the user purchased. The evaluated NDCG@10 and HIT@10 across 100 users are presented in a table.

**Setup Steps:**

1. **Create baseline runs and copy the results to `MoA_recsys`:**

   ```bash
   cd templates/SingleLLM_recsys
   python experiment.py --out_dir run_0
   cp -R run_0 MoA_recsys/run_0/
   ```

## Run AI Agentic Engineer Report Generation Experiments

Since this project is for automatic testing, not for scientific discovery, idea is fixed to 1 idea, skipping idea generation and novelty check steps.

**Note:** Please ensure the setup steps above are completed before running these experiments.

```bash
conda activate ai_agentic_engineer
# Run the experiments and report generation.
python launch_scientist.py --model "gpt-4o-mini-2024-07-18" --experiment MoA_code --num-ideas 1 --skip-novelty-check --skip-idea-generation
```
