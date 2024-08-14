# AutoTune: Fine-Tuning Pipeline for OpenAI Models with Structured Outputs

This repository contains a Jupyter notebook that automates the process of fine-tuning OpenAI models for structured outputs using the OpenAI API. The pipeline uses GPT-4o as the teacher model and GPT-4o-mini as the student model.

## Overview

The `autotune.ipynb` notebook guides you through the following steps to fine-tune models for structured outputs:

1. Preparing a batch file with prompts
2. Processing the batch using the teacher model (GPT-4o)
3. Processing batch results
4. Preparing fine-tuning data
5. Starting the fine-tuning process with the student model (GPT-4o-mini)

## Prerequisites

- Python 3.12
- Jupyter Notebook
- OpenAI API key

## Setup

1. Clone this repository.
2. Install the required dependencies `openai`, `pydantic`, `python-dotenv`.
3. In `utils.py`, set a path to an `.env` file containing your export `OPENAI_API_KEY`="".

## Usage

1. Open `autotune.ipynb` in Jupyter Notebook.
2. Define a structured output response model using Pydantic in the second code cell.
3. Set constants in the third code cell to control prompts, models, and other parameters.
4. Update the Pydantic model name in the fourth code cell to match your defined response model.
5. Run the cells in sequence.

The final cell will print a job ID that you can use to monitor the fine-tuning job and evaluate the model in the OpenAI dashboard.
  - `prompts/`: Directory for prompt files
  - `batch_files/`: Directory for batch input and output files
  - `finetune_files/`: Directory for fine-tuning data files


## Structured Outputs

This pipeline is specifically designed for fine-tuning models to produce structured outputs. The structured output is defined using a Pydantic model, which ensures that the model learns to generate responses in a consistent, predefined format. This is particularly useful for tasks that require specific data structures or formats in the output.

## Note

This pipeline is designed to work with OpenAI's API and the recently released GPT-4o and GPT-4o-mini models. Make sure to update the model names and API endpoints if they change in the future.

