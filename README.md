# A Naturalness-based Approach For codecompletionmodels to distinguish CTdata and CLdata

# Code Completion with RoBERTa

This repository contains scripts and instructions for training a RoBERTa-based model to complete Java code. The project involves preprocessing Java files, training the model, and evaluating it on incomplete code snippets.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
  - [1. Convert Java Files to Text](#1-convert-java-files-to-text)
  - [2. Preprocess Text Files](#2-preprocess-text-files)
  - [3. Train the Model](#3-train-the-model)
  - [4. Evaluate the Model](#4-evaluate-the-model)
  - [5. Test the Model](#5-test-the-model)
- [Scripts](#scripts)
  - [java_to_txt.py](#java_to_txtpy)
  - [preprocess.py](#preprocesspy)
  - [run.py](#runpy)
- [License](#license)

## Prerequisites

- Python 3.6 or higher
- PyTorch
- Transformers (HuggingFace)
- tqdm
- javalang
- Other dependencies as listed in `requirements.txt`

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/roberta-code-completion.git
    cd roberta-code-completion
    ```

2. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

### 1. Convert Java Files to Text

Convert Java files into a single text file with the `java_to_txt.py` script.

```sh
python java_to_txt.py --input_dir path/to/java/files --output_file train.txt


![Class Hierarchy Diagram](codeparrot_cover.png)
# Codeparrot as CodecompletionModel

This repository contains scripts to train and evaluate a Java code completion model using GPT-2 architecture. The project is divided into four main parts: dataset preparation, tokenizer training, model training, and model evaluation.

## Directory Structure


![Class Hierarchy Diagram](codeparrot.PNG)


## Setup

1. **Clone the Repository**: Clone the repository and navigate to the project directory.

    ```sh
    git clone https://github.com/yourusername/java-code-completion.git
    cd java-code-completion
    ```

2. **Create a Virtual Environment and Install Dependencies**: Create a virtual environment and install the required dependencies.

    ```sh
    python -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    ```

3. **Place Java Files**: Place your `.java` files in the `data/java_files` directory.

## Usage

### 1. Prepare the Dataset
```sh
Run the dataset preparation script to read `.java` files, remove duplicates, and save the contaminated dataset

python prepare_java_dataset.py
```
### 2. Train the Tokenizer
```sh
Train a new tokenizer using the contaminated dataset.
python train_java_tokenizer.py
```
### 3. Train the Model
```sh
Train the language model using the tokenizer and dataset.
python train_java_model.py --output_dir codeparrot_java_model --num_train_epochs 3 --per_device_train_batch_size 4 --save_steps 1000 --eval_steps 1000 --logging_dir logs
```
```sh
Alternatively, you can use the shell script to run the entire process:
bash run_java_training.sh
```
### 4. Evaluate the Model
```sh
Evaluate the trained model on both datasets(contaminated and cleaned).
python evaluate_model.py
```

