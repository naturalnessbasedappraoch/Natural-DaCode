# Code Completion with UniXcoder

This repository contains scripts and instructions for training a RoBERTa-based model to complete Java code. The project involves preprocessing Java files, training the model, and evaluating it on incomplete code snippets.

## Table of Contents
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
  - [1. Convert Java Files to Text](#1-convert-java-files-to-text)
  - [2. Preprocess Text Files](#2-preprocess-text-files)
  - [3. Train the Model](#3-train-the-model)
  - [4. Test the Model](#4-test-the-model)
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
- Dependencies as listed in `requirements.txt`

## Installation
1. Clone the repository:
    ```sh
    git clone https://github.com/naturalnessbasedappraoch/A-naturalnessbasedappraoch-for-Contamination-Detection-.git
    cd UniXcoder
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
```
## Usage

### 1. Convert Java Files to Text

Convert Java files into a single text file with the `java_to_txt.py` script.

```sh
python java_to_txt.py --input_dir path/to/java/files --output_file train.txt
```
### 2. Preprocess Text Files
Preprocess the text files to tokenize and format the data using preprocess.py.
```sh
python preprocess.py --base_dir . --output_dir preprocessed_data
```
### 3. Train the Model
Train the model using the run.py script. Make sure to specify the directories containing the preprocessed files
```sh
python run.py \
    --do_train \
    --do_eval \
    --lang java \
    --model_name_or_path roberta-base \
    --train_filename dataset/javaCorpus/train.txt \
    --dev_filename dataset/javaCorpus/dev.json \
    --output_dir saved_models/javaCorpus \
    --max_source_length 936 \
    --max_target_length 64 \
    --beam_size 5 \
    --train_batch_size 32 \
    --gradient_accumulation_steps 1 \
    --eval_batch_size 32 \
    --learning_rate 2e-5 \
    --num_train_epochs 10

```
### 4. Test the Model
Test the model with incomplete code snippets.
```sh
python run.py \
	--do_test \
	--lang java \
	--model_name_or_path roberta-base \
	--load_model_path saved_models/javaCorpus/checkpoint-best-acc/pytorch_model.bin \
	--test_filename dataset/javaCorpus/test.json \
  --output_dir saved_models/javaCorpus \
  --max_source_length 936 \
  --max_target_length 64 \
  --beam_size 5 \
  --eval_batch_size 328
```



