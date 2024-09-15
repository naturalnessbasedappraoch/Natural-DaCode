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
- Other dependencies as listed in `requirements.txt`

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

