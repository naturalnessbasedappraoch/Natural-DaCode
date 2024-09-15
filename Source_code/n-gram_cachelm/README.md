# Computation of Naturalness(same as Ray at al)

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Preprocessing](#preprocessing)
    - [Step 1: Extract and Preprocess Java Files](#step-1-extract-and-preprocess-java-files)
4. [Evaluation](#evaluation)
    - [Step 2: Prepare Evaluation Directory](#step-2-prepare-evaluation-directory)
5. [Training](#training)
    - [Step 3: Train Language Model](#step-3-train-language-model)
6. [Testing](#testing)
    - [Step 4: Run Tests](#step-4-run-tests)
7. [Troubleshooting](#troubleshooting)
8. [License](#license)

## Overview

This project preprocesses Java files, trains a language model using n-grams, and evaluates the model's performance through 10-fold cross-validation.

## Prerequisites

- Python 3.x
- Java (for the Lexer)
- SRILM toolkit (for n-gram language modeling)

## Preprocessing

### Step 1: Extract and Preprocess Java Files

1. **Extract Java Files**: Ensure all Java files are in a directory called `input_dir`.

2. **Run Preprocessing Script**:
    ```bash
    ./walk.py input_dir output_dir
    ```
    - `output_dir` will contain:
        - `0.java`: Un-tokenized code without comments.
        - `0.java.lines`: Tokenized code stored in lines.
        - `0.java.tokens`: Tokenized code written in a single line (for training LM).

3. **Modify Lexer for Other Languages**: (Optional)
    - To process other programming languages, modify the lexer in `walk.py`:
        ```python
        os.system('java -jar bin/LexJava-1.0.jar %s' % output)
        ```

## Evaluation

### Step 2: Prepare Evaluation Directory

1. **Rename Output Folder**:
    ```bash
    mv output_dir files
    ```

2. **Create Evaluation Directory Structure**:
    ```bash
    mkdir -p evaluation/data/sample_project/files
    mv files/* evaluation/data/sample_project/files
    ```

## Training

### Step 3: Train Language Model

1. **Run Training Script**:
    ```bash
    ./bin/train.py evaluation/data/sample_project 3
    ```
    - `3` denotes the order of n-grams.
        ```

## Testing

### Step 4: Run Tests

1. **Find Full Command List**:
    ```bash
    ./completion
    ```

2. **Necessary Parameters**:
    ```bash
    -INPUT_FILE         the input file
    -NGRAM_FILE         the ngrams file
    -NGRAM_ORDER        the value of N (order of lm)
    ```

3. **Optional Parameters**:
    ```bash
    -ENTROPY            calculate the cross entropy of the test file rather than providing the suggestions
    -TEST               test mode, no output, no debug information
    -FILES              test on files or not, default on a single file
    -DEBUG              output debug information
    -OUTPUT_FILE        the output file
    -BACKOFF            use the back-off technique
    -CACHE              use the cache technique 
    -CACHE_ONLY         only use the cache technique without ngrams
    -CACHE_ORDER        the maximum order of ngrams used in the cache (default: 3)
    -CACHE_DYNAMIC_LAMBDA   dynamic interpolation weight for -CACHE (H/(H+1)), default option
    -CACHE_LAMBDA       interpolation weight for -CACHE
    -WINDOW_CACHE       build the cache on a window of n tokens (default n=1000)
    -WINDOW_SIZE        the size of cache, default: 1000 tokens
    -FILE_CACHE         build the cache on a file or related files
    -SCOPE_FILE         the scope file for scope cache on CLASS or METHOD
    -RELATED_FILE       when using cache on file scope, build the cache on the related files
                       FILE_DIR should be given
    -FILE_DIR           the directory that stores all files
    ```

4. **Example Scripts**:
    - Refer to `entropy.bat` and `suggestion.bat` for examples on calculating entropies and code suggestions.

## Troubleshooting

- **SRILM Tools Issue**: If `ngram` and `ngram_count` do not work, download the latest version from [SRILM](http://www.speech.sri.com/projects/srilm/download.html) and compile it on your machine. Replace the original tools with the compiled versions.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
