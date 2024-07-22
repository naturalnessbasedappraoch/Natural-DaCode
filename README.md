![Class Hierarchy Diagram](codeparrot_cover.png)
# Java Code Completion

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

Run the dataset preparation script to read `.java` files, remove duplicates, and save the cleaned dataset.

```sh
python prepare_java_dataset.py
**### 2. Train the Tokenizer**
python train_java_tokenizer.py

**### 3. Train the Model**
python train_java_model.py --output_dir codeparrot_java_model --num_train_epochs 3 --per_device_train_batch_size 4 --save_steps 1000 --eval_steps 1000 --logging_dir logs

