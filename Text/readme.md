# Contamination Detection of Textual Data with Natural-DaCoDe

This repository contains instructions on how to classify contaminated data with **Natural-DaCoDe**, including how to access the dataset.
 
### 1. `Downloading the Datasets`

```bash
python execute_download.py --output_folder <path_to_output_folder>
```
### 2. `Train Dataset (for N-gram Model)`
This folder contains the data used to train the n-gram model. It includes various text files:

- Example files: `file1.txt`, `file2.txt`, etc.
To train the N-gram model, you can use the code provided in the [Ngram Model](https://github.com/naturalnessbasedappraoch/Natural-DaCode/tree/main/Source_code/n-gram_cachelm) repository. Follow the instructions in the repository to preprocess the data and train the model.

### 3. `Test Dataset (for Text Completion )`

The folder you download from script contains the test datasets for **ChatGPT3.5**. The test data is split into two categories:

- **`Seen`** (Cleaned Data):
    - It contains text files not part of the CahtGPT3.5 training data.
    - Example files: `file1.txt`, `file2.txt`, etc.
  
- **`Unseen`** (Contaminated Data):
    - It contains text files which are part of the ChatGPT3.5.
    - Example files: `file1.txt`, `file2.txt`, etc.

## Steps for Contamination Detection

### 1. **Performance and Naturalness:**
  - **Performance:** Calculate the token-level accuracy from text Generation Models(ChatGPT3.5).
  - **Naturalness:** Evaluate the naturalness scores for these text snippets using the N-gram model.

### 2. **Train the Classifier**
    - Combine performance metrics and naturalness scores.
    - Train an SVM classifier to distinguish between contaminated and cleaned data.

### 3. **Predict Contamination Source**
-Use the trained SVM classifier to predict whether a text is contaminated (CTdata) or cleaned (CLdata).

### 4. **Usage**
To perform contamination detection use the following scripts available in the repository:

   ```bash
   python Classifier_performance.py --datasets_dir path_to_your_datasets_folder
```
