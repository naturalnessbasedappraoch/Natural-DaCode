# A Naturalness-based Approach For codecompletionmodels to distinguish CTdata and CLdata
![Overview](Overview.PNG)


# Dataset for Code Completion and N-gram Models

This folder contains all the datasets used for training and testing the models in the paper **"Has My Code Been Stolen for Model Training? A Naturalness-Based Approach to Code Contamination Detection"**. The dataset is organized into three main parts: **Train Dataset**, **Testing Dataset**, and **N-gram Train Dataset**.

## Folder Structure

### 1. `Train Dataset (for Code Completion Model)`
This folder contains the training datasets for two different code-completion models: **UniXcoder** and **CodeParrot**.

- **`UniXcoder/CodeParrot Training Data`**: 
    - Contains Java files that are used to train the UniXcoder model.
    - Example files: `train1.java`, `train2.java`, etc.
    
### 2. `N-gram Train Dataset`
This folder contains the data used to train the n-gram model. It includes various Java files:

- Example files: `ngram_train1.java`, `ngram_train2.java`, etc.


### 3. `Test Dataset (for Code Completion and Ngram Models)`
This folder contains the test datasets for both **(UniXcoder** and **CodeParrot)** models and Pretrain models **(ChatGPT3.5 and Claude)**. The test data is split into two categories:

- **`CLdata/`** (Cleaned Data):
    - Contains clean Java files that were not part of the model training data.
    - Example files: `CLtest1.java`, `CLtest2.java`, etc.
  
- **`CTdata/`** (Contaminated Data):
    - Contains Java files that were part of the model training data.
    - Example files: `CTtest1.java`, `CTtest2.java`, etc.


## Usage Instructions

- **Train Dataset**: Use the files in the `Train Dataset (for Code Completion Model)` folder for training the UniXcoder and CodeParrot models.
- **Test Dataset**: Use the files in the `Test Dataset (for Code Completion Model)` folder to evaluate the models. The `CLdata` folder contains clean test data, while the `CTdata` folder contains contaminated test data.
- **N-gram Train Dataset**: The `N-gram Train Dataset` folder contains the data required to train an n-gram model, which is used for code naturalness evaluation.

## Preprocessing

If needed, the datasets can be preprocessed using scripts found in the `src/` folder of the repository (e.g., for tokenizing the Java files or cleaning the data).

## External Datasets

If the dataset size exceeds the GitHub repository limits or is unavailable here, you can find the relevant links to download them in the `external_links.md` file.

## License

This dataset is shared under the MIT license. For more details, refer to the `LICENSE` file in the main repository.

## Acknowledgements

Please cite the paper **"Has My Code Been Stolen for Model Training? A Naturalness-Based Approach to Code Contamination Detection"** if you use this dataset in your work.
