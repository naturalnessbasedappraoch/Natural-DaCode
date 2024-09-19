import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
import os

# Define the relative file paths (assuming the files are in the 'datasets' folder in your repository)
datasets = {
    "unixcoder": os.path.join("datasets", "unixcoder.xlsx"),
    "codeparrot": os.path.join("datasets", "codeparrot.xlsx"),
    "chatgpt": os.path.join("datasets", "chatgpt.xlsx"),
    "claude": os.path.join("datasets", "claude.xlsx")
}

# Define the SVM classifier
classifier = SVC(probability=True)

# Function to evaluate the model and return metrics
def evaluate_model(classifier, X_train, X_test, y_train, y_test):
    # Fit the model on the training data
    classifier.fit(X_train, y_train)
    
    # Make predictions on the test data
    y_pred = classifier.predict(X_test)
    y_prob = classifier.predict_proba(X_test)[:, 1]
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    
    # Calculate confusion matrix and derive TPR and FPR
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    tpr = tp / (tp + fn)  # True Positive Rate
    fpr = fp / (fp + tn)  # False Positive Rate
    
    # Calculate AUC
    auc = roc_auc_score(y_test, y_prob)
    
    return accuracy, tpr, fpr, auc

# Dictionary to store the results
results = {'Accuracy': [], 'TPR': [], 'FPR': [], 'AUC': []}

# Iterate over each dataset
for dataset_name, file_path in datasets.items():
    # Load the dataset
    data = pd.read_excel(file_path)

    # Select relevant columns (using original column names)
    data_filtered = data[['Token-level Accuracy', 'Naturalness', 'Source']]

    # Filter for CTdata and CLdata labels in the 'Source' column
    data_filtered = data_filtered[data_filtered['Source'].isin(['CTdata', 'CLdata'])]

    # Prepare the input features and output target
    X = data_filtered[['Token-level Accuracy', 'Naturalness']]
    y = data_filtered['Source']

    # Encode the categorical target variable (CTdata, CLdata)
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)  # CTdata -> 0, CLdata -> 1

    # Split the data into train and test sets with a 50:50 ratio
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.5, random_state=42)

    # Scale the input features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Evaluate the classifier
    accuracy, tpr, fpr, auc = evaluate_model(classifier, X_train_scaled, X_test_scaled, y_train, y_test)
    results['Accuracy'].append(accuracy * 100)  # Convert to percentage
    results['TPR'].append(tpr)
    results['FPR'].append(fpr)
    results['AUC'].append(auc * 100)  # Convert to percentage

# Convert the results to a DataFrame for better visualization
results_df = pd.DataFrame(results, index=datasets.keys())

# Save the results to an Excel file (generic path for GitHub repository)
output_file_path = os.path.join("datasets", "results.xlsx")
results_df.to_excel(output_file_path, index=True)

# Print the results table
print(results_df)
