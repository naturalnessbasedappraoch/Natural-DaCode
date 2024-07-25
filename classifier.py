import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, roc_curve

# Function to load and preprocess the data
def load_and_preprocess_data(file_path):
    # Load the dataset
    data = pd.read_excel(file_path)
    
    # Select relevant columns and rename for clarity
    data_filtered = data[['Token-level Accuracy', 'Naturalness', 'Source']]
    data_filtered.columns = ['performance', 'naturalness', 'source']
    
    # Prepare the input features and output target
    X = data_filtered[['performance', 'naturalness']]
    y = data_filtered['source']
    
    return X, y

# Function to encode target labels and split the data
def encode_and_split_data(X, y):
    # Encode the categorical target variable
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # Split the data into train and test sets with an 80:20 ratio
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test, label_encoder

# Function to scale the features
def scale_features(X_train, X_test):
    # Scale the input features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled

# Function to evaluate the model and return metrics
def evaluate_model(classifier, X_train, X_test, y_train, y_test, label_encoder):
    # Fit the model on the training data
    classifier.fit(X_train, y_train)
    
    # Make predictions on the test data
    y_pred = classifier.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=label_encoder.classes_, output_dict=True)
    
    if hasattr(classifier, "predict_proba"):
        y_prob = classifier.predict_proba(X_test)[:, 1]
    elif hasattr(classifier, "decision_function"):
        y_prob = classifier.decision_function(X_test)
    else:  # For Naive Bayes which doesn't have predict_proba or decision_function
        y_prob = classifier.predict(X_test)
    
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)
    
    return {
        "Accuracy": accuracy,
        "TPR": np.mean(tpr),
        "FPR": np.mean(fpr),
        "AUC": auc
    }

# Main function to run the classification evaluation
def main(input_file_path, output_file_path):
    # Load and preprocess data
    X, y = load_and_preprocess_data(input_file_path)
    
    # Encode and split data
    X_train, X_test, y_train, y_test, label_encoder = encode_and_split_data(X, y)
    
    # Scale features
    X_train_scaled, X_test_scaled = scale_features(X_train, X_test)
    
    # Classifiers to evaluate
    classifiers = {
        'SVM': SVC(probability=True),
        'Decision Tree': DecisionTreeClassifier(),
        'Random Forest': RandomForestClassifier(),
        'KNN': KNeighborsClassifier(),
        'Logistic Regression': LogisticRegression(),
        'Naive Bayes': GaussianNB(),
        'Gradient Boosting': GradientBoostingClassifier(),
        'Linear SVC': LinearSVC()
    }
    
    # Evaluate the classifiers
    results = []
    for classifier_name, classifier in classifiers.items():
        metrics = evaluate_model(classifier, X_train_scaled, X_test_scaled, y_train, y_test, label_encoder)
        results.append({
            "Classifier": classifier_name,
            **metrics
        })
    
    # Convert the results to a DataFrame for better visualization
    results_df = pd.DataFrame(results)
    
    # Save the results to an Excel file
    results_df.to_excel(output_file_path, index=False)
    
    # Print the results table
    print(results_df)

# Example usage
if __name__ == "__main__":
    input_file_path = "path_to_input_file.xlsx"  # Update this to your file path
    output_file_path = "path_to_output_file.xlsx"  # Update this to your desired output path
    main(input_file_path, output_file_path)
