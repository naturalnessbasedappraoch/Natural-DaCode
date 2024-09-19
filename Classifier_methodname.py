import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import os

# Load the combined dataset (CLdata + CTdata) with a 'Source' column indicating 'CTdata' or 'CLdata'
combined_data = pd.read_excel(os.path.join('datasets', 'data.xlsx'))

# Check if 'Source' column exists and contains the values 'CTdata' and 'CLdata'
assert 'Source' in combined_data.columns, "'Source' column is missing from the dataset."
assert set(combined_data['Source'].unique()) == {'CTdata', 'CLdata'}, "Source column should only contain 'CTdata' and 'CLdata'."

# Drop rows with missing values
combined_data = combined_data.dropna()

# Features to be used for training
features = ['Method name naturalness', 'Method body Naturalness', 'Method Name Length', 'Method body Length', 'Edit Distance']

# Target (Source: CTdata or CLdata)
target = 'Source'

# Split the data into features (X) and target (y)
X = combined_data[features]
y = combined_data[target]

# Stratified train-test split ensures equal representation of both classes
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, stratify=y, random_state=42)

# Standardize the feature data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize the SVM model with RBF kernel and balanced class weights
svm_model = SVC(kernel='rbf', probability=True, class_weight='balanced')

# Train the model
svm_model.fit(X_train_scaled, y_train)

# Predict the source (CTdata or CLdata) on the test set using the model
y_pred = svm_model.predict(X_test_scaled)
y_pred_prob = svm_model.predict_proba(X_test_scaled)[:, 1]  # Probabilities for AUC calculation

# Output the classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred, labels=['CTdata', 'CLdata'])
TN, FP, FN, TP = conf_matrix.ravel()  # Unpack the confusion matrix correctly

# Overall Accuracy
accuracy = (TP + TN) / (TP + TN + FP + FN)

# True Positive Rate (TPR) for class CTdata
tpr = TP / (TP + FN) if (TP + FN) != 0 else 0  # Avoid division by zero

# False Positive Rate (FPR) for class CTdata
fpr = FP / (FP + TN) if (FP + TN) != 0 else 0  # Avoid division by zero

# ROC AUC Score for combined classes CTdata and CLdata
y_test_binary = [1 if x == 'CLdata' else 0 for x in y_test]  # Convert 'CLdata' to 1 and 'CTdata' to 0 for binary AUC calculation
auc = roc_auc_score(y_test_binary, y_pred_prob)

# Print combined results for overall evaluation
print(f"Accuracy: {accuracy:.4f}")
print(f"True Positive Rate (TPR): {tpr:.4f}")
print(f"False Positive Rate (FPR): {fpr:.4f}")
print(f"AUC: {auc:.4f}")
