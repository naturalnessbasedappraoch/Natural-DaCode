import os
import json
from tqdm import tqdm

# Directory containing Java files (you can replace this with a generic relative or absolute path)
JAVA_FILES_DIR = os.path.join("path", "to", "your", "java_files")

# Output directory for cleaned data (now saving to "Train_dataset" folder)
TRAIN_DATASET_DIR = os.path.join("path", "to", "Train_dataset")

def read_java_files(directory):
    java_files = []
    for root, _, files in os.walk(directory):
        for file in tqdm(files, desc="Reading files"):
            if file.endswith(".java"):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        java_files.append(f.read())
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
    return java_files

def remove_duplicates(data):
    unique_files = list(set(data))
    return unique_files

def save_cleaned_data(data, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    output_file_path = os.path.join(output_dir, "Train_dataset.json")
    with open(output_file_path, "w") as f:
        for entry in data:
            json.dump({"content": entry}, f)
            f.write("\n")

def main():
    # Read Java files
    java_files = read_java_files(JAVA_FILES_DIR)
    print(f"Read {len(java_files)} Java files.")

    # Remove duplicates
    cleaned_data = remove_duplicates(java_files)
    print(f"Reduced to {len(cleaned_data)} unique Java files.")

    # Save cleaned data to "Train_dataset.json"
    save_cleaned_data(cleaned_data, TRAIN_DATASET_DIR)
    print(f"Cleaned data saved to {TRAIN_DATASET_DIR} as Train_dataset.json.")

if __name__ == "__main__":
    main()
