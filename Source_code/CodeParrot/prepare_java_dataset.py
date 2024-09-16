import os
import json
from tqdm import tqdm

# Directory containing Java files
JAVA_FILES_DIR = os.path.join("data", "raw_java_files")

# Output directory for cleaned data
CLEANED_DATA_DIR = os.path.join("data", "cleaned")

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
    with open(os.path.join(output_dir, "cleaned_dataset.json"), "w") as f:
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

    # Save cleaned data
    save_cleaned_data(cleaned_data, CLEANED_DATA_DIR)
    print(f"Cleaned data saved to {CLEANED_DATA_DIR}.")

if __name__ == "__main__":
    main()
