from datasets import load_dataset
import os

# List of splits to check
splits = ['WikiMIA_length32', 'WikiMIA_length64', 'WikiMIA_length128', 'WikiMIA_length256']

# Function to save text files based on the label
def save_text_files(base_dir):
    # Directory path for saving the text files
    save_dir = base_dir
    
    # Ensure the 'seen' and 'unseen' subfolders exist
    seen_folder = os.path.join(save_dir, 'seen')
    unseen_folder = os.path.join(save_dir, 'unseen')
    os.makedirs(seen_folder, exist_ok=True)
    os.makedirs(unseen_folder, exist_ok=True)

    # Initialize counters for overall seen and unseen documents
    overall_seen_count = 0
    overall_unseen_count = 0

    # Iterate through each split and process the documents
    for split in splits:
        # Load the dataset for the current split
        wikimia_dataset = load_dataset("swj0419/WikiMIA", split=split)
        
        # Initialize counters for seen and unseen documents in this split
        seen_count = 0
        unseen_count = 0
        
        # Iterate over each example in the split and save text to corresponding folders
        for idx, example in enumerate(wikimia_dataset):
            label = example.get('label', -1)  # Default to -1 in case label is missing
            text = example['input']
            
            # Determine the folder based on the label
            if label == 1:
                seen_count += 1
                overall_seen_count += 1
                folder = seen_folder
            elif label == 0:
                unseen_count += 1
                overall_unseen_count += 1
                folder = unseen_folder
            else:
                continue  # Skip if label is not 0 or 1
            
            # Save the text content to a text file
            file_path = os.path.join(folder, f"{split}_doc_{idx}.txt")
            with open(file_path, 'w') as f:
                f.write(text)

        # Print the total number of seen and unseen documents for the current split
        print(f"Total seen documents in {split}: {seen_count}")
        print(f"Total unseen documents in {split}: {unseen_count}")

    # Print the overall counts for seen and unseen documents
    print(f"\nOverall total seen documents: {overall_seen_count}")
    print(f"Overall total unseen documents: {overall_unseen_count}")


# Example usage: replace this with the desired base directory
base_dir = '/path/to/your/directory'  # Change this to your desired path

# Call the function to save the text files
save_text_files(base_dir)

