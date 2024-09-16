import json
from transformers import GPT2Tokenizer
from datasets import load_dataset
from tqdm import tqdm

# Load the cleaned dataset
dataset = load_dataset("json", data_files="TRAIN_DATASET_DIR/Train_dataset.json", split="train", streaming=True)

# Base tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Function to yield data in batches
def batch_iterator(batch_size=10):
    for _ in tqdm(range(0, len(dataset), batch_size)):
        yield [example["content"] for example in dataset.take(batch_size)]

# Train the tokenizer
new_tokenizer = tokenizer.train_new_from_iterator(batch_iterator(), vocab_size=50257)

# Save the tokenizer
new_tokenizer.save_pretrained("models/codeparrot-java-tokenizer")
print("Tokenizer trained and saved.")
