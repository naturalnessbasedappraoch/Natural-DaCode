import os
import json
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from sklearn.metrics import accuracy_score

# Load the tokenizer and model from the output directory
tokenizer = AutoTokenizer.from_pretrained("models/codeparrot-java-model")
model = AutoModelForCausalLM.from_pretrained("models/codeparrot-java-model")

code_generator = pipeline('text-generation', model=model, tokenizer=tokenizer)

def generate_code_and_evaluate(file_path):
    with open(file_path, 'r') as f:
        data = [json.loads(line) for line in f]
    
    results = []
    total_accuracy = 0
    total_prompts = len(data)

    for entry in data:
        prompt = entry['input']
        ground_truth = entry['gt']
        
        generated_code = code_generator(prompt, max_new_tokens=50, num_return_sequences=1)[0]['generated_text']
        generated_text = generated_code[len(prompt):]
        
        ground_truth_tokens = tokenizer(ground_truth, return_tensors="pt")["input_ids"].squeeze().tolist()
        generated_tokens = tokenizer(generated_text, return_tensors="pt")["input_ids"].squeeze().tolist()
        
        min_length = min(len(ground_truth_tokens), len(generated_tokens))
        accuracy = accuracy_score(ground_truth_tokens[:min_length], generated_tokens[:min_length])
        total_accuracy += accuracy
        
        results.append({"prompt": prompt, "generated_code": generated_text, "ground_truth": ground_truth, "accuracy": accuracy})
    
    average_accuracy = total_accuracy / total_prompts
    print(f"Average Accuracy: {average_accuracy}")
    
    df = pd.DataFrame(results)
    df.to_excel("results/token_level_accuracy_results.xlsx", index=False)

# Path to the JSON file with inputs and ground truths
json_file_path = "data/test.json"
generate_code_and_evaluate(json_file_path)
