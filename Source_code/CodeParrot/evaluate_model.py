import json
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

def load_evaluation_data(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def evaluate_model(model_name, data_file, max_length=50):
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Load evaluation data
    eval_data = load_evaluation_data(data_file)

    # Initialize code generation pipeline
    code_completion = pipeline('text-generation', model=model, tokenizer=tokenizer, device=0 if torch.cuda.is_available() else -1)

    # Evaluate model
    correct_predictions = 0
    total_predictions = len(eval_data)

    for item in eval_data:
        input_code = item['input']
        expected_output = item['output']

        # Generate code completion
        generated_outputs = code_completion(input_code, max_length=len(tokenizer.encode(input_code)) + max_length, num_return_sequences=1)

        # Extract generated code
        generated_code = generated_outputs[0]['generated_text'][len(input_code):]

        # Compare with expected output
        if expected_output.strip() == generated_code.strip():
            correct_predictions += 1
        else:
            print(f"Input: {input_code}")
            print(f"Expected: {expected_output}")
            print(f"Generated: {generated_code}")
            print("----")

    accuracy = correct_predictions / total_predictions
    print(f"Accuracy: {accuracy * 100:.2f}%")

if __name__ == "__main__":
    model_name = "codeparrot_java_model"  # Replace with your model's name
    data_file = "evaluation_data.json"  # Path to your evaluation dataset
    evaluate_model(model_name, data_file)
