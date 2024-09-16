import torch
import argparse
from torch.utils.data import DataLoader, IterableDataset
from transformers import GPT2Tokenizer, AutoModelForCausalLM, AutoConfig, AdamW, get_scheduler
from accelerate import Accelerator
from datasets import load_dataset

class ConstantLengthDataset(IterableDataset):
    def __init__(self, tokenizer, dataset, infinite=False, seq_length=1024):
        self.tokenizer = tokenizer
        self.concat_token_id = tokenizer.bos_token_id
        self.dataset = dataset
        self.seq_length = seq_length

    def __iter__(self):
        iterator = iter(self.dataset)
        more_examples = True
        while more_examples:
            buffer, buffer_len = [], 0
            while True:
                if buffer_len >= self.seq_length:
                    break
                try:
                    buffer.append(next(iterator)["content"])
                    buffer_len += len(buffer[-1])
                except StopIteration:
                    more_examples = False
                    break
            tokenized_inputs = self.tokenizer(buffer, truncation=False)["input_ids"]
            all_token_ids = []
            for tokenized_input in tokenized_inputs:
                all_token_ids.extend(tokenized_input + [self.concat_token_id])
            for i in range(0, len(all_token_ids), self.seq_length):
                input_ids = all_token_ids[i: i + self.seq_length]
                if len(input_ids) == self.seq_length:
                    yield torch.tensor(input_ids)

def create_dataloaders(tokenizer, seq_length=1024, batch_size=4):
    train_data = load_dataset("json", data_files="TRAIN_DATASET_DIR/Train_dataset.json", split="train", streaming=True)
    train_dataset = ConstantLengthDataset(tokenizer, train_data, infinite=True, seq_length=seq_length)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
    return train_dataloader

def main(args):
    tokenizer = GPT2Tokenizer.from_pretrained("models/codeparrot-java-tokenizer")
    config = AutoConfig.from_pretrained("gpt2-large", vocab_size=len(tokenizer))
    model = AutoModelForCausalLM.from_config(config)
    
    accelerator = Accelerator()
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    lr_scheduler = get_scheduler(
        name="linear", 
        optimizer=optimizer, 
        num_warmup_steps=args.num_warmup_steps, 
        num_training_steps=args.num_training_steps
    )
    
    train_dataloader = create_dataloaders(tokenizer, seq_length=args.seq_length, batch_size=args.per_device_train_batch_size)
    model, optimizer, train_dataloader = accelerator.prepare(model, optimizer, train_dataloader)
    
    for step, batch in enumerate(train_dataloader):
        outputs = model(batch, labels=batch)
        loss = outputs.loss
        accelerator.backward(loss)
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
    
    model.save_pretrained(args.output_dir)
    print(f"Model trained and saved at {args.output_dir}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Add command-line arguments
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the trained model")
    parser.add_argument("--num_train_epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--per_device_train_batch_size", type=int, default=8, help="Batch size per device during training")
    parser.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every X updates steps")
    parser.add_argument("--eval_steps", type=int, default=500, help="Run evaluation every X steps")
    parser.add_argument("--logging_dir", type=str, default="logs", help="Logging directory")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Number of gradient accumulation steps")
    parser.add_argument("--seq_length", type=int, default=1024, help="Sequence length for the model")
    parser.add_argument("--num_warmup_steps", type=int, default=500, help="Number of warmup steps")
    parser.add_argument("--num_training_steps", type=int, default=10000, help="Total number of training steps")

    args = parser.parse_args()

    main(args)
