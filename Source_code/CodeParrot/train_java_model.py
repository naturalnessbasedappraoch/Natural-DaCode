import torch
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
    train_data = load_dataset("json", data_files="data/cleaned_dataset.json", split="train", streaming=True)
    train_dataset = ConstantLengthDataset(tokenizer, train_data, infinite=True, seq_length=seq_length)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
    return train_dataloader

def main():
    tokenizer = GPT2Tokenizer.from_pretrained("models/codeparrot-java-tokenizer")
    config = AutoConfig.from_pretrained("gpt2-large", vocab_size=len(tokenizer))
    model = AutoModelForCausalLM.from_config(config)
    
    accelerator = Accelerator()
    optimizer = AdamW(model.parameters(), lr=5e-5)
    lr_scheduler = get_scheduler(name="linear", optimizer=optimizer, num_warmup_steps=500, num_training_steps=10000)
    
    train_dataloader = create_dataloaders(tokenizer)
    model, optimizer, train_dataloader = accelerator.prepare(model, optimizer, train_dataloader)
    
    for step, batch in enumerate(train_dataloader):
        outputs = model(batch, labels=batch)
        loss = outputs.loss
        accelerator.backward(loss)
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
    
    model.save_pretrained("models/codeparrot-java-model")
    print("Model trained and saved.")

if __name__ == "__main__":
    main()
