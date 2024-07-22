import torch
from torch.utils.data import DataLoader, IterableDataset
from transformers import GPT2Tokenizer, AutoTokenizer, AutoModelForCausalLM, AutoConfig, AdamW, get_scheduler
from transformers import HfArgumentParser, TrainingArguments
from accelerate import Accelerator
from datasets import load_dataset

class ConstantLengthDataset(IterableDataset):
    def __init__(self, tokenizer, dataset, infinite=False, seq_length=1024, num_of_sequences=1024, chars_per_token=3.6):
        self.tokenizer = tokenizer
        self.concat_token_id = tokenizer.bos_token_id
        self.dataset = dataset
        self.seq_length = seq_length
        self.input_characters = seq_length * chars_per_token * num_of_sequences
        self.epoch = 0
        self.infinite = infinite

    def __iter__(self):
        iterator = iter(self.dataset)
        more_examples = True
        while more_examples:
            buffer, buffer_len = [], 0
            while True:
                if buffer_len >= self.input_characters:
                    break
                try:
                    buffer.append(next(iterator)["content"])
                    buffer_len += len(buffer[-1])
                except StopIteration:
                    if self.infinite:
                        iterator = iter(self.dataset)
                        self.epoch += 1
                    else:
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

def create_dataloaders(tokenizer, args):
    train_data = load_dataset("json", data_files="cleaned_data_java/cleaned_dataset.json", split="train", streaming=True)
    train_data = train_data.shuffle(buffer_size=1000)
    
    train_dataset = ConstantLengthDataset(tokenizer, train_data, infinite=True, seq_length=args.seq_length)
    train_dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size)
    
    return train_dataloader

def evaluate(model, eval_dataloader, accelerator):
    model.eval()
    losses = []
    for step, batch in enumerate(eval_dataloader):
        with torch.no_grad():
            outputs = model(batch, labels=batch)
        loss = outputs.loss.repeat(eval_dataloader.batch_size)
        losses.append(accelerator.gather(loss))
    loss = torch.mean(torch.cat(losses))
    return loss.item(), torch.exp(loss).item()

def main():
    parser = HfArgumentParser(TrainingArguments)
    args = parser.parse_args_into_dataclasses()[0]

    accelerator = Accelerator()
    set_seed(args.seed)
    
    tokenizer = AutoTokenizer.from_pretrained("codeparrot-java-tokenizer")
    
    config_kwargs = {"vocab_size": len(tokenizer),
                     "scale_attn_by_layer_idx": True,
                     "reorder_and_upcast_attn": True}
    config = AutoConfig.from_pretrained('gpt2-large', **config_kwargs)
    model = AutoModelForCausalLM.from_config(config)
    
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=args.max_steps,
    )

    train_dataloader = create_dataloaders(tokenizer, args)
    model, optimizer, train_dataloader = accelerator.prepare(model, optimizer, train_dataloader)

    model.train()
    for step, batch in enumerate(train_dataloader):
        outputs = model(batch, labels=batch)
        loss = outputs.loss / args.gradient_accumulation_steps
        accelerator.backward(loss)
        if step % args.gradient_accumulation_steps == 0:
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        if step % args.eval_steps == 0 and step != 0:
            eval_loss, perplexity = evaluate(model, train_dataloader, accelerator)
            print(f"Step {step}: Eval Loss {eval_loss}, Perplexity {perplexity}")
            
            accelerator.save_state("codeparrot_java_model")
    
    eval_loss, perplexity = evaluate(model, train_dataloader, accelerator)
    print(f"Final Eval Loss {eval_loss}, Perplexity {perplexity}")
    model.save_pretrained("codeparrot_java_model")

if __name__ == "__main__":
    main()
