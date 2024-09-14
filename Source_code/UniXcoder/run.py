import os
import random
import logging
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from transformers import (RobertaConfig, RobertaTokenizer, RobertaModel, AdamW, get_linear_schedule_with_warmup)
from tqdm import tqdm
import json
import re
import multiprocessing

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

class Example:
    def __init__(self, idx, source):
        self.idx = idx
        self.source = source

def read_examples_from_txt(filename):
    examples = []
    with open(filename, encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if line:
                examples.append(Example(idx=idx, source=line))
    return examples

class InputFeatures:
    def __init__(self, example_id, source_ids):
        self.example_id = example_id
        self.source_ids = source_ids

def tokenize(item):
    source, max_length, tokenizer = item
    source_tokens = [x for x in tokenizer.tokenize(source) if x != '\u0120']
    source_tokens = ["<s>", "<decoder-only>", "</s>"] + source_tokens[-(max_length-3):]
    source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
    padding_length = max_length - len(source_ids)
    source_ids += [tokenizer.pad_token_id] * padding_length
    return source_tokens, source_ids

def convert_examples_to_features(examples, tokenizer, args, pool=None, stage=None):
    features = []
    max_length = args.max_source_length + args.max_target_length if stage == "train" else args.max_source_length
    sources = [(x.source, max_length, tokenizer) for x in examples]
    if pool is not None:
        tokenize_tokens = pool.map(tokenize, tqdm(sources, total=len(sources)))
    else:
        tokenize_tokens = [tokenize(x) for x in sources]
    for example_index, (source_tokens, source_ids) in enumerate(tokenize_tokens):
        features.append(InputFeatures(example_index, source_ids))
    return features

def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

class Seq2Seq(torch.nn.Module):
    def __init__(self, encoder, decoder, config, beam_size, max_length, sos_id, eos_id):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.config = config
        self.beam_size = beam_size
        self.max_length = max_length
        self.sos_id = sos_id
        self.eos_id = eos_id

    def forward(self, source_ids, is_train):
        outputs = self.encoder(input_ids=source_ids)
        hidden_states = outputs[0]
        if is_train:
            pass
        else:
            pass
        return hidden_states

def generate_code(model, tokenizer, source_ids, device, max_length):
    model.eval()
    source_ids = source_ids.unsqueeze(0).to(device)  # Add batch dimension and move to device
    generated_ids = model.generate(source_ids, max_length=max_length, num_beams=5, early_stopping=True)
    generated_code = tokenizer.decode(generated_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
    return generated_code

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--train_dir", default=None, type=str, help="The file containing .txt files for training.")
    parser.add_argument("--dev_dir", default=None, type=str, help="The file containing .txt files for dev.")
    parser.add_argument("--test_dir", default=None, type=str, help="The file containing .txt files for test.")
    parser.add_argument("--test_output", default=None, type=str, help="The file containing the expected outputs for testing.")
    parser.add_argument("--max_source_length", default=64, type=int, help="The maximum total source sequence length after tokenization. Sequences longer will be truncated, sequences shorter will be padded.")
    parser.add_argument("--max_target_length", default=32, type=int, help="The maximum total target sequence length after tokenization. Sequences longer will be truncated, sequences shorter will be padded.")
    parser.add_argument("--do_train", action='store_true', help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true', help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action='store_true', help="Whether to run eval on the test set.")
    parser.add_argument("--train_batch_size", default=8, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=8, type=int, help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs", default=3, type=int, help="Total number of training epochs to perform.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--no_cuda", action='store_true', help="Avoid using CUDA when available")
    parser.add_argument('--seed', type=int, default=42, help="random seed for initialization")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device

    set_seed(args.seed)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    config = RobertaConfig(
        vocab_size=50265,
        max_position_embeddings=514,
        num_attention_heads=12,
        num_hidden_layers=12,
        type_vocab_size=1,
    )
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')  # Tokenizer can still be pre-trained
    encoder = RobertaModel(config)  # Initialize Roberta model from scratch
    
    model = Seq2Seq(encoder=encoder, decoder=encoder, config=config,
                    beam_size=5, max_length=args.max_target_length,
                    sos_id=tokenizer.cls_token_id, eos_id=tokenizer.sep_token_id)

    model.to(device)

    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    if args.do_train:
        train_examples = read_examples_from_txt(args.train_dir)
        train_features = convert_examples_to_features(train_examples, tokenizer, args, stage='train')
        all_source_ids = torch.tensor([f.source_ids for f in train_features], dtype=torch.long)
        train_data = TensorDataset(all_source_ids)
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size // args.gradient_accumulation_steps)

        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in ['bias', 'LayerNorm.weight'])],
             'weight_decay': 0.0},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in ['bias', 'LayerNorm.weight'])], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(train_dataloader) * args.num_train_epochs)

        model.train()
        for epoch in range(args.num_train_epochs):
            for batch in train_dataloader:
                batch = tuple(t.to(device) for t in batch)
                source_ids = batch[0]
                outputs = model(source_ids, True)
                loss = outputs[0]  # Assuming the loss is returned as the first element
                if args.n_gpu > 1:
                    loss = loss.mean()
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

    if args.do_eval:
        eval_examples = read_examples_from_txt(args.dev_dir)
        eval_features = convert_examples_to_features(eval_examples, tokenizer, args, stage='dev')
        all_source_ids = torch.tensor([f.source_ids for f in eval_features], dtype=torch.long)
        eval_data = TensorDataset(all_source_ids)
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

        model.eval()
        for batch in eval_dataloader:
            batch = tuple(t.to(device) for t in batch)
            source_ids = batch[0]
            with torch.no_grad():
                outputs = model(source_ids, True)
                loss = outputs[0]
                if args.n_gpu > 1:
                    loss = loss.mean()

    if args.do_test:
        test_examples = read_examples_from_txt(args.test_dir)
        test_features = convert_examples_to_features(test_examples, tokenizer, args, stage='test')
        all_source_ids = torch.tensor([f.source_ids for f in test_features], dtype=torch.long)
        test_data = TensorDataset(all_source_ids)
        test_sampler = SequentialSampler(test_data)
        test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=args.eval_batch_size)

        model.eval()
        predictions = []
        for batch in test_dataloader:
            batch = tuple(t.to(device) for t in batch)
            source_ids = batch[0]
            with torch.no_grad():
                generated_code = generate_code(model, tokenizer, source_ids, device, args.max_target_length)
                predictions.append(generated_code)

        # Read expected output for comparison
        expected_outputs = read_examples_from_txt(args.test_output)

        # Evaluate predictions
        for pred, expected in zip(predictions, expected_outputs):
            print(f"Generated: {pred}")
            print(f"Expected: {expected.source}")
            print()

if __name__ == "__main__":
    main()
