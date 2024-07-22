#!/bin/bash

# Prepare the dataset
python prepare_java_dataset.py

# Train the tokenizer
python train_java_tokenizer.py

# Train the model
python train_java_model.py --output_dir codeparrot_java_model --num_train_epochs 10 --per_device_train_batch_size 4 --save_steps 1000 --eval_steps 1000 --logging_dir logs
