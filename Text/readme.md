# Contamination Detection of Textual Data with Natural-DaCoDe

This repository contains instructions on how to classify contaminated data with **Natural-DaCoDe**, including how to access the dataset.
 

### Loading the Datasets:
```python
from datasets import load_dataset
LENGTH = 64
dataset = load_dataset("swj0419/WikiMIA", split=f"WikiMIA_length{LENGTH}")

