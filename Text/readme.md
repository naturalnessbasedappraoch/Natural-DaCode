The WikiMIA datasets serve as a benchmark, specifically in detecting pretraining data from extensive large language models. Access WikiMIA datasets directly on Hugging Face.

### Loading the Datasets:
```python
from datasets import load_dataset
LENGTH = 64
dataset = load_dataset("swj0419/WikiMIA", split=f"WikiMIA_length{LENGTH}")

