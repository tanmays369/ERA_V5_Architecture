# Dataset used for the 1,000-step experiment

Kaggle dataset:
https://www.kaggle.com/datasets/udisinghania98/team5-moe-pilot-v0

Tokenizer: BrahmicTokenizer-131K

Dataset manifest counts:
- Training: 200,530 documents; 31,664,159 content tokens
- Validation: 5,729 documents; 2,694,054 content tokens

Training source: train.jsonl
Validation source: validation.jsonl

The 1,000-step configuration makes the full corpus available for sampling.
At 512 scored tokens per update, the run processes 512,000 training tokens;
it does not cover the entire dataset.

Large dataset files and model checkpoints are stored outside this repository.
The run manifest.json records the actual input paths, hashes and configuration.
