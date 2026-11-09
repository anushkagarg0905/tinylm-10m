# Development Log — TinyLM-10M

## Project Information

Project: TinyLM-10M

Objective:
Build a decoder-only Transformer from scratch using JAX and log architectural choices.

---

## Session Log

| Date | Hours | Task | What I did | What happened |  
|------|-------|------|-------|----------|
| 10-11-2026 | | Repository setup | | |
| 11-11-2026 | 0.20 | Dataset Exploration | Ran length stats on 1000 stories from `roneneldan/TinyStories` | Min: 274 chars, Max: 4123 chars, Avg: 941.64 chars | 
| 11-11-2026 | 0.40 | Trained the BPE tokenizer | Trained a byte-level BPE tokenizer using the first 100,000 training stories. | Tokenizer training completed successfully with a vocabulary size of 4,096. EOS token ID was 0. Encountered a temporary network connection warning (peer closed connection), followed by an automatic retry. Training subsequently completed. | 
| 13-11-2026 | 0.10 | Tokenizer verification | Tested the trained BPE tokenizer |  Tokenizer loaded successfully. Vocabulary size was 4,096, EOS token ID was 0, and the sample story produced 166 tokens. The round-trip test returned True. | 
| 13-11-2026 | 0.30 | Verifying dataset encoding | Created and ran src/test_data.py to load the first 5 TinyStories training examples, tokenize each story using the trained BPE tokenizer, and append the EOS token. |  All 5 stories were processed successfully. EOS token ID: 0 was correctly appended to every story. The stories produced 937 tokens in total. Dataset loading and processing completed. | 
| 17-11-2026 | 1.00 | Implemented the TinyStories dataset tokenization and binary data preparation pipeline. | Created src/data.py and processed the TinyStories training dataset using the trained 4,096-vocabulary BPE tokenizer, appended EOS tokens, and saved token IDs in uint16 binary format. | Successfully processed 300,000 training stories, generating 69,380,9442 tokens. Processing progressed without errors in the displayed output. The Hugging Face authentication warning appeared.| 
| 17-11-2026 | - | Completed TinyStories dataset preparation. | Tokenized the first 300,000 training stories and 5,000 validation stories using the trained BPE tokenizer, appended EOS tokens, and saved the token IDs as uint16 binary files. | Successfully generated train.bin with 69,380,442 tokens and val.bin with 1,025,440 tokens. Verified that the saved files contain the expected number of tokens. The training dataset exceeds the required 65.5 million tokens | 
| 17-11-2026 | 0.20 | Implemented and tested the data batching pipeline. | Implemented memory-mapped loading of train.bin and a batching function to randomly sample 64 sequences of 256 tokens, creating input and target batches. | Successfully generated input and target batches with shapes (64, 256). The shifted-target test and same-seed reproducibility test both passed. All batching tests passed.| 