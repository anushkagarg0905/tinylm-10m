# Experimental Protocol

## Project
TinyLM-10M

## Objective
Investigate how Transformers design choices affect validation loss in a  small language model

## Framework
JAX

## Dataset
TinyStories

## Tokenizer
Byte-level BPE
Vocabulary size: 4096

## Baseline Architecture
- 12 Transformer layers
- Model dimension: 256
- Attention heads: 8
- Head dimensions: 32
- MLP hidden dimensions: 1024
- Pre-LayerNorm
- Learned positional embeddings
- Weight tying
- Dropout: 0

## Data
- Training stories: 300,000
- Validation stories: 5,000
- Tokenizer training stories: 100,000
- Context length: 256

## Reproducibility
- Random seeds: 0, 1, 2
- Precision: float32
- Record of experiment configurations.
- Record token counts.

