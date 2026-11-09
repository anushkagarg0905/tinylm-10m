from tokenizers import Tokenizer
from datasets import load_dataset

tokenizer = Tokenizer.from_file("configs/tinystories_tokenizer.json")

print("Vocabulary size:", tokenizer.get_vocab_size())
print("EOS token ID:", tokenizer.token_to_id("<|endoftext|>"))

dataset = load_dataset("roneneldan/TinyStories",split="train",streaming=True)
story = next(iter(dataset))["text"]

encoded = tokenizer.encode(story)

print("\nOriginal story:")
print(story[:300])

print("\nFirst 30 token IDs:")
print(encoded.ids[:30])

print("\nTotal tokens:", len(encoded.ids))

decoded = tokenizer.decode(encoded.ids)
print("\nRound-trip test:", decoded == story)
