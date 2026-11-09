from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
from datasets import load_dataset

train_data = load_dataset("roneneldan/TinyStories", split="train", streaming=True)

tokenizer = Tokenizer(models.BPE(unk_token=None))

tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False) # forces byte-level processing, so unusual spaces and puntuation dont throw errors
tokenizer.decoder = decoders.ByteLevel()

trainer = trainers.BpeTrainer(
    vocab_size=4096,
    special_tokens=["<|endoftext|>"], # acts as a boundry wall, so the stories don't mix into each other
    initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    show_progress=True
)

def training_iterator():
    iterator = iter(train_data)
    for _ in range(100_000):
        story = next(iterator)["text"]
        yield story

tokenizer.train_from_iterator(
    training_iterator(),
    trainer=trainer,
    length=100_000
)

tokenizer.save("tinystories_tokenizer.json")

print("training completed.")
print("Vocabulary size:", tokenizer.get_vocab_size())
print(
    "EOS token ID:",
    tokenizer.token_to_id("<|endoftext|>")
)
