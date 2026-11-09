from pathlib import Path
import numpy as np
from datasets import load_dataset
from tokenizers import Tokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOKENIZER_PATH = PROJECT_ROOT / "configs" / "tinystories_tokenizer.json"
DATA_DIR = PROJECT_ROOT / "data"

TRAIN_STORIES = 300_000
VAL_STORIES = 5_000

tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))
EOS_TOKEN_ID = tokenizer.token_to_id("<|endoftext|>")

if EOS_TOKEN_ID is None:
    raise ValueError("EOS token was not found in the tokenizer.")

def process_split(split_name, number_of_stories, output_path):
    dataset = load_dataset("roneneldan/TinyStories", split=split_name, streaming=True)

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    total_tokens = 0
    processed_stories = 0
    token_buffer = []

    with open(output_path, "wb") as output_file:
        for example in dataset:
            if processed_stories >= number_of_stories:
                break

            story = example["text"]
            token_ids = tokenizer.encode(story).ids

            token_ids.append(EOS_TOKEN_ID)

            token_buffer.extend(token_ids)
            total_tokens += len(token_ids)
            processed_stories += 1

            if processed_stories % 1000 == 0:
                np.asarray(token_buffer, dtype=np.uint16).tofile(output_file)
                token_buffer.clear()

            if processed_stories % 10000 == 0:
                print(
                    f"Processed {processed_stories:,} stories | "
                    f"{total_tokens:,} tokens"
                )

        if token_buffer:
            np.asarray(token_buffer, dtype=np.uint16).tofile(output_file)

    saved_data = np.memmap(
        output_path,
        dtype=np.uint16,
        mode="r"
    )

    print(f"\nFinished {split_name} split.")
    print("Stories processed:", processed_stories)
    print("Total tokens:", total_tokens)
    print("Saved file:", output_path)
    print("Verified tokens in file:", len(saved_data))

    return total_tokens


def load_token_data(file_path):
    """Load a binary token file using memory mapping."""

    return np.memmap(file_path, dtype=np.uint16, mode="r")


def get_batch(data, batch_size, seq_length, rng):
    """
    Randomly sample sequences and create input-target batches.
    """

    start_positions = rng.integers(0, len(data) - seq_length, size=batch_size)

    x = np.stack([
        data[start:start + seq_length]
        for start in start_positions
    ])

    y = np.stack([
        data[start + 1:start + seq_length + 1]
        for start in start_positions
    ])

    x = x.astype(np.int32)
    y = y.astype(np.int32)

    return x, y

def main():
    train_tokens = process_split("train", TRAIN_STORIES, DATA_DIR / "train.bin")
    val_tokens = process_split("validation", VAL_STORIES, DATA_DIR / "val.bin")

    print("\n DATASET SUMMARY ")
    print("Training tokens:", train_tokens)
    print("Validation tokens:", val_tokens)

if __name__ == "__main__":
    main()