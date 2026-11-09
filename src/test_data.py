from pathlib import Path
from datasets import load_dataset
from tokenizers import Tokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOKENIZER_PATH = PROJECT_ROOT / "configs" / "tinystories_tokenizer.json"

tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))
EOS_TOKEN_ID = tokenizer.token_to_id("<|endoftext|>")

def encode_story(story: str) -> list[int]:
  encoded = tokenizer.encode(story)  # breaking words into subwords and mapping them to numbers
  token_ids = encoded.ids + [EOS_TOKEN_ID] # Extracts list of numbers and tack EOS token ID to the end so the model knows the story is over. 
  return token_ids

def main():
  dataset = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
  total_tokens = 0

  for index, example in enumerate(dataset):
    if index == 5: # verify implementation works before processing all the stories.
      break

    story = example["text"]
    token_ids = encode_story(story)
    total_tokens += len(token_ids)

    print(f"Story {index + 1}")
    print("Original characters:", len(story))
    print("Tokens including EOS:", len(token_ids))
    print("Last token:", token_ids[-1])
    print("EOS correctly appended:", token_ids[-1] == EOS_TOKEN_ID)
    print("-" * 40) # separate each story so the output is easy to read

  print("\nTotal tokens across 5 stories:", total_tokens)

if __name__ == "__main__":
  main()