import numpy as np
from src.data import load_token_data, get_batch

data = load_token_data("data/train.bin")

BATCH_SIZE = 64
SEQ_LENGTH = 256
SEED = 42

rng = np.random.default_rng(SEED)

x, y = get_batch(data, BATCH_SIZE, SEQ_LENGTH, rng)

# Test 1
assert x.shape == (64, 256)
assert y.shape == (64, 256)

# Test 2
assert np.array_equal(x[:, 1:], y[:, :-1])

# Test 3
rng_again = np.random.default_rng(SEED)

x_again, y_again = get_batch(data, BATCH_SIZE, SEQ_LENGTH, rng_again)

assert np.array_equal(x, x_again)
assert np.array_equal(y, y_again)

print("Input shape:", x.shape)
print("Target shape:", y.shape)
print("Shifted target test: Passed")
print("Same seed reproducibility test: Passed")
print("All batching tests passed!")
