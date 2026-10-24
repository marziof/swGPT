"""
Dataloader: Turn saved binary token files into batches that the GPT can learn from
"""

import numpy as np
import torch


def get_batch(path, B, T, rnd_generator, device):
    """
    Select a random batch of data in given binary file
    Args:
    - path: path to binary file with train or validation data
    - B, T: number B of sequences in batch, length T of sequences
    - rnd_generator: used to select random starting indices of sequences
    """
    arr = np.memmap(path, dtype=np.uint16, mode="r") # array of all tokenized stories, converted from binary 
    N = len(arr) # total number of tokens
    assert N > T, "Insufficient number of tokens for specified sequence length T"
    indices = torch.randint(low=0, high=N-T, size=(B,), generator=rnd_generator)
    offsets = torch.arange(T)
    x_indices = indices[:, None] + offsets
    y_indices = x_indices + 1  # shifted by 1 as target is to predict next token
    x = arr[x_indices.numpy()]
    y = arr[y_indices.numpy()]
    x_batch = torch.from_numpy(x.astype(np.int64)).to(device)
    y_batch = torch.from_numpy(y.astype(np.int64)).to(device)

    return x_batch, y_batch


def get_val_batches(path, B, T, n_batches, device, seed=1234):
    """
    Fixed-seed helper for eval, builds whole list at once
    """
    g = torch.Generator().manual_seed(seed)
    return [get_batch(path, B, T, g, device) for _ in range(n_batches)]


def print_a():
    print("a")