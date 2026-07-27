import torch
import numpy as np
import itertools
import random

def get_dataset(p=97, operation="add", train_fraction=0.5, seed=0):

    PLUS_TOKEN = p
    EQUALS_TOKEN = p + 1

    inputs = []
    labels = []

    for a in range(p):
        for b in range(p):

            if operation == "add":
                c = (a + b) % p
            elif operation == "substract":
                c = (a - b) % p
            elif operation == "multiply":
                c = (a * b)
            else:
                raise ValueError(f"Unknown Operation: {operation}")
            
            inputs.append([a, PLUS_TOKEN, b, EQUALS_TOKEN])
            labels.append(c)

    inputs = np.array(inputs)
    labels = np.array(labels)

    rng = np.random.default_rng(seed)
    permutation = rng.permutation(len(inputs))
    inputs = inputs[permutation]
    labels = labels[permutation]

    split_index = int(len(inputs) * train_fraction)

    train_inputs = inputs[:split_index]
    train_labels = labels[:split_index]
    val_inputs = inputs[split_index:]
    val_labels = labels[split_index:]

    train_inputs = torch.tensor(train_inputs, dtype=torch.long)
    train_labels = torch.tensor(train_labels, dtype=torch.long)
    val_inputs = torch.tensor(val_inputs, dtype=torch.long)
    val_labels = torch.tensor(val_labels, dtype=torch.long)

    return train_inputs, train_labels, val_inputs, val_labels

if __name__ == "__main__":
    train_inputs, train_labels, val_inputs, val_labels = get_dataset(
        p=5, operation="add", train_fraction=0.5, seed=0
    )

    print("Train inputs shape:", train_inputs.shape)
    print("Train labels shape:", train_labels.shape)
    print("Val inputs shape:", val_inputs.shape)
    print("Val labels shape:", val_labels.shape)

    print("\nFirst 5 training models:")
    for i in range(5):
        print(train_inputs[i].tolist(), "-->", train_labels[i].item())