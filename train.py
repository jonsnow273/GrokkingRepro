import torch
import torch.nn as nn

from data.modular_arithmetic import get_dataset
from model.transformer import GrokkingTransformer

def get_accuracy(logits, labels):
    predictions = logits.argmax(dim=-1)
    correct = (predictions == labels).float()
    return correct.mean().item()

def train(
    p=97,
    operation="add",
    train_fraction=0.5,
    embed_dim=128,
    n_heads=4,
    n_layers=2,
    lr=1e-3,
    weight_decay=1.0,
    total_steps=10000,
    eval_every=100,
    seed=0,
):
    device = torch.device("cpu")

    train_inputs, train_labels, val_inputs, val_labels = get_dataset(
        p=p, operation=operation, train_fraction=train_fraction, seed=seed
    )

    train_inputs, train_labels = train_inputs.to(device), train_labels.to(device)
    val_inputs, val_labels = val_inputs.to(device), val_labels.to(device)

    vocab_size = p + 2
    seq_len = 4

    model = GrokkingTransformer(
        vocab_size=vocab_size, seq_len=seq_len, 
        embed_dim=embed_dim, n_heads=n_heads, n_layers=n_layers, p=p
    ).to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    history = {
        "step" : [],
        "train_acc" : [],
        "val_acc" : [],
    }

    for step in range(total_steps):
        model.train()
        logits = model(train_inputs)
        loss = loss_fn(logits, train_labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % eval_every == 0 or step == total_steps - 1:
            model.eval()
            with torch.no_grad():
                train_logits = model(train_inputs)
                val_logits = model(val_inputs)
 
                train_acc = get_accuracy(train_logits, train_labels)
                val_acc = get_accuracy(val_logits, val_labels)
 
            history["step"].append(step)
            history["train_acc"].append(train_acc)
            history["val_acc"].append(val_acc)
 
            print(f"Step {step:6d} | train acc: {train_acc:.3f} | val acc: {val_acc:.3f}")
 
    return model, history

import yaml

def load_config(path):
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    return config

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    model, history = train(**config)

import json

with open("results/history.json", "w") as f:
    json.dump(history, f)