import json
import matplotlib.pyplot as plt

with open("results/history.json", "r") as f:
    history = json.load(f)

steps = history["step"]
train_acc = [a * 100 for a in history["train_acc"]]
val_acc = [a * 100 for a in history["val_acc"]]

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(steps, train_acc, label="Train accuracy", color="#2E86AB", linewidth=2)
ax.plot(steps, val_acc, label="Validation accuracy", color="#E63946", linewidth=2)
ax.set_xlabel("Optimization steps")
ax.set_ylabel("Accuracy (%)")
ax.set_title("Grokking: Train vs Validation Accuracy")
ax.legend(loc="center right")
ax.grid(alpha=0.3)
ax.set_ylim(-5, 105)

plt.tight_layout()
plt.savefig("results/accuracy_curve.png", dpi=150)
print("Saved plot to results/accuracy_curve.png")