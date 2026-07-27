# grokking-repro

A from-scratch reproduction of **grokking** — the phenomenon where a small transformer trained on an algorithmic task memorizes the training data almost immediately, then appears stuck at random validation accuracy for a long time, and then suddenly generalizes to near-perfect validation accuracy long after training accuracy has saturated.

Originally reported in [Power et al., 2021 — "Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets"](https://arxiv.org/abs/2201.02177).

## What's happening here

A tiny transformer is trained on a modular arithmetic task, e.g.:

```
(a + b) mod 97 = c
```

Given `a` and `b` as input tokens, the model predicts `c`. The dataset is small and fully enumerable (all pairs `a, b` in `[0, 96]`), and only a fraction of it is used for training — the rest is held out for validation.

With standard training, the model quickly memorizes the training split (train accuracy → ~100%) while validation accuracy stays near chance. If you keep training well past that point — with weight decay turned up — validation accuracy eventually jumps to ~100% too. That delayed jump is grokking.

## Why weight decay matters

Weight decay keeps exerting pressure on the model even after training loss has bottomed out. Over many more steps, that pressure nudges the memorizing solution toward a simpler, generalizing one — which is the mechanism generally credited for causing grokking. The ablation configs in this repo let you see this directly: turn weight decay off, and grokking mostly disappears.

## Project structure

```
grokking-repro/
├── data/
│   └── modular_arithmetic.py     # generates (a op b) mod p pairs, train/val split
├── model/
│   └── transformer.py            # small 1-2 layer transformer, built from scratch
├── train.py                      # training loop; logs train/val acc every step
├── configs/
│   └── addition_mod97.yaml       # p, weight decay, lr, train fraction, steps
├── results/
│   └── accuracy_curve.png        # train vs val accuracy over training steps
├── notebooks/
│   └── reproduce_grokking.ipynb  # run experiment, plot curve, writeup
└── README.md
```

## Setup

```bash
git clone <repo-url>
cd grokking-repro
pip install -r requirements.txt
```

## Running an experiment

```bash
python train.py --config configs/addition_mod97.yaml
```

This will:
1. Generate the modular arithmetic dataset and split it into train/val.
2. Train the transformer, logging train and validation accuracy every step.
3. Save the accuracy curve to `results/accuracy_curve.png`.

## Config

Key parameters in `configs/addition_mod97.yaml`:

| Parameter | Description |
|---|---|
| `p` | Modulus for the arithmetic task (e.g. 97) |
| `operation` | `add`, `subtract`, `multiply`, etc. |
| `train_fraction` | Fraction of all `(a, b)` pairs used for training |
| `weight_decay` | Regularization strength — the key grokking lever |
| `lr` | Learning rate |
| `steps` | Total training steps (grokking can take 10k–100k+) |

## Results

The main output is the accuracy curve: training accuracy saturates early, validation accuracy lags and then jumps. See `notebooks/reproduce_grokking.ipynb` for the plotted results and a short writeup of what was observed, including any ablations run (varying weight decay or train fraction) and how they affected the grokking point.

## References

- Power, A. et al. (2021). [*Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets*](https://arxiv.org/abs/2201.02177)
- Nanda, N. et al. (2023). [*Progress Measures for Grokking via Mechanistic Interpretability*](https://arxiv.org/abs/2301.05217)

---

made by -- pranit bharat more 🐐
