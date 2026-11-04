"""Demo: an ``nn.Sequential`` network learns ``sin(x)`` from noisy samples.

Shows the other way to wrap a training process: write the native PyTorch step as a plain
function, then hand it to ``TrainingProcess`` - and track a model you did not define as
a class (``ht.track(instance, ...)``). Validation uses the default supervised step.

    python demos/sine_regression.py
    python demos/sine_regression.py --no-gui
"""

from __future__ import annotations

import argparse

import torch
from demo_data import make_sine, pick_device, split
from torch import nn
from torch.utils.data import DataLoader

import hollywood_training as ht


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--close-on-finish", action="store_true")
    parser.add_argument("--no-gui", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = pick_device(args.device)

    x, y = make_sine(seed=args.seed)
    train_set, val_set = split(x, y, validation=0.25, seed=args.seed)
    train_loader = DataLoader(train_set, batch_size=50, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=150)

    model = ht.track(
        nn.Sequential(
            nn.Linear(1, 24),
            nn.SiLU(),
            nn.Linear(24, 24),
            nn.Tanh(),
            nn.Linear(24, 1),
        ),
        name="Sine Approximator",
        input_labels=("x",),
        output_labels=("sin x",),
    ).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    loss_fn = nn.MSELoss()

    # 1. the training of one batch, in plain PyTorch
    def train_step(batch):
        inputs, targets = (tensor.to(device) for tensor in batch)
        optimizer.zero_grad()
        loss = loss_fn(model(inputs), targets)
        loss.backward()
        optimizer.step()
        return loss

    # 2. wrap it  3. hand it to the manager
    process = ht.TrainingProcess(
        model,
        optimizer,
        train_loader,
        val_data=val_loader,
        loss_fn=loss_fn,
        epochs=args.epochs,
        train_step=train_step,
        normalization="none (x already in [-π, π])",
    )
    result = ht.run(process, gui=not args.no_gui, close_on_finish=args.close_on_finish)
    print(f"{result.state.value}: {result.epochs_completed} epochs in {result.elapsed:.1f} s")
    if result.history:
        print(f"final validation MSE {result.val_losses[-1]:.4f}")


if __name__ == "__main__":
    main()
