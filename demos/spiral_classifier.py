"""Demo: a small MLP learns to tell the three arms of a spiral apart.

Everything below is ordinary PyTorch. Hollywood Training adds three things:

1. ``@ht.track`` on the network class (plus labels for the drawing),
2. a ``TrainingProcess`` that wraps the native train / validation steps,
3. ``ht.run(process)``, which opens the window and manages the run.

Run it from the project folder::

    python demos/spiral_classifier.py                 # GUI
    python demos/spiral_classifier.py --start-paused  # inspect first, then type 'resume'
    python demos/spiral_classifier.py --no-gui        # console only

Try these in the window's terminal: ``help``, ``pause``, ``step 5``, ``lr 0.005``,
``delay 100``, ``predict 0.5 -0.2``, ``weights fc2``, ``save spiral.pt``.
"""

from __future__ import annotations

import argparse

import torch
from demo_data import Standardizer, make_spirals, pick_device, split
from torch import nn
from torch.utils.data import DataLoader

import hollywood_training as ht

ARMS = ("arm A", "arm B", "arm C")


@ht.track(name="Spiral Classifier", input_labels=("x", "y"), output_labels=ARMS)
class SpiralNet(nn.Module):
    def __init__(self, hidden: int = 16) -> None:
        super().__init__()
        self.fc1 = nn.Linear(2, hidden)
        self.norm1 = nn.BatchNorm1d(hidden)
        self.act1 = nn.Tanh()
        self.fc2 = nn.Linear(hidden, hidden)
        self.act2 = nn.ReLU()
        self.dropout = nn.Dropout(0.05)
        self.fc3 = nn.Linear(hidden, len(ARMS))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act1(self.norm1(self.fc1(x)))
        x = self.dropout(self.act2(self.fc2(x)))
        return self.fc3(x)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--epochs", type=int, default=240)
    parser.add_argument("--hidden", type=int, default=16, help="neurons per hidden layer")
    parser.add_argument("--lr", type=float, default=0.02)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:1, mps, ...")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--start-paused", action="store_true", help="open paused")
    parser.add_argument("--close-on-finish", action="store_true", help="close when done")
    parser.add_argument("--no-gui", action="store_true", help="print to the console instead")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = pick_device(args.device)

    # data: three interleaved spiral arms, z-score normalised with training statistics
    points, labels = make_spirals(seed=args.seed)
    train_set, val_set = split(points, labels, validation=0.2, seed=args.seed)
    normalize = Standardizer(train_set.tensors[0])
    train_set.tensors = (normalize(train_set.tensors[0]), train_set.tensors[1])
    val_set.tensors = (normalize(val_set.tensors[0]), val_set.tensors[1])
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=256)

    # the usual PyTorch objects
    model = SpiralNet(args.hidden).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=max(1, args.epochs // 3), gamma=0.5
    )

    process = ht.TrainingProcess(
        model,
        optimizer,
        train_loader,
        val_data=val_loader,
        loss_fn=loss_fn,
        epochs=args.epochs,
        scheduler=scheduler,
        normalization="z-score (inputs)",
    )

    @process.train_step
    def train_step(batch):
        inputs, targets = (tensor.to(device) for tensor in batch)
        optimizer.zero_grad()
        logits = model(inputs)
        loss = loss_fn(logits, targets)
        loss.backward()
        optimizer.step()
        accuracy = (logits.argmax(dim=1) == targets).float().mean()
        return {"loss": loss, "accuracy": accuracy}

    @process.validation_step
    def validation_step(batch):
        inputs, targets = (tensor.to(device) for tensor in batch)
        logits = model(inputs)
        accuracy = (logits.argmax(dim=1) == targets).float().mean()
        return {"loss": loss_fn(logits, targets), "accuracy": accuracy}

    @process.command(
        "predict",
        summary="classify a point of the spiral plane",
        arguments=[
            ht.Argument("x", float, "x coordinate"),
            ht.Argument("y", float, "y coordinate"),
        ],
    )
    def predict(context, x, y):
        point = normalize(torch.tensor([[x, y]])).to(device)

        def classify(target):  # runs on the training thread, between batches
            was_training = target.model.training
            target.model.eval()
            with torch.no_grad():
                probabilities = target.model(point).softmax(dim=1)[0]
            target.model.train(was_training)
            arm = int(probabilities.argmax())
            return f"({x:g}, {y:g}) -> {ARMS[arm]} ({probabilities[arm]:.0%})"

        return context.submit(classify, f"predict ({x:g}, {y:g})")

    result = ht.run(
        process,
        gui=not args.no_gui,
        start_paused=args.start_paused,
        close_on_finish=args.close_on_finish,
    )
    best = result.best_epoch()
    print(f"{result.state.value}: {result.epochs_completed} epochs in {result.elapsed:.1f} s")
    if best is not None:
        accuracy = best.val_metrics.get("accuracy")
        suffix = f", validation accuracy {accuracy:.1%}" if accuracy is not None else ""
        print(f"best epoch {best.epoch}: validation loss {best.val_loss:.4f}{suffix}")


if __name__ == "__main__":
    main()
