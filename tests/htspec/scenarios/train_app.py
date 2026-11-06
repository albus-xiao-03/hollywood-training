"""Scenario: a real user script - native PyTorch training handed to ``ht.run``.

Started by the harness as ``python train_app.py '<json config>'``. It reports on stdout,
one JSON object per line:

* ``{"event": "started", "pid": ..., "t": ...}``
* ``{"event": "batch1", "epoch": n, "t": ...}`` when epoch n's first training batch starts
* ``{"event": "epoch", "epoch": n, "t": ..., "lr": ..., "train_loss": ..., "weights": ...}``
  from an ``on_epoch_end`` hook (``weights`` only with ``"report_weights": true``)
* ``{"event": "pong", "count": n}`` whenever the custom ``ping`` command runs
* ``{"event": "result", "state": ..., "epochs": ..., "error": ...}`` when ``ht.run`` returns

Times are ``time.monotonic()``. Config keys (all optional): ``epochs``, ``batches``,
``batch_size``, ``batch_sleep`` (seconds per training batch), ``hidden``, ``validation``,
``lr``, ``seed``, ``report_weights``, ``fail_at_epoch`` (raise in the step), ``run`` (keyword
arguments for ``ht.run``).
"""

from __future__ import annotations

import json
import os
import sys
import time

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import hollywood_training as ht


def say(**fields) -> None:
    print(json.dumps(fields), flush=True)


@ht.track(name="Scenario Net", input_labels=("x0", "x1"), output_labels=("A", "B", "C"))
class ScenarioNet(nn.Module):
    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(2, hidden)
        self.act = nn.Tanh()
        self.fc2 = nn.Linear(hidden, 3)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


def main() -> int:
    config = json.loads(sys.argv[1]) if len(sys.argv) > 1 else {}
    torch.manual_seed(int(config.get("seed", 0)))
    batches = int(config.get("batches", 6))
    batch_size = int(config.get("batch_size", 8))
    batch_sleep = float(config.get("batch_sleep", 0.0))
    fail_at = config.get("fail_at_epoch")
    inputs = torch.randn(batches * batch_size, 2)
    targets = (inputs[:, 0] > inputs[:, 1]).long() + (inputs[:, 0] > 0).long()
    train_loader = DataLoader(TensorDataset(inputs, targets), batch_size=batch_size)
    val_loader = None
    if config.get("validation", True):
        val_inputs = torch.randn(2 * batch_size, 2)
        val_targets = (val_inputs[:, 0] > val_inputs[:, 1]).long()
        val_loader = DataLoader(TensorDataset(val_inputs, val_targets), batch_size=batch_size)

    model = ScenarioNet(int(config.get("hidden", 6)))
    optimizer = torch.optim.SGD(model.parameters(), lr=float(config.get("lr", 0.05)))
    loss_fn = nn.CrossEntropyLoss()
    process = ht.TrainingProcess(
        model,
        optimizer,
        train_loader,
        val_data=val_loader,
        loss_fn=loss_fn,
        epochs=int(config.get("epochs", 20)),
        normalization="none (scenario)",
    )
    calls = {"n": 0, "pings": 0}

    @process.train_step
    def train_step(batch):
        epoch = calls["n"] // batches + 1
        if calls["n"] % batches == 0:
            say(event="batch1", epoch=epoch, t=time.monotonic())
        calls["n"] += 1
        if fail_at is not None and epoch == int(fail_at):
            raise RuntimeError(f"scenario failure in epoch {epoch}")
        if batch_sleep:
            time.sleep(batch_sleep)
        x, y = batch
        optimizer.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        optimizer.step()
        return loss

    @process.on_epoch_end
    def report(record):
        fields = {
            "event": "epoch",
            "epoch": record.epoch,
            "t": time.monotonic(),
            "lr": optimizer.param_groups[0]["lr"],
            "train_loss": record.train_loss,
            "val_loss": record.val_loss,
        }
        if config.get("report_weights"):
            fields["weights"] = {
                name: tensor.detach().tolist() for name, tensor in model.state_dict().items()
            }
        say(**fields)

    @process.command("ping", summary="answer pong (scenario)")
    def ping(context):
        calls["pings"] += 1
        say(event="pong", count=calls["pings"])
        return f"pong {calls['pings']}"

    say(event="started", pid=os.getpid(), t=time.monotonic())
    try:
        result = ht.run(process, **config.get("run", {}))
    except ht.TrainingFailedError as exc:
        say(event="result", state="failed", epochs=exc.result.epochs_completed, error=str(exc))
        return 3
    say(
        event="result",
        state=result.state.value,
        epochs=result.epochs_completed,
        error=None if result.error is None else str(result.error),
        t=time.monotonic(),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
