# Getting Started

## Install

Hollywood Training needs Python 3.10+ and PyTorch 2.0+. Install the PyTorch build for
your machine first (CPU, CUDA or Apple MPS) from
[pytorch.org](https://pytorch.org/get-started/locally/). Then, from the project folder:

```bash
pip install .                 # the package
pip install ".[gpu]"          # optional: GPU utilisation on NVIDIA (nvidia-ml-py)
pip install -e ".[dev]"       # for development: editable + pytest, pytest-qt, ruff
```

The GUI uses `PySide6-Essentials` (Qt 6), which `pip` installs automatically. On a
minimal Linux system, Qt may need extra system libraries; see
[Troubleshooting](Troubleshooting).

## A first run

```bash
python demos/spiral_classifier.py
```

A window opens and trains a small network to separate three spiral arms. Everything you
type in the window goes to its terminal (there is no mouse pointer inside it): type
`help`. The [Demos](Demos) page lists more things to try.

## The three steps

### 1. Track the network

```python
@ht.track
class Net(nn.Module):
    ...
```

`@ht.track` returns your class unchanged. Options add labels to the drawing:

```python
@ht.track(name="Spiral classifier", input_labels=("x", "y"), output_labels=("A", "B", "C"))
class Net(nn.Module): ...

model = ht.track(nn.Sequential(nn.Linear(1, 8), nn.Tanh(), nn.Linear(8, 1)), name="tiny")
```

The architecture is read from `named_modules()` and refined by observing the first
forward pass, through PyTorch's forward hooks. Linear, convolution, transposed
convolution and embedding layers are drawn as connections. Activations, normalisation,
dropout, pooling and reshape layers appear as labels between them. See
[Tracking](Tracking).

### 2. Wrap your training process

Write one batch of training in plain PyTorch, then wrap it, either with decorators:

```python
process = ht.TrainingProcess(model, optimizer, train_loader,
                             loss_fn=loss_fn, val_data=val_loader, epochs=200,
                             scheduler=scheduler, normalization="z-score")

@process.train_step
def train_step(batch):
    inputs, targets = batch
    optimizer.zero_grad()
    logits = model(inputs)
    loss = loss_fn(logits, targets)
    loss.backward()
    optimizer.step()
    return {"loss": loss, "accuracy": (logits.argmax(1) == targets).float().mean()}

@process.validation_step
def validation_step(batch):
    inputs, targets = batch
    return loss_fn(model(inputs), targets)
```

or by passing the plain functions:

```python
process = ht.TrainingProcess(model, optimizer, train_loader, val_data=val_loader, epochs=200,
                             train_step=train_step, validation_step=validation_step)
```

Without custom steps, the default supervised steps are used. They need `loss_fn` and
batches of `(inputs, targets)`. Details: [Training Process](Training-Process),
[Training Steps](Training-Steps).

### 3. Run it

```python
if __name__ == "__main__":
    result = ht.run(process)
```

`run()` returns a [`TrainingResult`](Training-Results) after the window has closed. The
window stays open after training, so you can inspect the results, and
`epochs <more>` continues the run. Use `close_on_finish=True` to close the window
automatically, or `gui=False` for console output. All options are on the
[Runner](Runner) page.

The `if __name__ == "__main__":` guard is good practice, and required if your
`DataLoader` uses `num_workers > 0` on macOS or Windows.

## What happens during a run

* Your steps run on your thread. After every epoch the weights are snapshotted and sent
  to the window, and training continues only once the window has drawn them (see
  [Architecture](Architecture)).
* The window runs in its own process, so it animates smoothly and never slows training
  down. While it's open, PyTorch's CPU threads are capped at half the cores; see
  [Performance](Performance).
* Commands typed in the window control the run safely; see [Commands](Commands).
