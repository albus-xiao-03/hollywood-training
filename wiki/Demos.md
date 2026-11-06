# Demos

The `demos/` folder sits next to the package. It contains two small demos with
synthetic data, generated on the fly so there is nothing to download. Run them from the
project folder after installing the package.

| Demo | Network and data | Shows |
|---|---|---|
| `demos/spiral_classifier.py` | `SpiralNet`: `Linear 2→16 · BatchNorm1d · Tanh · Linear 16→16 · ReLU · Dropout · Linear 16→3`, three interleaved spiral arms | `@ht.track` on a class with input and output labels; step **decorators**; train and validation accuracy as metrics; z-score input normalisation plus `BatchNorm1d` (the "normalisation strategy" panel); a `StepLR` scheduler; a custom `predict x y` command |
| `demos/sine_regression.py` | `nn.Sequential`: `Linear 1→24 · SiLU · Linear 24→24 · Tanh · Linear 24→1`, noisy `sin(x)` | `ht.track(instance, ...)`; a native step function passed to the **constructor**; the default validation step; `SGD` with momentum; `MSELoss` |
| `demos/demo_data.py` | — | `make_spirals`, `make_sine`, the `Standardizer` (z-score), `split` into train / validation, `pick_device` (`auto` → CUDA, MPS or CPU) |

```bash
python demos/spiral_classifier.py
python demos/spiral_classifier.py --start-paused      # inspect the initial weights, then `resume`
python demos/spiral_classifier.py --epochs 600 --hidden 24 --lr 0.01
python demos/spiral_classifier.py --no-gui            # console output
python demos/sine_regression.py --device cpu
```

| Option | Demos | Meaning |
|---|---|---|
| `--epochs N` | both | epochs to train (spiral 240, sine 150) |
| `--lr X` | both | learning rate |
| `--device D` | both | `auto`, `cpu`, `cuda`, `cuda:1`, `mps`, ... |
| `--seed S` | both | data and initialisation seed |
| `--close-on-finish` | both | close the window when training ends |
| `--no-gui` | both | print to the console instead |
| `--hidden H`, `--batch-size B`, `--start-paused` | spiral | hidden width, batch size, open paused |

## Things to try

```text
help                 list every command
pause / resume       freeze and continue: the window keeps animating
step 5               run five epochs, then pause again
lr 0.005             change the learning rate on the fly (the scheduler keeps stepping)
delay 150            slow down to watch the weights move; delay 0 = full speed
weights fc2          statistics of one layer
predict 0.4 -0.3     spiral demo: which arm does the point belong to?
epochs 400           extend the run, even after it finished
probe 50             epoch 50's values, marked on the loss chart (pause first to keep the line)
scale log            logarithmic loss axis; scale lin goes back
neuron fc3 2         spiral demo: an output neuron with its arm label and bias
neuron               the drawn columns of the network view
layout chart 60      give the loss chart more room; layout reset goes back
window full          full screen; window normal goes back
copy 20              copy the last 20 lines of the terminal
save spiral.pt       write the model's state_dict
screenshot           save a PNG of the window
```

The window is keyboard-only: everything typed goes to the terminal, PgUp / PgDn scroll it,
and Ctrl+Home / Ctrl+End jump to its top / bottom ([GUI](GUI)).

## How `predict` is built

The spiral demo registers a [custom command](Commands). It runs the model on the
training thread through `context.submit`, so it never overlaps an optimisation step,
and it restores the model's train/eval mode afterwards:

```python
@process.command("predict", summary="classify a point of the spiral plane",
                 arguments=[ht.Argument("x", float, "x coordinate"),
                            ht.Argument("y", float, "y coordinate")])
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
```
