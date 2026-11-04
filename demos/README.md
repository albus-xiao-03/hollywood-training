# Demos

Two small, self-contained demos. The data is synthetic and generated on the fly, so there
is nothing to download. Install the package first (see the main [README](../README.md)),
then run the demos from the project folder.

| Demo | What it shows |
|---|---|
| [`spiral_classifier.py`](spiral_classifier.py) | `@ht.track` on a network class with neuron labels; step **decorators**; accuracy metrics; z-score data normalisation plus `BatchNorm1d`; a `StepLR` scheduler; a custom `predict` terminal command |
| [`sine_regression.py`](sine_regression.py) | tracking an `nn.Sequential` **instance**; a native step function passed to the **constructor**; the default validation step; SGD with momentum |
| [`demo_data.py`](demo_data.py) | dataset generators (spirals, noisy sine), z-score `Standardizer`, train/validation split, device selection |

```bash
python demos/spiral_classifier.py                   # opens the window
python demos/spiral_classifier.py --start-paused    # look first, then type: resume
python demos/spiral_classifier.py --epochs 600 --hidden 24 --lr 0.01
python demos/spiral_classifier.py --no-gui          # console output only
python demos/sine_regression.py
```

Common options: `--epochs`, `--lr`, `--device` (`auto`, `cpu`, `cuda`, `mps`, ...),
`--seed`, `--close-on-finish` and `--no-gui`. Run a demo with `--help` to see all of them.

## Things to try in the terminal

```text
help                 list every command
pause / resume       freeze and continue (the window keeps animating)
step 5               run five epochs, then pause again
lr 0.005             change the learning rate on the fly
delay 150            slow training down to watch the weights move (delay 0 = full speed)
weights fc2          statistics of one layer's weights
predict 0.4 -0.3     (spiral demo) ask the network which arm a point belongs to
epochs 400           extend a run, even after it has finished
save spiral.pt       write the model's state_dict
screenshot           save a PNG of the window
```
