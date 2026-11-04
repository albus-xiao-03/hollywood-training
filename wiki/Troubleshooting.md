# Troubleshooting

## "no display found"

`run()` checks for a display before opening a window on Linux and BSD: `DISPLAY` (X11)
or `WAYLAND_DISPLAY` must be set. On a server, over SSH without X forwarding, or in a
container:

* use `ht.run(process, gui=False)` for console output, or
* set `QT_QPA_PLATFORM=offscreen` to render the window without a screen (useful for
  tests and screenshots), or
* run a virtual display: `xvfb-run -a python train.py`.

## Qt cannot load the platform plugin "xcb"

Qt 6's X11 plugin needs some system libraries that minimal Linux installations lack,
most often `libxcb-cursor0`:

```bash
sudo apt-get install libxcb-cursor0          # Debian / Ubuntu
sudo dnf install xcb-util-cursor             # Fedora
```

Run with `QT_DEBUG_PLUGINS=1` to see exactly which library is missing.

## "Failed to create wl_display" warnings

`WAYLAND_DISPLAY` is set, but no Wayland compositor is reachable, so Qt falls back to
another platform. This is harmless. Choose one explicitly to silence it:
`QT_QPA_PLATFORM=xcb` (X11) or `QT_QPA_PLATFORM=wayland`.

## "the GUI process exited ... before connecting"

The window runs in a child process (`python -m hollywood_training.gui.window_process`)
started with the same interpreter. Its error message appears just above this one in
your terminal. Common causes are a missing Qt system library (above), or an environment
where starting processes is restricted. Run the window in-process instead:

```python
ht.run(process, isolate_gui=False)
```

The in-process window is also the choice for embedding in an existing Qt application,
and for frozen executables (PyInstaller and similar), where `sys.executable -m ...`
cannot start the child.

## Training is slower with the window open

See [Performance](Performance). In short: start Python with `OMP_NUM_THREADS` set to
about half your cores, for example `OMP_NUM_THREADS=4 python train.py`. By default,
`run()` caps PyTorch's threads at half the cores while the window is open.

## The window freezes during long batches (in-process mode only)

With `isolate_gui=False`, the window shares the training thread and refreshes between
batches. A single very long batch, taking seconds, delays its updates. The default
isolated window does not have this limitation.

## Wide layers are not fully drawn

By design. At most `max_neurons` (default 20) neurons per column are drawn, evenly
sampled, and a caption such as `20 of 128 neurons` says so. Statistics (`weights`
command) always cover the full tensors. Raise the limit with
`ht.run(process, max_neurons=32)`.

## A layer is missing from the drawing

Only connection layers (`Linear`, `Conv*`, `ConvTranspose*`, `Embedding`, and modules
with a 2-D `weight`) are drawn as connections. Other leaf modules, such as RNNs, are
listed under "not drawn" in the terminal at start-up, and by the `arch` command.
Activations written as functions (`F.relu(x)`) instead of modules (`nn.ReLU()`) cannot be
seen through module hooks, so they don't appear as badges. See
[Tracking Layer Classification](Tracking-Layer-Classification).

## `run()` raised `TrainingFailedError`

Your step function raised an exception. The traceback is shown in the window's
terminal, and the original exception is attached as `__cause__`.
`error.result.history` holds the epochs completed before the failure.
