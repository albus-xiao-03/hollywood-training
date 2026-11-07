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

## There is no mouse pointer in the window

This is by design: the window is driven from the keyboard
([Pointer Guard](GUI-Pointer-Guard)). The pointer disappears while it is over the window
and works as usual everywhere else, including the window's title bar. What the mouse did
before is a command now (`scale`, `probe`, `neuron`, `layout`, `copy`, `window`); see the
table on the [GUI](GUI) page.

## Typing does not reach the terminal

Keys go to the terminal while the Hollywood Training window is the **active** window.
Switch to it with the system's window switcher (for example Alt+Tab), or click it once:
the window manager activates it, and then every key goes to the command line
([Keyboard Focus](GUI-Keyboard-Focus)). Some desktops keep a newly opened window in the
background while you type elsewhere.

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

Every module that holds a weight tensor (a parameter with two or more dimensions) is
drawn as connections: `Linear`, `Conv*`, `ConvTranspose*`, `Embedding`, `Bilinear`,
recurrent and attention layers, and your own modules with such parameters. Modules
without weights that are not activations, normalisation, dropout, pooling or reshape
layers (padding, for example) are listed under "not drawn" in the terminal at start-up,
and by the `arch` command. Activations written as functions (`F.relu(x)`) instead of
modules (`nn.ReLU()`) cannot be seen through module hooks, so they don't appear as
badges. Tensors that are not parameters of a module are not tracked. See
[Tracking Layer Classification](Tracking-Layer-Classification).

## `run()` raised `TrainingFailedError`

Your step function raised an exception. The traceback is shown in the window's
terminal, and the original exception is attached as `__cause__`.
`error.result.history` holds the epochs completed before the failure.
