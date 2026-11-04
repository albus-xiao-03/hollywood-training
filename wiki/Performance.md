# Performance

The SPEC asks for training to run "as fast as synchronization allows". Two things
decide how fast that is: the cost of the per-epoch hand-off, and PyTorch's CPU thread
pool.

## The per-epoch hand-off

After every epoch the training thread blocks until the window has drawn that epoch (see
[Architecture](Architecture)). The wait is one complete redraw of the network view,
the loss chart, the status panel and the header, plus a local-socket round trip:
typically **10 to 15 ms**. Everything else is free for the training thread:

* the window runs in its own process ([GUI Remote Frontend](GUI-Remote-Frontend)), so
  animations, input and usage sampling never interrupt training;
* `BatchProgress` is throttled to about 30 messages a second and is never waited for;
* no artificial delay is added (`delay <ms>` adds one on request).

How the redraw is kept cheap:

| Technique | Where | Effect |
|---|---|---|
| Static layer cached in a pixmap, rebuilt only when weights or layout change | [Network View](GUI-Network-View), [Loss Chart](GUI-Loss-Chart), [Header](GUI-Header) | an animation frame only blits it and draws the light overlay (~0.4 ms) |
| Connections drawn only with 1 px pens, thickness from parallel offset lines | [Network View](GUI-Network-View) | Qt's raster engine has a fast path for ≤ 1 px antialiased lines; wider pens use the general stroker. A rebuild went from ~81 ms to ~10 ms for 336 connections |
| Glowing digits from cached per-character sprites | [GUI Theme](GUI-Theme), [Status Panel](GUI-Status-Panel) | the ticking timer no longer strokes text paths every frame (2.6 → 0.7 ms) |
| One shared animation clock | [GUI Animation](GUI-Animation) | a single 60 fps timer for every panel |

## PyTorch's CPU threads

By default, PyTorch (GNU OpenMP on Linux) runs one intra-op worker per core, and the
workers spin-wait after each operation. With every core taken by that pool, **any**
other work on the machine stalls its barriers. The window drawing on the same
machine is such work, and small-batch CPU training becomes several times slower. These
numbers were measured on a 10-core Linux VM, for the spiral demo (milliseconds per
epoch):

| Setup | Epoch time |
|---|---|
| console, no window (default threads) | ~9 |
| window, default threads (10) | ~84 |
| window, `torch.set_num_threads(5)` at runtime (`torch_threads="auto"`) | ~40 |
| window, `OMP_NUM_THREADS=5` set before Python starts | ~21 (compute as fast as console, plus the hand-off) |

And with pure PyTorch, no GUI at all, and a 2 ms busy gap between batches:

| PyTorch threads (10 cores) | 2 | 4 | 6 | 8 | 10 |
|---|---|---|---|---|---|
| epoch time, ms | 3.2 | 3.2 | 3.9 | 5.3 | 41.7 |

So `run()` caps PyTorch's intra-op threads at **half the CPU cores** while a window is
open (`torch_threads="auto"`). It never raises your setting, and restores it when
`run()` returns. The device field of the status panel shows the count in effect, e.g.
`cpu (5 threads)`.

A runtime cap can't shrink a thread pool that already exists, so it recovers only part
of the speed. For the best result, size the pool before Python starts:

```bash
OMP_NUM_THREADS=4 python train.py          # Linux / macOS
set OMP_NUM_THREADS=4 && python train.py   # Windows (cmd)
```

Pass `torch_threads=None` to keep your own setting, or an `int` to choose the count.
GPU training is unaffected. These effects are largest on virtual machines and for
small batches. With large models, each operation runs long enough that the spinning
matters less.

## Display density

The cost of the hand-off grows with the number of drawn connections, about
`max_neurons²` per layer. The default `max_neurons=20` keeps any layer to at most
400 drawn connections. Snapshots hold only the drawn (sampled) connections; the weight
statistics always cover the full tensors.
