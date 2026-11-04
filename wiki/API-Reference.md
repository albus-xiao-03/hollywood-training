# API Reference

Everything `import hollywood_training as ht` exposes. Names that need torch load on
first access (PEP 562), so `import hollywood_training` itself is cheap.

| Name | Kind | Page | Purpose |
|---|---|---|---|
| `ht.track` | function / decorator | [Tracking Decorator](Tracking-Decorator) | declare a network class or instance for tracking, with optional labels |
| `ht.is_tracked` | function | [Tracking Decorator](Tracking-Decorator) | whether a model (or its class) was declared |
| `ht.inspect_network` | function | [Tracking Network Tracker](Tracking-Network-Tracker) | the static architecture of a model |
| `ht.NetworkTracker` | class | [Tracking Network Tracker](Tracking-Network-Tracker) | architecture + per-epoch weight snapshots of one model |
| `ht.TrainingProcess` | class | [Training Process](Training-Process) | your model, optimizer, data and step functions, wrapped |
| `ht.run` | function | [Runner](Runner) | the management method: run a process with the window, or the console |
| `ht.TrainingResult` | class | [Training Results](Training-Results) | what `run()` returns: state, history, elapsed time, final weights |
| `ht.EpochRecord` | class | [Sync Events](Sync-Events) | one epoch's losses, metrics, learning rate and duration |
| `ht.TrainingState` | enum | [Sync Events](Sync-Events) | `PREPARING`, `RUNNING`, `PAUSED`, `FINISHED`, `STOPPED`, `FAILED` |
| `ht.TrainingFailedError` | exception | [Runner](Runner) | raised by `run()` when your step code raised |
| `ht.ProcessError` | exception | [Training Process](Training-Process) | an incomplete process (e.g. no step and no `loss_fn`) |
| `ht.Argument` | class | [Commands Registry](Commands-Registry) | a typed argument of a custom command |
| `ht.Command` | class | [Commands Registry](Commands-Registry) | a command declaration |
| `ht.CommandContext` | class | [Commands Context](Commands-Context) | what a command handler receives |
| `ht.CommandError` | exception | [Commands Registry](Commands-Registry) | raise it in a handler to report a refused request |
| `ht.__version__` | str | — | package version (`hollywood_training/_version.py`) |

The package also ships `py.typed`, so type checkers use its annotations.

## Signatures at a glance

```python
ht.track(target=None, /, *, name=None, input_labels=None, output_labels=None)

ht.TrainingProcess(model, optimizer, train_data, *, epochs, loss_fn=None, val_data=None,
                   scheduler=None, train_step=None, validation_step=None,
                   on_epoch_end=None, batch_size=None, normalization=None, name=None)
process.train_step(function)        # decorator
process.validation_step(function)   # decorator
process.on_epoch_end(function)      # decorator, any number of hooks
process.command(name, *, summary, arguments=(), aliases=(), details="")   # decorator

ht.run(process, *, gui=True, isolate_gui=True, start_paused=False, close_on_finish=False,
       max_neurons=20, title=None, log_every=None, torch_threads="auto") -> TrainingResult
```

## Subpackages

For advanced use: custom frontends, embedding, tooling.

| Package | Page |
|---|---|
| `hollywood_training.tracking` | [Tracking](Tracking) |
| `hollywood_training.training` | [Training](Training) |
| `hollywood_training.sync` | [Synchronization](Synchronization) |
| `hollywood_training.commands` | [Commands](Commands) |
| `hollywood_training.monitor` | [Monitoring](Monitoring) |
| `hollywood_training.gui` | [GUI](GUI) |
| `hollywood_training.console` | [Console Frontend](Console-Frontend) |
| `hollywood_training.runner` | [Runner](Runner) |
| `hollywood_training.formatting` | [Formatting](Formatting) |
