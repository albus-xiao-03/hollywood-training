"""Support package of the SPEC harness (``tests/``) - helpers only, no tests.

The harness verifies the package against ``specs/R1.md`` and ``specs/R2.md``. Modules:

* :mod:`.paths`      project locations; every temporary file goes to ``<project>/.temp/``
* :mod:`.env`        environment of the test process and of subprocesses (no real display)
* :mod:`.models`     small native PyTorch networks, data and training setups
* :mod:`.events`     :class:`~.events.SessionFeeder` - real-shaped session events
* :mod:`.rig`        :class:`~.rig.GuiRig` - an offscreen window wired to the real commands
* :mod:`.painting`   :class:`~.painting.PaintCapture` - what the panels actually draw
* :mod:`.headless`   :class:`~.headless.HeadlessRun` - a real manager, recording frontend
* :mod:`.keys`       key sequences in a small text notation, typed with ``QTest``
"""
