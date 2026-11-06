"""Support package of the SPEC harness (``tests/``) - helpers only, no tests.

The harness verifies the package against ``specs/R1.md`` and ``specs/R2.md``. Modules:

* :mod:`.paths`      project locations; every temporary file goes to ``<project>/.temp/``
* :mod:`.env`        environment of the test process and of subprocesses (no real display)
* :mod:`.models`     small native PyTorch networks, data and training setups
* :mod:`.events`     :class:`~.events.SessionFeeder` - real-shaped session events
* :mod:`.rig`        :class:`~.rig.GuiRig` - an offscreen window wired to the real commands
* :mod:`.painting`   :class:`~.painting.PaintCapture` - what the panels actually draw
* :mod:`.gui_state`  observable window state and pointer gestures (R2 tests)
* :mod:`.headless`   :class:`~.headless.HeadlessRun` - a real manager, recording frontend
* :mod:`.pilot`      :class:`~.pilot.Pilot` - drives a real in-process ``ht.run`` by keyboard
* :mod:`.probe_log`  enables and reads the window-process probe (``htspec/probe``)
* :mod:`.subproc`    Python subprocesses with the harness environment
* :mod:`.x11`        a private Xvfb server and an XTest / XFixes client (ctypes)
* :mod:`.live`       :class:`~.live.LiveApp` - a real user script on the private X server
* :mod:`.keys`       key sequences in a small text notation, typed with ``QTest``
"""
