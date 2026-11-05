"""The SPEC.md requirements this harness verifies: one catalog for traceability.

``REQUIREMENTS`` maps each ID used in ``@pytest.mark.spec(...)`` to the SPEC text it stands
for. ``MANUAL_CHECKS`` lists what automated tests cannot (fully) verify; the terminal
summary prints it after every run so it is never forgotten.
"""

from __future__ import annotations

REQUIREMENTS: dict[str, str] = {
    "R1": "Automatically track the network architecture, including the weights of each layer "
    "(decorator on the class or slot dataclasses preferred)",
    "R2": "Automatically manage training: native PyTorch-style steps, wrapped in an object that "
    "is passed to the package's management method",
    "R3": "Tracking (R1) and management (R2) must not significantly change the native PyTorch "
    "workflow",
    "R4": "A native GUI window (not a web page) with smooth animations of the REAL training",
    "R5": "Every training update is synchronized with the GUI: updated at EVERY epoch, "
    "completely, BEFORE the next epoch",
    "R6": "Training runs as fast as the synchronization allows (no regard for animation "
    "readability)",
    "R7": "A set of well-defined commands controls the managed training at runtime",
    "R8": "No 'dark magic' (e.g. retrieving and regex-analysing the user's function source)",
    "R9": "The package is completely decomposed into modules with a clear architecture",
    "R10": "All package files are inside the package folder 'hollywood_training'",
    "R11": "A demo with an illustrative network and dataset; all demo files in 'demos/' "
    "directly under the project folder, outside the package",
    "R12": "A README.md for quick start",
    "R13": "A 'wiki' folder with separate markdown files explaining EVERY component AND the "
    "overall architecture (GitHub-wiki compatibility preferred)",
    "G1": "GUI shows the architecture, the connections between each layer's neurons and the "
    "REAL weights, changing with the REAL training at EACH epoch",
    "G2": "GUI shows the training status: total epoch, current epoch, batch size, learning "
    "rate, loss function, optimizer, normalization strategy, progress bar, process timer",
    "G3": "GUI draws training and validation loss DYNAMICALLY with EACH epoch",
    "G4": "GUI shows memory, CPU and (when applicable) GPU usage",
    "G5": "GUI has an integrated terminal to control the training with the defined commands",
    "G6": "GUI shows the name (class name) of the training network",
    "G7": "GUI is blue based",
}

MANUAL_CHECKS: dict[str, str] = {
    "R4": "Perceived smoothness and the 'Hollywood' look need a human on a real display; tests "
    "only bound the frame rate and check that frames change.",
    "R6": "Absolute speed depends on the machine; the thread-policy comparison is an opt-in "
    "benchmark (HT_RUN_BENCHMARKS=1), and GPU training speed is not measured.",
    "R8": "Static scans cover the known 'dark magic' techniques; a reviewer should still read "
    "new code for novel ones.",
    "R9": "'Clear architecture' is partly a judgement call; tests check module boundaries, "
    "import cycles and documented layering only.",
    "R12": "Whether the README is a *good* quick start is a judgement call; tests run its code "
    "and check required sections.",
    "R13": "Tests check that every module and export is mentioned and links resolve, not that "
    "the explanations are correct or complete; publishing to a real GitHub wiki is manual.",
    "G4": "GPU rows are tested with synthetic readings; real CUDA / MPS readings need such a "
    "device (gpu-marked tests are skipped without one).",
    "G7": "Colour checks are numeric (hue bands); the overall visual impression is manual.",
    "ALL": "Real desktop sessions (Windows, macOS, Wayland, X11 compositors, HiDPI, multiple "
    "monitors) are not covered by the headless 'offscreen' runs.",
}
