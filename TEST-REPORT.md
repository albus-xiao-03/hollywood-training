============================= test session starts ==============================
platform linux -- Python 3.13.5, pytest-9.2.0, pluggy-1.7.0
cachedir: .temp/pytest-cache
PySide6 6.12.0 -- Qt runtime 6.12.0 -- Qt compiled 6.12.0
rootdir: /home/agent/workspace/hollywood-training
configfile: pyproject.toml
testpaths: tests
plugins: qt-4.5.0, timeout-2.4.0, cov-7.1.0
timeout: 120.0s
timeout method: signal
timeout func_only: False
collected 1083 items

tests/test_r1_01_tracking.py ........................................... [  3%]
...                                                                      [  4%]
tests/test_r1_02_training_management.py ................................ [  7%]
.......................................                                  [ 10%]
tests/test_r1_03_native_workflow.py ..............                       [ 12%]
tests/test_r1_04_native_gui.py ..................                        [ 13%]
tests/test_r1_05_synchronization.py ..........................           [ 16%]
tests/test_r1_06_speed.py ..........                                     [ 17%]
tests/test_r1_07_commands.py ........................................... [ 21%]
.................................                                        [ 24%]
tests/test_r1_08_no_dark_magic.py ...................................... [ 27%]
.........................                                                [ 29%]
tests/test_r1_09_modular_architecture.py ............................... [ 32%]
........................................................................ [ 39%]
.................................................                        [ 43%]
tests/test_r1_10_package_folder.py ....                                  [ 44%]
tests/test_r1_11_demos.py ..........                                     [ 45%]
tests/test_r1_12_readme.py .....                                         [ 45%]
tests/test_r1_13_wiki.py ............................................... [ 50%]
.........................                                                [ 52%]
tests/test_r1_g1_network_view.py .............FFF                        [ 53%]
tests/test_r1_g2_status_panel.py ..........................              [ 56%]
tests/test_r1_g3_loss_chart.py ...........                               [ 57%]
tests/test_r1_g4_usage.py ................                               [ 58%]
tests/test_r1_g5_terminal.py ...............                             [ 60%]
tests/test_r1_g6_network_name.py ..........                              [ 61%]
tests/test_r1_g7_blue_theme.py ......................                    [ 63%]
tests/test_r2_1_no_pointer.py .....................................      [ 66%]
tests/test_r2_2_mouse_disabled.py ...................................... [ 69%]
........................................................................ [ 76%]
..                                                                       [ 76%]
tests/test_r2_3_keyboard_only.py ....................................... [ 80%]
..............                                                           [ 81%]
tests/test_r2_4_view_commands.py ....................................... [ 85%]
........................................................................ [ 91%]
.............                                                            [ 93%]
tests/test_r2_5_r1_unaffected.py ..........................              [ 95%]
tests/test_r2_6_outside_window.py ..............                         [ 96%]
tests/test_r2_7_keyboard_reroute.py ..................................   [100%]

=================================== FAILURES ===================================
_____ test_every_panel_stays_fully_visible_at_small_window_sizes[960x680] ______

rig = <htspec.rig.GuiRig object at 0xffff4f8ed9f0>, size = (960, 680)

    @pytest.mark.req("R1.G2", "R1.G3", "R1.G4", "R1.G5")
    @pytest.mark.parametrize(
        "size", [(960, 680), (1202, 680), (1366, 720)], ids=lambda s: f"{s[0]}x{s[1]}"
    )
    def test_every_panel_stays_fully_visible_at_small_window_sizes(rig, size):
        """The window allows these sizes (its minimum is 960 x 680; a 1366 x 768 laptop screen
        opens it at 680 px high): no panel may overlap another, leave the window or be squeezed
        below its own minimum size - the information of every panel must stay visible."""
        window = rig.window
        window.resize(*size)
        rig.process_events(100)
        assert (window.width(), window.height()) == size
        rects = _visible_rects(window)
        problems = []
        for name, (x, y, w, h) in rects.items():
            widget = getattr(window, name)
            if x < 0 or y < 0 or x + w > window.width() or y + h > window.height():
                problems.append(f"{name} leaves the window: {(x, y, w, h)}")
            if w < widget.minimumWidth() or h < widget.minimumHeight():
                problems.append(f"{name} squeezed to {w}x{h} (minimum {widget.minimumWidth()}x"
                                f"{widget.minimumHeight()})")  # fmt: skip
        for (a, ra), (b, rb) in itertools.combinations(rects.items(), 2):
            overlap_w = min(ra[0] + ra[2], rb[0] + rb[2]) - max(ra[0], rb[0])
            overlap_h = min(ra[1] + ra[3], rb[1] + rb[3]) - max(ra[1], rb[1])
            if overlap_w > 0 and overlap_h > 0:
                problems.append(f"{a} and {b} overlap")
>       assert problems == []
E       AssertionError: assert ['status and chart overlap'] == []
E         
E         Left contains one more item: 'status and chart overlap'
E         Use -v to get more diff

tests/test_r1_g1_network_view.py:223: AssertionError
----------------------------- Captured Qt messages -----------------------------
QtWarningMsg: This plugin does not support raise()
_____ test_every_panel_stays_fully_visible_at_small_window_sizes[1202x680] _____

rig = <htspec.rig.GuiRig object at 0xffff4f8ef070>, size = (1202, 680)

    @pytest.mark.req("R1.G2", "R1.G3", "R1.G4", "R1.G5")
    @pytest.mark.parametrize(
        "size", [(960, 680), (1202, 680), (1366, 720)], ids=lambda s: f"{s[0]}x{s[1]}"
    )
    def test_every_panel_stays_fully_visible_at_small_window_sizes(rig, size):
        """The window allows these sizes (its minimum is 960 x 680; a 1366 x 768 laptop screen
        opens it at 680 px high): no panel may overlap another, leave the window or be squeezed
        below its own minimum size - the information of every panel must stay visible."""
        window = rig.window
        window.resize(*size)
        rig.process_events(100)
        assert (window.width(), window.height()) == size
        rects = _visible_rects(window)
        problems = []
        for name, (x, y, w, h) in rects.items():
            widget = getattr(window, name)
            if x < 0 or y < 0 or x + w > window.width() or y + h > window.height():
                problems.append(f"{name} leaves the window: {(x, y, w, h)}")
            if w < widget.minimumWidth() or h < widget.minimumHeight():
                problems.append(f"{name} squeezed to {w}x{h} (minimum {widget.minimumWidth()}x"
                                f"{widget.minimumHeight()})")  # fmt: skip
        for (a, ra), (b, rb) in itertools.combinations(rects.items(), 2):
            overlap_w = min(ra[0] + ra[2], rb[0] + rb[2]) - max(ra[0], rb[0])
            overlap_h = min(ra[1] + ra[3], rb[1] + rb[3]) - max(ra[1], rb[1])
            if overlap_w > 0 and overlap_h > 0:
                problems.append(f"{a} and {b} overlap")
>       assert problems == []
E       AssertionError: assert ['status and chart overlap'] == []
E         
E         Left contains one more item: 'status and chart overlap'
E         Use -v to get more diff

tests/test_r1_g1_network_view.py:223: AssertionError
----------------------------- Captured Qt messages -----------------------------
QtWarningMsg: This plugin does not support raise()
_____ test_every_panel_stays_fully_visible_at_small_window_sizes[1366x720] _____

rig = <htspec.rig.GuiRig object at 0xffff4f8ef930>, size = (1366, 720)

    @pytest.mark.req("R1.G2", "R1.G3", "R1.G4", "R1.G5")
    @pytest.mark.parametrize(
        "size", [(960, 680), (1202, 680), (1366, 720)], ids=lambda s: f"{s[0]}x{s[1]}"
    )
    def test_every_panel_stays_fully_visible_at_small_window_sizes(rig, size):
        """The window allows these sizes (its minimum is 960 x 680; a 1366 x 768 laptop screen
        opens it at 680 px high): no panel may overlap another, leave the window or be squeezed
        below its own minimum size - the information of every panel must stay visible."""
        window = rig.window
        window.resize(*size)
        rig.process_events(100)
        assert (window.width(), window.height()) == size
        rects = _visible_rects(window)
        problems = []
        for name, (x, y, w, h) in rects.items():
            widget = getattr(window, name)
            if x < 0 or y < 0 or x + w > window.width() or y + h > window.height():
                problems.append(f"{name} leaves the window: {(x, y, w, h)}")
            if w < widget.minimumWidth() or h < widget.minimumHeight():
                problems.append(f"{name} squeezed to {w}x{h} (minimum {widget.minimumWidth()}x"
                                f"{widget.minimumHeight()})")  # fmt: skip
        for (a, ra), (b, rb) in itertools.combinations(rects.items(), 2):
            overlap_w = min(ra[0] + ra[2], rb[0] + rb[2]) - max(ra[0], rb[0])
            overlap_h = min(ra[1] + ra[3], rb[1] + rb[3]) - max(ra[1], rb[1])
            if overlap_w > 0 and overlap_h > 0:
                problems.append(f"{a} and {b} overlap")
>       assert problems == []
E       AssertionError: assert ['status and chart overlap'] == []
E         
E         Left contains one more item: 'status and chart overlap'
E         Use -v to get more diff

tests/test_r1_g1_network_view.py:223: AssertionError
----------------------------- Captured Qt messages -----------------------------
QtWarningMsg: This plugin does not support raise()
=============================== warnings summary ===============================
tests/test_r1_03_native_workflow.py::test_the_network_still_scripts_like_an_untracked_one
  /home/agent/workspace/hollywood-training/.venv/lib/python3.13/site-packages/torch/jit/_script.py:1491: FutureWarning: `torch.jit.script` is deprecated. Please switch to `torch.compile` or `torch.export`.
    warnings.warn(

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
============= SPEC requirement coverage (specs/R1.md, specs/R2.md) =============
req      pass  fail  skip     requirement
R1.1       46     0     0     track the network architecture, including the weights of each layer
R1.2       71     0     0     manage the training: native PyTorch definition, wrapped, passed to the manager
R1.3       14     0     0     requirements 1 and 2 do not significantly change the native PyTorch workflow
R1.4       20     0     0     native GUI window (not a web page) with smooth animations of the real training
R1.5       26     0     0     training synchronised with the GUI: updated every epoch, before the next epoch
R1.6       10     0     0     training runs as fast as the synchronisation allows
R1.7      200     0     0     well-defined commands control the training at runtime
R1.8       63     0     0     no 'dark magic' Python methods
R1.9      152     0     0     completely decomposed into modules with a clear architecture
R1.10       4     0     0     all package files in the package folder hollywood_training
R1.11      10     0     0     a demo with an illustrative network and dataset, in demos/
R1.12       5     0     0     a README.md for quick start
R1.13      72     0     0     a wiki folder: a page for every component and the overall architecture
R1.G1      15     3     0  !! GUI: architecture, neuron connections, real weights changing every epoch
R1.G2      26     3     0  !! GUI: training status (epochs, batch size, lr, loss, optimizer, ..., timer)
R1.G3      11     3     0  !! GUI: training and validation loss drawn dynamically every epoch
R1.G4      16     3     0  !! GUI: memory, CPU and (when applicable) GPU usage
R1.G5      15     3     0  !! GUI: an integrated terminal controls the training with the commands
R1.G6      10     0     0     GUI: the name of the training network (its class name)
R1.G7      22     0     0     GUI: blue based
R2.1       37     0     0     no mouse pointer displayed anywhere inside the window
R2.2      112     0     0     all mouse-based interactions disabled within the application
R2.3       53     0     0     all implemented functionality accessible and operational without the mouse
R2.4      130     0     0     additional commands may replace the mouse-based interactions
R2.5      830     3     0  !! all other R1 features not affected (every R1 test counts here as well)
R2.6       14     0     0     pointer and mouse interactions outside the window not affected
R2.7       34     0     0     all keyboard input rerouted to the internal terminal while on the window
coverage report: /home/agent/workspace/hollywood-training/.temp/tests/reports/coverage-20261107-010000.json
================================ tests coverage ================================
_______________ coverage: platform linux, python 3.13.5-final-0 ________________

Name                                             Stmts   Miss  Cover   Missing
------------------------------------------------------------------------------
hollywood_training/__init__.py                      15      1    93%   70
hollywood_training/_version.py                       1      0   100%
hollywood_training/commands/__init__.py             15      7    53%   43-48, 52
hollywood_training/commands/builtin.py             284     17    94%   118, 146, 154, 169, 248, 251, 257-258, 264, 272, 293-299
hollywood_training/commands/context.py              39      0   100%
hollywood_training/commands/registry.py            134      9    93%   109, 130, 137-142, 149
hollywood_training/commands/view.py                  7      0   100%
hollywood_training/console.py                       63      3    95%   47, 73-74
hollywood_training/formatting.py                    57      4    93%   39, 58, 64, 70
hollywood_training/gui/__init__.py                  14      7    50%   36-41, 45
hollywood_training/gui/animation.py                 35      2    94%   32, 53
hollywood_training/gui/app.py                      166     31    81%   57, 64, 109-111, 115-117, 156, 183-185, 191, 196, 212, 226-227, 239-240, 247-252, 255-260
hollywood_training/gui/keyboard_focus.py            41      2    95%   36, 47
hollywood_training/gui/link.py                      37      9    76%   47, 52, 56-58, 65-66, 71-72
hollywood_training/gui/pointer_guard.py             45      1    98%   78
hollywood_training/gui/remote.py                   203     41    80%   110-112, 114-116, 120, 134, 145-146, 155, 161-162, 166-167, 173-174, 176-182, 196-197, 201, 207, 220-221, 231, 236-239, 283-284, 288-289, 292-293
hollywood_training/gui/theme.py                    149      1    99%   95
hollywood_training/gui/view_actions.py             148      5    97%   53, 57, 63, 237-238
hollywood_training/gui/widgets/__init__.py           8      0   100%
hollywood_training/gui/widgets/header.py           126      0   100%
hollywood_training/gui/widgets/hud.py              106      4    96%   79, 83, 87-88
hollywood_training/gui/widgets/loss_chart.py       316      4    99%   43, 167, 223, 385
hollywood_training/gui/widgets/network_view.py     545     18    97%   105, 123-124, 190-191, 342, 417, 515-519, 538, 552-557, 584, 788
hollywood_training/gui/widgets/status_panel.py     117      2    98%   46, 169
hollywood_training/gui/widgets/terminal.py         127      3    98%   63, 140, 185
hollywood_training/gui/widgets/usage_panel.py      128      5    96%   51-52, 66, 69-70
hollywood_training/gui/window.py                   210      4    98%   186-187, 314, 338
hollywood_training/gui/window_process.py           106    106     0%   16-147
hollywood_training/monitor/__init__.py              13      4    69%   32-34, 38
hollywood_training/monitor/gpu.py                   54     22    59%   34-35, 49-63, 75-76, 78-80
hollywood_training/monitor/system.py                25      2    92%   40-41
hollywood_training/monitor/usage.py                 22      0   100%
hollywood_training/runner.py                        77      5    94%   38-39, 61, 63-64
hollywood_training/sync/__init__.py                  6      0   100%
hollywood_training/sync/channel.py                  57      2    96%   51, 92
hollywood_training/sync/control.py                 116      3    97%   36, 71, 94
hollywood_training/sync/events.py                  115      1    99%   191
hollywood_training/sync/pump.py                     13      0   100%
hollywood_training/sync/session.py                 119      2    98%   159, 165
hollywood_training/tracking/__init__.py             15      4    73%   89-91, 95
hollywood_training/tracking/architecture.py         93      5    95%   49-52, 157
hollywood_training/tracking/decorator.py            46      0   100%
hollywood_training/tracking/inspector.py           119     18    85%   81, 176-177, 231, 235, 240, 271-274, 277-286
hollywood_training/tracking/layers.py              179     14    92%   108-109, 187, 191, 201, 205, 228, 232, 235, 272, 300, 308, 327, 335
hollywood_training/tracking/roles.py                10      0   100%
hollywood_training/tracking/snapshot.py            180     15    92%   82, 85, 148-149, 168, 194, 213, 236, 244, 260, 294-295, 301, 309, 330
hollywood_training/tracking/tracker.py              60      7    88%   38, 52, 56, 68, 72-73, 89
hollywood_training/tracking/weights.py              57      2    96%   92, 94
hollywood_training/training/__init__.py              8      0   100%
hollywood_training/training/describe.py             77      6    92%   52-53, 57, 104, 114-115
hollywood_training/training/history.py              22      0   100%
hollywood_training/training/manager.py             278     23    92%   120, 124, 128, 136, 149-150, 159-160, 178, 193-194, 225, 289-290, 323-324, 369-370, 398-400, 443-444
hollywood_training/training/process.py             132      3    98%   273-275
hollywood_training/training/steps.py                71     11    85%   69-71, 78-84, 91
hollywood_training/training/timing.py               20      1    95%   17
------------------------------------------------------------------------------
TOTAL                                             5226    436    92%
=========================== short test summary info ============================
FAILED tests/test_r1_g1_network_view.py::test_every_panel_stays_fully_visible_at_small_window_sizes[960x680]
FAILED tests/test_r1_g1_network_view.py::test_every_panel_stays_fully_visible_at_small_window_sizes[1202x680]
FAILED tests/test_r1_g1_network_view.py::test_every_panel_stays_fully_visible_at_small_window_sizes[1366x720]
============ 3 failed, 1080 passed, 1 warning in 158.48s (0:02:38) =============
