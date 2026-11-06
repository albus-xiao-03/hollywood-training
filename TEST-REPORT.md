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
collected 959 items / 1 error

tests/test_r1_01_tracking.py ........................................FFF [  4%]
FFF                                                                      [  4%]
tests/test_r1_02_training_management.py ................................ [  8%]
.......................................                                  [ 12%]
tests/test_r1_03_native_workflow.py ..............                       [ 13%]
tests/test_r1_04_native_gui.py ..................                        [ 15%]
tests/test_r1_05_synchronization.py ..........................           [ 18%]
tests/test_r1_06_speed.py ..........                                     [ 19%]
tests/test_r1_07_commands.py ........................................... [ 23%]
.................................                                        [ 27%]
tests/test_r1_08_no_dark_magic.py ...................................... [ 31%]
.........................                                                [ 33%]
tests/test_r1_09_modular_architecture.py ............................... [ 37%]
........................................................................ [ 44%]
.................................................                        [ 49%]
tests/test_r1_10_package_folder.py ....                                  [ 50%]
tests/test_r1_11_demos.py ..........                                     [ 51%]
tests/test_r1_12_readme.py .....                                         [ 51%]
tests/test_r1_13_wiki.py ............................................... [ 56%]
.........................                                                [ 59%]
tests/test_r1_g1_network_view.py ..........FF.FFF                        [ 60%]
tests/test_r1_g2_status_panel.py ..........................              [ 63%]
tests/test_r1_g3_loss_chart.py ...........                               [ 64%]
tests/test_r1_g4_usage.py ................                               [ 66%]
tests/test_r1_g5_terminal.py ...............                             [ 67%]
tests/test_r1_g6_network_name.py .........F                              [ 68%]
tests/test_r1_g7_blue_theme.py ......................                    [ 71%]
tests/test_r2_1_no_pointer.py ...................................FF      [ 75%]
tests/test_r2_2_mouse_disabled.py ...................................... [ 79%]
...................................................................EEEEE [ 86%]
EE                                                                       [ 86%]
tests/test_r2_3_keyboard_only.py ....................................... [ 90%]
..............                                                           [ 92%]
tests/test_r2_5_r1_unaffected.py ..........................              [ 94%]
tests/test_r2_6_outside_window.py ............FF                         [ 96%]
tests/test_r2_7_keyboard_reroute.py ............................F.FFF.   [100%]

==================================== ERRORS ====================================
______________ ERROR collecting tests/test_r2_4_view_commands.py _______________
tests/test_r2_4_view_commands.py::test_each_form_hands_the_window_one_validated_request: 'request' is a reserved name and cannot be used in @pytest.mark.parametrize
_______ ERROR at setup of test_real_clicks_on_every_panel_change_nothing _______

xserver = <htspec.x11.XServer object at 0xffff2936b620>
xclient = <htspec.x11.XClient object at 0xffff2936b380>
tmp_path_factory = TempPathFactory(_given_basetemp=None, _trace=<pluggy._tracing.TagTracerSub object at 0xffffaa307360>, _basetemp=PosixP...e/agent/workspace/hollywood-training/.temp/tmp/pytest-of-agent/pytest-0'), _retention_count=3, _retention_policy='all')

    @pytest.fixture(scope="module")
    def live(xserver, xclient, tmp_path_factory):
        app = LiveApp(xserver, xclient, tmp_path_factory.mktemp("r2_2_live"))
        for _ in range(3):  # a log long enough to scroll and to select from
>           app.enter("help", expect="type 'help <command>' for details")

tests/test_r2_2_mouse_disabled.py:317: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

self = <htspec.live.LiveApp object at 0xffff29115ae0>, line = 'help'
expect = "type 'help <command>' for details", timeout = 20.0

    def enter(self, line: str, *, expect: str | None = None, timeout: float = 20.0):
        """Type ``line`` + Return on the real keyboard and wait for the command's answer.
    
        Waits until the window process has sent this line to the training process and the
        answer has come back (and, with ``expect``, until the answer - or the window-only
        outcome of a view command - contains ``expect``); returns a probe snapshot taken
        after that.
        """
        started = self.now()
        self.type(line + "\n")
>       sent = self.wait(
            lambda r: r.get("msg") == "command" and r.get("value") == line,
            kind="send", after=started, timeout=timeout,
        )  # fmt: skip
E       TypeError: LiveApp.wait() got an unexpected keyword argument 'kind'

tests/htspec/live.py:136: TypeError
__ ERROR at setup of test_real_drags_on_every_splitter_handle_change_nothing ___

xserver = <htspec.x11.XServer object at 0xffff2936b620>
xclient = <htspec.x11.XClient object at 0xffff2936b380>
tmp_path_factory = TempPathFactory(_given_basetemp=None, _trace=<pluggy._tracing.TagTracerSub object at 0xffffaa307360>, _basetemp=PosixP...e/agent/workspace/hollywood-training/.temp/tmp/pytest-of-agent/pytest-0'), _retention_count=3, _retention_policy='all')

    @pytest.fixture(scope="module")
    def live(xserver, xclient, tmp_path_factory):
        app = LiveApp(xserver, xclient, tmp_path_factory.mktemp("r2_2_live"))
        for _ in range(3):  # a log long enough to scroll and to select from
>           app.enter("help", expect="type 'help <command>' for details")

tests/test_r2_2_mouse_disabled.py:317: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

self = <htspec.live.LiveApp object at 0xffff29115ae0>, line = 'help'
expect = "type 'help <command>' for details", timeout = 20.0

    def enter(self, line: str, *, expect: str | None = None, timeout: float = 20.0):
        """Type ``line`` + Return on the real keyboard and wait for the command's answer.
    
        Waits until the window process has sent this line to the training process and the
        answer has come back (and, with ``expect``, until the answer - or the window-only
        outcome of a view command - contains ``expect``); returns a probe snapshot taken
        after that.
        """
        started = self.now()
        self.type(line + "\n")
>       sent = self.wait(
            lambda r: r.get("msg") == "command" and r.get("value") == line,
            kind="send", after=started, timeout=timeout,
        )  # fmt: skip
E       TypeError: LiveApp.wait() got an unexpected keyword argument 'kind'

tests/htspec/live.py:136: TypeError
__ ERROR at setup of test_real_text_selection_with_the_mouse_selects_nothing ___

xserver = <htspec.x11.XServer object at 0xffff2936b620>
xclient = <htspec.x11.XClient object at 0xffff2936b380>
tmp_path_factory = TempPathFactory(_given_basetemp=None, _trace=<pluggy._tracing.TagTracerSub object at 0xffffaa307360>, _basetemp=PosixP...e/agent/workspace/hollywood-training/.temp/tmp/pytest-of-agent/pytest-0'), _retention_count=3, _retention_policy='all')

    @pytest.fixture(scope="module")
    def live(xserver, xclient, tmp_path_factory):
        app = LiveApp(xserver, xclient, tmp_path_factory.mktemp("r2_2_live"))
        for _ in range(3):  # a log long enough to scroll and to select from
>           app.enter("help", expect="type 'help <command>' for details")

tests/test_r2_2_mouse_disabled.py:317: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

self = <htspec.live.LiveApp object at 0xffff29115ae0>, line = 'help'
expect = "type 'help <command>' for details", timeout = 20.0

    def enter(self, line: str, *, expect: str | None = None, timeout: float = 20.0):
        """Type ``line`` + Return on the real keyboard and wait for the command's answer.
    
        Waits until the window process has sent this line to the training process and the
        answer has come back (and, with ``expect``, until the answer - or the window-only
        outcome of a view command - contains ``expect``); returns a probe snapshot taken
        after that.
        """
        started = self.now()
        self.type(line + "\n")
>       sent = self.wait(
            lambda r: r.get("msg") == "command" and r.get("value") == line,
            kind="send", after=started, timeout=timeout,
        )  # fmt: skip
E       TypeError: LiveApp.wait() got an unexpected keyword argument 'kind'

tests/htspec/live.py:136: TypeError
____________ ERROR at setup of test_real_wheel_turns_scroll_nothing ____________

xserver = <htspec.x11.XServer object at 0xffff2936b620>
xclient = <htspec.x11.XClient object at 0xffff2936b380>
tmp_path_factory = TempPathFactory(_given_basetemp=None, _trace=<pluggy._tracing.TagTracerSub object at 0xffffaa307360>, _basetemp=PosixP...e/agent/workspace/hollywood-training/.temp/tmp/pytest-of-agent/pytest-0'), _retention_count=3, _retention_policy='all')

    @pytest.fixture(scope="module")
    def live(xserver, xclient, tmp_path_factory):
        app = LiveApp(xserver, xclient, tmp_path_factory.mktemp("r2_2_live"))
        for _ in range(3):  # a log long enough to scroll and to select from
>           app.enter("help", expect="type 'help <command>' for details")

tests/test_r2_2_mouse_disabled.py:317: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

self = <htspec.live.LiveApp object at 0xffff29115ae0>, line = 'help'
expect = "type 'help <command>' for details", timeout = 20.0

    def enter(self, line: str, *, expect: str | None = None, timeout: float = 20.0):
        """Type ``line`` + Return on the real keyboard and wait for the command's answer.
    
        Waits until the window process has sent this line to the training process and the
        answer has come back (and, with ``expect``, until the answer - or the window-only
        outcome of a view command - contains ``expect``); returns a probe snapshot taken
        after that.
        """
        started = self.now()
        self.type(line + "\n")
>       sent = self.wait(
            lambda r: r.get("msg") == "command" and r.get("value") == line,
            kind="send", after=started, timeout=timeout,
        )  # fmt: skip
E       TypeError: LiveApp.wait() got an unexpected keyword argument 'kind'

tests/htspec/live.py:136: TypeError
___________ ERROR at setup of test_real_middle_click_does_not_paste ____________

xserver = <htspec.x11.XServer object at 0xffff2936b620>
xclient = <htspec.x11.XClient object at 0xffff2936b380>
tmp_path_factory = TempPathFactory(_given_basetemp=None, _trace=<pluggy._tracing.TagTracerSub object at 0xffffaa307360>, _basetemp=PosixP...e/agent/workspace/hollywood-training/.temp/tmp/pytest-of-agent/pytest-0'), _retention_count=3, _retention_policy='all')

    @pytest.fixture(scope="module")
    def live(xserver, xclient, tmp_path_factory):
        app = LiveApp(xserver, xclient, tmp_path_factory.mktemp("r2_2_live"))
        for _ in range(3):  # a log long enough to scroll and to select from
>           app.enter("help", expect="type 'help <command>' for details")

tests/test_r2_2_mouse_disabled.py:317: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

self = <htspec.live.LiveApp object at 0xffff29115ae0>, line = 'help'
expect = "type 'help <command>' for details", timeout = 20.0

    def enter(self, line: str, *, expect: str | None = None, timeout: float = 20.0):
        """Type ``line`` + Return on the real keyboard and wait for the command's answer.
    
        Waits until the window process has sent this line to the training process and the
        answer has come back (and, with ``expect``, until the answer - or the window-only
        outcome of a view command - contains ``expect``); returns a probe snapshot taken
        after that.
        """
        started = self.now()
        self.type(line + "\n")
>       sent = self.wait(
            lambda r: r.get("msg") == "command" and r.get("value") == line,
            kind="send", after=started, timeout=timeout,
        )  # fmt: skip
E       TypeError: LiveApp.wait() got an unexpected keyword argument 'kind'

tests/htspec/live.py:136: TypeError
_____ ERROR at setup of test_real_hovering_shows_no_tooltip_and_no_details _____

xserver = <htspec.x11.XServer object at 0xffff2936b620>
xclient = <htspec.x11.XClient object at 0xffff2936b380>
tmp_path_factory = TempPathFactory(_given_basetemp=None, _trace=<pluggy._tracing.TagTracerSub object at 0xffffaa307360>, _basetemp=PosixP...e/agent/workspace/hollywood-training/.temp/tmp/pytest-of-agent/pytest-0'), _retention_count=3, _retention_policy='all')

    @pytest.fixture(scope="module")
    def live(xserver, xclient, tmp_path_factory):
        app = LiveApp(xserver, xclient, tmp_path_factory.mktemp("r2_2_live"))
        for _ in range(3):  # a log long enough to scroll and to select from
>           app.enter("help", expect="type 'help <command>' for details")

tests/test_r2_2_mouse_disabled.py:317: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

self = <htspec.live.LiveApp object at 0xffff29115ae0>, line = 'help'
expect = "type 'help <command>' for details", timeout = 20.0

    def enter(self, line: str, *, expect: str | None = None, timeout: float = 20.0):
        """Type ``line`` + Return on the real keyboard and wait for the command's answer.
    
        Waits until the window process has sent this line to the training process and the
        answer has come back (and, with ``expect``, until the answer - or the window-only
        outcome of a view command - contains ``expect``); returns a probe snapshot taken
        after that.
        """
        started = self.now()
        self.type(line + "\n")
>       sent = self.wait(
            lambda r: r.get("msg") == "command" and r.get("value") == line,
            kind="send", after=started, timeout=timeout,
        )  # fmt: skip
E       TypeError: LiveApp.wait() got an unexpected keyword argument 'kind'

tests/htspec/live.py:136: TypeError
_ ERROR at setup of test_training_and_keyboard_still_work_after_all_that_mouse_input _

xserver = <htspec.x11.XServer object at 0xffff2936b620>
xclient = <htspec.x11.XClient object at 0xffff2936b380>
tmp_path_factory = TempPathFactory(_given_basetemp=None, _trace=<pluggy._tracing.TagTracerSub object at 0xffffaa307360>, _basetemp=PosixP...e/agent/workspace/hollywood-training/.temp/tmp/pytest-of-agent/pytest-0'), _retention_count=3, _retention_policy='all')

    @pytest.fixture(scope="module")
    def live(xserver, xclient, tmp_path_factory):
        app = LiveApp(xserver, xclient, tmp_path_factory.mktemp("r2_2_live"))
        for _ in range(3):  # a log long enough to scroll and to select from
>           app.enter("help", expect="type 'help <command>' for details")

tests/test_r2_2_mouse_disabled.py:317: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

self = <htspec.live.LiveApp object at 0xffff29115ae0>, line = 'help'
expect = "type 'help <command>' for details", timeout = 20.0

    def enter(self, line: str, *, expect: str | None = None, timeout: float = 20.0):
        """Type ``line`` + Return on the real keyboard and wait for the command's answer.
    
        Waits until the window process has sent this line to the training process and the
        answer has come back (and, with ``expect``, until the answer - or the window-only
        outcome of a view command - contains ``expect``); returns a probe snapshot taken
        after that.
        """
        started = self.now()
        self.type(line + "\n")
>       sent = self.wait(
            lambda r: r.get("msg") == "command" and r.get("value") == line,
            kind="send", after=started, timeout=timeout,
        )  # fmt: skip
E       TypeError: LiveApp.wait() got an unexpected keyword argument 'kind'

tests/htspec/live.py:136: TypeError
=================================== FAILURES ===================================
_________ test_every_weight_matrix_of_the_network_is_tracked[Bilinear] _________

kind = 'Bilinear'

    @pytest.mark.parametrize("kind", list(LAYER_TYPES))
    def test_every_weight_matrix_of_the_network_is_tracked(kind):
        """Strict reading of "including the weights of each layers": every weight tensor of a
        layer (a parameter with two or more dimensions) must appear in the tracked weights."""
        model = Holder(LAYER_TYPES[kind]())
        tracked = set(NetworkTracker(model).snapshot(0).names())
        untracked = [
            name
            for name, parameter in model.named_parameters()
            if parameter.dim() >= 2 and name.rsplit(".", 1)[0] not in tracked
        ]
>       assert untracked == [], f"{kind}: weights not tracked: {untracked} (tracked: {sorted(tracked)})"
E       AssertionError: Bilinear: weights not tracked: ['layer.weight'] (tracked: [])
E       assert ['layer.weight'] == []
E         
E         Left contains one more item: 'layer.weight'
E         Use -v to get more diff

tests/test_r1_01_tracking.py:406: AssertionError
___________ test_every_weight_matrix_of_the_network_is_tracked[RNN] ____________

kind = 'RNN'

    @pytest.mark.parametrize("kind", list(LAYER_TYPES))
    def test_every_weight_matrix_of_the_network_is_tracked(kind):
        """Strict reading of "including the weights of each layers": every weight tensor of a
        layer (a parameter with two or more dimensions) must appear in the tracked weights."""
        model = Holder(LAYER_TYPES[kind]())
        tracked = set(NetworkTracker(model).snapshot(0).names())
        untracked = [
            name
            for name, parameter in model.named_parameters()
            if parameter.dim() >= 2 and name.rsplit(".", 1)[0] not in tracked
        ]
>       assert untracked == [], f"{kind}: weights not tracked: {untracked} (tracked: {sorted(tracked)})"
E       AssertionError: RNN: weights not tracked: ['layer.weight_ih_l0', 'layer.weight_hh_l0'] (tracked: [])
E       assert ['layer.weigh...weight_hh_l0'] == []
E         
E         Left contains 2 more items, first extra item: 'layer.weight_ih_l0'
E         Use -v to get more diff

tests/test_r1_01_tracking.py:406: AssertionError
___________ test_every_weight_matrix_of_the_network_is_tracked[LSTM] ___________

kind = 'LSTM'

    @pytest.mark.parametrize("kind", list(LAYER_TYPES))
    def test_every_weight_matrix_of_the_network_is_tracked(kind):
        """Strict reading of "including the weights of each layers": every weight tensor of a
        layer (a parameter with two or more dimensions) must appear in the tracked weights."""
        model = Holder(LAYER_TYPES[kind]())
        tracked = set(NetworkTracker(model).snapshot(0).names())
        untracked = [
            name
            for name, parameter in model.named_parameters()
            if parameter.dim() >= 2 and name.rsplit(".", 1)[0] not in tracked
        ]
>       assert untracked == [], f"{kind}: weights not tracked: {untracked} (tracked: {sorted(tracked)})"
E       AssertionError: LSTM: weights not tracked: ['layer.weight_ih_l0', 'layer.weight_hh_l0'] (tracked: [])
E       assert ['layer.weigh...weight_hh_l0'] == []
E         
E         Left contains 2 more items, first extra item: 'layer.weight_ih_l0'
E         Use -v to get more diff

tests/test_r1_01_tracking.py:406: AssertionError
___________ test_every_weight_matrix_of_the_network_is_tracked[GRU] ____________

kind = 'GRU'

    @pytest.mark.parametrize("kind", list(LAYER_TYPES))
    def test_every_weight_matrix_of_the_network_is_tracked(kind):
        """Strict reading of "including the weights of each layers": every weight tensor of a
        layer (a parameter with two or more dimensions) must appear in the tracked weights."""
        model = Holder(LAYER_TYPES[kind]())
        tracked = set(NetworkTracker(model).snapshot(0).names())
        untracked = [
            name
            for name, parameter in model.named_parameters()
            if parameter.dim() >= 2 and name.rsplit(".", 1)[0] not in tracked
        ]
>       assert untracked == [], f"{kind}: weights not tracked: {untracked} (tracked: {sorted(tracked)})"
E       AssertionError: GRU: weights not tracked: ['layer.weight_ih_l0', 'layer.weight_hh_l0'] (tracked: [])
E       assert ['layer.weigh...weight_hh_l0'] == []
E         
E         Left contains 2 more items, first extra item: 'layer.weight_ih_l0'
E         Use -v to get more diff

tests/test_r1_01_tracking.py:406: AssertionError
____ test_every_weight_matrix_of_the_network_is_tracked[MultiheadAttention] ____

kind = 'MultiheadAttention'

    @pytest.mark.parametrize("kind", list(LAYER_TYPES))
    def test_every_weight_matrix_of_the_network_is_tracked(kind):
        """Strict reading of "including the weights of each layers": every weight tensor of a
        layer (a parameter with two or more dimensions) must appear in the tracked weights."""
        model = Holder(LAYER_TYPES[kind]())
        tracked = set(NetworkTracker(model).snapshot(0).names())
        untracked = [
            name
            for name, parameter in model.named_parameters()
            if parameter.dim() >= 2 and name.rsplit(".", 1)[0] not in tracked
        ]
>       assert untracked == [], f"{kind}: weights not tracked: {untracked} (tracked: {sorted(tracked)})"
E       AssertionError: MultiheadAttention: weights not tracked: ['layer.in_proj_weight'] (tracked: ['layer.out_proj'])
E       assert ['layer.in_proj_weight'] == []
E         
E         Left contains one more item: 'layer.in_proj_weight'
E         Use -v to get more diff

tests/test_r1_01_tracking.py:406: AssertionError
_ test_every_weight_matrix_of_the_network_is_tracked[TransformerEncoderLayer] __

kind = 'TransformerEncoderLayer'

    @pytest.mark.parametrize("kind", list(LAYER_TYPES))
    def test_every_weight_matrix_of_the_network_is_tracked(kind):
        """Strict reading of "including the weights of each layers": every weight tensor of a
        layer (a parameter with two or more dimensions) must appear in the tracked weights."""
        model = Holder(LAYER_TYPES[kind]())
        tracked = set(NetworkTracker(model).snapshot(0).names())
        untracked = [
            name
            for name, parameter in model.named_parameters()
            if parameter.dim() >= 2 and name.rsplit(".", 1)[0] not in tracked
        ]
>       assert untracked == [], f"{kind}: weights not tracked: {untracked} (tracked: {sorted(tracked)})"
E       AssertionError: TransformerEncoderLayer: weights not tracked: ['layer.self_attn.in_proj_weight'] (tracked: ['layer.linear1', 'layer.linear2', 'layer.self_attn.out_proj'])
E       assert ['layer.self_..._proj_weight'] == []
E         
E         Left contains one more item: 'layer.self_attn.in_proj_weight'
E         Use -v to get more diff

tests/test_r1_01_tracking.py:406: AssertionError
___________ test_the_weights_of_every_layer_are_shown[RecurrentNet] ____________

make_rig = <function make_rig.<locals>.factory at 0xffff588d4540>
network = <class 'test_r1_g1_network_view.RecurrentNet'>

    @pytest.mark.parametrize("network", [RecurrentNet, AttentionNet], ids=lambda c: c.__name__)
    def test_the_weights_of_every_layer_are_shown(make_rig, network):
        """Strict: every weight matrix of the network is shown as connections."""
        rig = rig_for(make_rig, network())
        shown = {link.name for link in drawn_links(rig)}
        missing = [
            name for name, parameter in rig.setup.model.named_parameters()
            if parameter.dim() >= 2 and name.rsplit(".", 1)[0] not in shown
        ]  # fmt: skip
>       assert missing == [], f"weights not shown: {missing}"
E       AssertionError: weights not shown: ['rnn.weight_ih_l0', 'rnn.weight_hh_l0']
E       assert ['rnn.weight_...weight_hh_l0'] == []
E         
E         Left contains 2 more items, first extra item: 'rnn.weight_ih_l0'
E         Use -v to get more diff

tests/test_r1_g1_network_view.py:176: AssertionError
----------------------------- Captured Qt messages -----------------------------
QtWarningMsg: This plugin does not support raise()
___________ test_the_weights_of_every_layer_are_shown[AttentionNet] ____________

make_rig = <function make_rig.<locals>.factory at 0xffff588d4400>
network = <class 'test_r1_g1_network_view.AttentionNet'>

    @pytest.mark.parametrize("network", [RecurrentNet, AttentionNet], ids=lambda c: c.__name__)
    def test_the_weights_of_every_layer_are_shown(make_rig, network):
        """Strict: every weight matrix of the network is shown as connections."""
        rig = rig_for(make_rig, network())
        shown = {link.name for link in drawn_links(rig)}
        missing = [
            name for name, parameter in rig.setup.model.named_parameters()
            if parameter.dim() >= 2 and name.rsplit(".", 1)[0] not in shown
        ]  # fmt: skip
>       assert missing == [], f"weights not shown: {missing}"
E       AssertionError: weights not shown: ['attn.in_proj_weight']
E       assert ['attn.in_proj_weight'] == []
E         
E         Left contains one more item: 'attn.in_proj_weight'
E         Use -v to get more diff

tests/test_r1_g1_network_view.py:176: AssertionError
----------------------------- Captured Qt messages -----------------------------
QtWarningMsg: This plugin does not support raise()
_____ test_every_panel_stays_fully_visible_at_small_window_sizes[960x680] ______

rig = <htspec.rig.GuiRig object at 0xffff588d5e50>, size = (960, 680)

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

rig = <htspec.rig.GuiRig object at 0xffff588d67b0>, size = (1202, 680)

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

rig = <htspec.rig.GuiRig object at 0xffff588d7110>, size = (1366, 720)

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
___________ test_a_compiled_network_is_still_shown_by_its_own_class ____________

make_rig = <function make_rig.<locals>.factory at 0xffff58884360>
paint_capture = <htspec.painting.PaintCapture object at 0xffff59ac68d0>

    def test_a_compiled_network_is_still_shown_by_its_own_class(make_rig, paint_capture):
        """Strict: ``torch.compile`` wraps the user's network in ``OptimizedModule``; the GUI must
        still show the training network's class name. (Nothing is compiled here: the name is
        read before any forward pass.)"""
        compiled = torch.compile(TinyNet(), backend="eager")
        texts, rig = header_texts(make_rig, paint_capture, compiled)
>       assert "TinyNet" in texts, f"the header shows {[t for t in texts if t.isidentifier()]}"
E       AssertionError: the header shows ['OptimizedModule', 'TRAINING']
E       assert 'TinyNet' in ['HOLLYWOOD TRAINING', 'NEURAL TRAINING VISUALIZER · v0.1.0', 'TRAINING NETWORK', 'OptimizedModule', '51 parameters', 'TRAINING', ...]

tests/test_r1_g6_network_name.py:91: AssertionError
----------------------------- Captured Qt messages -----------------------------
QtWarningMsg: This plugin does not support raise()
__ test_real_cursor_stays_invisible_when_paused_and_after_window_mode_changes __

live = <htspec.live.LiveApp object at 0xffff5a210b90>
xclient = <htspec.x11.XClient object at 0xffff2936b380>

    @pytest.mark.x11
    @pytest.mark.subprocess
    def test_real_cursor_stays_invisible_when_paused_and_after_window_mode_changes(live, xclient):
>       live.enter("pause", expect="paus")

tests/test_r2_1_no_pointer.py:274: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

self = <htspec.live.LiveApp object at 0xffff5a210b90>, line = 'pause'
expect = 'paus', timeout = 20.0

    def enter(self, line: str, *, expect: str | None = None, timeout: float = 20.0):
        """Type ``line`` + Return on the real keyboard and wait for the command's answer.
    
        Waits until the window process has sent this line to the training process and the
        answer has come back (and, with ``expect``, until the answer - or the window-only
        outcome of a view command - contains ``expect``); returns a probe snapshot taken
        after that.
        """
        started = self.now()
        self.type(line + "\n")
>       sent = self.wait(
            lambda r: r.get("msg") == "command" and r.get("value") == line,
            kind="send", after=started, timeout=timeout,
        )  # fmt: skip
E       TypeError: LiveApp.wait() got an unexpected keyword argument 'kind'

tests/htspec/live.py:136: TypeError
________ test_real_cursor_is_invisible_after_typing_and_terminal_output ________

live = <htspec.live.LiveApp object at 0xffff29116470>
xclient = <htspec.x11.XClient object at 0xffff2936b380>

    @pytest.mark.x11
    @pytest.mark.subprocess
    def test_real_cursor_is_invisible_after_typing_and_terminal_output(live, xclient):
        """Terminal output and typed text (widgets repaint, the log grows) do not bring the
        pointer back."""
>       live.enter("help", expect="type 'help <command>' for details")

tests/test_r2_1_no_pointer.py:292: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

self = <htspec.live.LiveApp object at 0xffff29116470>, line = 'help'
expect = "type 'help <command>' for details", timeout = 20.0

    def enter(self, line: str, *, expect: str | None = None, timeout: float = 20.0):
        """Type ``line`` + Return on the real keyboard and wait for the command's answer.
    
        Waits until the window process has sent this line to the training process and the
        answer has come back (and, with ``expect``, until the answer - or the window-only
        outcome of a view command - contains ``expect``); returns a probe snapshot taken
        after that.
        """
        started = self.now()
        self.type(line + "\n")
>       sent = self.wait(
            lambda r: r.get("msg") == "command" and r.get("value") == line,
            kind="send", after=started, timeout=timeout,
        )  # fmt: skip
E       TypeError: LiveApp.wait() got an unexpected keyword argument 'kind'

tests/htspec/live.py:136: TypeError
_________________ test_nobody_holds_a_pointer_or_keyboard_grab _________________

outside = (<htspec.live.LiveApp object at 0xffff291a2050>, <htspec.subproc.Spawned object at 0xffff291a1b50>, {'event': 'ready', 'pid': 41054, 'win_id': 8388615, 'geometry': [20, 20, 380, 220]})
xclient = <htspec.x11.XClient object at 0xffff2936b380>

    @pytest.mark.x11
    @pytest.mark.subprocess
    def test_nobody_holds_a_pointer_or_keyboard_grab(outside, xclient):
        app, _neighbour, _ready = outside
        app.focus()
        assert xclient.try_grab_pointer() == 0
        assert xclient.try_grab_keyboard() == 0
>       app.enter("pause", expect="paus")

tests/test_r2_6_outside_window.py:343: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

self = <htspec.live.LiveApp object at 0xffff291a2050>, line = 'pause'
expect = 'paus', timeout = 20.0

    def enter(self, line: str, *, expect: str | None = None, timeout: float = 20.0):
        """Type ``line`` + Return on the real keyboard and wait for the command's answer.
    
        Waits until the window process has sent this line to the training process and the
        answer has come back (and, with ``expect``, until the answer - or the window-only
        outcome of a view command - contains ``expect``); returns a probe snapshot taken
        after that.
        """
        started = self.now()
        self.type(line + "\n")
>       sent = self.wait(
            lambda r: r.get("msg") == "command" and r.get("value") == line,
            kind="send", after=started, timeout=timeout,
        )  # fmt: skip
E       TypeError: LiveApp.wait() got an unexpected keyword argument 'kind'

tests/htspec/live.py:136: TypeError
______________ test_outside_stays_normal_after_the_session_ended _______________

outside = (<htspec.live.LiveApp object at 0xffff291a2050>, <htspec.subproc.Spawned object at 0xffff291a1b50>, {'event': 'ready', 'pid': 41054, 'win_id': 8388615, 'geometry': [20, 20, 380, 220]})
xclient = <htspec.x11.XClient object at 0xffff2936b380>

    @pytest.mark.x11
    @pytest.mark.subprocess
    def test_outside_stays_normal_after_the_session_ended(outside, xclient):
        """Last test of the module: it ends the module's session (``quit``)."""
        app, neighbour, ready = outside
        x, y, w, h = app.window_rect()
        result = app.quit()
        assert result["state"] in ("stopped", "finished")
        time.sleep(0.3)
>       assert not xclient.cursor_at(x + w // 2, y + h // 2).blank  # the window is gone
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E       assert not True
E        +  where True = Cursor(width=16, height=16, opaque=0, x=1279, y=719, serial=9).blank
E        +    where Cursor(width=16, height=16, opaque=0, x=1279, y=719, serial=9) = cursor_at((439 + (1680 // 2)), (209 + (1020 // 2)))
E        +      where cursor_at = <htspec.x11.XClient object at 0xffff2936b380>.cursor_at

tests/test_r2_6_outside_window.py:361: AssertionError
______________ test_real_keys_reach_the_terminal_without_a_click _______________

live = <htspec.live.LiveApp object at 0xffff59cccb50>

    @pytest.mark.x11
    @pytest.mark.subprocess
    def test_real_keys_reach_the_terminal_without_a_click(live):
        """Switching to the window (what the window manager does) is all it takes."""
        state = live.focus()
        assert state["focus"]["class"] == "CommandLine"
>       live.enter("ping", expect="pong 1")

tests/test_r2_7_keyboard_reroute.py:273: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

self = <htspec.live.LiveApp object at 0xffff59cccb50>, line = 'ping'
expect = 'pong 1', timeout = 20.0

    def enter(self, line: str, *, expect: str | None = None, timeout: float = 20.0):
        """Type ``line`` + Return on the real keyboard and wait for the command's answer.
    
        Waits until the window process has sent this line to the training process and the
        answer has come back (and, with ``expect``, until the answer - or the window-only
        outcome of a view command - contains ``expect``); returns a probe snapshot taken
        after that.
        """
        started = self.now()
        self.type(line + "\n")
>       sent = self.wait(
            lambda r: r.get("msg") == "command" and r.get("value") == line,
            kind="send", after=started, timeout=timeout,
        )  # fmt: skip
E       TypeError: LiveApp.wait() got an unexpected keyword argument 'kind'

tests/htspec/live.py:136: TypeError
_________________ test_real_command_keys_control_the_training __________________

live = <htspec.live.LiveApp object at 0xffff59cccb50>

    @pytest.mark.x11
    @pytest.mark.subprocess
    def test_real_command_keys_control_the_training(live):
        live.focus()
>       live.enter("pause", expect="paus")

tests/test_r2_7_keyboard_reroute.py:302: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

self = <htspec.live.LiveApp object at 0xffff59cccb50>, line = 'pause'
expect = 'paus', timeout = 20.0

    def enter(self, line: str, *, expect: str | None = None, timeout: float = 20.0):
        """Type ``line`` + Return on the real keyboard and wait for the command's answer.
    
        Waits until the window process has sent this line to the training process and the
        answer has come back (and, with ``expect``, until the answer - or the window-only
        outcome of a view command - contains ``expect``); returns a probe snapshot taken
        after that.
        """
        started = self.now()
        self.type(line + "\n")
>       sent = self.wait(
            lambda r: r.get("msg") == "command" and r.get("value") == line,
            kind="send", after=started, timeout=timeout,
        )  # fmt: skip
E       TypeError: LiveApp.wait() got an unexpected keyword argument 'kind'

tests/htspec/live.py:136: TypeError
__________________ test_real_special_keys_act_in_the_terminal __________________

live = <htspec.live.LiveApp object at 0xffff59cccb50>
xclient = <htspec.x11.XClient object at 0xffff2936b380>

    @pytest.mark.x11
    @pytest.mark.subprocess
    def test_real_special_keys_act_in_the_terminal(live, xclient):
        live.focus()
>       live.enter("history 1")

tests/test_r2_7_keyboard_reroute.py:315: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

self = <htspec.live.LiveApp object at 0xffff59cccb50>, line = 'history 1'
expect = None, timeout = 20.0

    def enter(self, line: str, *, expect: str | None = None, timeout: float = 20.0):
        """Type ``line`` + Return on the real keyboard and wait for the command's answer.
    
        Waits until the window process has sent this line to the training process and the
        answer has come back (and, with ``expect``, until the answer - or the window-only
        outcome of a view command - contains ``expect``); returns a probe snapshot taken
        after that.
        """
        started = self.now()
        self.type(line + "\n")
>       sent = self.wait(
            lambda r: r.get("msg") == "command" and r.get("value") == line,
            kind="send", after=started, timeout=timeout,
        )  # fmt: skip
E       TypeError: LiveApp.wait() got an unexpected keyword argument 'kind'

tests/htspec/live.py:136: TypeError
_____ test_real_keys_belong_to_another_application_while_it_has_the_focus ______

live = <htspec.live.LiveApp object at 0xffff59cccb50>
xserver = <htspec.x11.XServer object at 0xffff2936b620>
xclient = <htspec.x11.XClient object at 0xffff2936b380>

    @pytest.mark.x11
    @pytest.mark.subprocess
    def test_real_keys_belong_to_another_application_while_it_has_the_focus(live, xserver, xclient):
        neighbour = spawn_python(
            [str(paths.SCENARIO_DIR / "other_app.py"), "20", "20", "380", "220"],
            display=xserver.display,
        )
        try:
            ready = neighbour.wait_message(lambda m: m.get("event") == "ready", timeout=60)
            before = live.state()["input"]["text"]
            xclient.focus(ready["win_id"])
            time.sleep(0.3)
            start = len(neighbour.messages)
            x, y = live.center("network")
            xclient.move(x, y)  # the pointer over the Hollywood window does not matter
            live.type("theirs")
            time.sleep(0.4)
            typed = "".join(m["text"] for m in neighbour.messages[start:] if m["event"] == "key")
            assert typed == "theirs"
            assert live.state()["input"]["text"] == before
            live.focus()  # back on the Hollywood window: its terminal gets the keys again
>           live.enter("ping", expect="pong")

tests/test_r2_7_keyboard_reroute.py:360: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

self = <htspec.live.LiveApp object at 0xffff59cccb50>, line = 'ping'
expect = 'pong', timeout = 20.0

    def enter(self, line: str, *, expect: str | None = None, timeout: float = 20.0):
        """Type ``line`` + Return on the real keyboard and wait for the command's answer.
    
        Waits until the window process has sent this line to the training process and the
        answer has come back (and, with ``expect``, until the answer - or the window-only
        outcome of a view command - contains ``expect``); returns a probe snapshot taken
        after that.
        """
        started = self.now()
        self.type(line + "\n")
>       sent = self.wait(
            lambda r: r.get("msg") == "command" and r.get("value") == line,
            kind="send", after=started, timeout=timeout,
        )  # fmt: skip
E       TypeError: LiveApp.wait() got an unexpected keyword argument 'kind'

tests/htspec/live.py:136: TypeError
=============================== warnings summary ===============================
tests/test_r1_03_native_workflow.py::test_the_network_still_scripts_like_an_untracked_one
  /home/agent/workspace/hollywood-training/.venv/lib/python3.13/site-packages/torch/jit/_script.py:1491: FutureWarning: `torch.jit.script` is deprecated. Please switch to `torch.compile` or `torch.export`.
    warnings.warn(

tests/test_r2_2_mouse_disabled.py::test_dropping_text_onto_the_window_inserts_nothing
tests/test_r2_2_mouse_disabled.py::test_dropping_text_onto_the_window_inserts_nothing
tests/test_r2_2_mouse_disabled.py::test_dropping_text_onto_the_window_inserts_nothing
tests/test_r2_2_mouse_disabled.py::test_dropping_text_onto_the_window_inserts_nothing
  /home/agent/workspace/hollywood-training/tests/test_r2_2_mouse_disabled.py:195: DeprecationWarning: Function: 'QDragEnterEvent.QDragEnterEvent(const QPoint &pos, QFlags<Qt.DropAction> actions, const QMimeData *data, QFlags<Qt.MouseButton> buttons, QFlags<Qt.KeyboardModifier> modifiers)' is marked as deprecated, please check the documentation for more information.
    QApplication.sendEvent(widget, QDragEnterEvent(pos, copy, mime, LEFT, NO_MOD))

tests/test_r2_2_mouse_disabled.py::test_dropping_text_onto_the_window_inserts_nothing
tests/test_r2_2_mouse_disabled.py::test_dropping_text_onto_the_window_inserts_nothing
tests/test_r2_2_mouse_disabled.py::test_dropping_text_onto_the_window_inserts_nothing
tests/test_r2_2_mouse_disabled.py::test_dropping_text_onto_the_window_inserts_nothing
  /home/agent/workspace/hollywood-training/tests/test_r2_2_mouse_disabled.py:196: DeprecationWarning: Function: 'QDragMoveEvent.QDragMoveEvent(const QPoint &pos, QFlags<Qt.DropAction> actions, const QMimeData *data, QFlags<Qt.MouseButton> buttons, QFlags<Qt.KeyboardModifier> modifiers, QEvent.Type type)' is marked as deprecated, please check the documentation for more information.
    QApplication.sendEvent(widget, QDragMoveEvent(pos, copy, mime, LEFT, NO_MOD))

tests/test_r2_6_outside_window.py::test_another_window_of_the_application_accepts_drops_and_context_menus
  /home/agent/workspace/hollywood-training/tests/test_r2_6_outside_window.py:201: DeprecationWarning: Function: 'QDragEnterEvent.QDragEnterEvent(const QPoint &pos, QFlags<Qt.DropAction> actions, const QMimeData *data, QFlags<Qt.MouseButton> buttons, QFlags<Qt.KeyboardModifier> modifiers)' is marked as deprecated, please check the documentation for more information.
    QApplication.sendEvent(edit, QDragEnterEvent(position, copy, mime, LEFT, NO_MOD))

tests/test_r2_6_outside_window.py::test_another_window_of_the_application_accepts_drops_and_context_menus
  /home/agent/workspace/hollywood-training/tests/test_r2_6_outside_window.py:202: DeprecationWarning: Function: 'QDragMoveEvent.QDragMoveEvent(const QPoint &pos, QFlags<Qt.DropAction> actions, const QMimeData *data, QFlags<Qt.MouseButton> buttons, QFlags<Qt.KeyboardModifier> modifiers, QEvent.Type type)' is marked as deprecated, please check the documentation for more information.
    QApplication.sendEvent(edit, QDragMoveEvent(position, copy, mime, LEFT, NO_MOD))

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
============= SPEC requirement coverage (specs/R1.md, specs/R2.md) =============
req      pass  fail  skip     requirement
R1.1       40     6     0  !! track the network architecture, including the weights of each layer
R1.2       71     0     0     manage the training: native PyTorch definition, wrapped, passed to the manager
R1.3       14     0     0     requirements 1 and 2 do not significantly change the native PyTorch workflow
R1.4       20     0     0     native GUI window (not a web page) with smooth animations of the real training
R1.5       26     0     0     training synchronised with the GUI: updated every epoch, before the next epoch
R1.6       10     0     0     training runs as fast as the synchronisation allows
R1.7       76     0     0     well-defined commands control the training at runtime
R1.8       63     0     0     no 'dark magic' Python methods
R1.9      152     0     0     completely decomposed into modules with a clear architecture
R1.10       4     0     0     all package files in the package folder hollywood_training
R1.11      10     0     0     a demo with an illustrative network and dataset, in demos/
R1.12       5     0     0     a README.md for quick start
R1.13      72     0     0     a wiki folder: a page for every component and the overall architecture
R1.G1      13     5     0  !! GUI: architecture, neuron connections, real weights changing every epoch
R1.G2      26     3     0  !! GUI: training status (epochs, batch size, lr, loss, optimizer, ..., timer)
R1.G3      11     3     0  !! GUI: training and validation loss drawn dynamically every epoch
R1.G4      16     3     0  !! GUI: memory, CPU and (when applicable) GPU usage
R1.G5      15     3     0  !! GUI: an integrated terminal controls the training with the commands
R1.G6       9     1     0  !! GUI: the name of the training network (its class name)
R1.G7      22     0     0     GUI: blue based
R2.1       35     2     0  !! no mouse pointer displayed anywhere inside the window
R2.2      105     7     0  !! all mouse-based interactions disabled within the application
R2.3       53     0     0     all implemented functionality accessible and operational without the mouse
R2.4        6     0     0     additional commands may replace the mouse-based interactions
R2.5      697    12     0  !! all other R1 features not affected (every R1 test counts here as well)
R2.6       12     2     0  !! pointer and mouse interactions outside the window not affected
R2.7       30     4     0  !! all keyboard input rerouted to the internal terminal while on the window
coverage report: /home/agent/workspace/hollywood-training/.temp/tests/reports/coverage-20261106-144042.json
================================ tests coverage ================================
_______________ coverage: platform linux, python 3.13.5-final-0 ________________

Name                                             Stmts   Miss  Cover   Missing
------------------------------------------------------------------------------
hollywood_training/__init__.py                      15      1    93%   70
hollywood_training/_version.py                       1      0   100%
hollywood_training/commands/__init__.py             15      7    53%   43-48, 52
hollywood_training/commands/builtin.py             276     18    93%   80, 87, 96-97, 99, 118, 146, 154, 169, 248, 251, 257-258, 264, 272, 315, 320, 359
hollywood_training/commands/context.py              39      1    97%   87
hollywood_training/commands/registry.py            134     10    93%   42, 109, 130, 137-142, 149
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
hollywood_training/gui/view_actions.py             148     25    83%   53, 57, 63, 66-67, 79, 89-90, 98-101, 104, 118, 134, 139-140, 142-143, 161, 208-209, 223, 237-238
hollywood_training/gui/widgets/__init__.py           8      0   100%
hollywood_training/gui/widgets/header.py           126      0   100%
hollywood_training/gui/widgets/hud.py              106      4    96%   79, 83, 87-88
hollywood_training/gui/widgets/loss_chart.py       316      5    98%   43, 167, 223, 385, 391
hollywood_training/gui/widgets/network_view.py     490     12    98%   89, 107-108, 267, 332, 342, 437-441, 445, 614, 651
hollywood_training/gui/widgets/status_panel.py     117      2    98%   46, 169
hollywood_training/gui/widgets/terminal.py         127      3    98%   63, 140, 185
hollywood_training/gui/widgets/usage_panel.py      128      5    96%   51-52, 66, 69-70
hollywood_training/gui/window.py                   210      5    98%   186-187, 201-202, 338
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
hollywood_training/tracking/__init__.py             15      4    73%   83-85, 89
hollywood_training/tracking/architecture.py         91      5    95%   49-52, 151
hollywood_training/tracking/decorator.py            46      0   100%
hollywood_training/tracking/inspector.py            75      2    97%   160, 164
hollywood_training/tracking/layers.py               85      4    95%   109, 126, 132, 149
hollywood_training/tracking/roles.py                10      0   100%
hollywood_training/tracking/snapshot.py             81      5    94%   56, 59, 95, 117, 133
hollywood_training/tracking/tracker.py              60      7    88%   38, 52, 56, 68, 72-73, 89
hollywood_training/tracking/weights.py              44      2    95%   68, 70
hollywood_training/training/__init__.py              8      0   100%
hollywood_training/training/describe.py             77      6    92%   52-53, 57, 104, 114-115
hollywood_training/training/history.py              22      0   100%
hollywood_training/training/manager.py             278     23    92%   120, 124, 128, 136, 149-150, 159-160, 178, 193-194, 225, 289-290, 323-324, 369-370, 398-400, 443-444
hollywood_training/training/process.py             130      2    98%   271-272
hollywood_training/training/steps.py                71     11    85%   69-71, 78-84, 91
hollywood_training/training/timing.py               20      1    95%   17
------------------------------------------------------------------------------
TOTAL                                             4909    418    91%
=========================== short test summary info ============================
ERROR tests/test_r2_4_view_commands.py - Failed: tests/test_r2_4_view_command...
ERROR tests/test_r2_2_mouse_disabled.py::test_real_clicks_on_every_panel_change_nothing
ERROR tests/test_r2_2_mouse_disabled.py::test_real_drags_on_every_splitter_handle_change_nothing
ERROR tests/test_r2_2_mouse_disabled.py::test_real_text_selection_with_the_mouse_selects_nothing
ERROR tests/test_r2_2_mouse_disabled.py::test_real_wheel_turns_scroll_nothing
ERROR tests/test_r2_2_mouse_disabled.py::test_real_middle_click_does_not_paste
ERROR tests/test_r2_2_mouse_disabled.py::test_real_hovering_shows_no_tooltip_and_no_details
ERROR tests/test_r2_2_mouse_disabled.py::test_training_and_keyboard_still_work_after_all_that_mouse_input
FAILED tests/test_r1_01_tracking.py::test_every_weight_matrix_of_the_network_is_tracked[Bilinear]
FAILED tests/test_r1_01_tracking.py::test_every_weight_matrix_of_the_network_is_tracked[RNN]
FAILED tests/test_r1_01_tracking.py::test_every_weight_matrix_of_the_network_is_tracked[LSTM]
FAILED tests/test_r1_01_tracking.py::test_every_weight_matrix_of_the_network_is_tracked[GRU]
FAILED tests/test_r1_01_tracking.py::test_every_weight_matrix_of_the_network_is_tracked[MultiheadAttention]
FAILED tests/test_r1_01_tracking.py::test_every_weight_matrix_of_the_network_is_tracked[TransformerEncoderLayer]
FAILED tests/test_r1_g1_network_view.py::test_the_weights_of_every_layer_are_shown[RecurrentNet]
FAILED tests/test_r1_g1_network_view.py::test_the_weights_of_every_layer_are_shown[AttentionNet]
FAILED tests/test_r1_g1_network_view.py::test_every_panel_stays_fully_visible_at_small_window_sizes[960x680]
FAILED tests/test_r1_g1_network_view.py::test_every_panel_stays_fully_visible_at_small_window_sizes[1202x680]
FAILED tests/test_r1_g1_network_view.py::test_every_panel_stays_fully_visible_at_small_window_sizes[1366x720]
FAILED tests/test_r1_g6_network_name.py::test_a_compiled_network_is_still_shown_by_its_own_class
FAILED tests/test_r2_1_no_pointer.py::test_real_cursor_stays_invisible_when_paused_and_after_window_mode_changes
FAILED tests/test_r2_1_no_pointer.py::test_real_cursor_is_invisible_after_typing_and_terminal_output
FAILED tests/test_r2_6_outside_window.py::test_nobody_holds_a_pointer_or_keyboard_grab
FAILED tests/test_r2_6_outside_window.py::test_outside_stays_normal_after_the_session_ended
FAILED tests/test_r2_7_keyboard_reroute.py::test_real_keys_reach_the_terminal_without_a_click
FAILED tests/test_r2_7_keyboard_reroute.py::test_real_command_keys_control_the_training
FAILED tests/test_r2_7_keyboard_reroute.py::test_real_special_keys_act_in_the_terminal
FAILED tests/test_r2_7_keyboard_reroute.py::test_real_keys_belong_to_another_application_while_it_has_the_focus
====== 20 failed, 932 passed, 11 warnings, 8 errors in 129.65s (0:02:09) =======
