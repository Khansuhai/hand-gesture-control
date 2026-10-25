"""Config loading, One Euro filter, hotkeys and action names."""

import json
import math
import random

import pytest
from pynput import keyboard

from src.actions import validate_action
from src.config import ConfigError, DEFAULTS, load_config
from src.hotkeys import VK_P, VK_Q, Hotkeys
from src.smoothing import OneEuroFilter

# ---- config -------------------------------------------------------------------


def test_defaults_load():
    cfg, raw, warnings = load_config(None)
    assert raw == DEFAULTS
    assert cfg.camera.width == 640
    assert cfg.gesture_actions["pinch"] == "left_click"


def test_shipped_config_json_is_valid():
    cfg, _, warnings = load_config("config.json")
    assert warnings == []
    assert cfg.tracking.hand == "right"


def test_partial_config_merges_with_defaults(tmp_path):
    p = tmp_path / "c.json"
    p.write_text(json.dumps({"scroll": {"speed": 3}, "_note": "comments are ignored"}))
    cfg, _, warnings = load_config(p)
    assert cfg.scroll.speed == 3
    assert cfg.scroll.mode == "joystick"
    assert warnings == []


def test_unknown_key_warns(tmp_path):
    p = tmp_path / "c.json"
    p.write_text(json.dumps({"camera": {"idx": 1}}))
    _, _, warnings = load_config(p)
    assert any("camera.idx" in w for w in warnings)


@pytest.mark.parametrize(
    "bad",
    [
        {"camera": {"width": "640"}},
        {"tracking": {"hand": "middle"}},
        {"gestures": {"pinch_on_ratio": 0.5, "pinch_off_ratio": 0.3}},
        {"active_zone": {"left": 0.45, "right": 0.45}},
        {"gesture_actions": {"pinch": "explode"}},
        {"gesture_actions": {"thumbs_up": "key:notakey"}},
        {"timing": {"hold_times_s": {"fist": -1}}},
    ],
)
def test_invalid_config_is_rejected(tmp_path, bad):
    p = tmp_path / "c.json"
    p.write_text(json.dumps(bad))
    with pytest.raises(ConfigError):
        load_config(p)


def test_broken_json_gives_clear_error(tmp_path):
    p = tmp_path / "c.json"
    p.write_text('{"camera": {"index": 0,}}')
    with pytest.raises(ConfigError, match="not valid JSON"):
        load_config(p)


@pytest.mark.parametrize(
    "spec, ok",
    [
        ("left_click", True), ("double_click", True), ("key:enter", True), ("key:volumeup", True),
        ("key:playpause", True), ("hotkey:alt+tab", True), ("hotkey:ctrl+shift+esc", True),
        ("type:hello", True), ("pause", True), ("none", True),
        ("key:", False), ("hotkey:ctrl+nope", False), ("launch:rocket", False), ("jump", False),
    ],
)
def test_action_names(spec, ok):
    assert (validate_action(spec) is None) == ok


# ---- One Euro filter ----------------------------------------------------------


def test_one_euro_constant_signal_unchanged():
    f = OneEuroFilter(1.0, 0.01)
    for i in range(100):
        out = f(5.0, i / 30)
    assert out == pytest.approx(5.0)


def test_one_euro_reduces_jitter():
    random.seed(1)
    f = OneEuroFilter(1.0, 0.005)
    raw, filt = [], []
    for i in range(300):
        x = 500 + random.uniform(-5, 5)
        raw.append(x)
        filt.append(f(x, i / 30))

    def spread(v):
        v = v[50:]
        m = sum(v) / len(v)
        return math.sqrt(sum((a - m) ** 2 for a in v) / len(v))

    assert spread(filt) < spread(raw) / 3


def test_one_euro_follows_fast_moves():
    f = OneEuroFilter(1.0, 0.005)
    for i in range(30):
        f(0.0, i / 30)
    out = 0.0
    for i in range(30, 45):  # half a second after a big jump
        out = f(1000.0, i / 30)
    assert out > 950


# ---- hotkeys ------------------------------------------------------------------


def _hk():
    calls = []
    hk = Hotkeys(on_quit=lambda: calls.append("quit"), on_toggle_pause=lambda: calls.append("pause"))
    return hk, calls


def test_ctrl_alt_q_quits():
    hk, calls = _hk()
    hk.press(keyboard.Key.ctrl_l)
    hk.press(keyboard.Key.alt_l)
    hk.press(keyboard.KeyCode.from_vk(VK_Q))
    assert calls == ["quit"]


def test_ctrl_alt_p_toggles_once_even_with_key_repeat():
    hk, calls = _hk()
    hk.press(keyboard.Key.ctrl_r)
    hk.press(keyboard.Key.alt_l)
    for _ in range(5):  # holding P auto-repeats
        hk.press(keyboard.KeyCode.from_vk(VK_P))
    hk.release(keyboard.KeyCode.from_vk(VK_P))
    hk.press(keyboard.KeyCode.from_vk(VK_P))
    assert calls == ["pause", "pause"]


def test_plain_q_or_ctrl_q_does_nothing():
    hk, calls = _hk()
    hk.press(keyboard.KeyCode.from_vk(VK_Q))
    hk.release(keyboard.KeyCode.from_vk(VK_Q))
    hk.press(keyboard.Key.ctrl_l)
    hk.press(keyboard.KeyCode.from_vk(VK_Q))
    hk.release(keyboard.KeyCode.from_vk(VK_Q))
    hk.release(keyboard.Key.ctrl_l)
    hk.press(keyboard.Key.alt_l)
    hk.press(keyboard.KeyCode.from_vk(VK_Q))
    assert calls == []
