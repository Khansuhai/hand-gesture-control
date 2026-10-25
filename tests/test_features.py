"""Swipes, volume, two-hand zoom, per-app profiles, shortcuts and calibration."""

import json

import numpy as np
import pytest

from src.calibrate import CalibrationError, compute_thresholds, measure, save_thresholds
from src.config import ConfigError, load_config
from src.foreground import foreground_exe
from src.gestures import GestureRecognizer, SwipeDetector, TwoHandPinch
from tests.synthetic import make_hand, pose_hand
from tests.test_controller import DT, Sim, result

# ---- swipes -------------------------------------------------------------------


def _swipe(det, start, end, seconds=0.2, pose="open_palm", t0=0.0):
    frames = max(2, round(seconds / DT))
    found = None
    for i in range(frames + 1):
        a = i / frames
        x = start[0] + (end[0] - start[0]) * a
        y = start[1] + (end[1] - start[1]) * a
        out = det.update(result(pose, palm=(x, y)), t0 + i * DT)
        found = found or out
    return found


@pytest.fixture
def swipe_cfg():
    cfg, _, _ = load_config(None)
    return cfg.swipe


@pytest.mark.parametrize(
    "start, end, expected",
    [
        ((0.3, 0.5), (0.7, 0.5), "swipe_right"),
        ((0.7, 0.5), (0.3, 0.5), "swipe_left"),
        ((0.5, 0.7), (0.5, 0.3), "swipe_up"),
        ((0.5, 0.3), (0.5, 0.7), "swipe_down"),
    ],
)
def test_fast_palm_moves_are_swipes(swipe_cfg, start, end, expected):
    assert _swipe(SwipeDetector(swipe_cfg), start, end) == expected


def test_slow_palm_move_is_not_a_swipe(swipe_cfg):
    assert _swipe(SwipeDetector(swipe_cfg), (0.3, 0.5), (0.7, 0.5), seconds=1.5) is None


def test_still_palm_is_not_a_swipe(swipe_cfg):
    assert _swipe(SwipeDetector(swipe_cfg), (0.5, 0.5), (0.51, 0.5)) is None


def test_moving_with_other_pose_is_not_a_swipe(swipe_cfg):
    assert _swipe(SwipeDetector(swipe_cfg), (0.3, 0.5), (0.7, 0.5), pose="point") is None


def test_diagonal_move_is_not_a_swipe(swipe_cfg):
    assert _swipe(SwipeDetector(swipe_cfg), (0.3, 0.3), (0.6, 0.6)) is None


def test_one_swipe_fires_once(swipe_cfg):
    det = SwipeDetector(swipe_cfg)
    assert _swipe(det, (0.2, 0.5), (0.8, 0.5)) == "swipe_right"
    # Hand comes back quickly (return stroke) inside the cooldown: ignored.
    assert _swipe(det, (0.8, 0.5), (0.2, 0.5), t0=0.25) is None


def test_swipe_fires_mapped_action_and_is_ignored_when_paused():
    s = Sim()
    assert s.c.trigger("swipe_right", s.t)
    assert s.ex.actions() == ["hotkey:alt+tab"]
    s.c.set_paused(True, s.t)
    assert not s.c.trigger("swipe_left", s.t + 1)
    assert s.ex.actions() == ["hotkey:alt+tab"]


# ---- volume / media / shortcuts ---------------------------------------------


def test_call_me_moved_up_raises_volume():
    s = Sim()
    s.run(result("call_me", palm=(0.5, 0.6)), frames=1)
    s.run(result("call_me", palm=(0.5, 0.45)), seconds=0.5)
    acts = s.ex.actions()
    assert acts and set(acts) == {"key:volumeup"}


def test_call_me_moved_down_lowers_volume():
    s = Sim()
    s.run(result("call_me", palm=(0.5, 0.45)), frames=1)
    s.run(result("call_me", palm=(0.5, 0.62)), seconds=0.5)
    assert set(s.ex.actions()) == {"key:volumedown"}


@pytest.mark.parametrize(
    "pose, action",
    [("rock", "key:playpause"), ("thumbs_down", "hotkey:win+ctrl+o"), ("pinky", "hotkey:win+shift+s")],
)
def test_hold_shortcuts(pose, action):
    s = Sim()
    s.run(result(pose), seconds=0.3)
    assert s.ex.actions() == []  # must be held
    s.run(result(pose), seconds=1.5)
    assert s.ex.actions() == [action]  # and fires once


# ---- two-hand zoom ------------------------------------------------------------


def _pinch_hand(x, handedness):
    return pose_hand("pinch", center=(x, 420), scale=70, handedness=handedness)


def test_two_hand_pinch_reports_distance():
    cfg, _, _ = load_config(None)
    th = TwoHandPinch(cfg.gestures)
    hands = [_pinch_hand(150, "Left"), _pinch_hand(450, "Right")]
    out = [th.update(hands) for _ in range(cfg.gestures.stable_frames)]
    assert out[-1] == pytest.approx(300 / 640, abs=0.01)
    # Only one hand pinching: nothing.
    th2 = TwoHandPinch(cfg.gestures)
    mixed = [_pinch_hand(150, "Left"), pose_hand("point", center=(450, 420), handedness="Right")]
    assert all(th2.update(mixed) is None for _ in range(5))


def test_two_hand_pull_apart_zooms_in_and_push_zooms_out():
    s = Sim()
    for i in range(20):
        s.c.update(result("pinch", pinch_ratio=0.1), s.t, two_hand=0.3 + i * 0.02)
        s.t += DT
    assert sum(s.ex.zoom_events()) > 0
    before = sum(s.ex.zoom_events())
    for i in range(20):
        s.c.update(result("pinch", pinch_ratio=0.1), s.t, two_hand=0.7 - i * 0.025)
        s.t += DT
    assert sum(s.ex.zoom_events()) < before


def test_two_hand_zoom_never_clicks_or_drags():
    s = Sim()
    s.run(result("pinch", pinch_ratio=0.1), seconds=0.6)  # one hand starts a drag...
    assert s.ex.down
    for i in range(10):  # ...then the other hand joins
        s.c.update(result("pinch", pinch_ratio=0.1), s.t, two_hand=0.3 + i * 0.03)
        s.t += DT
    assert not s.ex.down
    s.run(result("pinch", pinch_ratio=0.1), seconds=0.5)  # second hand leaves, first still pinched
    s.run(result("point"), seconds=0.5)  # then opens
    assert "left_click" not in s.ex.actions()
    assert s.ex.events.count("mouse_down") == 1


# ---- per-app profiles ---------------------------------------------------------


def test_profiles_switch_with_foreground_app():
    s = Sim()
    s.c.set_app("chrome.exe", s.t)
    assert s.c.profile == "browser"
    s.c.trigger("swipe_left", s.t)
    assert s.ex.actions() == ["hotkey:alt+left"]
    s.c.set_app("notepad.exe", s.t + 1)
    assert s.c.profile == "default"
    s.c.trigger("swipe_left", s.t + 2)
    assert s.ex.actions()[-1] == "hotkey:alt+shift+tab"


def test_google_earth_zooms_with_plain_wheel():
    s = Sim()
    s.c.set_app("GoogleEarth.exe", s.t)
    s.run(result("three_fingers", palm=(0.5, 0.6)), frames=1)
    s.run(result("three_fingers", palm=(0.5, 0.42)), seconds=0.8)
    assert s.ex.scroll_total() > 0 and s.ex.zoom_events() == []


def test_switching_profile_releases_held_buttons():
    s = Sim()
    s.run(result("four_fingers"), seconds=0.3)
    assert s.ex.buttons == {"middle"}
    s.c.set_app("powerpnt.exe", s.t)
    assert s.ex.buttons == set()


def test_custom_profile_in_config(tmp_path):
    p = tmp_path / "c.json"
    p.write_text(json.dumps({"profiles": {"vlc": {"match": ["vlc.exe"], "gesture_actions": {"rock": "key:space"}}}}))
    cfg, _, _ = load_config(p)
    assert cfg.profiles["vlc"].cfg.gesture_actions["rock"] == "key:space"
    assert "browser" in cfg.profiles  # built-in profiles are kept


@pytest.mark.parametrize(
    "bad",
    [
        {"profiles": {"x": {"gesture_actions": {"rock": "key:space"}}}},  # no match list
        {"profiles": {"x": {"match": ["a.exe"], "gesture_actions": {"rock": "key:nope"}}}},
        {"profiles": {"x": {"match": ["a.exe"], "rotate": {"button": "side"}}}},
        {"gesture_actions": {"two_hand_pinch": "left_click"}},
    ],
)
def test_bad_profiles_rejected(tmp_path, bad):
    p = tmp_path / "c.json"
    p.write_text(json.dumps(bad))
    with pytest.raises(ConfigError):
        load_config(p)


def test_foreground_exe_returns_a_program_name_or_none():
    exe = foreground_exe()
    assert exe is None or exe.endswith(".exe")


# ---- calibration --------------------------------------------------------------


def _samples(scale=100.0, jitter=0.01, n=25):
    rng = np.random.default_rng(0)
    out = {}
    for step, pose in [("open", "open_palm"), ("fist", "fist"), ("point", "point"), ("pinch", "pinch")]:
        hands = []
        for _ in range(n):
            h = pose_hand(pose, scale=scale)
            h.landmarks[:, :2] += rng.normal(0, jitter * scale, size=(21, 2))
            hands.append(measure(h))
        out[step] = hands
    return out


def test_calibration_produces_working_thresholds():
    values, _ = compute_thresholds(_samples())
    cfg, _, _ = load_config(None, {"gestures": values})
    rec = GestureRecognizer(cfg.gestures)
    for pose in ["open_palm", "fist", "point", "pinch", "two_fingers"]:
        assert rec.classify(rec.features(pose_hand(pose))) == pose
        rec.reset()


def test_calibration_rejects_missing_data():
    s = _samples()
    s["pinch"] = s["pinch"][:3]
    with pytest.raises(CalibrationError):
        compute_thresholds(s)


def test_calibration_rejects_unclear_pinch():
    s = _samples()
    s["pinch"] = s["point"]  # user never actually pinched
    with pytest.raises(CalibrationError):
        compute_thresholds(s)


def test_save_thresholds_keeps_other_settings_and_backs_up(tmp_path):
    p = tmp_path / "config.json"
    p.write_text(json.dumps({"camera": {"index": 1}, "gestures": {"stable_frames": 4}}))
    backup = save_thresholds(p, {"pinch_on_ratio": 0.2})
    data = json.loads(p.read_text())
    assert data["camera"]["index"] == 1
    assert data["gestures"] == {"stable_frames": 4, "pinch_on_ratio": 0.2}
    assert json.loads(backup.read_text())["gestures"] == {"stable_frames": 4}
