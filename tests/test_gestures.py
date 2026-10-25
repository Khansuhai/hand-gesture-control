import numpy as np
import pytest

from src import gestures
from src.config import load_config
from src.gestures import (
    INDEX_TIP,
    THUMB_TIP,
    GestureRecognizer,
    HandFeatures,
    HandSelector,
    Stabilizer,
    hand_size,
    register_pose,
)
from tests.synthetic import FINGERS, make_hand, pose_hand


@pytest.fixture
def gcfg():
    cfg, _, _ = load_config(None)
    return cfg.gestures


def classify(gcfg, hand):
    rec = GestureRecognizer(gcfg)
    return rec.classify(rec.features(hand))


@pytest.mark.parametrize(
    "pose",
    ["point", "two_fingers", "three_fingers", "four_fingers", "open_palm", "fist", "thumbs_up",
     "thumbs_down", "call_me", "rock", "pinky", "pinch", "middle_pinch"],
)
def test_each_pose_is_recognized(gcfg, pose):
    assert classify(gcfg, pose_hand(pose)) == pose


@pytest.mark.parametrize(
    "pose, count",
    [("fist", 0), ("point", 1), ("two_fingers", 2), ("three_fingers", 3), ("four_fingers", 4), ("open_palm", 5)],
)
def test_finger_count(gcfg, pose, count):
    rec = GestureRecognizer(gcfg)
    assert rec.features(pose_hand(pose)).count() == count


@pytest.mark.parametrize("pose", ["point", "two_fingers", "open_palm", "fist", "pinch", "middle_pinch"])
@pytest.mark.parametrize("scale", [45.0, 90.0, 180.0])  # far from / near the camera
def test_poses_work_at_any_distance(gcfg, pose, scale):
    assert classify(gcfg, pose_hand(pose, scale=scale, center=(320, 300))) == pose


@pytest.mark.parametrize("pose", ["point", "two_fingers", "open_palm", "fist", "pinch"])
@pytest.mark.parametrize("angle", [-30, 25])
def test_poses_survive_hand_tilt(gcfg, pose, angle):
    assert classify(gcfg, pose_hand(pose, rotation_deg=angle)) == pose


def test_no_hand_shape_is_none(gcfg):
    # middle + ring + pinky up, index down is not a configured pose
    hand = make_hand(thumb="tucked", index="curl", middle="ext", ring="ext", pinky="ext")
    assert classify(gcfg, hand) == "none"


def test_fist_with_thumb_on_index_is_not_a_pinch(gcfg):
    hand = pose_hand("fist")
    rec = GestureRecognizer(gcfg)
    f = rec.features(hand)
    assert f.pinch_ratio < gcfg.pinch_on_ratio  # thumb really is touching the curled index
    assert not f.index_pinched
    assert rec.classify(f) == "fist"


def test_thumb_resting_on_curled_middle_does_not_freeze_cursor(gcfg):
    # Pointing with the thumb tip on the curled middle finger (very common).
    curled_middle_tip = (0.01, -0.85)
    hand = make_hand(thumb="tucked", index="ext", thumb_tip=(curled_middle_tip[0] - 0.05, curled_middle_tip[1]))
    rec = GestureRecognizer(gcfg)
    f = rec.features(hand)
    assert f.middle_pinch_ratio < gcfg.pinch_freeze_ratio
    assert not f.middle_pinch_near
    assert rec.classify(f) == "point"


def test_fingers_reaching_for_thumb_count_as_near(gcfg):
    rec = GestureRecognizer(gcfg)
    assert rec.features(pose_hand("pinch")).index_pinch_near
    assert rec.features(pose_hand("middle_pinch")).middle_pinch_near


def test_upright_fist_with_thumb_sideways_is_fist_not_thumbs_up(gcfg):
    hand = make_hand(thumb="out", index="curl")
    assert classify(gcfg, hand) == "fist"


def test_thumbs_down_is_not_thumbs_up(gcfg):
    hand = pose_hand("thumbs_up", rotation_deg=-90)
    assert classify(gcfg, hand) != "thumbs_up"


def test_hand_size_scales_linearly():
    a = hand_size(make_hand(scale=50).landmarks[:, :2])
    b = hand_size(make_hand(scale=150).landmarks[:, :2])
    assert b / a == pytest.approx(3.0, rel=1e-6)


def _hand_with_pinch_ratio(ratio: float):
    """Point pose with the thumb tip placed `ratio` hand sizes from the index tip."""
    base = make_hand(thumb="pinch_index", index="pinch", scale=1.0, center=(0, 0))
    size = hand_size(base.landmarks[:, :2])
    tip = np.array(FINGERS["index"]["pinch"][3])
    return make_hand(
        thumb="pinch_index", index="pinch", thumb_tip=tuple(tip + np.array([-ratio * size, 0.0]))
    )


def test_pinch_hysteresis(gcfg):
    rec = GestureRecognizer(gcfg)
    between = (gcfg.pinch_on_ratio + gcfg.pinch_off_ratio) / 2
    # Not pinched yet: "between" is not close enough to start.
    assert not rec.features(_hand_with_pinch_ratio(between)).index_pinched
    # Start the pinch.
    assert rec.features(_hand_with_pinch_ratio(gcfg.pinch_on_ratio * 0.5)).index_pinched
    # Already pinched: "between" keeps it pinched (no flicker).
    assert rec.features(_hand_with_pinch_ratio(between)).index_pinched
    # Clearly open: released.
    assert not rec.features(_hand_with_pinch_ratio(gcfg.pinch_off_ratio * 1.3)).index_pinched


def test_pinch_ratio_is_distance_relative(gcfg):
    rec = GestureRecognizer(gcfg)
    near = rec.features(pose_hand("pinch", scale=180)).pinch_ratio
    far = rec.features(pose_hand("pinch", scale=45)).pinch_ratio
    assert near == pytest.approx(far, rel=1e-6)


def test_stabilizer_needs_consecutive_frames():
    s = Stabilizer(3)
    assert s.update("point") == "none"
    assert s.update("point") == "none"
    assert s.update("point") == "point"
    assert s.update("fist") == "point"  # one odd frame is ignored
    assert s.update("point") == "point"
    assert s.update("fist") == "point"
    assert s.update("fist") == "point"
    assert s.update("fist") == "fist"


def test_recognizer_reports_stable_pose_and_points(gcfg):
    rec = GestureRecognizer(gcfg)
    hand = pose_hand("point")
    results = [rec.update(hand, now=i / 30) for i in range(gcfg.stable_frames)]
    assert results[0].raw_pose == "point"
    assert results[-1].pose == "point"
    w, h = hand.image_size
    assert results[-1].index_tip == pytest.approx((hand.landmarks[INDEX_TIP, 0] / w, hand.landmarks[INDEX_TIP, 1] / h))
    mid = (hand.landmarks[INDEX_TIP, :2] + hand.landmarks[THUMB_TIP, :2]) / 2
    assert results[-1].pinch_point == pytest.approx((mid[0] / w, mid[1] / h))


def test_short_dropout_keeps_state_long_dropout_resets(gcfg):
    rec = GestureRecognizer(gcfg, lost_reset_s=0.3)
    for i in range(5):
        rec.update(pose_hand("pinch"), now=i / 30)
    assert rec.stabilizer.stable == "pinch"
    rec.update(None, now=0.25)
    assert rec.stabilizer.stable == "pinch"
    rec.update(None, now=0.8)
    assert rec.stabilizer.stable == "none"


def test_register_custom_pose(gcfg, monkeypatch):
    monkeypatch.setattr(gestures, "POSES", list(gestures.POSES))
    register_pose(
        "three_up_no_index",
        lambda f: not f.fingers["index"] and f.fingers["middle"] and f.fingers["ring"] and f.fingers["pinky"],
        before="point",
    )
    hand = make_hand(thumb="tucked", index="curl", middle="ext", ring="ext", pinky="ext")
    assert classify(gcfg, hand) == "three_up_no_index"


RIGHT = make_hand(handedness="Right", center=(450, 400))
LEFT = make_hand(handedness="Left", center=(170, 400))


def test_selector_prefers_right_hand_when_both_visible():
    sel = HandSelector("right", allow_other=True)
    assert sel.select([LEFT, RIGHT], 0.0) is RIGHT


def test_selector_uses_left_hand_when_right_is_not_there():
    sel = HandSelector("right", allow_other=True)
    assert sel.select([LEFT], 0.0) is LEFT


def test_selector_waits_before_switching_from_right_to_left():
    sel = HandSelector("right", allow_other=True, switch_delay_s=0.3)
    assert sel.select([RIGHT, LEFT], 0.0) is RIGHT
    assert sel.select([LEFT], 0.1) is None  # right just blinked out: don't flip yet
    assert sel.select([RIGHT, LEFT], 0.15) is RIGHT
    assert sel.select([LEFT], 0.2) is None
    assert sel.select([LEFT], 0.6) is LEFT  # right really gone: left takes over
    assert sel.select([LEFT], 0.7) is LEFT


def test_selector_goes_back_to_right_immediately():
    sel = HandSelector("right", allow_other=True)
    sel.select([LEFT], 0.0)
    assert sel.select([LEFT, RIGHT], 0.1) is RIGHT


def test_selector_right_only_ignores_left():
    sel = HandSelector("right", allow_other=False)
    assert sel.select([LEFT], 5.0) is None
    assert sel.select([LEFT, RIGHT], 5.1) is RIGHT


def test_relabeled_hand_in_same_place_is_the_same_hand():
    sel = HandSelector("right", allow_other=True)
    right = make_hand(handedness="Right", center=(320, 400))
    same_but_called_left = make_hand(handedness="Left", center=(330, 405))
    assert sel.select([right], 0.0) is right
    assert sel.select([same_but_called_left], 0.03) is same_but_called_left  # no waiting
    assert not sel.switched  # so a drag/rotate is NOT interrupted
    assert sel.select([right], 0.06) is right
    assert not sel.switched


def test_other_hand_elsewhere_is_a_real_switch():
    sel = HandSelector("right", allow_other=True, switch_delay_s=0.3)
    sel.select([make_hand(handedness="Right", center=(450, 400))], 0.0)
    left = make_hand(handedness="Left", center=(150, 400))
    assert sel.select([left], 0.1) is None
    assert sel.select([left], 0.5) is left
    assert sel.switched


def test_right_hand_coming_back_is_a_switch():
    sel = HandSelector("right", allow_other=True)
    left = make_hand(handedness="Left", center=(150, 400))
    right = make_hand(handedness="Right", center=(450, 400))
    sel.select([left], 0.0)
    assert sel.select([left, right], 0.1) is right
    assert sel.switched


def test_selector_any_sticks_with_current_hand():
    sel = HandSelector("any", allow_other=False)
    first = sel.select([LEFT], 0.0)
    assert first is LEFT
    assert sel.select([RIGHT, LEFT], 0.1) is LEFT


def test_four_curled_helper():
    f = HandFeatures(
        fingers={"thumb": True, "index": False, "middle": False, "ring": False, "pinky": False},
        thumb_up=False, hand_size=1, pinch_ratio=1, middle_pinch_ratio=1,
        index_pinched=False, middle_pinched=False,
    )
    assert f.four_curled()
