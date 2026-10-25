"""Hand-gesture PC controller - entry point.

    python main.py              control the mouse/keyboard with your hand
    python main.py --dry-run    only show detected gestures, control nothing
    python main.py --no-preview run without the camera window

Everything runs locally. No video is saved or sent anywhere.
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
WINDOW = "Hand Gesture Control"


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Control your mouse and keyboard with hand gestures.")
    p.add_argument("--config", default=str(BASE_DIR / "config.json"), help="path to config.json")
    p.add_argument("--dry-run", action="store_true", help="detect gestures but do not move/click anything")
    p.add_argument("--no-preview", action="store_true", help="hide the camera window (quit with Ctrl+Alt+Q)")
    p.add_argument("--camera", type=int, help="camera index (overrides config)")
    p.add_argument("--hand", choices=["right", "left", "any"], help="which hand to track (overrides config)")
    p.add_argument("--start-paused", action="store_true", help="start in PAUSED mode")
    p.add_argument("--calibrate", action="store_true", help="measure your hand and save matching thresholds")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    from src.config import ConfigError, load_config

    overrides: dict = {}
    if args.camera is not None:
        overrides.setdefault("camera", {})["index"] = args.camera
    if args.hand:
        overrides.setdefault("tracking", {})["hand"] = args.hand
    try:
        cfg, _, warnings = load_config(args.config, overrides)
    except ConfigError as e:
        print(f"Config error: {e}")
        return 2
    for w in warnings:
        print(f"Warning: {w}")

    # pyautogui must be imported before any window is created: it makes the
    # process DPI-aware so cursor coordinates match the real screen pixels.
    import pyautogui

    import cv2

    from src import overlay
    from src.actions import ActionExecutor
    from src.camera import Camera, CameraError
    from src.controller import Controller
    from src.foreground import foreground_exe
    from src.gestures import GestureRecognizer, HandSelector, SwipeDetector, TwoHandPinch
    from src.hand_tracker import HandTracker
    from src.hotkeys import Hotkeys

    if args.calibrate:
        from src.calibrate import run_calibration

        return run_calibration(cfg, Path(args.config), BASE_DIR)

    executor = ActionExecutor(dry_run=args.dry_run)
    controller = Controller(cfg, executor)
    recognizer = GestureRecognizer(cfg.gestures, lost_reset_s=cfg.timing.hand_lost_grace_s)
    selector = HandSelector(cfg.tracking.hand, cfg.tracking.use_other_hand_if_missing, cfg.tracking.switch_delay_s)
    swipes = SwipeDetector(cfg.swipe)
    two_hands = TwoHandPinch(cfg.gestures)

    stop = threading.Event()
    toggle = threading.Event()
    hotkeys = Hotkeys(on_quit=stop.set, on_toggle_pause=toggle.set)

    preview = cfg.ui.preview and not args.no_preview
    camera = Camera(cfg.camera.index, cfg.camera.width, cfg.camera.height, cfg.camera.fps, cfg.camera.backend)
    tracker = None
    exit_code = 0
    try:
        tracker = HandTracker(cfg.tracking, BASE_DIR)
        camera.open()
        hotkeys.start()
        if args.start_paused:
            controller.set_paused(True, time.monotonic())

        w, h = camera.size
        print(f"Camera {cfg.camera.index} open ({w}x{h}, {camera.backend}).  Screen {executor.screen_size[0]}x{executor.screen_size[1]}.")
        print("Mode:", "DRY RUN (nothing will be clicked)" if args.dry_run else "LIVE CONTROL")
        print("Quit: Ctrl+Alt+Q (or ESC in the preview).  Pause/resume: Ctrl+Alt+P.")

        if preview:
            cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
            cv2.setWindowProperty(WINDOW, cv2.WND_PROP_TOPMOST, 1)

        fps = 0.0
        last_t = time.monotonic()
        last_printed = ("", 0.0)
        show_help = False
        next_app_check = 0.0
        while not stop.is_set():
            frame = camera.read(timeout=2.0)
            if frame is None:
                if camera.failed:
                    raise CameraError("The camera stopped sending frames (was it unplugged or taken by another app?).")
                continue
            now = time.monotonic()
            if cfg.camera.mirror:
                frame = cv2.flip(frame, 1)

            all_hands = tracker.process(frame, int(now * 1000))
            hand = selector.select(all_hands, now)
            if selector.switched:
                # Other hand took over: start its gestures fresh and
                # release anything the previous hand was holding.
                recognizer.reset()
                controller.hand_changed(now)
                print(f"[{time.strftime('%H:%M:%S')}] {hand.handedness} hand now in control")
            result = recognizer.update(hand, now)
            if toggle.is_set():
                toggle.clear()
                controller.toggle_pause(now)
            if now >= next_app_check:  # per-app profile, checked twice a second
                next_app_check = now + 0.5
                controller.set_app(foreground_exe(), now)
            controller.update(result, now, two_hand=two_hands.update(all_hands))
            swipes.cfg = controller.cfg.swipe
            swipe = swipes.update(result, now)
            if swipe:
                controller.trigger(swipe, now)

            dt = now - last_t
            last_t = now
            if dt > 0:
                fps = 1.0 / dt if fps == 0 else 0.9 * fps + 0.1 * (1.0 / dt)

            text, t = controller.last_action
            # Continuous actions repeat every frame; print them once per burst.
            repeat = text == last_printed[0] and text.startswith(("SCROLL", "ZOOM", "VOLUME"))
            if text and t != last_printed[1] and not repeat:
                print(f"[{time.strftime('%H:%M:%S')}] {'(dry-run) ' if args.dry_run else ''}{text}")
            if text:
                last_printed = (text, t)

            if preview:
                overlay.draw(
                    frame,
                    zone=cfg.active_zone,
                    all_hands=all_hands,
                    hand=hand,
                    result=result,
                    controller=controller,
                    fps=fps,
                    now=now,
                    dry_run=args.dry_run,
                    show_debug=cfg.ui.show_debug,
                    show_help=show_help,
                )
                if cfg.ui.window_scale != 1.0:
                    frame = cv2.resize(frame, None, fx=cfg.ui.window_scale, fy=cfg.ui.window_scale)
                cv2.imshow(WINDOW, frame)
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC
                    break
                if key in (ord("h"), ord("H")):
                    show_help = not show_help
                if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                    break  # window closed with the X button
    except pyautogui.FailSafeException:
        print("Fail-safe: the mouse was moved into a screen corner. Stopped.")
    except CameraError as e:
        print(f"Camera error: {e}")
        exit_code = 1
    except KeyboardInterrupt:
        pass
    finally:
        controller.shutdown()  # releases any held mouse button
        camera.release()
        if tracker is not None:
            tracker.close()
        hotkeys.stop()
        cv2.destroyAllWindows()
        print("Stopped. Camera closed, mouse buttons released.")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
