"""
Audio Sound Notification System for LPR & RFID Gates.

Plays audio feedback from res/sound/:
- successful.mp3: When a scan / license plate is found/verified in the system.
- denied.mp3: When a plate or RFID tag is unregistered, mismatch, or denied.
- welcome.mp3: When both license plate and RFID are completed at the Entrance gate.
- goodbye.mp3: When both license plate and RFID are completed at the Exit gate.
"""

import os
import shutil
import subprocess
import threading
import time
from typing import Any

# Directory containing the audio files
SOUND_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "res", "sound")

# Valid sound identifiers
VALID_SOUNDS = {
    "successful": "successful.mp3",
    "denied": "denied.mp3",
    "welcome": "welcome.mp3",
    "goodbye": "goodbye.mp3",
}

# State to track latest sound event and prevent duplicate rapid playback
_sound_lock = threading.Lock()
_last_sound_times: dict[str, float] = {}
_last_sound_event: dict[str, Any] = {
    "sound": None,
    "gate": None,
    "timestamp": None,
    "epoch": 0.0,
    "played_on_host": False,
}


def _get_audio_player() -> list[str] | None:
    """Find an available CLI audio player on Linux."""
    if shutil.which("ffplay"):
        return ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet"]
    if shutil.which("mpv"):
        return ["mpv", "--no-video", "--really-quiet"]
    if shutil.which("mpg123"):
        return ["mpg123", "-q"]
    if shutil.which("paplay"):
        return ["paplay"]
    if shutil.which("aplay"):
        return ["aplay", "-q"]
    return None


def play_sound(sound_name: str, gate: str | None = None, delay_seconds: float = 0.0) -> None:
    """
    Play a sound file asynchronously on host speakers and broadcast to web clients.
    sound_name: 'successful' | 'denied' | 'welcome' | 'goodbye'
    gate: 'entrance' | 'exit' | None
    delay_seconds: optional delay before playing (e.g. for sequential welcome/goodbye after successful)
    """
    clean_name = str(sound_name or "").strip().lower()
    if clean_name not in VALID_SOUNDS:
        print(f"[sound] Unknown sound name: '{sound_name}'")
        return

    filename = VALID_SOUNDS[clean_name]
    sound_path = os.path.join(SOUND_DIR, filename)
    if not os.path.exists(sound_path):
        print(f"[sound] File not found: {sound_path}")
        return

    # Deduplicate rapid repeated sound requests within 1.0s
    now = time.time()
    with _sound_lock:
        last_time = _last_sound_times.get(clean_name, 0.0)
        if (now + delay_seconds) - last_time < 1.0:
            print(f"[sound] Suppressed duplicate '{clean_name}' sound trigger.")
            return
        _last_sound_times[clean_name] = now + delay_seconds

    def _play_worker():
        if delay_seconds > 0:
            time.sleep(delay_seconds)

        player_cmd = _get_audio_player()
        played_on_host = False

        if player_cmd:
            try:
                cmd = player_cmd + [sound_path]
                subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=10.0)
                played_on_host = True
            except Exception as exc:
                print(f"[sound] Playback error for {filename}: {exc}")

        # Update last sound event for browser clients
        with _sound_lock:
            global _last_sound_event
            _last_sound_event = {
                "sound": clean_name,
                "file": filename,
                "gate": gate,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "epoch": time.time(),
                "played_on_host": played_on_host,
            }

    threading.Thread(target=_play_worker, daemon=True, name=f"sound-{clean_name}").start()


def play_gate_success(gate: str) -> None:
    """
    Helper when both plate and RFID are complete for a gate:
    1. Play 'successful.mp3' immediately.
    2. Play 'welcome.mp3' for Entrance OR 'goodbye.mp3' for Exit.
    3. Trigger Green light relay (switches from Red NC to Green NO).
    """
    clean_gate = str(gate or "").strip().lower()
    play_sound("successful", gate=clean_gate, delay_seconds=0.0)
    if clean_gate == "entrance":
        play_sound("welcome", gate=clean_gate, delay_seconds=1.1)
    elif clean_gate == "exit":
        play_sound("goodbye", gate=clean_gate, delay_seconds=1.1)

    # Trigger relay hardware to switch to Green light
    try:
        from relay_system import trigger_gate_relay
        trigger_gate_relay(clean_gate)
    except Exception as exc:
        print(f"[relay] Trigger error: {exc}")


def get_last_sound_event() -> dict[str, Any]:
    """Return the most recent sound event for dashboard polling / web audio."""
    with _sound_lock:
        return dict(_last_sound_event)
