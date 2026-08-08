"""
relay_system.py — Raspberry Pi GPIO Relay & Traffic Light Controller.

Controls 4 lights (2 Red, 2 Green) across 2 gates (Entrance and Exit) using 2 GPIO pins.

Hardware Wiring & Relay Configuration:
========================================================================================
1. Relay Module (2-Channel Relay Board with SPDT contacts: COM, NC, NO):
   - Entrance Relay Channel:
       * COM (Common)           --> Power Source (+) for Lights (e.g. +12V DC or Live)
       * NC (Normally Closed)   --> Entrance RED Light (+)     [Normal / Idle State: ON]
       * NO (Normally Open)     --> Entrance GREEN Light (+)   [Access Granted: ON]
   - Exit Relay Channel:
       * COM (Common)           --> Power Source (+) for Lights (e.g. +12V DC or Live)
       * NC (Normally Closed)   --> Exit RED Light (+)         [Normal / Idle State: ON]
       * NO (Normally Open)     --> Exit GREEN Light (+)       [Access Granted: ON]
   - Light Ground / Return:
       * Return (-) of all 4 lights --> Connected directly to Power Source Ground / Return (-)

2. Raspberry Pi GPIO Pin Connections (40-Pin Header):
   -------------------------------------------------------------------------------------
   Function               | BCM GPIO   | Physical Pin Header | Relay Module Terminal
   -------------------------------------------------------------------------------------
   Entrance Relay Control | GPIO 17    | Pin 11              | IN1 (Entrance Relay)
   Exit Relay Control     | GPIO 27    | Pin 13              | IN2 (Exit Relay)
   Relay Power VCC        | 5V Power   | Pin 2 (or Pin 4)    | VCC / DC+
   Relay Ground GND       | Ground     | Pin 6 (or Pin 9/14) | GND / DC-
   -------------------------------------------------------------------------------------

3. Operating Behavior:
   - Normal / Idle State:
       Relay is De-energized (GPIO LOW / 0).
       COM is connected to NC terminal -> RED lights are naturally ON (both sides).
       GREEN lights are OFF.
   - Successful Entry / Exit (ACCESS_GRANTED):
       Relay is Energized (GPIO HIGH / 1) for the specific gate.
       COM switches to NO terminal -> GREEN light turns ON, RED light turns OFF.
       After GATE_OPEN_DURATION (default 5.0s), relay de-energizes -> RED light turns back ON.
========================================================================================
"""

from __future__ import annotations

import os
import threading
import time
from typing import Any

# Default GPIO Pin configuration (BCM numbering)
DEFAULT_ENTRANCE_PIN = 17  # Physical Pin 11
DEFAULT_EXIT_PIN = 27      # Physical Pin 13
DEFAULT_GATE_DURATION = 5.0  # Seconds green light stays active

# Physical pin mapping for reference
BCM_TO_PHYSICAL: dict[int, int] = {
    17: 11,
    27: 13,
    22: 15,
    23: 16,
    24: 18,
    25: 22,
    5: 29,
    6: 31,
    12: 32,
    13: 33,
    16: 36,
    26: 37,
}


def _get_env_pin(var_name: str, default: int) -> int:
    try:
        return int(os.getenv(var_name, str(default)).strip())
    except (ValueError, TypeError):
        return default


def _get_env_duration(var_name: str, default: float) -> float:
    try:
        return float(os.getenv(var_name, str(default)).strip())
    except (ValueError, TypeError):
        return default


# Read configuration
ENTRANCE_PIN = _get_env_pin("ENTRANCE_RELAY_PIN", DEFAULT_ENTRANCE_PIN)
EXIT_PIN = _get_env_pin("EXIT_RELAY_PIN", DEFAULT_EXIT_PIN)
GATE_OPEN_DURATION = _get_env_duration("GATE_OPEN_DURATION", DEFAULT_GATE_DURATION)
ACTIVE_HIGH = os.getenv("RELAY_ACTIVE_LOW", "false").strip().lower() not in ("true", "1", "yes")

# Active level (1 = High for standard Active-High relay coils)
LEVEL_ACTIVE = 1 if ACTIVE_HIGH else 0
LEVEL_INACTIVE = 0 if ACTIVE_HIGH else 1


# ---------------------------------------------------------------------------
# Hardware Driver Layer with Automatic Fallback
# ---------------------------------------------------------------------------
class _BaseRelayDriver:
    name: str = "base"

    def setup_pin(self, pin: int) -> None:
        pass

    def write_pin(self, pin: int, level: int) -> None:
        pass

    def cleanup(self) -> None:
        pass


class _GpiozeroDriver(_BaseRelayDriver):
    name = "gpiozero"

    def __init__(self):
        from gpiozero import OutputDevice  # type: ignore
        self._OutputDevice = OutputDevice
        self._devices: dict[int, Any] = {}

    def setup_pin(self, pin: int) -> None:
        if pin not in self._devices:
            # active_high matches our LEVEL_ACTIVE logic
            dev = self._OutputDevice(pin, active_high=ACTIVE_HIGH, initial_value=False)
            self._devices[pin] = dev

    def write_pin(self, pin: int, level: int) -> None:
        dev = self._devices.get(pin)
        if dev is None:
            self.setup_pin(pin)
            dev = self._devices.get(pin)
        if dev:
            if level == LEVEL_ACTIVE:
                dev.on()
            else:
                dev.off()

    def cleanup(self) -> None:
        for dev in self._devices.values():
            try:
                dev.off()
                dev.close()
            except Exception:
                pass
        self._devices.clear()


class _RPiGPIODriver(_BaseRelayDriver):
    name = "RPi.GPIO"

    def __init__(self):
        import RPi.GPIO as GPIO  # type: ignore
        self._GPIO = GPIO
        self._GPIO.setwarnings(False)
        self._GPIO.setmode(self._GPIO.BCM)
        self._pins: set[int] = set()

    def setup_pin(self, pin: int) -> None:
        init_state = self._GPIO.HIGH if LEVEL_INACTIVE == 1 else self._GPIO.LOW
        self._GPIO.setup(pin, self._GPIO.OUT, initial=init_state)
        self._pins.add(pin)

    def write_pin(self, pin: int, level: int) -> None:
        state = self._GPIO.HIGH if level == 1 else self._GPIO.LOW
        self._GPIO.output(pin, state)

    def cleanup(self) -> None:
        try:
            for pin in self._pins:
                self.write_pin(pin, LEVEL_INACTIVE)
            self._GPIO.cleanup()
        except Exception:
            pass
        self._pins.clear()


class _GpiodDriver(_BaseRelayDriver):
    name = "gpiod"

    def __init__(self):
        import gpiod  # type: ignore
        self._gpiod = gpiod
        self._chip = None
        self._lines: dict[int, Any] = {}
        # Try opening default gpiochip
        for chip_name in ("gpiochip4", "gpiochip0", "0"):
            try:
                self._chip = gpiod.Chip(chip_name)
                break
            except Exception:
                continue

    def setup_pin(self, pin: int) -> None:
        if self._chip is None or pin in self._lines:
            return
        try:
            line = self._chip.get_line(pin)
            line.request(consumer="lpr_relay", type=self._gpiod.LINE_REQ_DIR_OUT, default_val=LEVEL_INACTIVE)
            self._lines[pin] = line
        except Exception as exc:
            print(f"[relay] gpiod setup error for pin {pin}: {exc}")

    def write_pin(self, pin: int, level: int) -> None:
        line = self._lines.get(pin)
        if line:
            try:
                line.set_value(level)
            except Exception as exc:
                print(f"[relay] gpiod write error for pin {pin}: {exc}")

    def cleanup(self) -> None:
        for line in self._lines.values():
            try:
                line.set_value(LEVEL_INACTIVE)
                line.release()
            except Exception:
                pass
        self._lines.clear()
        if self._chip:
            try:
                self._chip.close()
            except Exception:
                pass


class _SysfsDriver(_BaseRelayDriver):
    name = "sysfs"

    def __init__(self):
        self._exported_pins: set[int] = set()

    def setup_pin(self, pin: int) -> None:
        gpio_dir = f"/sys/class/gpio/gpio{pin}"
        if not os.path.exists(gpio_dir):
            try:
                with open("/sys/class/gpio/export", "w") as f:
                    f.write(str(pin))
            except Exception:
                pass
        # Set direction
        dir_file = os.path.join(gpio_dir, "direction")
        if os.path.exists(dir_file):
            try:
                with open(dir_file, "w") as f:
                    f.write("out")
            except Exception:
                pass
        self._exported_pins.add(pin)
        self.write_pin(pin, LEVEL_INACTIVE)

    def write_pin(self, pin: int, level: int) -> None:
        val_file = f"/sys/class/gpio/gpio{pin}/value"
        if os.path.exists(val_file):
            try:
                with open(val_file, "w") as f:
                    f.write(str(level))
            except Exception:
                pass

    def cleanup(self) -> None:
        for pin in self._exported_pins:
            self.write_pin(pin, LEVEL_INACTIVE)
            try:
                with open("/sys/class/gpio/unexport", "w") as f:
                    f.write(str(pin))
            except Exception:
                pass
        self._exported_pins.clear()


class _MockDriver(_BaseRelayDriver):
    name = "mock_simulator"

    def __init__(self):
        self._states: dict[int, int] = {}

    def setup_pin(self, pin: int) -> None:
        self._states[pin] = LEVEL_INACTIVE

    def write_pin(self, pin: int, level: int) -> None:
        self._states[pin] = level

    def cleanup(self) -> None:
        self._states.clear()


def _init_driver() -> _BaseRelayDriver:
    """Auto-detect available Raspberry Pi GPIO library or fall back to simulation."""
    # 1. Try gpiozero
    try:
        drv = _GpiozeroDriver()
        print("[relay] Initialized hardware driver: gpiozero")
        return drv
    except Exception:
        pass

    # 2. Try RPi.GPIO
    try:
        drv = _RPiGPIODriver()
        print("[relay] Initialized hardware driver: RPi.GPIO")
        return drv
    except Exception:
        pass

    # 3. Try gpiod
    try:
        drv = _GpiodDriver()
        if drv._chip is not None:
            print("[relay] Initialized hardware driver: gpiod")
            return drv
    except Exception:
        pass

    # 4. Try Linux sysfs if on Linux with /sys/class/gpio
    if os.path.exists("/sys/class/gpio/export"):
        try:
            drv = _SysfsDriver()
            print("[relay] Initialized hardware driver: Linux sysfs")
            return drv
        except Exception:
            pass

    # 5. Mock simulator
    print("[relay] No Raspberry Pi GPIO hardware driver found — running in Simulated Mode.")
    return _MockDriver()


# ---------------------------------------------------------------------------
# Module Controller & Thread-Safe Gate Relay State
# ---------------------------------------------------------------------------
_driver: _BaseRelayDriver | None = None
_lock = threading.Lock()

# Gate state tracking
# 'active': True = Green Light ON (Relay Energized), False = Red Light ON (NC Idle)
_gate_timers: dict[str, threading.Timer | None] = {"entrance": None, "exit": None}
_gate_states: dict[str, dict[str, Any]] = {
    "entrance": {
        "pin_bcm": ENTRANCE_PIN,
        "pin_physical": BCM_TO_PHYSICAL.get(ENTRANCE_PIN, 11),
        "active": False,
        "light": "RED",
        "last_triggered": None,
        "until_epoch": 0.0,
        "duration": GATE_OPEN_DURATION,
    },
    "exit": {
        "pin_bcm": EXIT_PIN,
        "pin_physical": BCM_TO_PHYSICAL.get(EXIT_PIN, 13),
        "active": False,
        "light": "RED",
        "last_triggered": None,
        "until_epoch": 0.0,
        "duration": GATE_OPEN_DURATION,
    },
}


def _get_driver() -> _BaseRelayDriver:
    global _driver
    if _driver is None:
        _driver = _init_driver()
        # Setup pins
        _driver.setup_pin(ENTRANCE_PIN)
        _driver.setup_pin(EXIT_PIN)
        _driver.write_pin(ENTRANCE_PIN, LEVEL_INACTIVE)
        _driver.write_pin(EXIT_PIN, LEVEL_INACTIVE)
    return _driver


def init_relays() -> None:
    """Initialize GPIO pins and ensure all relays are in normal idle state (Red ON)."""
    with _lock:
        drv = _get_driver()
        drv.write_pin(ENTRANCE_PIN, LEVEL_INACTIVE)
        drv.write_pin(EXIT_PIN, LEVEL_INACTIVE)
        _gate_states["entrance"]["active"] = False
        _gate_states["entrance"]["light"] = "RED"
        _gate_states["exit"]["active"] = False
        _gate_states["exit"]["light"] = "RED"
        print(
            f"[relay] System initialized: Entrance (GPIO {ENTRANCE_PIN} / Pin {BCM_TO_PHYSICAL.get(ENTRANCE_PIN, 11)}), "
            f"Exit (GPIO {EXIT_PIN} / Pin {BCM_TO_PHYSICAL.get(EXIT_PIN, 13)}) -> Normal state: RED LIGHTS ON (NC)."
        )


def _deactivate_gate_worker(gate: str) -> None:
    """Deactivate relay after timer expires, returning gate to Red light (NC)."""
    with _lock:
        gate_key = str(gate or "").strip().lower()
        if gate_key not in _gate_states:
            return

        drv = _get_driver()
        pin = ENTRANCE_PIN if gate_key == "entrance" else EXIT_PIN
        drv.write_pin(pin, LEVEL_INACTIVE)

        _gate_states[gate_key]["active"] = False
        _gate_states[gate_key]["light"] = "RED"
        _gate_states[gate_key]["until_epoch"] = 0.0
        _gate_timers[gate_key] = None

        print(f"[relay-{gate_key}] Gate timer expired -> Relay OFF, RED LIGHT ON (NC).")


def trigger_gate_relay(gate: str, duration: float | None = None) -> dict[str, Any]:
    """
    Trigger green light for a specific gate ('entrance' or 'exit').
    - Energizes the relay (COM switches from NC to NO -> GREEN ON, RED OFF).
    - Starts a timer for `duration` seconds.
    - Automatically reverts back to RED light (NC) when the timer expires.
    - Thread-safe; safely refreshes timer if triggered again while already green.
    """
    gate_key = str(gate or "").strip().lower()
    if gate_key not in ("entrance", "exit"):
        raise ValueError("gate must be 'entrance' or 'exit'.")

    open_sec = float(duration) if (duration is not None and duration > 0) else GATE_OPEN_DURATION
    pin = ENTRANCE_PIN if gate_key == "entrance" else EXIT_PIN

    with _lock:
        drv = _get_driver()

        # Cancel existing active timer if any
        prev_timer = _gate_timers.get(gate_key)
        if prev_timer is not None:
            try:
                prev_timer.cancel()
            except Exception:
                pass
            _gate_timers[gate_key] = None

        # Energize relay -> Green ON, Red OFF
        drv.write_pin(pin, LEVEL_ACTIVE)

        now = time.time()
        until = now + open_sec
        now_str = time.strftime("%Y-%m-%d %H:%M:%S")

        _gate_states[gate_key]["active"] = True
        _gate_states[gate_key]["light"] = "GREEN"
        _gate_states[gate_key]["last_triggered"] = now_str
        _gate_states[gate_key]["until_epoch"] = until
        _gate_states[gate_key]["duration"] = open_sec

        # Start background timer to turn back to Red
        timer = threading.Timer(open_sec, _deactivate_gate_worker, args=(gate_key,))
        timer.daemon = True
        timer.name = f"relay-timer-{gate_key}"
        timer.start()
        _gate_timers[gate_key] = timer

        print(
            f"[relay-{gate_key}] ACCESS GRANTED -> GREEN LIGHT ON (GPIO {pin} HIGH, "
            f"duration={open_sec:.1f}s until {time.strftime('%H:%M:%S', time.localtime(until))})."
        )

        return {
            "ok": True,
            "gate": gate_key,
            "pin_bcm": pin,
            "pin_physical": BCM_TO_PHYSICAL.get(pin, 11 if gate_key == "entrance" else 13),
            "light": "GREEN",
            "duration": open_sec,
            "until": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(until)),
            "driver": drv.name,
        }


def set_relay_state(gate: str, active: bool) -> dict[str, Any]:
    """
    Manually lock/set relay state for a gate.
    active=True  -> Relay Energized (GREEN LIGHT ON, RED LIGHT OFF)
    active=False -> Relay De-energized (RED LIGHT ON, GREEN LIGHT OFF)
    """
    gate_key = str(gate or "").strip().lower()
    if gate_key not in ("entrance", "exit"):
        raise ValueError("gate must be 'entrance' or 'exit'.")

    pin = ENTRANCE_PIN if gate_key == "entrance" else EXIT_PIN

    with _lock:
        drv = _get_driver()

        # Cancel any active timer
        prev_timer = _gate_timers.get(gate_key)
        if prev_timer is not None:
            try:
                prev_timer.cancel()
            except Exception:
                pass
            _gate_timers[gate_key] = None

        if active:
            drv.write_pin(pin, LEVEL_ACTIVE)
            _gate_states[gate_key]["active"] = True
            _gate_states[gate_key]["light"] = "GREEN"
            _gate_states[gate_key]["until_epoch"] = 0.0
            print(f"[relay-{gate_key}] Manually set to GREEN (Relay ON).")
        else:
            drv.write_pin(pin, LEVEL_INACTIVE)
            _gate_states[gate_key]["active"] = False
            _gate_states[gate_key]["light"] = "RED"
            _gate_states[gate_key]["until_epoch"] = 0.0
            print(f"[relay-{gate_key}] Manually set to RED (Relay OFF / NC).")

        return {
            "ok": True,
            "gate": gate_key,
            "pin_bcm": pin,
            "pin_physical": BCM_TO_PHYSICAL.get(pin, 11 if gate_key == "entrance" else 13),
            "light": _gate_states[gate_key]["light"],
            "active": _gate_states[gate_key]["active"],
            "driver": drv.name,
        }


def get_relay_status() -> dict[str, Any]:
    """Return live status of both gate relays, active light colors, and pin mapping."""
    with _lock:
        drv = _get_driver()
        now = time.time()

        entrance_info = dict(_gate_states["entrance"])
        exit_info = dict(_gate_states["exit"])

        # Calculate remaining seconds if green
        for info in (entrance_info, exit_info):
            until = info.get("until_epoch", 0.0)
            if info.get("active") and until > now:
                info["remaining_seconds"] = round(until - now, 1)
            else:
                info["remaining_seconds"] = 0.0

        return {
            "driver": drv.name,
            "active_high": ACTIVE_HIGH,
            "default_duration": GATE_OPEN_DURATION,
            "entrance": entrance_info,
            "exit": exit_info,
            "wiring": {
                "entrance": {
                    "bcm": ENTRANCE_PIN,
                    "physical": BCM_TO_PHYSICAL.get(ENTRANCE_PIN, 11),
                    "nc_light": "RED (Normally Closed / Idle)",
                    "no_light": "GREEN (Normally Open / Success)",
                },
                "exit": {
                    "bcm": EXIT_PIN,
                    "physical": BCM_TO_PHYSICAL.get(EXIT_PIN, 13),
                    "nc_light": "RED (Normally Closed / Idle)",
                    "no_light": "GREEN (Normally Open / Success)",
                },
                "vcc": "Physical Pin 2 or 4 (5V Power)",
                "gnd": "Physical Pin 6, 9, or 14 (Ground)",
            },
        }


def cleanup_relays() -> None:
    """Safe shutdown: cancel timers, return to Red state, and release pins."""
    with _lock:
        for gate in ("entrance", "exit"):
            timer = _gate_timers.get(gate)
            if timer is not None:
                try:
                    timer.cancel()
                except Exception:
                    pass
                _gate_timers[gate] = None

        global _driver
        if _driver is not None:
            try:
                _driver.cleanup()
            except Exception:
                pass
            _driver = None
        print("[relay] Hardware relays cleaned up safely.")
