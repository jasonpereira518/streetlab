"""What the ego can actually *see* about traffic controls.

The simulation owns ground-truth signal phases (`SignalController`). That is the
right source for rendering lamp colours and for NPC traffic, which is meant to
be a reactive background. It is the wrong source for the ego planner: a real
driver (or a real AV stack) does not read the master clock — it reads the
device in front of it, when that device is in view.

This module is the sensing seam for that. Given ego pose, the scene's light and
stop-sign props, optional building occluders, and the truth phases, it returns
only the phases the car could resolve from its windscreen. Devices that fall
outside range or field of view, or sit behind a building, are simply absent —
and `plan/behavior.py` treats a missing signal phase as "unknown, so stop".

A short memory keeps a phase alive for a couple of seconds after the last
sighting so a pole briefly clipped by a tree does not flicker the FSM. That is
still a claim about what the driver recently saw, not about the sim clock.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Mapping, Sequence

from perception.visibility import line_of_sight_clear
from schema import Building, CameraParams, SignalState, StopSign, TrafficLight
from sim.vehicle import VehicleState

#: How far ahead a control device can still be read. Comfortably beyond the
#: behaviour layer's `APPROACH_M` (45 m) so a light is usually acquired before
#: the FSM starts caring about it.
MAX_DEVICE_RANGE_M = 70.0

#: Half-angle of the forward windscreen cone. 70° each side is a wide but
#: still forward-looking FOV — mirrors are not modelled here; rear devices
#: never govern the car's own stop line.
FOV_HALF_RAD = math.radians(70.0)

#: How long a last-seen phase is trusted after the device leaves view.
MEMORY_S = 2.0

#: Windscreen height used for the occlusion ray, matching the detector camera.
_CAMERA_Z_M = 1.33

#: Nominal lamp / sign face height for the sight-line sample when a device
#: does not carry its own height (stop signs).
_DEVICE_Z_M = 3.2


def _lamp_xy(light: TrafficLight) -> tuple[float, float]:
    """World XY of the signal head — mast-arm tip, matching the renderer."""
    # Mast arm reaches along heading rotated -90° (same as world.ts).
    arm_x = math.sin(light.heading) * light.mast_arm_m
    arm_y = -math.cos(light.heading) * light.mast_arm_m
    return light.position[0] + arm_x, light.position[1] + arm_y


def _lamp_z(light: TrafficLight) -> float:
    """Approximate lamp-housing height used by the 3D mast-arm head."""
    return max(light.height_m - 0.72, _DEVICE_Z_M)


@dataclass(slots=True)
class RoadRulesObserver:
    """Tracks which traffic devices ego has recently resolved by sight."""

    max_range_m: float = MAX_DEVICE_RANGE_M
    fov_half_rad: float = FOV_HALF_RAD
    memory_s: float = MEMORY_S
    #: device id -> (sim time last seen, last SignalState or None for a stop sign)
    _memory: dict[str, tuple[float, SignalState | None]] = field(default_factory=dict)

    def reset(self) -> None:
        self._memory.clear()

    def observe(
        self,
        ego: VehicleState,
        t: float,
        truth_signals: Mapping[str, SignalState],
        lights: Sequence[TrafficLight],
        stop_signs: Sequence[StopSign],
        buildings: Sequence[Building] = (),
    ) -> dict[str, SignalState]:
        """Return signal phases currently known from vision (+ short memory).

        Stop signs do not appear in the return map — the behaviour layer already
        treats `kind == "stop_sign"` as an unconditional stop. They are still
        sighted so callers can ask `seen(id)` if they want confirmation.
        """
        camera = CameraParams(
            x=ego.x,
            y=ego.y,
            z=_CAMERA_Z_M,
            yaw=ego.heading,
            pitch=0.0,
            roll=0.0,
            fov_y_deg=50.0,
            aspect=16.0 / 9.0,
        )

        currently_visible: set[str] = set()
        for light in lights:
            hx, hy = _lamp_xy(light)
            if not self._in_view(ego, hx, hy, camera, buildings, z=_lamp_z(light)):
                continue
            truth = truth_signals.get(light.id)
            if truth is None:
                continue
            currently_visible.add(light.id)
            self._memory[light.id] = (t, truth)

        for sign in stop_signs:
            if not self._in_view(
                ego, sign.position[0], sign.position[1], camera, buildings
            ):
                continue
            currently_visible.add(sign.id)
            self._memory[sign.id] = (t, None)

        out: dict[str, SignalState] = {}
        stale: list[str] = []
        for device_id, (seen_at, state) in self._memory.items():
            if t - seen_at > self.memory_s:
                stale.append(device_id)
                continue
            if state is None:
                continue
            # Out-of-view memory must not authorize GO. A last-seen green that
            # turns red while the pole is occluded would otherwise wave the
            # car through. Remembered red/yellow still hold (cautious).
            if device_id not in currently_visible and state.phase in (
                "green",
                "flashing_yellow",
                "off",
            ):
                continue
            out[device_id] = state
        for device_id in stale:
            del self._memory[device_id]
        return out

    def seen(self, device_id: str, t: float) -> bool:
        """True if this device was sighted within the memory window."""
        entry = self._memory.get(device_id)
        return entry is not None and t - entry[0] <= self.memory_s

    def _in_view(
        self,
        ego: VehicleState,
        x: float,
        y: float,
        camera: CameraParams,
        buildings: Sequence[Building],
        *,
        z: float = _DEVICE_Z_M,
    ) -> bool:
        dx, dy = x - ego.x, y - ego.y
        dist = math.hypot(dx, dy)
        if dist < 1e-3 or dist > self.max_range_m:
            return False
        bearing = math.atan2(dy, dx)
        if abs(math.remainder(bearing - ego.heading, math.tau)) > self.fov_half_rad:
            return False
        return line_of_sight_clear(camera, x, y, z, buildings)
