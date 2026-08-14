"""StreetLab wire schema — the Python mirror of ``streetlab/src/schema.ts``.

The TypeScript file is the single source of truth. This module is a hand-written
pydantic v2 transcription of it, laid out in the same order and with the same
section headings so the two can be read side by side. Field names are verbatim,
including ``type`` and ``cls``, which shadow Python conventions but are what the
wire carries.

Conventions (unchanged from the TypeScript)
  - Distances in metres, speeds in m/s, accelerations in m/s^2.
  - Angles in radians. Headings are 0 at +x (east) and increase CCW.
  - World is a right-handed 2D plane: +x east, +y north.
  - Time ``t`` is simulator seconds since scenario start.

Three transcription hazards are handled deliberately; see the tests that pin them:

  - ``z.number()`` rejects NaN and Infinity, but pydantic allows both by default.
    Every numeric field therefore goes through the ``Num`` family of aliases,
    which set ``allow_inf_nan=False``. One non-finite float would otherwise fail
    zod validation on arrival and silently freeze the frontend.
  - ``.nullable()`` in zod means "present, possibly null" — not "may be absent".
    Never serialise with ``exclude_none=True``.
  - zod strips unknown keys rather than rejecting them, so a stray Python-only
    field is invisible to the frontend. The fixture round-trip tests compare full
    dictionaries, which catches additions and omissions alike.
"""

from typing import Annotated, Any, Generic, Literal, TypeVar, Union

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError

# The wire protocol version, mirroring PROTOCOL_VERSION in schema.ts. Every
# message carries it in a field named `protocol`.
PROTOCOL_VERSION = 2

# This Python package's own version. Deliberately distinct from the wire
# protocol and never serialised — the two version independently.
SCHEMA_VERSION = "0.1.0"

# --------------------------------------------------------------------------- #
# Primitives                                                                   #
# --------------------------------------------------------------------------- #

# `z.number()` — finite only.
Num = Annotated[float, Field(allow_inf_nan=False)]
NonNeg = Annotated[float, Field(ge=0, allow_inf_nan=False)]
Pos = Annotated[float, Field(gt=0, allow_inf_nan=False)]
# `z.number().min(0).max(1)`
Unit = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]

# `[x, y]` in world metres.
Vec2 = tuple[Num, Num]

HexColor = Annotated[str, Field(pattern=r"^#[0-9a-fA-F]{6}$")]


class Wire(BaseModel):
    """Base for every wire model. Unknown keys are ignored, matching zod's strip."""

    model_config = ConfigDict(extra="ignore")


class Pose(Wire):
    x: Num
    y: Num
    # radians, 0 = +x (east), CCW positive
    heading: Num


class Size(Wire):
    length: Pos
    width: Pos
    height: Pos


# --------------------------------------------------------------------------- #
# SceneDescription — static world, sent once per scenario load                  #
# --------------------------------------------------------------------------- #

SignalPhase = Literal["red", "yellow", "green", "flashing_yellow", "off"]
RoadClass = Literal["arterial", "collector", "residential", "service"]
LaneMarking = Literal["none", "dashed_white", "solid_white", "double_yellow"]


class Road(Wire):
    id: str
    name: str
    road_class: RoadClass
    # Ordered polyline down the middle of the carriageway.
    centerline: Annotated[list[Vec2], Field(min_length=2)]
    lanes_forward: Annotated[int, Field(ge=0)]
    lanes_backward: Annotated[int, Field(ge=0)]
    lane_width_m: Pos
    speed_limit_mps: NonNeg
    oneway: bool
    # Marking drawn on the centre divider.
    center_marking: LaneMarking
    has_sidewalk: bool


class Building(Wire):
    id: str
    # CCW footprint ring; not closed (first point is not repeated).
    footprint: Annotated[list[Vec2], Field(min_length=3)]
    height_m: Pos
    color: HexColor
    roof_color: HexColor


class Crosswalk(Wire):
    id: str
    center: Vec2
    # Direction pedestrians walk, radians.
    heading: Num
    # Depth of the striped band, measured perpendicular to `heading`.
    width_m: Pos
    # Extent along `heading` — i.e. the carriageway width being crossed.
    length_m: Pos
    style: Literal["ladder", "continental", "transverse"]


class TrafficLight(Wire):
    id: str
    position: Vec2
    # Direction the lamp faces — i.e. toward the traffic it governs.
    heading: Num
    # Horizontal mast-arm reach; 0 means a pole-mounted head.
    mast_arm_m: NonNeg
    height_m: Pos


class StopSign(Wire):
    id: str
    position: Vec2
    heading: Num


class Tree(Wire):
    id: str
    position: Vec2
    height_m: Pos
    canopy_radius_m: Pos
    trunk_radius_m: Pos
    # 0..1 hue jitter so a forest of clones does not look like one.
    variant: Unit


class StreetSign(Wire):
    id: str
    position: Vec2
    heading: Num
    text: str
    kind: Literal["street_name", "speed_limit", "no_parking"]


class ScenarioSummary(Wire):
    """One entry in the left sidebar's saved-scenario list."""

    id: str
    # 1-based display index, rendered as "01".."05".
    index: Annotated[int, Field(gt=0)]
    name: str
    location: str
    description: str
    duration_s: NonNeg
    bookmarked: bool
    difficulty: Literal["easy", "moderate", "hard"]
    # Road skeleton for the minimap thumbnail, in thumbnail-local units.
    preview_paths: list[list[Vec2]]
    # Ego route for the minimap thumbnail.
    preview_route: list[Vec2]


class Origin(Wire):
    lat: Num
    lon: Num


class Bounds(Wire):
    min_x: Num
    min_y: Num
    max_x: Num
    max_y: Num


class SceneDescription(Wire):
    type: Literal["scene_description"] = "scene_description"
    protocol: int = PROTOCOL_VERSION
    scene_id: str
    scenario_id: str
    name: str
    # Human-readable neighbourhood, e.g. "Nob Hill".
    location: str
    # ODbL requires crediting OpenStreetMap wherever its data is shown.
    attribution: str
    origin: Origin
    bounds: Bounds
    roads: list[Road]
    buildings: list[Building]
    crosswalks: list[Crosswalk]
    traffic_lights: list[TrafficLight]
    stop_signs: list[StopSign]
    trees: list[Tree]
    street_signs: list[StreetSign]
    # Scenarios the server can load; drives the left sidebar.
    catalog: list[ScenarioSummary]


# --------------------------------------------------------------------------- #
# StateUpdate — streamed at sim rate                                           #
# --------------------------------------------------------------------------- #

DetectionClass = Literal[
    "car", "truck", "bus", "motorcycle", "cyclist", "pedestrian", "unknown"
]


class Detection(Wire):
    id: str
    cls: DetectionClass
    pose: Pose
    size: Size
    # World-frame velocity `[vx, vy]` in m/s.
    velocity: Vec2
    speed_mps: Num
    confidence: Unit
    hazard: bool
    # Shown on the 3D billboard when `hazard` is true.
    hazard_label: str | None
    ttc_s: Num | None
    # Lane index relative to ego: -1 right, 0 same, +1 left, null if unknown.
    lane_offset: int | None


class RadarPoint(Wire):
    id: str | None
    # Ego-frame bearing, radians. 0 = dead ahead, + = left.
    azimuth: Num
    range_m: NonNeg
    # Closing rate; negative means approaching.
    range_rate_mps: Num
    rcs_db: Num
    # True for points associated with a tracked detection (vs clutter).
    tracked: bool


class LaneNeighbor(Wire):
    id: str
    cls: DetectionClass
    # -1 right, 0 same lane, +1 left.
    lane_offset: int
    # Signed distance ahead of ego along the lane.
    longitudinal_m: Num
    # Signed lateral offset from ego's lane centre, + = left.
    lateral_m: Num
    speed_mps: Num
    hazard: bool


class LaneState(Wire):
    lane_index: int
    lane_count: Annotated[int, Field(gt=0)]
    lane_width_m: Pos
    # Ego's signed lateral offset from its lane centre, + = left.
    offset_m: Num
    # Ego heading minus lane heading, radians.
    heading_error: Num
    left_marking: LaneMarking
    right_marking: LaneMarking
    neighbors: list[LaneNeighbor]


class Subsystem(Wire):
    key: str
    label: str
    status: Literal["ok", "warn", "fault"]
    detail: str | None


class VehicleStatus(Wire):
    battery_pct: Annotated[float, Field(ge=0, le=100, allow_inf_nan=False)]
    range_km: NonNeg
    motor_temp_c: Num
    # [FL, FR, RL, RR] in kPa.
    tire_pressure_kpa: tuple[Num, Num, Num, Num]
    subsystems: list[Subsystem]
    overall: Literal["ok", "warn", "fault"]


class TrajectorySample(Wire):
    # Seconds relative to this frame; negative is observed history.
    t: Num
    # Lateral offset from the current lane centre, + = left.
    lateral_m: Num


class TrajectoryPrediction(Wire):
    horizon_s: Pos
    planned: list[TrajectorySample]
    # Predicted path of the cutting-in agent, or null when nobody is cutting in.
    cutin: list[TrajectorySample] | None
    cutin_label: str | None


class Telemetry(Wire):
    radar: list[RadarPoint]
    lane: LaneState
    # Time-to-collision with the most critical object, seconds.
    ttc_s: Num | None
    vehicle: VehicleStatus
    trajectory: TrajectoryPrediction


Maneuver = Literal[
    "keep_lane",
    "turn_left",
    "turn_right",
    "lane_change_left",
    "lane_change_right",
    "stop",
    "yield",
]


class Plan(Wire):
    # Forward plan in world coordinates, ego-first.
    polyline: list[Vec2]
    target_speed_mps: NonNeg
    maneuver: Maneuver
    confidence: Unit


CruiseMode = Literal["off", "cruise", "autosteer", "fsd"]


class Cruise(Wire):
    mode: CruiseMode
    set_speed_mps: NonNeg


class Ego(Wire):
    pose: Pose
    speed_mps: Num
    accel_mps2: Num
    # Road-wheel angle in radians, + = left.
    steering_angle: Num
    yaw_rate: Num
    throttle: Unit
    brake: Unit
    gear: Literal["P", "R", "N", "D"]
    speed_limit_mps: NonNeg
    cruise: Cruise
    size: Size


class SignalState(Wire):
    id: str
    phase: SignalPhase
    time_to_change_s: Num | None


class SimEvent(Wire):
    t: Num
    level: Literal["info", "warn", "critical"]
    code: str
    message: str


class StateUpdate(Wire):
    type: Literal["state_update"] = "state_update"
    protocol: int = PROTOCOL_VERSION
    seq: Annotated[int, Field(ge=0)]
    t: NonNeg
    sim_rate_hz: Pos
    paused: bool
    assist_active: bool
    scenario_id: str
    ego: Ego
    detections: list[Detection]
    plan: Plan
    telemetry: Telemetry
    signals: list[SignalState]
    events: list[SimEvent]


# --------------------------------------------------------------------------- #
# Command — client -> server                                                   #
# --------------------------------------------------------------------------- #

LayerKey = Literal[
    "detections",
    "plan_path",
    "lane_markings",
    "crosswalks",
    "buildings",
    "trees",
    "traffic_lights",
    "radar_cone",
    "labels",
]

CameraView = Literal["chase", "overhead", "cockpit", "free"]

# `z.union([z.number(), z.string(), z.boolean()])`. `bool` precedes the numeric
# types because Python's bool is a subclass of int; pydantic's smart union
# matches exact types first, and `int` precedes `float` so an integral value
# survives the round trip as an integer rather than becoming 35.0.
ParamValue = Union[bool, int, float, str]


class _Cmd(Wire):
    """Every command carries a client-generated id so an Ack can be correlated."""

    id: str


class SetPaused(_Cmd):
    cmd: Literal["set_paused"] = "set_paused"
    paused: bool


class Step(_Cmd):
    cmd: Literal["step"] = "step"
    frames: Annotated[int, Field(gt=0)]


class Reset(_Cmd):
    cmd: Literal["reset"] = "reset"


class LoadScenario(_Cmd):
    cmd: Literal["load_scenario"] = "load_scenario"
    scenario_id: str


class LoadLocation(_Cmd):
    cmd: Literal["load_location"] = "load_location"
    query: Annotated[str, Field(min_length=1)]
    # Absent means "use the location's default". zod `.optional()` allows the
    # key to be missing, unlike `.nullable()` which would require it present.
    radius_m: Pos | None = None


class SetParam(_Cmd):
    cmd: Literal["set_param"] = "set_param"
    key: str
    value: ParamValue


class ToggleLayer(_Cmd):
    cmd: Literal["toggle_layer"] = "toggle_layer"
    layer: LayerKey
    visible: bool


class SetCamera(_Cmd):
    cmd: Literal["set_camera"] = "set_camera"
    view: CameraView


class InjectHazard(_Cmd):
    cmd: Literal["inject_hazard"] = "inject_hazard"
    kind: str


Command = Annotated[
    Union[
        SetPaused,
        Step,
        Reset,
        LoadScenario,
        LoadLocation,
        SetParam,
        ToggleLayer,
        SetCamera,
        InjectHazard,
    ],
    Field(discriminator="cmd"),
]


# --------------------------------------------------------------------------- #
# Ack — server -> client                                                       #
# --------------------------------------------------------------------------- #


class Ack(Wire):
    type: Literal["ack"] = "ack"
    protocol: int = PROTOCOL_VERSION
    # Echoes `Command.id`.
    id: str
    cmd: str
    ok: bool
    message: str | None
    t: Num


# --------------------------------------------------------------------------- #
# Envelope + helpers                                                           #
# --------------------------------------------------------------------------- #

ServerMessage = Annotated[
    Union[SceneDescription, StateUpdate, Ack], Field(discriminator="type")
]

_COMMAND_ADAPTER: TypeAdapter[Any] = TypeAdapter(Command)
_SERVER_MESSAGE_ADAPTER: TypeAdapter[Any] = TypeAdapter(ServerMessage)

T = TypeVar("T")


class ParseResult(BaseModel, Generic[T]):
    """Mirror of the TypeScript `ParseResult<T>` discriminated result."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    ok: bool
    value: T | None = None
    error: str | None = None


def format_issues(err: ValidationError) -> str:
    """Format a validation failure into a single compact line for logs."""
    return "; ".join(
        f"{'.'.join(str(p) for p in issue['loc']) or '<root>'}: {issue['msg']}"
        for issue in err.errors()[:4]
    )


def parse_server_message(raw: Any) -> ParseResult:
    """Validate an outbound server message. Never raises."""
    try:
        return ParseResult(ok=True, value=_SERVER_MESSAGE_ADAPTER.validate_python(raw))
    except ValidationError as err:
        return ParseResult(ok=False, error=format_issues(err))


def parse_command(raw: Any) -> ParseResult:
    """Validate an inbound command. Never raises — a malformed command from the
    wire must degrade to a logged warning and a failed ack, never a crash."""
    try:
        return ParseResult(ok=True, value=_COMMAND_ADAPTER.validate_python(raw))
    except ValidationError as err:
        return ParseResult(ok=False, error=format_issues(err))


# All layer keys, in the order the Layers tab should present them.
LAYER_KEYS: tuple[str, ...] = (
    "detections",
    "plan_path",
    "lane_markings",
    "crosswalks",
    "buildings",
    "trees",
    "traffic_lights",
    "radar_cone",
    "labels",
)
