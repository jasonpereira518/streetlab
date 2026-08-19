/**
 * StreetLab wire schema.
 *
 * This file is the single source of truth for the frontend<->simulator
 * protocol. The backend (a separate Python process) will emit exactly these
 * messages over a WebSocket; the in-process mock emits the same shapes so the
 * UI never learns whether it is talking to a real simulator or not.
 *
 * Conventions
 *   - Distances in metres, speeds in m/s, accelerations in m/s^2.
 *   - Angles in radians. Headings are 0 at +x (east) and increase CCW.
 *   - World is a right-handed 2D plane: +x east, +y north. The renderer maps
 *     this to three.js as (x, 0, -y) so that +y (north) points at -z.
 *   - Time `t` is simulator seconds since scenario start.
 */
import { z } from 'zod';

export const PROTOCOL_VERSION = 3;

/* ------------------------------------------------------------------ */
/* Primitives                                                          */
/* ------------------------------------------------------------------ */

/** `[x, y]` in world metres. Tuples keep 60 Hz polyline payloads compact. */
export const Vec2Schema = z.tuple([z.number(), z.number()]);

export const PoseSchema = z.object({
  x: z.number(),
  y: z.number(),
  /** radians, 0 = +x (east), CCW positive */
  heading: z.number(),
});

export const SizeSchema = z.object({
  length: z.number().positive(),
  width: z.number().positive(),
  height: z.number().positive(),
});

const HexColorSchema = z.string().regex(/^#[0-9a-fA-F]{6}$/, 'expected #rrggbb');

/* ------------------------------------------------------------------ */
/* SceneDescription — static world, sent once per scenario load        */
/* ------------------------------------------------------------------ */

export const SignalPhaseSchema = z.enum([
  'red',
  'yellow',
  'green',
  'flashing_yellow',
  'off',
]);

export const RoadClassSchema = z.enum([
  'arterial',
  'collector',
  'residential',
  'service',
]);

export const LaneMarkingSchema = z.enum([
  'none',
  'dashed_white',
  'solid_white',
  'double_yellow',
]);

export const RoadSchema = z.object({
  id: z.string(),
  name: z.string(),
  road_class: RoadClassSchema,
  /** Ordered polyline down the middle of the carriageway. */
  centerline: z.array(Vec2Schema).min(2),
  lanes_forward: z.number().int().min(0),
  lanes_backward: z.number().int().min(0),
  lane_width_m: z.number().positive(),
  speed_limit_mps: z.number().nonnegative(),
  oneway: z.boolean(),
  /** Marking drawn on the centre divider. */
  center_marking: LaneMarkingSchema,
  has_sidewalk: z.boolean(),
});

export const BuildingSchema = z.object({
  id: z.string(),
  /** CCW footprint ring; not closed (first point is not repeated). */
  footprint: z.array(Vec2Schema).min(3),
  height_m: z.number().positive(),
  color: HexColorSchema,
  roof_color: HexColorSchema,
});

export const CrosswalkSchema = z.object({
  id: z.string(),
  center: Vec2Schema,
  /** Direction pedestrians walk, radians. */
  heading: z.number(),
  /** Depth of the striped band, measured perpendicular to `heading`. */
  width_m: z.number().positive(),
  /** Extent along `heading` — i.e. the carriageway width being crossed. */
  length_m: z.number().positive(),
  style: z.enum(['ladder', 'continental', 'transverse']),
});

export const TrafficLightSchema = z.object({
  id: z.string(),
  position: Vec2Schema,
  /** Direction the lamp faces — i.e. toward the traffic it governs. */
  heading: z.number(),
  /** Horizontal mast-arm reach; 0 means a pole-mounted head. */
  mast_arm_m: z.number().nonnegative(),
  height_m: z.number().positive(),
});

export const StopSignSchema = z.object({
  id: z.string(),
  position: Vec2Schema,
  heading: z.number(),
});

export const TreeSchema = z.object({
  id: z.string(),
  position: Vec2Schema,
  height_m: z.number().positive(),
  canopy_radius_m: z.number().positive(),
  trunk_radius_m: z.number().positive(),
  /** 0..1 hue jitter so a forest of clones does not look like one. */
  variant: z.number().min(0).max(1),
});

export const StreetSignSchema = z.object({
  id: z.string(),
  position: Vec2Schema,
  heading: z.number(),
  text: z.string(),
  kind: z.enum(['street_name', 'speed_limit', 'no_parking']),
});

/** One entry in the left sidebar's saved-scenario list. */
export const ScenarioSummarySchema = z.object({
  id: z.string(),
  /** 1-based display index, rendered as "01".."05". */
  index: z.number().int().positive(),
  name: z.string(),
  location: z.string(),
  description: z.string(),
  duration_s: z.number().nonnegative(),
  bookmarked: z.boolean(),
  difficulty: z.enum(['easy', 'moderate', 'hard']),
  /** Road skeleton for the minimap thumbnail, in thumbnail-local units. */
  preview_paths: z.array(z.array(Vec2Schema)),
  /** Ego route for the minimap thumbnail. */
  preview_route: z.array(Vec2Schema),
});

export const SceneDescriptionSchema = z.object({
  type: z.literal('scene_description'),
  protocol: z.number().int(),
  scene_id: z.string(),
  scenario_id: z.string(),
  name: z.string(),
  /** Human-readable neighbourhood, e.g. "Nob Hill". */
  location: z.string(),
  // ODbL requires crediting OpenStreetMap wherever its data is shown.
  attribution: z.string(),
  origin: z.object({ lat: z.number(), lon: z.number() }),
  bounds: z.object({
    min_x: z.number(),
    min_y: z.number(),
    max_x: z.number(),
    max_y: z.number(),
  }),
  roads: z.array(RoadSchema),
  buildings: z.array(BuildingSchema),
  crosswalks: z.array(CrosswalkSchema),
  traffic_lights: z.array(TrafficLightSchema),
  stop_signs: z.array(StopSignSchema),
  trees: z.array(TreeSchema),
  street_signs: z.array(StreetSignSchema),
  /** Scenarios the server can load; drives the left sidebar. */
  catalog: z.array(ScenarioSummarySchema),
});

/* ------------------------------------------------------------------ */
/* StateUpdate — streamed at sim rate                                  */
/* ------------------------------------------------------------------ */

export const DetectionClassSchema = z.enum([
  'car',
  'truck',
  'bus',
  'motorcycle',
  'cyclist',
  'pedestrian',
  'unknown',
]);

export const DetectionSchema = z.object({
  id: z.string(),
  cls: DetectionClassSchema,
  pose: PoseSchema,
  size: SizeSchema,
  /** World-frame velocity `[vx, vy]` in m/s. */
  velocity: Vec2Schema,
  speed_mps: z.number(),
  confidence: z.number().min(0).max(1),
  hazard: z.boolean(),
  /** Shown on the 3D billboard when `hazard` is true. */
  hazard_label: z.string().nullable(),
  ttc_s: z.number().nullable(),
  /** Lane index relative to ego: -1 right, 0 same, +1 left, null if unknown. */
  lane_offset: z.number().int().nullable(),
});

export const PerceptionModeSchema = z.enum(['ground-truth', 'ml']);

/**
 * The camera that produced one frame, in WIRE world coordinates:
 * `+x` east, `+y` north, `+z` up, ground plane at `z = 0`.
 * The frontend converts out of Three.js's Y-up frame before sending, so the
 * backend never learns that a renderer convention exists.
 */
export const CameraParamsSchema = z.object({
  x: z.number(),
  y: z.number(),
  z: z.number(),
  /** radians, 0 = +x (east), CCW positive — same convention as Pose.heading */
  yaw: z.number(),
  pitch: z.number(),
  roll: z.number(),
  fov_y_deg: z.number().positive(),
  aspect: z.number().positive(),
});

/** Transport and quality numbers for the ML perception path. */
export const PerceptionStatsSchema = z.object({
  mode: PerceptionModeSchema,
  /** Model inference time. Null until Phase 2 lands a model. */
  detector_ms: z.number().nonnegative().nullable(),
  /** Frame render -> detections available. */
  e2e_ms: z.number().nonnegative().nullable(),
  frames_received: z.number().int().nonnegative(),
  frames_dropped: z.number().int().nonnegative(),
  /** Quality fields stay null until scoring lands in Phase 3. */
  precision: z.number().min(0).max(1).nullable(),
  recall: z.number().min(0).max(1).nullable(),
  mean_pos_err_m: z.number().nonnegative().nullable(),
});

export const RadarPointSchema = z.object({
  id: z.string().nullable(),
  /** Ego-frame bearing, radians. 0 = dead ahead, + = left. */
  azimuth: z.number(),
  range_m: z.number().nonnegative(),
  /** Closing rate; negative means approaching. */
  range_rate_mps: z.number(),
  rcs_db: z.number(),
  /** True for points associated with a tracked detection (vs clutter). */
  tracked: z.boolean(),
});

export const LaneNeighborSchema = z.object({
  id: z.string(),
  cls: DetectionClassSchema,
  /** -1 right, 0 same lane, +1 left. */
  lane_offset: z.number().int(),
  /** Signed distance ahead of ego along the lane. */
  longitudinal_m: z.number(),
  /** Signed lateral offset from ego's lane centre, + = left. */
  lateral_m: z.number(),
  speed_mps: z.number(),
  hazard: z.boolean(),
});

export const LaneStateSchema = z.object({
  lane_index: z.number().int(),
  lane_count: z.number().int().positive(),
  lane_width_m: z.number().positive(),
  /** Ego's signed lateral offset from its lane centre, + = left. */
  offset_m: z.number(),
  /** Ego heading minus lane heading, radians. */
  heading_error: z.number(),
  left_marking: LaneMarkingSchema,
  right_marking: LaneMarkingSchema,
  neighbors: z.array(LaneNeighborSchema),
});

export const SubsystemSchema = z.object({
  key: z.string(),
  label: z.string(),
  status: z.enum(['ok', 'warn', 'fault']),
  detail: z.string().nullable(),
});

export const VehicleStatusSchema = z.object({
  battery_pct: z.number().min(0).max(100),
  range_km: z.number().nonnegative(),
  motor_temp_c: z.number(),
  /** [FL, FR, RL, RR] in kPa. */
  tire_pressure_kpa: z.tuple([
    z.number(),
    z.number(),
    z.number(),
    z.number(),
  ]),
  subsystems: z.array(SubsystemSchema),
  overall: z.enum(['ok', 'warn', 'fault']),
});

export const TrajectorySampleSchema = z.object({
  /** Seconds relative to this frame; negative is observed history. */
  t: z.number(),
  /** Lateral offset from the current lane centre, + = left. */
  lateral_m: z.number(),
});

export const TrajectoryPredictionSchema = z.object({
  horizon_s: z.number().positive(),
  planned: z.array(TrajectorySampleSchema),
  /** Predicted path of the cutting-in agent, or null when nobody is cutting in. */
  cutin: z.array(TrajectorySampleSchema).nullable(),
  cutin_label: z.string().nullable(),
});

export const TelemetrySchema = z.object({
  radar: z.array(RadarPointSchema),
  lane: LaneStateSchema,
  /** Time-to-collision with the most critical object, seconds. */
  ttc_s: z.number().nullable(),
  vehicle: VehicleStatusSchema,
  trajectory: TrajectoryPredictionSchema,
});

export const ManeuverSchema = z.enum([
  'keep_lane',
  'turn_left',
  'turn_right',
  'lane_change_left',
  'lane_change_right',
  'stop',
  'yield',
]);

export const PlanSchema = z.object({
  /** Forward plan in world coordinates, ego-first. */
  polyline: z.array(Vec2Schema),
  target_speed_mps: z.number().nonnegative(),
  maneuver: ManeuverSchema,
  confidence: z.number().min(0).max(1),
});

export const CruiseModeSchema = z.enum(['off', 'cruise', 'autosteer', 'fsd']);

export const EgoSchema = z.object({
  pose: PoseSchema,
  speed_mps: z.number(),
  accel_mps2: z.number(),
  /** Road-wheel angle in radians, + = left. */
  steering_angle: z.number(),
  yaw_rate: z.number(),
  throttle: z.number().min(0).max(1),
  brake: z.number().min(0).max(1),
  gear: z.enum(['P', 'R', 'N', 'D']),
  speed_limit_mps: z.number().nonnegative(),
  cruise: z.object({
    mode: CruiseModeSchema,
    set_speed_mps: z.number().nonnegative(),
  }),
  size: SizeSchema,
});

export const SignalStateSchema = z.object({
  id: z.string(),
  phase: SignalPhaseSchema,
  time_to_change_s: z.number().nullable(),
});

export const SimEventSchema = z.object({
  t: z.number(),
  level: z.enum(['info', 'warn', 'critical']),
  code: z.string(),
  message: z.string(),
});

export const StateUpdateSchema = z.object({
  type: z.literal('state_update'),
  protocol: z.number().int(),
  seq: z.number().int().nonnegative(),
  t: z.number().nonnegative(),
  sim_rate_hz: z.number().positive(),
  paused: z.boolean(),
  assist_active: z.boolean(),
  scenario_id: z.string(),
  ego: EgoSchema,
  detections: z.array(DetectionSchema),
  plan: PlanSchema,
  telemetry: TelemetrySchema,
  signals: z.array(SignalStateSchema),
  events: z.array(SimEventSchema),
  /** Null when no ML perception is running — distinct from "measured, and zero". */
  perception: PerceptionStatsSchema.nullable(),
});

/* ------------------------------------------------------------------ */
/* Command — client -> server                                          */
/* ------------------------------------------------------------------ */

export const LayerKeySchema = z.enum([
  'detections',
  'plan_path',
  'lane_markings',
  'crosswalks',
  'buildings',
  'trees',
  'traffic_lights',
  'radar_cone',
  'labels',
]);

export const CameraViewSchema = z.enum(['chase', 'overhead', 'cockpit', 'free']);

export const ParamValueSchema = z.union([
  z.number(),
  z.string(),
  z.boolean(),
]);

/** Every command carries a client-generated id so an Ack can be correlated. */
const cmd = <S extends z.ZodRawShape>(shape: S) =>
  z.object({ id: z.string(), ...shape });

export const CommandSchema = z.discriminatedUnion('cmd', [
  cmd({ cmd: z.literal('set_paused'), paused: z.boolean() }),
  cmd({ cmd: z.literal('step'), frames: z.number().int().positive() }),
  cmd({ cmd: z.literal('reset') }),
  cmd({ cmd: z.literal('load_scenario'), scenario_id: z.string() }),
  cmd({
    cmd: z.literal('load_location'),
    query: z.string().min(1),
    radius_m: z.number().positive().optional(),
  }),
  cmd({
    cmd: z.literal('set_param'),
    key: z.string(),
    value: ParamValueSchema,
  }),
  cmd({
    cmd: z.literal('toggle_layer'),
    layer: LayerKeySchema,
    visible: z.boolean(),
  }),
  cmd({ cmd: z.literal('set_camera'), view: CameraViewSchema }),
  cmd({ cmd: z.literal('inject_hazard'), kind: z.string() }),
  cmd({ cmd: z.literal('set_perception'), mode: PerceptionModeSchema }),
  cmd({
    cmd: z.literal('camera_frame'),
    /** Monotonic per connection; the backend drops anything out of order. */
    seq: z.number().int().nonnegative(),
    /** Sim seconds the frame depicts. */
    t: z.number(),
    width: z.number().int().positive(),
    height: z.number().int().positive(),
    format: z.literal('jpeg'),
    /** base64. Capped: an uncapped field here is an OOM waiting for a bad client. */
    data: z.string().max(524288),
    camera: CameraParamsSchema,
  }),
]);

/* ------------------------------------------------------------------ */
/* Ack — server -> client                                              */
/* ------------------------------------------------------------------ */

export const AckSchema = z.object({
  type: z.literal('ack'),
  protocol: z.number().int(),
  /** Echoes `Command.id`. */
  id: z.string(),
  cmd: z.string(),
  ok: z.boolean(),
  message: z.string().nullable(),
  t: z.number(),
});

/* ------------------------------------------------------------------ */
/* Envelope + helpers                                                  */
/* ------------------------------------------------------------------ */

export const ServerMessageSchema = z.discriminatedUnion('type', [
  SceneDescriptionSchema,
  StateUpdateSchema,
  AckSchema,
]);

/* ---- inferred types ---- */

export type Vec2 = z.infer<typeof Vec2Schema>;
export type Pose = z.infer<typeof PoseSchema>;
export type Size = z.infer<typeof SizeSchema>;
export type SignalPhase = z.infer<typeof SignalPhaseSchema>;
export type RoadClass = z.infer<typeof RoadClassSchema>;
export type LaneMarking = z.infer<typeof LaneMarkingSchema>;
export type Road = z.infer<typeof RoadSchema>;
export type Building = z.infer<typeof BuildingSchema>;
export type Crosswalk = z.infer<typeof CrosswalkSchema>;
export type TrafficLight = z.infer<typeof TrafficLightSchema>;
export type StopSign = z.infer<typeof StopSignSchema>;
export type Tree = z.infer<typeof TreeSchema>;
export type StreetSign = z.infer<typeof StreetSignSchema>;
export type ScenarioSummary = z.infer<typeof ScenarioSummarySchema>;
export type SceneDescription = z.infer<typeof SceneDescriptionSchema>;

export type DetectionClass = z.infer<typeof DetectionClassSchema>;
export type Detection = z.infer<typeof DetectionSchema>;
export type PerceptionMode = z.infer<typeof PerceptionModeSchema>;
export type CameraParams = z.infer<typeof CameraParamsSchema>;
export type PerceptionStats = z.infer<typeof PerceptionStatsSchema>;
export type RadarPoint = z.infer<typeof RadarPointSchema>;
export type LaneNeighbor = z.infer<typeof LaneNeighborSchema>;
export type LaneState = z.infer<typeof LaneStateSchema>;
export type Subsystem = z.infer<typeof SubsystemSchema>;
export type VehicleStatus = z.infer<typeof VehicleStatusSchema>;
export type TrajectorySample = z.infer<typeof TrajectorySampleSchema>;
export type TrajectoryPrediction = z.infer<typeof TrajectoryPredictionSchema>;
export type Telemetry = z.infer<typeof TelemetrySchema>;
export type Maneuver = z.infer<typeof ManeuverSchema>;
export type Plan = z.infer<typeof PlanSchema>;
export type CruiseMode = z.infer<typeof CruiseModeSchema>;
export type Ego = z.infer<typeof EgoSchema>;
export type SignalState = z.infer<typeof SignalStateSchema>;
export type SimEvent = z.infer<typeof SimEventSchema>;
export type StateUpdate = z.infer<typeof StateUpdateSchema>;

export type LayerKey = z.infer<typeof LayerKeySchema>;
export type CameraView = z.infer<typeof CameraViewSchema>;
export type ParamValue = z.infer<typeof ParamValueSchema>;
export type Command = z.infer<typeof CommandSchema>;

/**
 * A command as callers write it: the correlation `id` is filled in by the store,
 * so UI code only supplies the payload. Distributes over the union so each
 * member keeps its own required fields.
 */
export type CommandInput = Command extends infer C
  ? C extends { id: string }
    ? Omit<C, 'id'> & { id?: string }
    : never
  : never;
export type Ack = z.infer<typeof AckSchema>;
export type ServerMessage = z.infer<typeof ServerMessageSchema>;

export type ParseResult<T> =
  | { ok: true; value: T }
  | { ok: false; error: string };

/** Format a zod failure into a single compact line for logs/UI. */
export function formatIssues(err: z.ZodError): string {
  return err.issues
    .slice(0, 4)
    .map((i) => `${i.path.join('.') || '<root>'}: ${i.message}`)
    .join('; ');
}

/**
 * Validate an inbound server message. Never throws — transports call this on
 * every frame and a malformed frame must degrade to a logged warning rather
 * than tearing down the socket.
 */
export function parseServerMessage(raw: unknown): ParseResult<ServerMessage> {
  const res = ServerMessageSchema.safeParse(raw);
  return res.success
    ? { ok: true, value: res.data }
    : { ok: false, error: formatIssues(res.error) };
}

/** Validate an outbound command before it hits the wire. */
export function parseCommand(raw: unknown): ParseResult<Command> {
  const res = CommandSchema.safeParse(raw);
  return res.success
    ? { ok: true, value: res.data }
    : { ok: false, error: formatIssues(res.error) };
}

/** All layer keys, in the order the Layers tab should present them. */
export const LAYER_KEYS = LayerKeySchema.options;
