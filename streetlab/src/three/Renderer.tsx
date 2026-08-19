/**
 * WebGPU viewport.
 *
 * Owns the renderer, the scene graph and the animation loop. React is used only
 * to mount the canvas and show the backend/FPS chip — every per-frame update is
 * imperative and reads `frameBus` directly, so the render loop never waits on
 * React or on the network.
 */
import { useEffect, useRef, useState } from 'react';
import * as THREE from 'three/webgpu';
import {
  color as tslColor,
  float,
  mix,
  positionLocal,
  positionWorld,
  smoothstep,
  uniform,
  uv,
} from 'three/tsl';
import type { CameraView, LayerKey, StateUpdate } from '../schema';
import { perfMetrics } from '../perf/perfMetrics';
import { frameBus, useSimStore } from '../store/simStore';
import type { SimStoreState } from '../store/simStore';
import { color as tokens, lighting } from '../ui/theme';
import type { LightingPreset } from '../ui/theme';
import { buildWorld } from './world';
import type { World } from './world';
import { EgoVehicle } from './ego';
import { TrafficFleet } from './agents';
import { ChaseCamera } from './chaseCam';
import { PathRibbon } from './pathRibbon';
import { HazardOverlay } from './hazardOverlay';
import { createDetectorCamera, DETECTOR_FRAME } from './detectorCamera';

const SKY_RADIUS = 900;
const GROUND_SIZE = 3000;
const SHADOW_EXTENT = 80;

export type Backend = 'webgpu' | 'webgl2';

interface RenderStats {
  backend: Backend;
  fps: number;
  drawCalls: number;
  triangles: number;
}

/* ------------------------------------------------------------------ */
/* Renderer construction                                               */
/* ------------------------------------------------------------------ */

async function createRenderer(
  host: HTMLElement,
): Promise<{ renderer: THREE.WebGPURenderer; canvas: HTMLCanvasElement; backend: Backend }> {
  const supportsWebGPU =
    typeof navigator !== 'undefined' && 'gpu' in navigator;

  const attempt = async (
    forceWebGL: boolean,
  ): Promise<{ renderer: THREE.WebGPURenderer; canvas: HTMLCanvasElement }> => {
    const canvas = document.createElement('canvas');
    canvas.className = 'viewport-canvas';
    host.appendChild(canvas);
    const renderer = new THREE.WebGPURenderer({
      canvas,
      antialias: true,
      alpha: false,
      forceWebGL,
    });
    try {
      await renderer.init();
    } catch (err) {
      canvas.remove();
      renderer.dispose();
      throw err;
    }
    return { renderer, canvas };
  };

  if (supportsWebGPU) {
    try {
      const { renderer, canvas } = await attempt(false);
      const isWebGPU = Boolean(
        (renderer.backend as { isWebGPUBackend?: boolean }).isWebGPUBackend,
      );
      return { renderer, canvas, backend: isWebGPU ? 'webgpu' : 'webgl2' };
    } catch (err) {
      console.warn('[streetlab] WebGPU init failed, falling back to WebGL2', err);
    }
  }

  const { renderer, canvas } = await attempt(true);
  return { renderer, canvas, backend: 'webgl2' };
}

/* ------------------------------------------------------------------ */
/* Static scene furniture                                              */
/* ------------------------------------------------------------------ */

interface SkyRig {
  mesh: THREE.Mesh;
  /** Mutated in place; the uniform node holds a reference to these. */
  skyColor: THREE.Color;
  horizonColor: THREE.Color;
  dispose(): void;
}

function createSky(): SkyRig {
  const geo = new THREE.SphereGeometry(SKY_RADIUS, 32, 16);
  const mat = new THREE.MeshBasicNodeMaterial({
    side: THREE.BackSide,
    depthWrite: false,
    fog: false,
    toneMapped: false,
  });
  const skyColor = new THREE.Color(lighting.midday.sky);
  const horizonColor = new THREE.Color(lighting.midday.horizon);

  // 0 at the nadir, 1 at the zenith.
  const h = positionLocal.y.div(SKY_RADIUS).mul(0.5).add(0.5);
  mat.colorNode = mix(
    uniform(horizonColor),
    uniform(skyColor),
    smoothstep(float(0.5), float(0.82), h),
  );

  const mesh = new THREE.Mesh(geo, mat);
  mesh.name = 'sky';
  mesh.frustumCulled = false;
  mesh.renderOrder = -100;
  return {
    mesh,
    skyColor,
    horizonColor,
    dispose() {
      geo.dispose();
      mat.dispose();
    },
  };
}

interface GroundRig {
  mesh: THREE.Mesh;
  nearColor: THREE.Color;
  farColor: THREE.Color;
  dispose(): void;
}

function createGround(): GroundRig {
  const geo = new THREE.PlaneGeometry(GROUND_SIZE, GROUND_SIZE, 1, 1);
  geo.rotateX(-Math.PI / 2);
  const mat = new THREE.MeshStandardNodeMaterial({
    roughness: 0.98,
    metalness: 0,
  });
  const nearColor = new THREE.Color('#DFE4DC');
  const farColor = new THREE.Color(lighting.midday.horizon);

  // Blend the block colour out to the horizon tint so the plane's edge never
  // reads as a hard line against the sky.
  const dist = positionWorld.xz.length();
  mat.colorNode = mix(
    uniform(nearColor),
    uniform(farColor),
    smoothstep(float(140), float(900), dist),
  );

  const mesh = new THREE.Mesh(geo, mat);
  mesh.name = 'ground';
  mesh.receiveShadow = true;
  mesh.position.y = -0.01;
  return {
    mesh,
    nearColor,
    farColor,
    dispose() {
      geo.dispose();
      mat.dispose();
    },
  };
}

/** Translucent forward sensor fan, anchored to the ego. */
function createRadarCone(): { mesh: THREE.Mesh; dispose(): void } {
  const RANGE = 58;
  const HALF_FOV = Math.PI / 3.6;
  const SEGMENTS = 48;
  const positions: number[] = [0, 0, 0];
  const uvs: number[] = [0, 0.5];
  const index: number[] = [];
  for (let i = 0; i <= SEGMENTS; i++) {
    const a = -HALF_FOV + (2 * HALF_FOV * i) / SEGMENTS;
    positions.push(Math.cos(a) * RANGE, 0, -Math.sin(a) * RANGE);
    uvs.push(1, i / SEGMENTS);
    if (i > 0) index.push(0, i, i + 1);
  }
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
  geo.setAttribute('uv', new THREE.Float32BufferAttribute(uvs, 2));
  geo.setIndex(index);

  const mat = new THREE.MeshBasicNodeMaterial({
    transparent: true,
    depthWrite: false,
    side: THREE.DoubleSide,
    toneMapped: false,
  });
  mat.colorNode = tslColor(tokens.accent);
  // Fade with range, and taper toward the edges of the arc.
  mat.opacityNode = smoothstep(float(1), float(0.05), uv().x)
    .mul(smoothstep(float(0), float(0.14), uv().y))
    .mul(smoothstep(float(1), float(0.86), uv().y))
    .mul(float(0.07));

  const mesh = new THREE.Mesh(geo, mat);
  mesh.name = 'radar-cone';
  mesh.position.y = 0.11;
  mesh.renderOrder = 1;
  mesh.frustumCulled = false;
  return {
    mesh,
    dispose() {
      geo.dispose();
      mat.dispose();
    },
  };
}

/* ------------------------------------------------------------------ */
/* Component                                                           */
/* ------------------------------------------------------------------ */

export function Renderer() {
  const hostRef = useRef<HTMLDivElement>(null);
  const [stats, setStats] = useState<RenderStats | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const host = hostRef.current;
    if (!host) return;

    let disposed = false;
    let cleanup: (() => void) | null = null;

    (async () => {
      let created: Awaited<ReturnType<typeof createRenderer>>;
      try {
        created = await createRenderer(host);
      } catch (err) {
        if (!disposed) setError(String(err));
        return;
      }
      if (disposed) {
        created.canvas.remove();
        created.renderer.dispose();
        return;
      }
      // The desktop shell has no visible console, so this line is the record of
      // which path the machine actually took. It is mirrored in the on-screen
      // stats chip.
      console.info(
        `[streetlab] renderer backend: ${created.backend}` +
          (created.backend === 'webgl2' ? ' (WebGPU unavailable — fell back)' : ''),
      );
      cleanup = mount(host, created, setStats);
    })();

    return () => {
      disposed = true;
      cleanup?.();
    };
  }, []);

  return (
    <div className="viewport" ref={hostRef}>
      {error && (
        <div className="viewport-error" role="alert">
          <strong>3D renderer unavailable</strong>
          <span>{error}</span>
        </div>
      )}
      {stats && (
        <div className="viewport-stats" title="Renderer backend, frame rate and draw calls">
          <span className={`stat-backend stat-backend--${stats.backend}`}>
            {stats.backend === 'webgpu' ? 'WebGPU' : 'WebGL2'}
          </span>
          <span>{stats.fps} fps</span>
          <span>{stats.drawCalls} draws</span>
        </div>
      )}
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Imperative mount                                                    */
/* ------------------------------------------------------------------ */

function mount(
  host: HTMLElement,
  created: { renderer: THREE.WebGPURenderer; canvas: HTMLCanvasElement; backend: Backend },
  setStats: (s: RenderStats) => void,
): () => void {
  const { renderer, canvas, backend } = created;

  renderer.setPixelRatio(Math.min(2, window.devicePixelRatio || 1));
  renderer.toneMapping = THREE.NeutralToneMapping;
  renderer.toneMappingExposure = 1.05;
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;

  const scene = new THREE.Scene();
  const sky = createSky();
  const ground = createGround();
  const radar = createRadarCone();
  scene.add(sky.mesh, ground.mesh);

  const preset0 = lighting.midday;
  scene.fog = new THREE.Fog(new THREE.Color(preset0.horizon), 190, 780);

  const hemi = new THREE.HemisphereLight(
    new THREE.Color(preset0.sky),
    new THREE.Color(preset0.ground),
    preset0.ambient,
  );
  const sun = new THREE.DirectionalLight(
    new THREE.Color(preset0.sun),
    preset0.intensity,
  );
  sun.castShadow = true;
  sun.shadow.mapSize.set(2048, 2048);
  sun.shadow.camera.near = 1;
  sun.shadow.camera.far = 320;
  sun.shadow.camera.left = -SHADOW_EXTENT;
  sun.shadow.camera.right = SHADOW_EXTENT;
  sun.shadow.camera.top = SHADOW_EXTENT;
  sun.shadow.camera.bottom = -SHADOW_EXTENT;
  sun.shadow.bias = -0.0008;
  sun.shadow.normalBias = 0.06;
  const sunTarget = new THREE.Object3D();
  sun.target = sunTarget;
  scene.add(hemi, sun, sunTarget);

  const cam = new ChaseCamera(host.clientWidth / Math.max(1, host.clientHeight));
  // Deliberately a separate camera from `cam`: switching the user's view
  // between chase/overhead/cockpit/free must never change what perception
  // sees, nor the rate frames are emitted at.
  const detectorCamera = createDetectorCamera(scene, renderer);
  const ego = new EgoVehicle({ length: 4.9, width: 1.96, height: 1.44 });
  const fleet = new TrafficFleet();
  const ribbon = new PathRibbon();
  const hazards = new HazardOverlay();
  ego.group.add(radar.mesh);
  scene.add(ego.group, fleet.group, ribbon.mesh, hazards.group);

  let world: World | null = null;
  let builtEpoch = -1;
  let sunDir = new THREE.Vector3(0.6, 0.8, 0.4);
  // The merged building mesh (see world.ts), reused as the chase camera's
  // occlusion geometry so it never has to build its own spatial structure.
  // Captured once per scene build rather than looked up every frame.
  let buildings: THREE.Object3D | null = null;

  /* ---- store wiring (imperative, no React re-render) ---- */

  const applyLighting = (presetName: string) => {
    const preset =
      lighting[(presetName as LightingPreset) in lighting
        ? (presetName as LightingPreset)
        : 'midday'];
    sky.skyColor.set(preset.sky);
    sky.horizonColor.set(preset.horizon);
    ground.farColor.set(preset.horizon);
    (scene.fog as THREE.Fog).color.set(preset.horizon);
    hemi.color.set(preset.sky);
    hemi.groundColor.set(preset.ground);
    hemi.intensity = preset.ambient;
    sun.color.set(preset.sun);
    sun.intensity = preset.intensity;
    sunDir = new THREE.Vector3(
      Math.cos(preset.azim) * Math.cos(preset.elev),
      Math.sin(preset.elev),
      Math.sin(preset.azim) * Math.cos(preset.elev),
    ).normalize();
  };

  const applyLayers = (layers: Record<LayerKey, boolean>) => {
    world?.setLayerVisible('lane_markings', layers.lane_markings);
    world?.setLayerVisible('crosswalks', layers.crosswalks);
    world?.setLayerVisible('buildings', layers.buildings);
    world?.setLayerVisible('trees', layers.trees);
    world?.setLayerVisible('traffic_lights', layers.traffic_lights);
    world?.setLayerVisible('labels', layers.labels);
    fleet.setVisible(layers.detections);
    hazards.setVisible(layers.detections);
    hazards.setLabelsVisible(layers.labels);
    ribbon.setVisible(layers.plan_path);
    radar.mesh.visible = layers.radar_cone;
  };

  const applyParams = (params: Record<string, unknown>) => {
    ribbon.setOpacity(Number(params.plan_opacity ?? 0.55));
    hazards.setLabelScale(Number(params.label_scale ?? 1));
    hazards.setAccent(String(params.hazard_color ?? tokens.warn));
    applyLighting(String(params.time_of_day ?? 'midday'));
  };

  const rebuildWorld = (state: SimStoreState) => {
    if (!state.scene || state.sceneEpoch === builtEpoch) return;
    world?.dispose();
    if (world) scene.remove(world.root);
    world = buildWorld(state.scene);
    scene.add(world.root);
    builtEpoch = state.sceneEpoch;
    buildings = world.root.getObjectByName('buildings') ?? null;
    applyLayers(state.layers);
    cameraReset = true;
  };

  let cameraReset = true;
  let cameraView: CameraView = useSimStore.getState().cameraView;

  const unsubscribe = useSimStore.subscribe((state, prev) => {
    if (state.sceneEpoch !== prev.sceneEpoch) rebuildWorld(state);
    if (state.layers !== prev.layers) applyLayers(state.layers);
    if (state.params !== prev.params) applyParams(state.params);
    if (state.cameraView !== prev.cameraView) cameraView = state.cameraView;
  });

  {
    const s = useSimStore.getState();
    rebuildWorld(s);
    applyParams(s.params);
    applyLayers(s.layers);
  }

  /* ---- resize ---- */

  const resize = () => {
    const w = Math.max(1, host.clientWidth);
    const h = Math.max(1, host.clientHeight);
    renderer.setSize(w, h, false);
    cam.setAspect(w / h);
  };
  resize();
  const ro = new ResizeObserver(resize);
  ro.observe(host);

  /* ---- free-view pointer input ---- */

  let dragging = false;
  let lastX = 0;
  let lastY = 0;
  const onPointerDown = (e: PointerEvent) => {
    if (cameraView !== 'free') return;
    dragging = true;
    lastX = e.clientX;
    lastY = e.clientY;
    canvas.setPointerCapture(e.pointerId);
  };
  const onPointerMove = (e: PointerEvent) => {
    if (!dragging) return;
    cam.orbit(e.clientX - lastX, e.clientY - lastY);
    lastX = e.clientX;
    lastY = e.clientY;
  };
  const onPointerUp = (e: PointerEvent) => {
    dragging = false;
    if (canvas.hasPointerCapture(e.pointerId)) {
      canvas.releasePointerCapture(e.pointerId);
    }
  };
  const onWheel = (e: WheelEvent) => {
    if (cameraView !== 'free') return;
    e.preventDefault();
    cam.zoom(e.deltaY);
  };
  canvas.addEventListener('pointerdown', onPointerDown);
  canvas.addEventListener('pointermove', onPointerMove);
  canvas.addEventListener('pointerup', onPointerUp);
  canvas.addEventListener('pointercancel', onPointerUp);
  canvas.addEventListener('wheel', onWheel, { passive: false });

  /* ---- animation loop ---- */

  let lastTime = performance.now();
  let fpsAccum = 0;
  let fpsFrames = 0;
  let lastStatsAt = lastTime;
  let lastSeq = -1;

  // Detector-camera throttle. Accumulated (not reset to zero) so a slow or
  // variable display frame time doesn't drift the emitted rate away from the
  // fixed ~10 Hz — the same accumulator pattern MockSim uses for its own
  // fixed-rate loop, just against wall-clock dt instead of a fixed DT.
  let sinceCaptureMs = 0;
  let captureSeq = 0;

  const applyFrame = (frame: StateUpdate, dt: number) => {
    ego.setPose(frame.ego.pose);
    ego.setAttitude(frame.ego.steering_angle, frame.ego.accel_mps2);
    fleet.update(frame.detections, dt);
    ribbon.update(frame.plan.polyline);
    hazards.update(frame.detections, cam.camera);
    world?.updateSignals(frame.signals, frame.t);

    // Keep the shadow frustum centred on the car so a 160 m box is enough.
    sunTarget.position.set(frame.ego.pose.x, 0, -frame.ego.pose.y);
    sun.position.copy(sunTarget.position).addScaledVector(sunDir, 150);
  };

  renderer.setAnimationLoop(() => {
    const now = performance.now();
    const dt = Math.min(0.1, (now - lastTime) / 1000);
    lastTime = now;

    const frame = frameBus.latest;
    if (frame) {
      if (cameraReset) {
        cam.reset(frame.ego.pose, buildings);
        cameraReset = false;
      }
      // Re-run scene-graph updates every display frame even if the simulator
      // has not produced a new one: damping still has work to do.
      applyFrame(frame, dt);
      cam.update(frame.ego.pose, frame.ego.speed_mps, cameraView, dt, buildings);
      lastSeq = frame.seq;

      // Detector capture: driven off the ego pose directly, never off `cam`
      // (the user's view camera) or `cameraView`, so this is unaffected by
      // which view the user has selected.
      sinceCaptureMs += dt * 1000;
      if (sinceCaptureMs >= DETECTOR_FRAME.intervalMs) {
        sinceCaptureMs -= DETECTOR_FRAME.intervalMs;
        detectorCamera.update({
          x: frame.ego.pose.x,
          // Three.js is Y-up with +z south; the wire pose is +y north. See
          // detectorCamera.ts's own conversion, and the identical z = -y used
          // for the sun target above.
          z: -frame.ego.pose.y,
          heading: frame.ego.pose.heading,
        });
        const capturedAtT = frame.t;
        const seq = captureSeq++;
        void detectorCamera
          .capture()
          .then((captured) => {
            if (!captured) return;
            useSimStore.getState().send({
              cmd: 'camera_frame',
              seq,
              t: capturedAtT,
              width: DETECTOR_FRAME.width,
              height: DETECTOR_FRAME.height,
              format: 'jpeg',
              data: captured.data,
              camera: captured.camera,
            });
          })
          .catch((err) => {
            // GPU readback can fail transiently (e.g. context loss); never let
            // that become an unhandled rejection that takes down the loop.
            console.warn('[streetlab] detector camera capture failed', err);
          });
      }
    }

    renderer.render(scene, cam.camera);

    fpsFrames++;
    fpsAccum += dt;
    if (now - lastStatsAt > 500) {
      const fps = Math.round(fpsFrames / Math.max(1e-3, fpsAccum));
      setStats({
        backend,
        fps,
        drawCalls: renderer.info.render.drawCalls,
        triangles: renderer.info.render.triangles,
      });
      perfMetrics.reportFps(fps);
      fpsFrames = 0;
      fpsAccum = 0;
      lastStatsAt = now;
    }
  });

  void lastSeq;

  return () => {
    renderer.setAnimationLoop(null);
    unsubscribe();
    ro.disconnect();
    canvas.removeEventListener('pointerdown', onPointerDown);
    canvas.removeEventListener('pointermove', onPointerMove);
    canvas.removeEventListener('pointerup', onPointerUp);
    canvas.removeEventListener('pointercancel', onPointerUp);
    canvas.removeEventListener('wheel', onWheel);
    world?.dispose();
    ego.dispose();
    fleet.dispose();
    ribbon.dispose();
    hazards.dispose();
    sky.dispose();
    ground.dispose();
    radar.dispose();
    detectorCamera.dispose();
    renderer.dispose();
    canvas.remove();
  };
}
