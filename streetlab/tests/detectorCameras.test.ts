/**
 * The detector camera layouts the wire names (`PerceptionStats.camera_set`) are pinned to
 * contract/detector_cameras.json, which the backend's tests/test_detector_cameras.py reads
 * too: the renderer's cameras and the backend's model of them cannot drift apart.
 */
import { describe, expect, it } from 'vitest';
import pin from '../../contract/detector_cameras.json';
import { cameraParamsFromThree, DETECTOR_CAMERA_SETS, MOUNT_PITCH_RAD } from '../src/three/detectorCamera';

describe('detector camera sets', () => {
  it('has exactly the layouts the contract names', () => {
    expect(Object.keys(DETECTOR_CAMERA_SETS).sort()).toEqual(Object.keys(pin.sets).sort());
  });

  for (const [name, cams] of Object.entries(pin.sets)) {
    it(`${name} matches the pin`, () => {
      const ours = DETECTOR_CAMERA_SETS[name as keyof typeof DETECTOR_CAMERA_SETS];
      expect(ours.length).toBe(cams.length);
      cams.forEach((c, i) => {
        expect(ours[i].name).toBe(c.name);
        expect((ours[i].yawRad * 180) / Math.PI).toBeCloseTo(c.yaw_deg, 9);
        expect(ours[i].fovYDeg).toBeCloseTo(c.fov_y_deg, 9);
        expect([ours[i].width, ours[i].height]).toEqual([c.width, c.height]);
      });
    });
  }

  it('reports each camera\'s own yaw, fov and aspect on the wire', () => {
    const left = DETECTOR_CAMERA_SETS['front+sides100'][1];
    const p = cameraParamsFromThree({ x: 1, y: 1.33, z: -2 }, 0.5, MOUNT_PITCH_RAD, left);
    expect(p.yaw).toBeCloseTo(0.5 + (80 * Math.PI) / 180, 12);
    expect(p.fov_y_deg).toBe(100);
    expect(p.aspect).toBe(1);
    expect(p.pitch).toBe(MOUNT_PITCH_RAD);
    // The front camera's call (three arguments) is unchanged.
    expect(cameraParamsFromThree({ x: 1, y: 1.33, z: -2 }, 0.5, MOUNT_PITCH_RAD).yaw).toBe(0.5);
  });
});
