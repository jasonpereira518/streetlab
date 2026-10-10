# ML camera coverage study (headless, 2026-10-09)

Question: the front detector camera sees about +-38 deg; the ground-truth driving feed sees +-75 deg plus a 15 m shoulder check. Gate S's ML-attributable collisions all had the other vehicle outside the front camera. Which cheapest camera layout closes them?

Method: `NoisyTruthPerception(cameras=...)` (`perception/noisy_truth.py` `CAMERA_SETS`), nominal noise unchanged, 15 cells (`scripts/gate_s.py run --fov`): the 9 cells whose collision was a field-of-view limit or GT-shared, `nobhill_slow` seed 4 `red_light_runner`, three hazard-free controls and the two hazard-free runs that still fire AEB. Ground truth rows are the full run's (`cells_c`), noisy rows are re-run per layout. Pixel size is modelled: a camera with fewer pixels per radian (`f_px`) gets proportionally larger range sigma (it falls out of the projection and the localizer) and its p(detect) is evaluated at `dist x REFERENCE_F_PX / f_px`. Observations from different cameras that agree within their combined 95 % covariance are fused into one before the tracker (`ml_source.fuse_cameras`). The criteria are Gate S's, applied to this subset only (so "pass" here is not Gate S).

| layout | cameras | inference cost | collisions (GT same cells: 5) | remaining collisions: bearing 1 s before | c2 | c3 AEB (hazard-free) | c4 | c5 | c7 min |
|---|---|---|---|---|---|---|---|---|---|
| front (baseline) | 1 x 640x384, 75.7 deg | 1.0x | 10 | grid cut_in s4 107, s5 -95/-88, runner s5 -73; nobhill runner x4 -66; nobhill_slow oncoming 57, cut_in 103 | pass | 3 | 1/39 | 2/10 | 0.824 |
| wide110 | 1 x 640x384, 110 deg (angular resolution x0.58) | 1.0x | 10 | beyond +-55 deg: nobhill runner -66, -66, -82; cut_in/runner at -87..-145; two grid_slow free runs now collide at -50, -62 (worse range accuracy) | pass (5/6 runner) | 2 | 10/39 | 4/10 | 0.758 |
| front+sides76 | front + 2 x 640x640 76 deg at +-74 | 3.0x | 3 | grid cut_in s5 -96/-113, grid runner s5 -96 (the 2 deg overlap seam and behind) | pass | 5 | 1/39 | 5/10 | 0.824 |
| **front+sides100** | front + 2 x 640x640 100 deg at +-80 (coarser sides, x0.65 pixels/rad) | 3.0x | **0** | none | pass | 6 | 4/39 | 3/10 | 0.765 |

Findings

- A single wide camera does not work: it still stops at +-55 deg and costs range accuracy everywhere (it introduced two collisions into hazard-free runs).
- Two 76 deg side cameras leave a 2 deg seam between cameras and stop at +-112: the three remaining collisions are objects at 96-113 deg.
- Front + two 100 deg side cameras closes every collision in the subset, including the five where the ground-truth feed also collides (the cameras reach farther behind than the GT feed's 40 deg rear cone).
- It is the only layout that passes the FOV-attributable cases, so it is the pick. Its cost is 3 inferences per frame set.
- It does not fix, and slightly worsens, hazard-free AEB (6 vs 3 activations over the same two runs): three cameras triple the false-positive rate (0.2 per frame per camera in this model). Diagnosed in the Gate S amendment.

Inference cost (measured this session, int8 `rtdetr_r18vd_quantized`, CPU, on a machine at load average ~200 so absolute times are not quoted as such): one frame vs three frames in a batch of 3 = 2.9x; three sequential runs = 4.8x (contended). Earlier sessions measured int8 at 58-78 ms per frame against a 100 ms frame interval (`2026-08-26-cycle5-phase2-gates.md`, `2026-08-28-cycle5-latency-floor.md`), i.e. 3 cameras need 1.8-2.3x the frame interval on one CPU worker unless inferences are batched or run on separate workers. The simulated observation latency (170 ms) is therefore optimistic for the 3-camera layout on this CPU; the real implementation measures it.
