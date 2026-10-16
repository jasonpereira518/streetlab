// Live closed-loop runner (SKELETON; spec 2026-10-04-streetlab-ml-driving-design.md, 0c / phase A).
//
// Intended flow, one run per (scene, seed):
//   1. start `streetlab serve --perception ml --record <path>` with the scene/seed,
//   2. drive the frontend with Playwright (NOT the Browser pane, which throttles
//      background tabs to ~1 frame/minute) so real detector frames exist,
//   3. send `set_perception ml` on the first step and let the run play out,
//   4. `scripts/scorecard.py --from-recording <path>` scores what the backend wrote.
//
// Not yet implemented: `--record` does not exist on the CLI and the recording
// format (the `evaluation.driving_metrics.Run` fields) is unwritten. Live runs
// are paired by seed but not bit-reproducible (frame arrival is wall-clock), so
// the eventual runner must report every seed, never a best-of.
const args = Object.fromEntries(
  process.argv.slice(2).map((a) => a.replace(/^--/, '').split('=')),
);

if (!args.scene || !args.seed || !args.out) {
  console.error('usage: node closed_loop_eval.cjs --scene=<key> --seed=<n> --out=<recording path>');
  process.exit(2);
}
console.error('closed_loop_eval: live runner not implemented yet (phase A); see header.');
process.exit(1);
