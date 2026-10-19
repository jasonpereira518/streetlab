# Buses back on the block loops: set-back stop lines and a tail check

**Date:** 2026-10-17
**Branch:** `claude/ecstatic-hodgkin-54b71b` (follows PR #50, which kept buses off the loops)

## The problem

An 11.5 m bus on the grid's block loop takes the lane's 6 m corner as a rigid
box, posed along the chord between its axles. Its front overhang sweeps the
receiving street's oncoming approach: in dense traffic it swung 0.38 m into a
car waiting at that line (grid-merge seed 7). PR #50 kept buses off loops with
corners tighter than 11 m. This puts them back.

## What was measured first

The bus waits in lane at its line. Its centre rests 1 m short of the line, so
its front is 4.75 m past it. From there it was swept through every corner of
both bus loops (grid-merge, grid-arterial) against a car and a truck resting at
every other approach's line.

**One approach per corner conflicts:** the lane on the receiving street coming
toward the junction. The setback that clears it by 0.3 m:

| Turn the bus takes from its line                  | Peak radius | Setback needed       |
|---------------------------------------------------|-------------|----------------------|
| The lane's own corner (the 6 m fillet)            | 5.6–5.8 m   | **4.0–7.5 m**        |
| A true wide arc from the line, exiting in lane    | 7.1–10.5 m  | 5.5–11 m             |
| The wide arc plus a 0.9 m swing on the exit       | 7.9–10.5 m  | 13–17 m, or none fits |

Every case keeps the body on the carriageway (0.00 m over the kerb). Turning
wider makes things worse here: a rigid box on a bigger arc pushes its front
further into the receiving lane. A real bus turns wide to keep its rear wheels
off the inside kerb, and this model has no off-tracking rear to protect.

(A trap in measuring this: `Route.fillet` caps an arc at half its adjoining
leg. A local polyline needs legs twice the tangent length, or the radius
silently halves.)

An earlier attempt swung the bus out before the corner, with traffic
negotiating around it. Every grid corner has a sign or a signal, so the bus
waited at its line already swung into the oncoming lane. The negotiation rules
moved contacts and deadlocks from place to place without converging, and that
design was dropped.

## What it does

- **Buses on the loops.** `_PROFILES[i]` again, with no exclusion.
- **Set-back lines.** `map.scene_build._bus_setbacks` re-runs the measurement
  at build time, from the pose the sim draws. It runs only for scenarios that
  put a bus on the loop:
  - **grid-merge:** `tl_0_0_s` and `tl_0_80_w` by 4.0 m; `ss_80_80_n` and
    `tl_80_0_e` by 7.5 m;
  - **grid-arterial:** all four corners' receiving approaches by 7.5 m.

  Those approaches' stop lines move back by that much. The ego's own lines
  never do: its approach is never the receiving one.
- **Stop bars.** `StopSign` and `TrafficLight` gain `stop_bar_extra_m`
  (default 0, no protocol bump, like `edge_m`). `world.ts` paints the bar that
  much further back, and a stop sign moves with its bar.
- **Tail check.** On the 6 m corner the bus's tail swings up to ~1.4 m over
  the centreline into the oncoming lane. Two oncoming motorcycles were struck
  in dense grid-merge runs. A vehicle 10 m or longer, at rest short of a
  corner tighter than 11 m, now pulls off only when nothing outside its own
  lane is in the corner's sweep, or will be within 4 s, accelerating as IDM
  does. Others are not told to yield.

## Latent defects found on the way

- **The ego rolling into an all-way-stop box.** Having won the tie, the ego
  crept in from 6 m back. The box stood empty past `BOX_TIMEOUT_S`, and a
  truck was released across its path. Traffic now treats an ego that waited
  here and is rolling in as holding the box.
- **A car queued behind you as a rival.** A car queued behind this one on its
  own approach, noted as stopped first, beat it for the turn, and the junction
  waited on a car that could not move. It is no longer counted.
- **A body straddling into your lane.** A truck easing back from the kerb lane
  sat 1.8 m aside at 9°. Its centre was just outside "same lane" and its tail
  in it, and the motorcycle behind drove into it. `_merging_gap` now also
  counts a body that reaches into the follower's width.
- **A car-sized corridor for a motorcycle.** A motorcycle stopped for a
  turning bus's tail 1.16 m from its lane centre, with 0.76 m to spare, and
  the tail swept into it. The intrusion corridor is now the body's half-width
  plus 0.55 m, capped at the old 1.5 m, so cars, trucks and buses are
  unchanged.

## Dense traffic, 12 per lane-km, 180 s

| Run                | origin/main | This branch |
|--------------------|-------------|-------------|
| grid-merge seed 7  | −0.59 m     | **+0.74 m** |
| grid-merge seed 11 | —           | +0.52 m     |
| grid-arterial s7   | —           | +1.04 m     |
| grid-arterial s11  | —           | +1.02 m     |
| grid-loop seed 7   | —           | +0.25 m     |
| grid-signals s7    | —           | +0.24 m     |

All runs: 0.00 m over the kerb, no building or prop contact. Longest ego wait
is 8.3 s (grid-arterial) and 7.2 s (grid-loop). grid-arterial is now one of
the dense clearance cases.
