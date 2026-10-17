# Bridges and tunnels drawn off the ground

**Date:** 2026-10-17
**Branch:** `claude/bridges-tunnels` (phase 7a of the realism plan)

## What was wrong

- **Grading filled the ground up to the bridge.** `terrain.grade` treated
  Hyde St's bridge as a street on the ground and filled the Broadway cutting
  beneath it up towards the deck. The elevation tiles (~3 m pixels) had
  already smeared that cutting to 1–2 m below the deck, so Broadway ran
  through the bridge.
- **Junctions between layers.** The renderer's `findCrossings` ignored
  `layer`. A bridge and the street beneath got a junction: the centre line
  under the deck was cut, and stop bars could be painted.
- **Tunnels drawn on the hill.** A tunnel's tarmac was laid over the top of
  the hill above it. Its pavement also clipped the pavements of the streets
  overhead.

## What it does now

- **`Road.elevation_m`:** surface height at each centreline vertex, for a way
  carried off the ground. Additive and defaulted to `[]`, so the protocol stays
  11. OSM bridges and tunnels get it whenever the scene has terrain: a straight
  line between the graded ground at the way's two ends. Phase 7b's grid ramps
  will use the same field.
- **Grading:**
  - **Bridges** are left out, like tunnels.
  - **A street passing under a bridge** is pinned, where it crosses, at least
    `UNDERPASS_CLEARANCE_M` = 5.5 m below the deck (4.4 m legal clearance plus
    the deck), with 0.5 m spare. The deck's ends settle on the graded ground,
    and the cut beside them pulls those down by up to 0.3 m.
  - **Approaches** touching a bridge's end are not cut.
- **Renderer:**
  - **Layers:** crossings, and the carriageways a pavement stops at, are per
    layer.
  - **Bridges:** tarmac, pavements and markings are laid on the deck, with a
    deck fascia and soffit, railings, and piers every 20 m. A pier is dropped
    where its foot would stand within a metre of a carriageway beneath. Hyde
    St's 21 m span gets none.
  - **Tunnels:** tarmac and markings are hidden. A portal (headwall and dark
    mouth) stands at each end that opens to daylight. Two tunnel ways joined
    underground get none at the joint.
- **Vehicles:** `World.groundFor(heading)` returns the surface of a structure
  the vehicle is driving along (within its half-width and 40° of its
  direction), and the ground otherwise. A car on Hyde St's deck sits on it; one
  on Broadway beneath stays on Broadway. The ego, traffic, the sun target and
  the detector camera's mount all use it.

| Hyde St over Broadway (bundled Nob Hill) | Before | Now |
|---|---|---|
| Deck to the ground under it | 1.2–2.2 m | ≥ 5.4 m (test-pinned) |
| Deck end to the ground the approaches are drawn on | – | within 0.3 m |

Checked live on a Hyde St extract: the deck spans both Broadway carriageways
in the cutting, and the Broadway Tunnel's west portal stands at the end of
its bore.

## Known, not fixed here

- **The driven line** (reference path) is still laid on the ground, so on a
  route over a bridge it runs under the deck.
- **The chase camera in a tunnel** keeps above the ground, so it rises above
  the hill while the car is underground.
- **Live extracts differ from the bundled one:** a live Hyde St extract tags
  only one of Broadway's two bores as a tunnel. The other carriageway is drawn
  over the hill, as its tags say.

## Tests

- **`test_bridges_tunnels.py`:**
  - the deck is straight between its ends;
  - the street beneath is cut down and the approaches are not;
  - a higher layer is not cut;
  - Hyde St clears Broadway on the bundled extract.
- **`renderInvariants.test.ts`:**
  - the deck sits at its height and the street beneath it on the ground;
  - there is no junction between layers: the centre line is unbroken and
    there is no stop bar;
  - railings are drawn;
  - `groundFor` follows heading;
  - tunnel tarmac is hidden, and portals stand at both open ends.
