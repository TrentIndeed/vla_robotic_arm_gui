# Sim-match verification: replay a real demo in Isaac Sim

Drive a recorded demo's joint trajectory through the SO-101 in Isaac Sim and overlay the
sim desk camera against the real frames. If they line up, your sim scene (arm, bottle,
basket, board, camera) matches your real setup. This is **open-loop joint replay** — it
verifies *geometry + camera*, not grasp physics, which is exactly what you want for
validating the environment.

## Two steps (two environments)

**1. Export a demo — in the LeRobot env (this repo's `.venv`):**
```bash
python isaac_tools/export_episode.py --episode 0
# -> isaac_replay_export/ep0/{joints.npy, actions.npy, real_desk/####.png, meta.json}
```

**2. Replay + overlay — inside the Isaac Lab workshop env (Brev launchable):**
Copy `isaac_replay_export/ep0/` into the workshop, then:
```bash
python isaac_tools/replay_in_isaac.py --export isaac_replay_export/ep0 \
    --task Lerobot-So101-Teleop-Bottle-To-Basket --out replay_compare
# -> replay_compare/####.png  (sim | real, side by side)
```
Scrub `replay_compare/` — left (sim) should match right (real).

## The one thing you tune: the joint mapping

The real arm reports **normalized −100…100** (gripper 0…100); the sim USD uses **radians**.
`replay_in_isaac.py` has a `SIGN / RANGE / OFFSET` block per joint — adjust until the sim arm
matches the real frames. **Best source:** plug in the numbers from the workshop's
"Calibrate the SO-101" step. Tune order: get each joint's **sign** right first (does it move
the correct direction?), then **range** (does ±100 reach the same extremes?), then **offset**
(is the neutral pose aligned?).

## Reading the overlay
- **Arm offset/scaled/mirrored** → fix the joint mapping (`SIGN/RANGE/OFFSET`).
- **Arm matches but bottle/basket are in the wrong place** → adjust object positions in the
  env cfg (see [../docs/isaac_bottle_basket_task.md](../docs/isaac_bottle_basket_task.md)).
- **Everything matches but framing/perspective is off** → adjust the camera pose/FOV.

## Notes
- `replay_in_isaac.py` is a template: the `TODO` spots (task id, robot/camera entity names,
  gripper convention) depend on the workshop's exact assets — confirm them in the launchable.
- `--stride` replays every Nth frame for speed while tuning; drop to 1 for a smooth pass.
