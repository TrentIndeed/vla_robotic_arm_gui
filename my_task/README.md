# Bottle → Basket — Trenton's SO-101 adaptation

Adapts the workshop's "vials → rack" task to **pick a pill bottle and place it in an open
basket** on a white foam board, matching a real recorded setup.

## What's here / what changed
- **`source/sim_to_real_so101/tasks/bottle_to_basket_env_cfg.py`** — new task config, adapted
  from `vials_to_rack_env_cfg.py` (single bottle instead of 3 vials, basket target, reuses the
  workshop's grasp/placement MDP terms).
- **`source/sim_to_real_so101/tasks/__init__.py`** — registers the new gym envs:
  - `Lerobot-So101-Teleop-Bottle-To-Basket` (+ `-DR`, `-Eval`, `-DR-Eval`)
- **`my_task/TASK_SPEC.md`** — real-world dimensions (board, bottle, basket, camera) converted
  to meters + the coordinate frame.
- **`my_task/isaac_tools/`** — verify the sim matches reality by replaying a recorded real demo:
  `export_episode.py` (run in the LeRobot env) → `replay_in_isaac.py` (run here) → side-by-side
  sim/real overlay.

## Finish these (marked `TODO` in the cfg)
1. **Assets** — the bottle uses `Vial_opaque.usda` and the basket uses `tray.usda` as stand-ins
   so the env loads. Build a real **pill-bottle USD** (2 in dia × 3 in tall) and an **open-basket
   USD** (10 × 4 × 3 in, ⅛ in walls). Simplest = primitives/CAD (see `TASK_SPEC.md`). The basket
   must be an *open* box with wall collision, not a solid hull.
2. **Positions** — tune `bottle.init_state.pos` / `basket.init_state.pos` to where the arm
   reaches; refine against your real frames with the replay overlay.
3. **Success box** — set the basket's inner footprint + rim height in the `bottle_placed` /
   `success` params (basket-local frame), `vertical_threshold=0.0` (a bottle needn't be upright).
4. **Camera** — match the desk camera to your real photo (`TASK_SPEC.md` has the pose).

## Run it
After Isaac Lab is set up in the launchable, list/teleop the env by its id, e.g.:
```bash
# (use the workshop's teleop/record script with the new task id)
... --task Lerobot-So101-Teleop-Bottle-To-Basket
```
Start with the base env (placeholders) to confirm it loads, then swap assets + tune positions,
then use the `-DR` variant for domain randomization and `-Eval` for the success-terminated eval.

## Verify the sim matches your real setup
1. In your LeRobot project: `python isaac_tools/export_episode.py --episode 0`
2. Copy `isaac_replay_export/ep0/` here, then run `my_task/isaac_tools/replay_in_isaac.py` and
   tune the joint mapping until the sim arm overlays your real frames. That validates kinematics,
   camera, and scene geometry at once.
