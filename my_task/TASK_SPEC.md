# Isaac Lab task spec — bottle → basket (your real setup)

Converted from your measurements for the `Sim-to-Real-SO-101-Workshop` adaptation.
Isaac/USD units are **meters**. Coordinate frame = **robot base at origin**: `x` forward
(into the board), `y` to the robot's left, `z` up, board surface at `z = 0`.

## Dimensions (imperial → meters)

| Thing | Imperial | Meters |
|---|---|---|
| Foam board (all white, no black mat) | 3.5 ft × 2.5 ft | **1.067 (long/Y) × 0.762 (deep/X)** |
| Bottle (cylinder) | 2 in dia × 3 in tall | **r = 0.0254, h = 0.0762** |
| Basket outer (L×W×H) | 10 × 4 × 3 in | **0.254 (Y) × 0.1016 (X) × 0.0762 (Z)** |
| Basket wall thickness | 1/8 in | **0.003175** |
| Basket inner footprint | — | **0.2476 (Y) × 0.0953 (X)**, open top |
| Camera height | 16 in | **0.4064** |
| Robot from right short edge | 1 ft | **0.3048** |

## Layout (robot base = origin, faces +X into the board)

```
            LEFT short edge (Y = +0.762)
   back-left corner ___________________________ back-right corner
   [CAMERA here]   |                           |   (Y = -0.3048)
   x=0, y=+0.762   |        workspace          |
   z=0.4064        |   (reachable ~0.1–0.4 m)  |
                   |        [ROBOT base]       |  <-- back long edge (X=0)
                   |         origin (0,0)      |      1 ft (0.3048) from right edge
                   |___________________________|
                front long edge (X = +0.762)
```

- Board surface spans **X ∈ [0, 0.762]**, **Y ∈ [−0.3048, +0.762]** (the robot sits on the
  back edge, 1 ft in from the right short edge).
- SO-101 reach is ~0.35 m, so the usable area is roughly **X ∈ [0.10, 0.40], Y ∈ [−0.25, +0.30]**.
- **Robot is white** — set the SO-101 USD material to white (the workshop's is grey/black).
- **Replace the black mat** (`mat.usda`) with a plain **white board** surface.

## Scene config (Isaac Lab style — adapt to the workshop's `*_env_cfg.py`)

```python
import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg, AssetBaseCfg

WHITE = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.9, 0.9))
BLACK = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.05, 0.05, 0.05))

# --- White foam board (static surface; replaces the black mat) ---
board = AssetBaseCfg(
    prim_path="{ENV_REGEX_NS}/Board",
    spawn=sim_utils.CuboidCfg(
        size=(0.762, 1.067, 0.01),                 # X deep, Y long, 1 cm thick
        visual_material=WHITE,
        collision_props=sim_utils.CollisionPropertiesCfg(),
    ),
    # center the board: robot on back edge -> board extends +X; 1 ft from right edge
    init_state=AssetBaseCfg.InitialStateCfg(pos=(0.381, 0.2286, -0.005)),
)

# --- Pill bottle (the pick object) ---
bottle = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/Bottle",
    spawn=sim_utils.CylinderCfg(
        radius=0.0254, height=0.0762, axis="Z",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        mass_props=sim_utils.MassPropertiesCfg(mass=0.04),   # ~40 g; tune to real
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.15, 0.35, 0.8)),  # blue cap-ish
    ),
    init_state=RigidObjectCfg.InitialStateCfg(pos=(0.30, 0.0, 0.0381)),  # bottom on table -> z = h/2
)
```

### Basket = open box (floor + 4 walls). Author as a small USD, OR 5 static cuboids.

Walls **relative to basket center**, sizes `(X, Y, Z)` (t = 0.003175):

| Part | size (X, Y, Z) | center offset (x, y, z) |
|---|---|---|
| floor | (0.1016, 0.254, 0.003175) | (0, 0, 0.00159) |
| wall +X (front) | (0.003175, 0.254, 0.0762) | (+0.0492, 0, 0.0381) |
| wall −X (back) | (0.003175, 0.254, 0.0762) | (−0.0492, 0, 0.0381) |
| wall +Y (end) | (0.1016, 0.003175, 0.0762) | (0, +0.1254, 0.0381) |
| wall −Y (end) | (0.1016, 0.003175, 0.0762) | (0, −0.1254, 0.0381) |

Place the basket center at e.g. **(0.16, 0.16, 0)** (back-left, like your photo — long axis
along Y), black material. Make it **static/fixed** (it's the target, shouldn't move) — a
single convex hull would seal the opening, so collision must be the **walls**, not a hull.

## Success condition (basket containment) — for `mdp/terms.py`

Replace the rack's slot-seating check. Bottle counts as placed when its center is inside the
basket's inner footprint and below the rim:

```python
def bottle_in_basket(env, bottle_cfg, basket_pos, rim_z=0.0762):
    p = env.scene[bottle_cfg].data.root_pos_w[:, :3]      # (N,3) world
    bx, by, bz = basket_pos                               # basket center (world)
    half_x = 0.0953 / 2.0   # inner footprint half-widths (X = 4in side)
    half_y = 0.2476 / 2.0   # (Y = 10in side)
    inside_xy = (torch.abs(p[:, 0] - bx) < half_x) & (torch.abs(p[:, 1] - by) < half_y)
    below_rim = (p[:, 2] < (bz + rim_z)) & (p[:, 2] > (bz - 0.02))
    return inside_xy & below_rim
```

## Desk camera (back-left corner, aimed at opposite corner +15°)

- **Position (world):** `(0.0, 0.762, 0.4064)` — back-left corner, 16 in up.
- **Nominal look-at:** front-right corner `(0.762, -0.3048, 0.0)`.
- **+15° to the right:** yaw the look direction ~15° clockwise (top view) → look-at ≈
  `(0.46, -0.46, 0.0)`. **Dial this in visually** in the launchable against your real photo —
  matching the sim camera to the real desk camera is what makes sim-to-real transfer work.

```python
camera = CameraCfg(
    prim_path="{ENV_REGEX_NS}/desk_cam",
    height=480, width=640,
    spawn=sim_utils.PinholeCameraCfg(focal_length=18.0),   # widen/narrow to match real FOV
    offset=CameraCfg.OffsetCfg(pos=(0.0, 0.762, 0.4064), convention="world"),
    # set rotation from the look-at above (or use a look_at helper), then nudge +15° yaw
)
```

## Reset randomization (in `mdp/resets.py` style)

- **Bottle** start pose (so it doesn't memorize one spot): `x ∈ (0.22, 0.38)`, `y ∈ (−0.15, 0.25)`,
  `yaw ∈ (−π, π)` (cylinder is symmetric, yaw is free).
- **Basket**: keep mostly fixed; small jitter `x,y ∈ (−0.03, 0.03)`, `yaw ∈ (−0.2, 0.2)`.
- Keep both within the reachable area above.

## To refine in the launchable (visual)
1. Nudge **basket + bottle positions** so they sit where the arm comfortably reaches (watch the
   IK/teleop).
2. **Match the camera** to your real desk-cam photo (position is set; tune yaw + focal length).
3. Set the **bottle mass** to the real bottle's weight.
4. Confirm the **board origin/offset** so the robot sits on the back edge as measured.
