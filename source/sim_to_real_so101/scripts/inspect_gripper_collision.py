# SPDX-License-Identifier: Apache-2.0
# Print the collision approximation of every collider under the robot's gripper/jaw, so we
# can see WHY objects get trapped/laggy on contact (e.g. a single convex hull of the whole
# hand that encloses the grasp gap). Headless, fast, no streaming.
#   python source/sim_to_real_so101/scripts/inspect_gripper_collision.py
import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Lerobot-So101-Teleop-Bottle-To-Basket")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
args_cli.enable_cameras = False

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import isaaclab_tasks  # noqa: F401,E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
import sim_to_real_so101.tasks  # noqa: F401,E402
import isaacsim.core.utils.stage as stage_utils  # noqa: E402
from pxr import UsdPhysics  # noqa: E402


def main():
    env_cfg = parse_env_cfg(args_cli.task, num_envs=1)
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    env.reset()
    stage = stage_utils.get_current_stage()

    root = "/World/envs/env_0/Robot"
    print("\n================ ROBOT COLLIDERS (gripper region) ================")
    for prim in stage.Traverse():
        p = str(prim.GetPath())
        if not p.startswith(root):
            continue
        has_col = prim.HasAPI(UsdPhysics.CollisionAPI)
        approx = None
        if prim.HasAPI(UsdPhysics.MeshCollisionAPI):
            a = UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr()
            approx = a.Get() if a and a.IsValid() else None
        raw = prim.GetAttribute("physics:approximation")
        raw = raw.Get() if raw and raw.IsValid() else None
        # show every collider, plus any mesh in the gripper/jaw region
        low = p.lower()
        in_hand = any(k in low for k in ["jaw", "gripper", "finger", "moving", "fixed", "wrist"])
        if has_col or approx or raw or (in_hand and prim.GetTypeName() == "Mesh"):
            tag = "<-- HAND" if in_hand else ""
            print(f"{str(prim.GetTypeName()):10s} collider={has_col!s:5s} approx={approx or raw}  {p}  {tag}")
    print("=================================================================\n")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
