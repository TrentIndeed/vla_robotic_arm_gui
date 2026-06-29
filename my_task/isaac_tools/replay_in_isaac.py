"""Replay a recorded demo's joints through the SO-101 in Isaac Lab and overlay the sim
desk camera against the real frames — to verify the sim scene matches your real setup.

Run this INSIDE the Sim-to-Real-SO-101-Workshop (Isaac Lab) environment, after exporting a
demo with export_episode.py (in the LeRobot env):

    python isaac_tools/replay_in_isaac.py --export isaac_replay_export/ep0 \
        --task Lerobot-So101-Teleop-Bottle-To-Basket --out replay_compare

It steps the arm through the recorded joint trajectory (open-loop) and writes side-by-side
images (sim | real) to <out>/####.png. If the arm, bottle, basket and board line up with
the real frames, your environment matches. If not, adjust the JOINT MAPPING below (or the
object/camera placement) until the overlay agrees — that IS the calibration.

NOTE: this is a template. The parts marked TODO depend on the workshop's exact asset/env
names (task id, robot joint names, camera key, gripper convention) — confirm them in the
launchable. The normalized->radian mapping is the thing you'll actually tune.
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Replay a demo in Isaac Lab and overlay vs real.")
parser.add_argument("--export", required=True, help="dir from export_episode.py (has joints.npy, real_desk/)")
parser.add_argument("--task", default="Lerobot-So101-Teleop-Bottle-To-Basket", help="registered Isaac Lab task id")
parser.add_argument("--cam-key", default="desk_cam", help="camera entity name in the scene")
parser.add_argument("--out", default="replay_compare")
parser.add_argument("--stride", type=int, default=2, help="replay every Nth frame (faster)")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
sim_app = app_launcher.app

# ---- after the app is up, the heavy imports are safe ----
import json
from pathlib import Path

import cv2
import gymnasium as gym
import numpy as np
import torch

import isaaclab_tasks  # noqa: F401  (registers the workshop tasks)
from isaaclab_tasks.utils import parse_env_cfg

# =====================================================================================
# JOINT MAPPING — normalized (-100..100, gripper 0..100) -> sim radians. TUNE THESE.
# The real arm reports normalized joint positions; the sim USD uses radians. Adjust
# SIGN / RANGE / OFFSET per joint until the sim arm matches the real frames in the overlay.
# Best source of truth: the numbers from the workshop's "Calibrate the SO-101" step — plug
# them in here. These defaults assume norm=±100 -> ±RANGE rad about OFFSET.
# =====================================================================================
ORDER = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
SIGN   = {"shoulder_pan": +1, "shoulder_lift": +1, "elbow_flex": +1, "wrist_flex": +1, "wrist_roll": +1}
RANGE  = {"shoulder_pan": 1.9, "shoulder_lift": 1.9, "elbow_flex": 1.9, "wrist_flex": 1.9, "wrist_roll": 2.6}  # rad @ norm=100
OFFSET = {"shoulder_pan": 0.0, "shoulder_lift": 0.0, "elbow_flex": 0.0, "wrist_flex": 0.0, "wrist_roll": 0.0}  # rad
GRIP_AT_100 = 0.04   # gripper sim joint value at norm=100 (rad or m — asset specific). TODO confirm.


def normalized_to_sim(row: np.ndarray) -> dict[str, float]:
    q = {}
    for name, v in zip(ORDER, row):
        if name == "gripper":
            q[name] = float(v) / 100.0 * GRIP_AT_100
        else:
            q[name] = SIGN[name] * (float(v) / 100.0) * RANGE[name] + OFFSET[name]
    return q


def main() -> None:
    exp = Path(args.export)
    joints = np.load(exp / "joints.npy")              # (T, 6) normalized
    meta = json.loads((exp / "meta.json").read_text())
    print(f"Loaded {len(joints)} frames, joint order {meta['joint_names']}")

    env_cfg = parse_env_cfg(args.task, num_envs=1)
    env = gym.make(args.task, cfg=env_cfg).unwrapped
    env.reset()

    robot = env.scene["robot"]                        # TODO: confirm the robot entity name
    sim_joint_names = list(robot.joint_names)
    # Map our 6 joint names to the robot's joint indices (sim asset names may differ slightly).
    name_to_idx = {}
    for j in ORDER:
        match = [i for i, sj in enumerate(sim_joint_names) if j in sj or sj in j]
        if match:
            name_to_idx[j] = match[0]
        else:
            print(f"WARN: joint '{j}' not found in sim joints {sim_joint_names}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    for t in range(0, len(joints), args.stride):
        q = normalized_to_sim(joints[t])
        q_full = robot.data.joint_pos.clone()         # current full joint vector (1, n)
        for j, ji in name_to_idx.items():
            q_full[0, ji] = q[j]
        robot.write_joint_position_to_sim(q_full)
        robot.write_joint_velocity_to_sim(torch.zeros_like(q_full))
        env.sim.step(render=True)                     # advance + render one frame

        # --- grab sim camera, load the matching real frame, save side-by-side ---
        cam = env.scene[args.cam_key]                 # TODO: confirm camera entity name
        sim_rgb = cam.data.output["rgb"][0, ..., :3].cpu().numpy().astype("uint8")
        sim_bgr = cv2.cvtColor(sim_rgb, cv2.COLOR_RGB2BGR)
        real = cv2.imread(str(exp / "real_desk" / f"{t:04d}.png"))
        if real is not None:
            h = min(sim_bgr.shape[0], real.shape[0])
            sim_bgr = cv2.resize(sim_bgr, (int(sim_bgr.shape[1] * h / sim_bgr.shape[0]), h))
            real = cv2.resize(real, (int(real.shape[1] * h / real.shape[0]), h))
            combo = np.hstack([sim_bgr, real])
            cv2.putText(combo, "SIM", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(combo, "REAL", (sim_bgr.shape[1] + 10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        else:
            combo = sim_bgr
        cv2.imwrite(str(out / f"{t:04d}.png"), combo)

    print(f"Wrote overlay frames to {out}. Scrub them: sim (left) should match real (right).")
    print("If the arm is offset/scaled, tune SIGN/RANGE/OFFSET above. If objects/camera are off,")
    print("adjust their positions in the env cfg (see docs/isaac_bottle_basket_task.md).")
    env.close()
    sim_app.close()


if __name__ == "__main__":
    main()
