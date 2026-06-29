"""Export one recorded demo (joint trajectory + real desk-camera frames) for sim replay.

Runs in the LeRobot env (this repo's .venv) so the Isaac side needs no lerobot. Pulls one
episode from the dataset and writes:

    <out>/joints.npy        (T, 6) normalized joint positions, order in meta.json
    <out>/actions.npy       (T, 6) commanded actions (optional reference)
    <out>/real_desk/####.png  the real desk-camera frame at each timestep
    <out>/meta.json         joint names, fps, episode, frame count

Then drive these joints through the SO-101 in Isaac Sim and overlay the sim camera against
real_desk/ to verify the scene matches your setup (see replay_in_isaac.py).

    python isaac_tools/export_episode.py --episode 0
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")   # local dataset, never hit the Hub

import numpy as np
from PIL import Image


def main() -> None:
    p = argparse.ArgumentParser(description="Export a demo's joints + real frames for sim replay")
    p.add_argument("--episode", type=int, default=0)
    p.add_argument("--dataset", default="local/so101_pick_place")
    p.add_argument("--root", default="data/local__so101_pick_place")
    p.add_argument("--cam", default="observation.images.desk", help="which camera to export")
    p.add_argument("--out", default=None)
    a = p.parse_args()

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset(a.dataset, root=Path(a.root))
    names = ds.meta.features["observation.state"]["names"]
    fps = int(ds.fps)

    # Find the (contiguous) frame indices for this episode without decoding video.
    ep_col = [int(e) for e in ds.hf_dataset["episode_index"]]
    idxs = [i for i, e in enumerate(ep_col) if e == a.episode]
    if not idxs:
        raise SystemExit(f"Episode {a.episode} not found (dataset has {ds.num_episodes}).")

    out = Path(a.out or f"isaac_replay_export/ep{a.episode}")
    (out / "real_desk").mkdir(parents=True, exist_ok=True)

    joints, actions = [], []
    for n, i in enumerate(idxs):
        item = ds[i]
        joints.append(item["observation.state"].numpy())
        actions.append(item["action"].numpy())
        img = item[a.cam]                                   # CHW float [0,1]
        img = (img.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype("uint8")
        Image.fromarray(img).save(out / "real_desk" / f"{n:04d}.png")

    joints = np.stack(joints)
    np.save(out / "joints.npy", joints)
    np.save(out / "actions.npy", np.stack(actions))
    (out / "meta.json").write_text(json.dumps({
        "episode": a.episode, "fps": fps, "num_frames": len(idxs),
        "joint_names": names, "camera": a.cam,
        "state_min": joints.min(0).tolist(), "state_max": joints.max(0).tolist(),
    }, indent=2), encoding="utf-8")

    print(f"Exported episode {a.episode}: {len(idxs)} frames -> {out}")
    print(f"  joints {joints.shape}, joint order: {names}")
    print(f"  per-joint min: {np.round(joints.min(0), 1)}")
    print(f"  per-joint max: {np.round(joints.max(0), 1)}")


if __name__ == "__main__":
    main()
