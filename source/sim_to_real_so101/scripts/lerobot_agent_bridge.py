# SPDX-License-Identifier: Apache-2.0
#
# Teleoperate the Isaac Sim SO-101 with a REAL arm that lives on a *different*
# machine — your follower arm, back-driven by hand as a "leader" (torque off).
#
# The workshop's lerobot_agent.py reads the arm over a local serial port, which only
# works when the arm is plugged into the same box as the sim. Here the arm is on your
# Windows PC and the sim is on the cloud GPU box, so this script instead receives the
# arm's joint positions over a TCP socket from a tiny local sender
# (tools/follower_bridge_sender.py on your PC) and feeds them into the sim. Everything
# downstream — the real->sim joint mapping and the dataset recorder — is reused
# unchanged from the workshop.
#
# Run on the CLOUD box (where Isaac Sim is):
#   ./isaaclab.sh -p source/sim_to_real_so101/scripts/lerobot_agent_bridge.py \
#       --task Lerobot-So101-Teleop-Bottle-To-Basket --livestream 2 \
#       --bind_host 0.0.0.0 --bind_port 5556
#   # add --repo_id <id> --repo_root <dir> --task_name <name> to record a dataset
#
# Then on your PC run tools/follower_bridge_sender.py pointing at this box:port.
# In the viewer: 'R' resets, 'S' starts/stops recording, 'C' cancels.
import argparse
import json
import os
import socket
import threading

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Isaac Lab SO-101 teleop over a network bridge (remote real arm).")
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--task", type=str, default="Lerobot-So101-Teleop-Bottle-To-Basket")
parser.add_argument("--bind_host", type=str, default=os.getenv("BRIDGE_HOST", "0.0.0.0"),
                    help="Interface to listen on for the local arm sender.")
parser.add_argument("--bind_port", type=int, default=int(os.getenv("BRIDGE_PORT", "5556")))
parser.add_argument("--repo_id", type=str, default=None)
parser.add_argument("--repo_root", type=str, default=None)
parser.add_argument("--task_name", type=str, default=None)
parser.add_argument("--save_mp4", action="store_true", default=False)
parser.add_argument("--depth", action="store_true", default=False)
parser.add_argument("--instance_id_seg", action="store_true", default=False)
parser.add_argument("--seed", type=int, default=101)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True  # always render cameras (needed for recording + the policy view)

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest follows."""

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg
import sim_to_real_so101.tasks  # noqa: F401
from sim_to_real_so101.utils.keyboard import KeyboardControl
from sim_to_real_so101.utils.lerobot_interface import LeRobotSO101Interface
from sim_to_real_so101.utils.lerobot_recorder import LeRobotRecorder

JOINT_ORDER = LeRobotSO101Interface.SO101_JOINT_ORDER


class ActionServer:
    """Background TCP server. Accepts one sender at a time and reads newline-delimited
    JSON action dicts ({"shoulder_pan.pos": float, ...}), keeping only the LATEST. The
    network rate is fully decoupled from the sim rate — the sim always reads the most
    recent pose and never blocks on the socket."""

    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
        self._latest = None
        self._lock = threading.Lock()
        self.connected = False
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()

    def _serve(self):
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind((self.host, self.port))
        srv.listen(1)
        print(f"[BRIDGE] Listening for the follower-arm sender on {self.host}:{self.port} ...")
        while True:
            conn, addr = srv.accept()
            print(f"[BRIDGE] Sender connected from {addr}. Move the arm by hand to drive the sim.")
            self.connected = True
            buf = b""
            try:
                with conn:
                    conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                    while True:
                        data = conn.recv(4096)
                        if not data:
                            break
                        buf += data
                        # keep only the last complete line in the buffer (drop stale frames)
                        while b"\n" in buf:
                            line, buf = buf.split(b"\n", 1)
                            line = line.strip()
                            if not line:
                                continue
                            try:
                                act = json.loads(line.decode("utf-8"))
                                with self._lock:
                                    self._latest = act
                            except json.JSONDecodeError as exc:
                                print(f"[BRIDGE] dropped malformed packet: {exc}")
            except OSError as exc:
                print(f"[BRIDGE] connection error: {exc}")
            self.connected = False
            print("[BRIDGE] Sender disconnected — waiting for reconnect...")

    def latest(self):
        with self._lock:
            return self._latest


def main():
    keyboard_control = KeyboardControl()

    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    env_cfg.seed = args_cli.seed
    env = gym.make(args_cli.task, cfg=env_cfg)

    print(f"[INFO]: Action space: {env.action_space}")
    print(f"[INFO]: 'R' reset world | 'S' start/stop recording | 'C' cancel recording")
    env.reset()

    # discover cameras (same as lerobot_agent.py)
    cameras = {}
    for obj in env.unwrapped.scene.keys():
        if obj.startswith("camera_"):
            camera_cfg = getattr(env.unwrapped.scene.cfg, obj)
            cameras[obj.replace("camera_", "")] = {"height": camera_cfg.height, "width": camera_cfg.width}
            print(f"[INFO]: Found Camera: {obj.replace('camera_', '')}")
    if not cameras:
        print("[INFO]: No cameras found - videos will not be recorded")

    # Interface used ONLY for the real->sim joint mapping. No serial robot on this box,
    # so we deliberately do NOT call init_device()/connect(); the mapping tables are
    # built in __init__ and need no hardware.
    iface = LeRobotSO101Interface(
        device=env.unwrapped.device, port="", id="bridge", cameras=cameras, fps=30, kind="leader",
    )

    action_server = ActionServer(args_cli.bind_host, args_cli.bind_port)
    actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)

    recording_mode = all([args_cli.repo_id, args_cli.repo_root, args_cli.task_name])
    recorder = None
    if recording_mode:
        recorder = LeRobotRecorder(
            task_name=args_cli.task_name, repo_id=args_cli.repo_id, dataset_root=args_cli.repo_root,
            fps=30, device=env.unwrapped.device, cameras=cameras, save_mp4=args_cli.save_mp4,
            depth=args_cli.depth, instance_id_seg=args_cli.instance_id_seg,
        )
        try:
            recorder.init_dataset()
        except ValueError:
            print("[ERROR]: Failed to initialize dataset. Folder already exists.")
            env.close()
            simulation_app.close()
            return

    last_dict = None     # hold the last valid pose so the arm freezes (not snaps) if packets pause
    warned_wait = False
    while simulation_app.is_running():
        with torch.inference_mode():
            act_dict = action_server.latest()
            if isinstance(act_dict, dict) and all(j in act_dict for j in JOINT_ORDER):
                last_dict = act_dict
            elif last_dict is None and not warned_wait:
                print("[BRIDGE] No arm packets yet — sim is holding still. Start the local sender.")
                warned_wait = True

            real_action = None
            if last_dict is not None:
                real_action, mapped_action = iface.real_to_sim_obs_processor(last_dict)
                actions[:] = mapped_action

            obs, _, _, _, _ = env.step(actions)

            if keyboard_control.reset_world:
                keyboard_control.reset_world = False
                env.reset()
                continue

            if recording_mode and keyboard_control.recording and real_action is not None:
                visual_obs = obs.get("visual", None)
                if visual_obs is None:
                    print("[WARNING]: No 'visual' observation group - recording needs a task with cameras")
                    keyboard_control.recording = False
                    continue
                joint_pos_obs = obs["policy"]["joint_pos_obs"][0]
                visual_obs = obs["visual"]
                real_obs, visual_buffers, depth_buffers, instance_id_seg_buffers = (
                    iface.sim_to_real_dataset_processor(joint_pos_obs, visual_obs)
                )
                recorder.push_frame_to_buffer(
                    real_action, real_obs, visual_buffers, depth_buffers, instance_id_seg_buffers
                )

    env.close()


if __name__ == "__main__":
    main()
    while True:
        simulation_app.update()
