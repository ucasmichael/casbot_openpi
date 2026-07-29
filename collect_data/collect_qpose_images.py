#!/usr/bin/env python3
"""
Interactive ADU-centered data collection.

Run this script on ADU. It subscribes to ADU and HDU ROS topics from ADU and
stores every episode on ADU only; HDU is never used for persistent storage.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import shlex
import signal
import subprocess
import sys
import tempfile
import termios
import time
import tty
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable


ROS_SETUP = "/opt/ros/jazzy/setup.bash"
AGIBOT_ENV = "/agibot/software/v0/entry/env/env.sh"

DEFAULT_HEAD_CAMERAS = ["head_front_dual_fisheye_camera"]
DEFAULT_HEAD_STREAM = "h265"
DEFAULT_CHEST_CAMERAS = ["chest_front_d457_camera"]
DEFAULT_JOINT_TOPICS = [
    "/motion/control/arm_joint_state",
    "/motion/control/hand_joint_state",
    "/motion/control/neck_joint_state",
]


@dataclass
class Recorder:
    name: str
    out: Path
    log: Path


@dataclass
class LocalRecorder(Recorder):
    proc: subprocess.Popen[str] | None = None

    def stop(self) -> None:
        if self.proc is None or self.proc.poll() is not None:
            return
        os.killpg(self.proc.pid, signal.SIGINT)
        wait_process(self.proc, timeout_s=15)
        if self.proc.poll() is None:
            os.killpg(self.proc.pid, signal.SIGTERM)
            wait_process(self.proc, timeout_s=5)


@dataclass
class RemoteRecorder(Recorder):
    host: str = ""
    key: str | None = None
    batch_mode: bool = True
    pid_file: str = ""

    def stop(self) -> None:
        if not self.pid_file:
            return
        stop_script = f"""
set +e
if [[ -f {shlex.quote(self.pid_file)} ]]; then
  PID="$(cat {shlex.quote(self.pid_file)} 2>/dev/null)"
  if [[ -n "$PID" ]]; then
    kill -INT "$PID" 2>/dev/null || true
  fi
fi
for _ in $(seq 1 30); do
  if [[ -z "$PID" ]] || ! kill -0 "$PID" 2>/dev/null; then
    exit 0
  fi
  sleep 0.5
done
if [[ -n "$PID" ]]; then
  kill -TERM "$PID" 2>/dev/null || true
fi
"""
        run_ssh(self.host, stop_script, key=self.key, batch_mode=self.batch_mode)


def main() -> int:
    args = parse_args()
    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    run_local_shell("command -v ros2 >/dev/null", "ros2 not found. Did you source the robot environment?")
    if args.stop_agivslam:
        run_local_shell("command -v aimcli >/dev/null && aimcli em stop agivslam || true")

    head_topics = camera_topics(args.head_cameras, kind="head", streams=[args.head_stream])
    chest_topics = camera_topics(args.chest_cameras, kind="chest", streams=args.chest_streams)
    joint_topics = [] if args.no_joints else args.joint_topics

    if not chest_topics and not head_topics and not joint_topics:
        raise SystemExit("No topics selected.")

    selected_topics = [*chest_topics, *head_topics, *joint_topics]
    missing_publishers = topics_without_publishers(selected_topics)
    if missing_publishers:
        joined = "\n  ".join(missing_publishers)
        raise SystemExit(f"这些 topic 当前没有 publisher，停止采集：\n  {joined}")

    print("将采集这些 topic：")
    for topic in chest_topics:
        print(f"  ADU  {topic}")
    for topic in [*head_topics, *joint_topics]:
        print(f"  HDU  {topic}")
    print()

    session_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = output_root / f"collection_{session_stamp}"
    session_dir.mkdir(parents=True, exist_ok=True)

    session_meta = {
        "created_at": session_stamp,
        "adu_topics": chest_topics,
        "hdu_topics": [*head_topics, *joint_topics],
        "hdu_storage": "disabled",
        "storage": "all bags are written on ADU",
        "episodes": args.episodes,
        "head_stream": args.head_stream,
        "chest_streams": args.chest_streams,
        "compression_note": "head camera uses existing h265 topic; chest rgb/depth remain raw Image topics unless compressed publishers exist",
    }

    fixed_prompt = args.prompt
    if fixed_prompt is None:
        fixed_prompt = input("本次采集固定 prompt: ").strip()
    session_meta["prompt"] = fixed_prompt
    write_json(session_dir / "session_meta.json", session_meta)

    for index in range(1, args.episodes + 1):
        episode_dir = session_dir / f"episode_{index:03d}"
        episode_dir.mkdir(parents=True, exist_ok=True)
        (episode_dir / "prompt.txt").write_text(fixed_prompt + "\n", encoding="utf-8")

        print(f"\n[{index}/{args.episodes}] prompt: {fixed_prompt}")
        print("准备好后按 Enter 开始采集。")
        input()

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        started_at = time.time()
        all_topics = [*chest_topics, *head_topics, *joint_topics]
        recorder = None
        try:
            recorder = start_local_recorder(
                name="adu_all",
                topics=all_topics,
                out=episode_dir / f"adu_all_{stamp}",
                log=episode_dir / "adu_all.log",
            )

            time.sleep(args.startup_wait)
            assert_recorders_started([recorder])
            print("采集中。按空格、Enter 或 q 结束本条；Ctrl+C 会先停止当前录制再退出。")
            wait_for_space()
        except KeyboardInterrupt:
            print("\n收到 Ctrl+C，正在停止当前录制。")
            raise
        finally:
            stop_recorders([r for r in [recorder] if r is not None])

        ended_at = time.time()
        meta = {
            "index": index,
            "prompt": fixed_prompt,
            "started_at_unix": started_at,
            "ended_at_unix": ended_at,
            "duration_s": round(ended_at - started_at, 3),
            "bag": str(recorder.out) if recorder else None,
            "stored_on": "ADU",
            "adu_topics": chest_topics,
            "hdu_source_topics": [*head_topics, *joint_topics],
            "recorded_topics": all_topics,
        }
        write_json(episode_dir / "metadata.json", meta)

        print("\n===== ros2 bag info =====")
        if recorder:
            print_bag_info(recorder.out)
        print(f"完成 episode {index}: {episode_dir}")

    print(f"\n全部采集完成：{session_dir}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect qpose joint states and ADU/HDU images into MCAP ros bags stored on ADU only.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--output-root", type=Path, default=Path("/agibot/data/home/agi/Desktop/bags"))
    parser.add_argument("--episodes", type=positive_int, default=1, help="采集条数")
    parser.add_argument("--prompt", default=None, help="本次采集所有 episode 共用的固定 prompt")
    parser.add_argument("--head-cameras", nargs="*", default=DEFAULT_HEAD_CAMERAS, help="头部相机名，或直接传 /topic")
    parser.add_argument("--head-stream", default=DEFAULT_HEAD_STREAM, choices=["h265", "image"], help="头部相机 stream；默认记录现有 H.265 压缩视频流")
    parser.add_argument("--chest-cameras", nargs="*", default=DEFAULT_CHEST_CAMERAS, help="胸部相机名，或直接传 /topic")
    parser.add_argument("--chest-streams", nargs="*", default=["rgb", "depth"], help="胸部相机 stream 名")
    parser.add_argument("--joint-topics", nargs="*", default=DEFAULT_JOINT_TOPICS, help="qpose/关节 topic")
    parser.add_argument("--no-joints", action="store_true", help="不记录关节 topic")
    parser.add_argument("--startup-wait", type=float, default=2.0, help="启动 recorder 后等待多少秒再允许动作")
    parser.add_argument("--no-stop-agivslam", dest="stop_agivslam", action="store_false", help="不自动停止 ADU agivslam")
    parser.set_defaults(stop_agivslam=True)
    return parser.parse_args()


def positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be >= 1")
    return number


def camera_topics(cameras: Iterable[str], kind: str, streams: list[str]) -> list[str]:
    topics: list[str] = []
    for camera in cameras:
        if camera.startswith("/"):
            topics.append(camera)
            continue
        if kind == "head":
            for stream in streams:
                topics.append(f"/hal/{camera}/{stream}")
        elif kind == "chest":
            for stream in streams:
                topics.append(f"/hal/{camera}/{stream}")
        else:
            raise ValueError(f"unknown camera kind: {kind}")
    return topics


def start_local_recorder(name: str, topics: list[str], out: Path, log: Path) -> LocalRecorder:
    qos_path = Path(tempfile.gettempdir()) / f"collect_{name}_{os.getpid()}.yaml"
    qos_path.write_text(qos_yaml(topics), encoding="utf-8")
    out.parent.mkdir(parents=True, exist_ok=True)
    log.parent.mkdir(parents=True, exist_ok=True)
    cmd = ros_shell(
        [
            "exec",
            "ros2",
            "bag",
            "record",
            "--qos-profile-overrides-path",
            str(qos_path),
            "-s",
            "mcap",
            "-o",
            str(out),
            "--topics",
            *topics,
        ]
    )
    log_handle = log.open("w", encoding="utf-8")
    proc = subprocess.Popen(
        ["bash", "-lc", cmd],
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    return LocalRecorder(name=name, out=out, log=log, proc=proc)


def start_remote_recorder(
    host: str,
    key: str | None,
    batch_mode: bool,
    name: str,
    topics: list[str],
    out: Path,
    log: Path,
) -> RemoteRecorder:
    pid_file = f"/tmp/collect_{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{os.getpid()}.pid"
    qos_path = f"/tmp/collect_{name}_{os.getpid()}.yaml"
    quoted_topics = " ".join(shlex.quote(t) for t in topics)
    record_cmd = ros_shell(
        [
            "exec",
            "ros2",
            "bag",
            "record",
            "--qos-profile-overrides-path",
            qos_path,
            "-s",
            "mcap",
            "-o",
            str(out),
            "--topics",
            *topics,
        ]
    )
    remote_script = f"""
set -Eeo pipefail
mkdir -p {shlex.quote(str(out.parent))}
cat > {shlex.quote(qos_path)} <<'YAML'
{qos_yaml(topics)}
YAML
nohup bash -lc {shlex.quote(record_cmd)} > {shlex.quote(str(log))} 2>&1 &
echo $! > {shlex.quote(pid_file)}
echo $!
"""
    result = run_ssh(host, remote_script, key=key, batch_mode=batch_mode, capture=True)
    pid = result.stdout.strip().splitlines()[-1]
    print(f"HDU recorder pid={pid}, topics={quoted_topics}")
    return RemoteRecorder(name=name, out=out, log=log, host=host, key=key, batch_mode=batch_mode, pid_file=pid_file)


def qos_yaml(topics: list[str]) -> str:
    lines = []
    for topic in topics:
        depth = 50 if "joint_state" in topic else 10
        lines.append(f"{topic}: {{history: keep_last, depth: {depth}, reliability: best_effort, durability: volatile}}")
    return "\n".join(lines) + "\n"


def ros_shell(argv: list[str]) -> str:
    command = " ".join(shlex.quote(arg) for arg in argv)
    return (
        "set +u; "
        f"source {shlex.quote(ROS_SETUP)}; "
        f"source {shlex.quote(AGIBOT_ENV)}; "
        "set -e; "
        f"{command}"
    )


def run_local_shell(command: str, error: str | None = None) -> subprocess.CompletedProcess[str]:
    full_command = (
        "set +u; "
        f"source {shlex.quote(ROS_SETUP)}; "
        f"source {shlex.quote(AGIBOT_ENV)}; "
        "set -e; "
        f"{command}"
    )
    result = subprocess.run(["bash", "-lc", full_command], text=True)
    if result.returncode != 0 and error:
        raise SystemExit(error)
    return result


def ssh_command(host: str, key: str | None, batch_mode: bool) -> list[str]:
    cmd = ["ssh", "-o", "ConnectTimeout=8"]
    if batch_mode:
        cmd += ["-o", "BatchMode=yes"]
    if key:
        cmd += ["-i", key]
    cmd += [host, "bash", "-s"]
    return cmd


def run_ssh(
    host: str,
    script: str,
    key: str | None,
    batch_mode: bool,
    capture: bool = False,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ssh_command(host, key, batch_mode),
        input=script,
        text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.STDOUT if capture else None,
        check=True,
    )


def topics_without_publishers(topics: list[str]) -> list[str]:
    missing: list[str] = []
    for topic in topics:
        result = subprocess.run(
            ["bash", "-lc", ros_shell(["ros2", "topic", "info", topic])],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        if result.returncode != 0 or "Publisher count: 0" in result.stdout:
            missing.append(topic)
    return missing


def assert_recorders_started(recorders: list[Recorder]) -> None:
    for recorder in recorders:
        if isinstance(recorder, LocalRecorder):
            if recorder.proc and recorder.proc.poll() is not None:
                raise RuntimeError(f"{recorder.name} recorder exited early. See {recorder.log}")
        if recorder.log.exists():
            text = recorder.log.read_text(encoding="utf-8", errors="replace")
            if "error" in text.lower() or "no such" in text.lower():
                print(f"警告：{recorder.name} 日志里出现异常字样，请检查 {recorder.log}")


def stop_recorders(recorders: list[Recorder]) -> None:
    for recorder in recorders:
        try:
            recorder.stop()
        except Exception as exc:
            print(f"停止 {recorder.name} 时出错：{exc}", file=sys.stderr)


def wait_process(proc: subprocess.Popen[str], timeout_s: float) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if proc.poll() is not None:
            return
        time.sleep(0.2)


def wait_for_space() -> None:
    stream = None
    should_close = False
    try:
        stream = open("/dev/tty", "r", encoding="utf-8", errors="ignore")
        should_close = True
    except OSError:
        if not sys.stdin.isatty():
            input("stdin 不是 TTY，按 Enter 结束本条。")
            return
        stream = sys.stdin

    fd = stream.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        while True:
            char = stream.read(1)
            if char in {" ", "\r", "\n", "q", "Q"}:
                print()
                return
            if char == "\x03":
                raise KeyboardInterrupt
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        if should_close and stream is not None:
            stream.close()


def print_bag_info(out: Path) -> None:
    result = subprocess.run(
        ["bash", "-lc", ros_shell(["ros2", "bag", "info", str(out)])],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    print(f"--- {out} ---")
    print(result.stdout.rstrip() or f"ros2 bag info failed with code {result.returncode}")


def print_remote_bag_info(host: str, out: Path, key: str | None, batch_mode: bool) -> None:
    script = ros_shell(["ros2", "bag", "info", str(out)])
    print(f"--- {host}:{out} ---")
    try:
        result = run_ssh(host, script, key=key, batch_mode=batch_mode, capture=True)
        print(result.stdout.rstrip())
    except subprocess.CalledProcessError as exc:
        print(exc.stdout or f"remote ros2 bag info failed with code {exc.returncode}")


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\n已退出。")
        raise SystemExit(130)
