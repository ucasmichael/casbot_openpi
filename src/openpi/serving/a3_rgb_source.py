"""Buffered RGB capture for the external A3 D455 inference mode."""

from __future__ import annotations

from collections import deque
import dataclasses
import logging
import threading
import time
from typing import Any

import numpy as np

from openpi.serving import a3_time_alignment

LOGGER = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class RGBSourceConfig:
    serial: str
    width: int = 640
    height: int = 480
    fps: int = 30
    buffer_frames: int = 45
    mode: str = "realsense"


@dataclasses.dataclass(frozen=True)
class RGBFrame:
    host_wall_ns: int
    host_monotonic_ns: int
    frame_id: int
    device_timestamp_ms: float
    timestamp_domain: str
    color: np.ndarray
    stream_generation: int = 0


class BufferedRGBSource:
    def __init__(self, config: RGBSourceConfig) -> None:
        if config.mode not in {"realsense", "synthetic"}:
            raise ValueError("RGB source mode must be realsense or synthetic")
        if min(config.width, config.height, config.fps, config.buffer_frames) <= 0:
            raise ValueError("RGB dimensions, FPS, and buffer size must be positive")
        self.config = config
        self._frames: deque[RGBFrame] = deque(maxlen=config.buffer_frames)
        self._condition = threading.Condition()
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._error: BaseException | None = None
        self._last_capture_error: str | None = None
        self._restart_count = 0
        self._stream_generation = 0
        self._thread = threading.Thread(target=self._capture_loop, name="openpi-a3-d455-rgb", daemon=True)

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "serial": self.config.serial,
            "width": self.config.width,
            "height": self.config.height,
            "fps": self.config.fps,
        }

    def start(self) -> None:
        self._thread.start()

    def wait_ready(self, timeout_s: float = 8.0) -> RGBFrame:
        if not self._ready.wait(timeout_s):
            detail = f"; last_error={self._last_capture_error}" if self._last_capture_error else ""
            raise TimeoutError(f"external D455 did not produce an RGB frame{detail}")
        with self._condition:
            self.assert_healthy()
            if not self._frames:
                raise RuntimeError("external D455 reported ready without an RGB frame")
            return self._frames[-1]

    def stop(self) -> None:
        self._stop.set()
        with self._condition:
            self._condition.notify_all()
        if self._thread.ident is not None:
            self._thread.join(timeout=3.0)
            if self._thread.is_alive():
                raise RuntimeError("external D455 capture thread did not stop")

    def assert_healthy(self) -> None:
        if self._error is not None:
            raise RuntimeError(f"external D455 capture failed: {self._error}") from self._error
        if self._thread.ident is not None and not self._thread.is_alive() and not self._stop.is_set():
            raise RuntimeError("external D455 capture thread is not running")

    def frames(self) -> list[RGBFrame]:
        with self._condition:
            self.assert_healthy()
            return list(self._frames)

    def aligned_observation(
        self,
        source_state_window: dict[str, Any],
        clock: a3_time_alignment.ClockSync,
    ) -> a3_time_alignment.AlignedObservation:
        result = a3_time_alignment.select_aligned_frame(self.frames(), source_state_window, clock)
        result.metadata.update(
            self.metadata,
            camera_stream_generation=result.frame.stream_generation,
            camera_restart_count=self._restart_count,
            camera_last_error=self._last_capture_error,
        )
        return result

    def _append(self, frame: RGBFrame) -> None:
        with self._condition:
            self._frames.append(frame)
            self._ready.set()
            self._condition.notify_all()

    def _capture_loop(self) -> None:
        try:
            if self.config.mode == "synthetic":
                self._synthetic_loop()
            else:
                self._realsense_loop()
        except BaseException as error:
            self._error = error
            self._ready.set()
            with self._condition:
                self._condition.notify_all()

    def _synthetic_loop(self) -> None:
        period_s = 1.0 / self.config.fps
        sequence = 0
        deadline = time.monotonic()
        while not self._stop.is_set():
            wall_ns = time.time_ns()
            monotonic_ns = time.monotonic_ns()
            self._append(
                RGBFrame(
                    host_wall_ns=wall_ns,
                    host_monotonic_ns=monotonic_ns,
                    frame_id=sequence,
                    device_timestamp_ms=wall_ns / 1_000_000.0,
                    timestamp_domain="global_time",
                    color=np.zeros((self.config.height, self.config.width, 3), dtype=np.uint8),
                    stream_generation=self._stream_generation,
                )
            )
            sequence += 1
            deadline += period_s
            self._stop.wait(max(0.0, deadline - time.monotonic()))

    def _realsense_loop(self) -> None:
        backoff_s = 0.25
        while not self._stop.is_set():
            generation = self._stream_generation
            try:
                self._run_realsense_pipeline(generation)
                if self._stop.is_set():
                    return
                raise RuntimeError("RealSense pipeline stopped unexpectedly")
            except Exception as error:
                if self._stop.is_set():
                    return
                with self._condition:
                    self._last_capture_error = f"{type(error).__name__}: {error}"
                    self._restart_count += 1
                    self._stream_generation += 1
                    self._frames.clear()
                    self._condition.notify_all()
                LOGGER.warning(
                    "D455 %s capture failed; restart=%d backoff_s=%.2f error=%s",
                    self.config.serial,
                    self._restart_count,
                    backoff_s,
                    self._last_capture_error,
                )
                self._stop.wait(backoff_s)
                backoff_s = min(backoff_s * 2.0, 2.0)

    def _run_realsense_pipeline(self, generation: int) -> None:
        try:
            import pyrealsense2 as rs
        except ImportError as error:
            raise RuntimeError("external-d455 mode requires pyrealsense2 in the policy-server environment") from error

        context = rs.context()
        serials = {device.get_info(rs.camera_info.serial_number) for device in context.query_devices()}
        if self.config.serial not in serials:
            raise RuntimeError(f"RealSense {self.config.serial} not found; available={sorted(serials)}")
        pipeline = rs.pipeline()
        rs_config = rs.config()
        rs_config.enable_device(self.config.serial)
        rs_config.enable_stream(
            rs.stream.color,
            self.config.width,
            self.config.height,
            rs.format.bgr8,
            self.config.fps,
        )
        started = False
        restored_options: list[tuple[Any, float]] = []
        previous_wall_offset_ns: int | None = None
        try:
            for device in context.query_devices():
                if device.get_info(rs.camera_info.serial_number) != self.config.serial:
                    continue
                for sensor in device.query_sensors():
                    if sensor.supports(rs.option.global_time_enabled):
                        old = sensor.get_option(rs.option.global_time_enabled)
                        if old != 1:
                            sensor.set_option(rs.option.global_time_enabled, 1)
                            restored_options.append((sensor, old))
            pipeline.start(rs_config)
            started = True
            while not self._stop.is_set():
                frames = pipeline.wait_for_frames(timeout_ms=1000)
                color_frame = frames.get_color_frame()
                if not color_frame:
                    continue
                host_wall_ns = time.time_ns()
                host_monotonic_ns = time.monotonic_ns()
                wall_offset_ns = host_wall_ns - host_monotonic_ns
                if previous_wall_offset_ns is not None and abs(wall_offset_ns - previous_wall_offset_ns) > 5_000_000:
                    raise RuntimeError("policy-server wall clock stepped by more than 5 ms")
                previous_wall_offset_ns = wall_offset_ns
                bgr = np.asanyarray(color_frame.get_data())
                if bgr.shape != (self.config.height, self.config.width, 3) or bgr.dtype != np.uint8:
                    raise RuntimeError(f"unexpected D455 RGB frame shape or dtype: {bgr.shape} {bgr.dtype}")
                self._append(
                    RGBFrame(
                        host_wall_ns=host_wall_ns,
                        host_monotonic_ns=host_monotonic_ns,
                        frame_id=int(color_frame.get_frame_number()),
                        device_timestamp_ms=float(color_frame.get_timestamp()),
                        timestamp_domain=str(color_frame.get_frame_timestamp_domain()).rsplit(".", 1)[-1],
                        color=np.ascontiguousarray(bgr[:, :, ::-1]),
                        stream_generation=generation,
                    )
                )
        finally:
            try:
                if started:
                    pipeline.stop()
            finally:
                for sensor, old in restored_options:
                    sensor.set_option(rs.option.global_time_enabled, old)
