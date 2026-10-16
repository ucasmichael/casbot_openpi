"""Small timeout-bounded OpenPI WebSocket client used by A3 deployment."""

from __future__ import annotations

from collections.abc import Callable
import contextlib
from dataclasses import dataclass
import queue
import threading
import time
from typing import Any

import msgpack
import numpy as np
from websockets.sync.client import ClientConnection
from websockets.sync.client import connect

A3_PROTOCOL = "openpi.a3.external-camera.v1"
CAMERA_SOURCE_ADU_D457 = "adu-d457"
CAMERA_SOURCE_EXTERNAL_D455 = "external-d455"
CAMERA_SOURCES = {CAMERA_SOURCE_ADU_D457, CAMERA_SOURCE_EXTERNAL_D455}


def _pack_numpy(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        if value.dtype.kind in ("V", "O", "c"):
            raise ValueError(f"unsupported NumPy dtype {value.dtype}")
        return {
            b"__ndarray__": True,
            b"data": value.tobytes(),
            b"dtype": value.dtype.str,
            b"shape": value.shape,
        }
    if isinstance(value, np.generic):
        return {
            b"__npgeneric__": True,
            b"data": value.item(),
            b"dtype": value.dtype.str,
        }
    raise TypeError(f"cannot serialize {type(value)!r}")


def _unpack_numpy(value: dict) -> Any:
    if b"__ndarray__" in value:
        return np.ndarray(
            buffer=value[b"data"],
            dtype=np.dtype(value[b"dtype"]),
            shape=value[b"shape"],
        )
    if b"__npgeneric__" in value:
        return np.dtype(value[b"dtype"]).type(value[b"data"])
    return value


def pack_message(value: Any) -> bytes:
    return msgpack.packb(value, default=_pack_numpy)


def unpack_message(raw: bytes) -> Any:
    return msgpack.unpackb(raw, object_hook=_unpack_numpy)


@dataclass(frozen=True)
class PolicyRequest:
    observation: dict[str, Any]
    source_state_window: dict[str, Any] | None = None


@dataclass(frozen=True)
class ClockSample:
    server_minus_actor_monotonic_ns: int
    monotonic_uncertainty_ns: int
    server_sample_monotonic_ns: int


class PolicyClient:
    def __init__(
        self,
        host: str,
        port: int,
        *,
        connect_timeout_s: float,
        infer_timeout_s: float,
        expected_camera_source: str | None = None,
    ) -> None:
        self.uri = f"ws://{host}:{port}"
        self.connect_timeout_s = connect_timeout_s
        self.infer_timeout_s = infer_timeout_s
        self.expected_camera_source = expected_camera_source
        self.connection: ClientConnection | None = None
        self.metadata: dict[str, Any] = {}
        self.camera_source = CAMERA_SOURCE_ADU_D457
        self.clock_sync_interval_s = 5.0
        self.clock_probe_count = 7
        self._next_clock_sync = 0.0
        self._sequence = 0

    def open(self) -> dict[str, Any]:
        self.close()
        self.connection = connect(
            self.uri,
            open_timeout=self.connect_timeout_s,
            close_timeout=1.0,
            compression=None,
            max_size=None,
        )
        raw = self.connection.recv(timeout=self.connect_timeout_s)
        if not isinstance(raw, bytes):
            raise RuntimeError(f"policy server returned text metadata: {raw}")
        metadata = unpack_message(raw)
        if not isinstance(metadata, dict):
            raise RuntimeError(f"policy metadata must be a dict, got {type(metadata)!r}")
        self.metadata = metadata
        transport = metadata.get("a3_observation")
        if transport is None:
            self.camera_source = CAMERA_SOURCE_ADU_D457
        elif not isinstance(transport, dict):
            raise RuntimeError("policy a3_observation metadata must be a dict")
        else:
            self.camera_source = str(transport.get("camera_source", ""))
            if self.camera_source not in CAMERA_SOURCES:
                raise RuntimeError(f"unsupported server camera source {self.camera_source!r}")
            if self.camera_source == CAMERA_SOURCE_EXTERNAL_D455:
                if transport.get("protocol") != A3_PROTOCOL:
                    raise RuntimeError("external-camera protocol mismatch")
                clock = transport.get("clock_sync")
                if not isinstance(clock, dict) or clock.get("method") != "ntp_min_rtt":
                    raise RuntimeError("external-camera server requires unsupported clock synchronization")
                self.clock_probe_count = int(clock.get("probe_count", 7))
                self.clock_sync_interval_s = float(clock.get("interval_s", 5.0))
                if not 3 <= self.clock_probe_count <= 15:
                    raise RuntimeError("clock probe count must be in [3,15]")
                if not 1.0 <= self.clock_sync_interval_s <= 30.0:
                    raise RuntimeError("clock sync interval must be in [1,30] seconds")
                self._synchronize_clock()
                self._next_clock_sync = time.monotonic() + self.clock_sync_interval_s
        if self.expected_camera_source is not None and self.camera_source != self.expected_camera_source:
            raise RuntimeError(
                f"policy camera source changed: expected={self.expected_camera_source!r} actual={self.camera_source!r}"
            )
        self._sequence = 0
        return metadata

    def infer(
        self,
        request_value: PolicyRequest | Callable[[], PolicyRequest],
    ) -> tuple[dict[str, Any], int, int]:
        if self.connection is None:
            self.open()
        assert self.connection is not None
        if self.camera_source == CAMERA_SOURCE_EXTERNAL_D455 and time.monotonic() >= self._next_clock_sync:
            self._synchronize_clock()
            self._next_clock_sync = time.monotonic() + self.clock_sync_interval_s
        request_value = request_value() if callable(request_value) else request_value
        if not isinstance(request_value, PolicyRequest):
            raise TypeError("policy inference requires PolicyRequest")
        observation = request_value.observation
        if self.camera_source == CAMERA_SOURCE_EXTERNAL_D455:
            if request_value.source_state_window is None:
                raise RuntimeError("external-d455 inference requires an ADU source-state window")
            if "images" in observation:
                raise RuntimeError("external-d455 inference must not send an ADU image")
            payload: dict[str, Any] = {
                "type": "inference",
                "protocol": A3_PROTOCOL,
                "sequence": self._sequence,
                "observation": observation,
                "source_state_window": request_value.source_state_window,
            }
        else:
            if request_value.source_state_window is not None:
                raise RuntimeError("adu-d457 inference must not send an external alignment window")
            if "images" not in observation:
                raise RuntimeError("adu-d457 inference requires an ADU image")
            payload = observation
        request = pack_message(payload)
        self.connection.send(request)
        response = self.connection.recv(timeout=self.infer_timeout_s)
        if isinstance(response, str):
            raise RuntimeError(f"policy server error: {response}")
        result = unpack_message(response)
        if not isinstance(result, dict):
            raise RuntimeError(f"policy result must be a dict, got {type(result)!r}")
        self._sequence += 1
        return result, len(request), len(response)

    def _receive_mapping(self, *, timeout_s: float) -> dict[str, Any]:
        assert self.connection is not None
        raw = self.connection.recv(timeout=timeout_s)
        if isinstance(raw, str):
            raise RuntimeError(f"policy server error during handshake: {raw}")
        value = unpack_message(raw)
        if not isinstance(value, dict):
            raise RuntimeError(f"policy handshake response must be a dict, got {type(value)!r}")
        return value

    def _synchronize_clock(self) -> ClockSample:
        assert self.connection is not None
        samples: list[ClockSample] = []
        for sequence in range(self.clock_probe_count):
            actor_send_ns = time.monotonic_ns()
            self.connection.send(
                pack_message(
                    {
                        "type": "clock_probe",
                        "protocol": A3_PROTOCOL,
                        "sequence": sequence,
                        "actor_send_monotonic_ns": actor_send_ns,
                    }
                )
            )
            acknowledgement = self._receive_mapping(timeout_s=self.connect_timeout_s)
            actor_receive_ns = time.monotonic_ns()
            if acknowledgement.get("type") != "clock_probe_ack":
                raise RuntimeError("expected clock_probe_ack")
            if acknowledgement.get("protocol") != A3_PROTOCOL:
                raise RuntimeError("clock probe protocol mismatch")
            if int(acknowledgement.get("sequence", -1)) != sequence:
                raise RuntimeError("clock probe sequence mismatch")
            server_receive_ns = int(acknowledgement.get("server_receive_monotonic_ns", -1))
            server_send_ns = int(acknowledgement.get("server_send_monotonic_ns", -1))
            server_elapsed_ns = server_send_ns - server_receive_ns
            round_trip_ns = (actor_receive_ns - actor_send_ns) - server_elapsed_ns
            if server_elapsed_ns < 0 or round_trip_ns < 0:
                raise RuntimeError("invalid monotonic clock exchange")
            samples.append(
                ClockSample(
                    server_minus_actor_monotonic_ns=(
                        (server_receive_ns - actor_send_ns) + (server_send_ns - actor_receive_ns)
                    )
                    // 2,
                    monotonic_uncertainty_ns=(round_trip_ns + 1) // 2,
                    server_sample_monotonic_ns=(server_receive_ns + server_send_ns) // 2,
                )
            )
        selected = min(samples, key=lambda sample: sample.monotonic_uncertainty_ns)
        if selected.monotonic_uncertainty_ns > 5_000_000:
            values = [round(sample.monotonic_uncertainty_ns / 1_000_000.0, 3) for sample in samples]
            raise RuntimeError(f"clock synchronization uncertainty exceeds 5 ms: half_rtt_ms={values}")
        self.connection.send(
            pack_message(
                {
                    "type": "clock_sync_result",
                    "protocol": A3_PROTOCOL,
                    "server_minus_actor_monotonic_ns": selected.server_minus_actor_monotonic_ns,
                    "monotonic_uncertainty_ns": selected.monotonic_uncertainty_ns,
                    "server_sample_monotonic_ns": selected.server_sample_monotonic_ns,
                    "probe_count": len(samples),
                }
            )
        )
        acknowledgement = self._receive_mapping(timeout_s=self.connect_timeout_s)
        if acknowledgement.get("type") != "clock_sync_ack" or not acknowledgement.get("accepted"):
            raise RuntimeError("policy server rejected clock synchronization")
        return selected

    def close(self) -> None:
        if self.connection is not None:
            with contextlib.suppress(Exception):
                self.connection.close()
            self.connection = None

    def __enter__(self) -> PolicyClient:
        self.open()
        return self

    def __exit__(self, *_args: Any) -> None:
        self.close()


@dataclass(frozen=True)
class InferenceResponse:
    sequence: int
    result: dict[str, Any] | None
    error: Exception | None
    round_trip_ms: float
    request_bytes: int
    response_bytes: int


class InferenceWorker:
    """Own the WebSocket in one thread so the control loop never blocks on inference."""

    def __init__(self, client: PolicyClient) -> None:
        self.client = client
        self.requests: queue.Queue[tuple[int, PolicyRequest | Callable[[], PolicyRequest]] | None] = queue.Queue(
            maxsize=1
        )
        self.responses: queue.Queue[InferenceResponse] = queue.Queue()
        self.thread = threading.Thread(target=self._run, name="a3-policy-inference", daemon=True)

    def start(self) -> None:
        self.thread.start()

    def submit(
        self,
        sequence: int,
        request: PolicyRequest | Callable[[], PolicyRequest],
    ) -> None:
        self.requests.put_nowait((sequence, request))

    def poll(self) -> InferenceResponse | None:
        try:
            return self.responses.get_nowait()
        except queue.Empty:
            return None

    def stop(self) -> None:
        with contextlib.suppress(queue.Full):
            self.requests.put_nowait(None)
        self.client.close()
        self.thread.join(timeout=2.0)

    def _run(self) -> None:
        while True:
            request = self.requests.get()
            if request is None:
                break
            sequence, inference_request = request
            started = time.perf_counter()
            try:
                result, request_bytes, response_bytes = self.client.infer(inference_request)
                response = InferenceResponse(
                    sequence=sequence,
                    result=result,
                    error=None,
                    round_trip_ms=(time.perf_counter() - started) * 1000,
                    request_bytes=request_bytes,
                    response_bytes=response_bytes,
                )
            except Exception as error:
                self.client.close()
                response = InferenceResponse(
                    sequence=sequence,
                    result=None,
                    error=error,
                    round_trip_ms=(time.perf_counter() - started) * 1000,
                    request_bytes=0,
                    response_bytes=0,
                )
            self.responses.put(response)
        self.client.close()
