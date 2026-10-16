"""Small timeout-bounded OpenPI WebSocket client used by A3 deployment."""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass
from typing import Any

import msgpack
import numpy as np
from websockets.sync.client import ClientConnection, connect


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


class PolicyClient:
    def __init__(
        self,
        host: str,
        port: int,
        *,
        connect_timeout_s: float,
        infer_timeout_s: float,
    ) -> None:
        self.uri = f"ws://{host}:{port}"
        self.connect_timeout_s = connect_timeout_s
        self.infer_timeout_s = infer_timeout_s
        self.connection: ClientConnection | None = None
        self.metadata: dict[str, Any] = {}

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
        return metadata

    def infer(self, observation: dict[str, Any]) -> tuple[dict[str, Any], int, int]:
        if self.connection is None:
            self.open()
        assert self.connection is not None
        request = pack_message(observation)
        self.connection.send(request)
        response = self.connection.recv(timeout=self.infer_timeout_s)
        if isinstance(response, str):
            raise RuntimeError(f"policy server error: {response}")
        result = unpack_message(response)
        if not isinstance(result, dict):
            raise RuntimeError(f"policy result must be a dict, got {type(result)!r}")
        return result, len(request), len(response)

    def close(self) -> None:
        if self.connection is not None:
            try:
                self.connection.close()
            except Exception:
                pass
            self.connection = None

    def __enter__(self) -> "PolicyClient":
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
        self.requests: queue.Queue[tuple[int, dict[str, Any]] | None] = queue.Queue(maxsize=1)
        self.responses: queue.Queue[InferenceResponse] = queue.Queue()
        self.thread = threading.Thread(target=self._run, name="a3-policy-inference", daemon=True)

    def start(self) -> None:
        self.thread.start()

    def submit(self, sequence: int, observation: dict[str, Any]) -> None:
        self.requests.put_nowait((sequence, observation))

    def poll(self) -> InferenceResponse | None:
        try:
            return self.responses.get_nowait()
        except queue.Empty:
            return None

    def stop(self) -> None:
        try:
            self.requests.put_nowait(None)
        except queue.Full:
            pass
        self.client.close()
        self.thread.join(timeout=2.0)

    def _run(self) -> None:
        while True:
            request = self.requests.get()
            if request is None:
                break
            sequence, observation = request
            started = time.perf_counter()
            try:
                result, request_bytes, response_bytes = self.client.infer(observation)
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

