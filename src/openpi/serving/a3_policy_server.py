"""A3 policy server that pairs external D455 RGB with ADU source-state samples."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
import http
import logging
import time
import traceback
from typing import Any

from openpi_client import base_policy as _base_policy
from openpi_client import msgpack_numpy
import websockets.asyncio.server as _server
import websockets.frames

from openpi.serving import a3_time_alignment

LOGGER = logging.getLogger(__name__)


def external_camera_metadata(base: Mapping[str, Any], camera: Any) -> dict[str, Any]:
    return {
        **base,
        "a3_observation": {
            "protocol": a3_time_alignment.PROTOCOL,
            "camera_source": a3_time_alignment.CAMERA_SOURCE_EXTERNAL_D455,
            "clock_sync": {
                "required": True,
                "method": "ntp_min_rtt",
                "probe_count": 7,
                "interval_s": 5.0,
            },
            "alignment": {
                "max_clock_uncertainty_ms": 5.0,
                "max_error_bound_ms": 20.0,
                "max_calibration_age_s": 8.0,
            },
            "camera": dict(camera.metadata),
        },
    }


def build_model_observation(
    observation: Mapping[str, Any],
    aligned: a3_time_alignment.AlignedObservation,
) -> dict[str, Any]:
    if "images" in observation:
        raise ValueError("external-d455 requests must not contain an ADU image")
    return {
        **observation,
        "state": aligned.state,
        # Keep the training/deployment key stable; in this mode its value is D455 RGB.
        "images": {"chest_rgb": aligned.frame.color},
    }


class A3ExternalCameraPolicyServer:
    def __init__(
        self,
        policy: _base_policy.BasePolicy,
        camera: Any,
        *,
        host: str = "0.0.0.0",
        port: int = 8000,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        self._policy = policy
        self._camera = camera
        self._host = host
        self._port = port
        self._metadata = external_camera_metadata(metadata or {}, camera)
        self._active_session = False
        logging.getLogger("websockets.server").setLevel(logging.INFO)

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata

    def serve_forever(self) -> None:
        self._camera.start()
        try:
            frame = self._camera.wait_ready()
            LOGGER.info(
                "external D455 ready serial=%s frame=%d size=%dx%d",
                self._camera.metadata["serial"],
                frame.frame_id,
                self._camera.metadata["width"],
                self._camera.metadata["height"],
            )
            asyncio.run(self.run())
        finally:
            self._camera.stop()

    async def run(self) -> None:
        async with _server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
            process_request=_health_check,
        ) as server:
            await server.serve_forever()

    async def _handler(self, websocket: _server.ServerConnection) -> None:
        if self._active_session:
            await websocket.close(code=1013, reason="another A3 policy session is active")
            return
        self._active_session = True
        packer = msgpack_numpy.Packer()
        clock: a3_time_alignment.ClockSync | None = None
        expected_sequence = 0
        session_generation: int | None = None
        previous_total_s: float | None = None
        LOGGER.info("A3 external-camera connection from %s opened", websocket.remote_address)
        try:
            await websocket.send(packer.pack(self._metadata))
            async for raw in websocket:
                server_receive_monotonic_ns = time.monotonic_ns()
                started = time.monotonic()
                message = msgpack_numpy.unpackb(raw)
                if not isinstance(message, dict):
                    raise ValueError("A3 transport message must be a mapping")
                message_type = message.get("type")
                if message_type == "clock_probe":
                    await websocket.send(packer.pack(self._clock_probe_ack(message, server_receive_monotonic_ns)))
                    continue
                if message_type == "clock_sync_result":
                    clock = self._parse_clock_sync(message)
                    await websocket.send(
                        packer.pack(
                            {
                                "type": "clock_sync_ack",
                                "protocol": a3_time_alignment.PROTOCOL,
                                "accepted": True,
                                "monotonic_uncertainty_ns": clock.monotonic_uncertainty_ns,
                            }
                        )
                    )
                    continue
                if message_type != "inference":
                    raise ValueError(f"unexpected A3 transport message type {message_type!r}")
                if clock is None:
                    raise RuntimeError("clock synchronization is required before inference")
                if message.get("protocol") != a3_time_alignment.PROTOCOL:
                    raise ValueError("A3 external-camera protocol mismatch")
                sequence = int(message.get("sequence", -1))
                if sequence != expected_sequence:
                    raise ValueError(f"inference sequence {sequence} != {expected_sequence}")
                observation = message.get("observation")
                if not isinstance(observation, dict):
                    raise ValueError("inference observation must be a mapping")
                select_started = time.monotonic()
                aligned = await asyncio.to_thread(
                    self._camera.aligned_observation,
                    message.get("source_state_window"),
                    clock,
                )
                generation = int(aligned.metadata["camera_stream_generation"])
                if session_generation is None:
                    session_generation = generation
                elif generation != session_generation:
                    raise RuntimeError(f"D455 stream restarted during session: {session_generation} -> {generation}")
                select_ms = (time.monotonic() - select_started) * 1000.0
                model_observation = build_model_observation(observation, aligned)
                infer_started = time.monotonic()
                result = dict(await asyncio.to_thread(self._policy.infer, model_observation))
                infer_ms = (time.monotonic() - infer_started) * 1000.0
                result["observation_alignment"] = {
                    **aligned.metadata,
                    "aligned_state": aligned.state.tolist(),
                }
                result["server_timing"] = {
                    "rgb_state_select_ms": select_ms,
                    "infer_ms": infer_ms,
                }
                LOGGER.info(
                    "external observation sequence=%d rgb_frame=%d max_skew_ms=%.3f "
                    "error_bound_ms=%.3f frame_age_ms=%.3f infer_ms=%.1f",
                    sequence,
                    aligned.metadata["rgb_frame_id"],
                    aligned.metadata["source_max_abs_skew_ms"],
                    aligned.metadata["source_max_error_bound_ms"],
                    aligned.metadata["frame_age_ms"],
                    infer_ms,
                )
                if previous_total_s is not None:
                    result["server_timing"]["prev_total_ms"] = previous_total_s * 1000.0
                await websocket.send(packer.pack(result))
                expected_sequence += 1
                previous_total_s = time.monotonic() - started
        except websockets.ConnectionClosed:
            LOGGER.info("A3 external-camera connection from %s closed", websocket.remote_address)
        except Exception:
            await websocket.send(traceback.format_exc())
            await websocket.close(
                code=websockets.frames.CloseCode.INTERNAL_ERROR,
                reason="A3 external-camera server error; traceback sent in previous frame",
            )
            raise
        finally:
            self._active_session = False

    @staticmethod
    def _clock_probe_ack(message: Mapping[str, Any], receive_ns: int) -> dict[str, Any]:
        if message.get("protocol") != a3_time_alignment.PROTOCOL:
            raise ValueError("clock probe protocol mismatch")
        sequence = int(message.get("sequence", -1))
        actor_send_ns = int(message.get("actor_send_monotonic_ns", -1))
        if sequence < 0 or actor_send_ns <= 0:
            raise ValueError("invalid clock probe")
        return {
            "type": "clock_probe_ack",
            "protocol": a3_time_alignment.PROTOCOL,
            "sequence": sequence,
            "actor_send_monotonic_ns": actor_send_ns,
            "server_receive_monotonic_ns": receive_ns,
            "server_send_monotonic_ns": time.monotonic_ns(),
        }

    @staticmethod
    def _parse_clock_sync(message: Mapping[str, Any]) -> a3_time_alignment.ClockSync:
        if message.get("protocol") != a3_time_alignment.PROTOCOL:
            raise ValueError("clock synchronization protocol mismatch")
        uncertainty_ns = int(message.get("monotonic_uncertainty_ns", -1))
        probe_count = int(message.get("probe_count", 0))
        sample_ns = int(message.get("server_sample_monotonic_ns", -1))
        if not 0 <= uncertainty_ns <= a3_time_alignment.MAX_CLOCK_UNCERTAINTY_NS:
            raise ValueError(f"clock uncertainty exceeds 5 ms: {uncertainty_ns / 1_000_000.0:.3f} ms")
        if probe_count < 3 or sample_ns <= 0:
            raise ValueError("invalid clock synchronization result")
        return a3_time_alignment.ClockSync(
            server_minus_actor_monotonic_ns=int(message["server_minus_actor_monotonic_ns"]),
            monotonic_uncertainty_ns=uncertainty_ns,
            probe_count=probe_count,
            received_monotonic_ns=time.monotonic_ns(),
            sample_monotonic_ns=sample_ns,
        )


def _health_check(
    connection: _server.ServerConnection,
    request: _server.Request,
) -> _server.Response | None:
    if request.path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    return None
