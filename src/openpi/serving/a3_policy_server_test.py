import asyncio
import time
from types import SimpleNamespace
from typing import ClassVar

import numpy as np
from openpi_client import msgpack_numpy
import pytest

from openpi.serving import a3_policy_server
from openpi.serving import a3_time_alignment


def test_external_metadata_declares_camera_and_alignment_contract():
    camera = SimpleNamespace(metadata={"serial": "318122300934", "width": 640, "height": 480, "fps": 30})

    metadata = a3_policy_server.external_camera_metadata({"policy": "test"}, camera)

    assert metadata["policy"] == "test"
    contract = metadata["a3_observation"]
    assert contract["protocol"] == a3_time_alignment.PROTOCOL
    assert contract["camera_source"] == "external-d455"
    assert contract["clock_sync"]["probe_count"] == 7
    assert contract["alignment"]["max_error_bound_ms"] == 20.0


def test_external_adapter_replaces_state_and_injects_rgb_without_mutating_request():
    request_state = np.zeros(37, dtype=np.float32)
    aligned_state = np.arange(37, dtype=np.float32)
    image = np.full((8, 10, 3), 77, dtype=np.uint8)
    observation = {"state": request_state, "prompt": "move cup", "active_sides": ["right"]}
    aligned = a3_time_alignment.AlignedObservation(
        frame=SimpleNamespace(color=image),
        state=aligned_state,
        metadata={},
    )

    result = a3_policy_server.build_model_observation(observation, aligned)

    np.testing.assert_array_equal(result["state"], aligned_state)
    np.testing.assert_array_equal(result["images"]["chest_rgb"], image)
    np.testing.assert_array_equal(observation["state"], request_state)
    assert "images" not in observation


def test_external_adapter_rejects_client_image():
    aligned = a3_time_alignment.AlignedObservation(
        frame=SimpleNamespace(color=np.zeros((2, 2, 3), dtype=np.uint8)),
        state=np.zeros(37, dtype=np.float32),
        metadata={},
    )
    with pytest.raises(ValueError, match="must not contain"):
        a3_policy_server.build_model_observation({"images": {}}, aligned)


def test_server_handler_injects_aligned_observation_before_policy():
    class CapturingPolicy:
        def __init__(self):
            self.observation = None

        def infer(self, observation):
            self.observation = observation
            return {"actions": np.zeros((50, 37), dtype=np.float32)}

    image = np.full((6, 8, 3), 31, dtype=np.uint8)
    aligned_state = np.arange(37, dtype=np.float32)

    class Camera:
        metadata: ClassVar = {"serial": "test", "width": 8, "height": 6, "fps": 30}

        def aligned_observation(self, window, clock):
            assert window == {"test": "window"}
            assert clock.probe_count == 7
            return a3_time_alignment.AlignedObservation(
                frame=SimpleNamespace(color=image),
                state=aligned_state,
                metadata={
                    "camera_stream_generation": 0,
                    "rgb_frame_id": 7,
                    "source_max_abs_skew_ms": 2.0,
                    "source_max_error_bound_ms": 3.0,
                    "frame_age_ms": 4.0,
                },
            )

    packer = msgpack_numpy.Packer()
    now_ns = time.monotonic_ns()
    incoming = [
        packer.pack(
            {
                "type": "clock_sync_result",
                "protocol": a3_time_alignment.PROTOCOL,
                "server_minus_actor_monotonic_ns": 0,
                "monotonic_uncertainty_ns": 1_000_000,
                "server_sample_monotonic_ns": now_ns,
                "probe_count": 7,
            }
        ),
        packer.pack(
            {
                "type": "inference",
                "protocol": a3_time_alignment.PROTOCOL,
                "sequence": 0,
                "observation": {"state": np.zeros(37), "prompt": "move cup"},
                "source_state_window": {"test": "window"},
            }
        ),
    ]

    class Websocket:
        remote_address = ("test", 1)

        def __init__(self):
            self.sent = []

        def __aiter__(self):
            return self

        async def __anext__(self):
            if not incoming:
                raise StopAsyncIteration
            return incoming.pop(0)

        async def send(self, value):
            self.sent.append(value)

        async def close(self, **_kwargs):
            return None

    policy = CapturingPolicy()
    websocket = Websocket()
    server = a3_policy_server.A3ExternalCameraPolicyServer(policy, Camera(), metadata={})

    asyncio.run(server._handler(websocket))  # noqa: SLF001

    assert len(websocket.sent) == 3
    response = msgpack_numpy.unpackb(websocket.sent[-1])
    assert response["actions"].shape == (50, 37)
    assert response["observation_alignment"]["source_max_error_bound_ms"] == 3.0
    np.testing.assert_array_equal(response["observation_alignment"]["aligned_state"], aligned_state)
    np.testing.assert_array_equal(policy.observation["state"], aligned_state)
    np.testing.assert_array_equal(policy.observation["images"]["chest_rgb"], image)
