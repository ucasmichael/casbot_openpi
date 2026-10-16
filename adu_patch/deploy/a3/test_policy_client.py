from deploy.a3 import policy_client
import numpy as np


class FakeConnection:
    def __init__(self, metadata):
        self.metadata = metadata
        self.sent = []
        self.first_receive = True
        self.closed = False

    def send(self, raw):
        self.sent.append(policy_client.unpack_message(raw))

    def recv(self, timeout):
        del timeout
        if self.first_receive:
            self.first_receive = False
            return policy_client.pack_message(self.metadata)
        message = self.sent[-1]
        if message.get("type") == "clock_probe":
            server_ns = int(message["actor_send_monotonic_ns"]) + 2_000_000_000
            return policy_client.pack_message(
                {
                    "type": "clock_probe_ack",
                    "protocol": policy_client.A3_PROTOCOL,
                    "sequence": message["sequence"],
                    "server_receive_monotonic_ns": server_ns,
                    "server_send_monotonic_ns": server_ns,
                }
            )
        if message.get("type") == "clock_sync_result":
            return policy_client.pack_message(
                {
                    "type": "clock_sync_ack",
                    "protocol": policy_client.A3_PROTOCOL,
                    "accepted": True,
                }
            )
        return policy_client.pack_message({"actions": np.zeros((50, 37), dtype=np.float32)})

    def close(self):
        self.closed = True


def make_client(monkeypatch, metadata):
    connection = FakeConnection(metadata)
    monkeypatch.setattr(policy_client, "connect", lambda *_args, **_kwargs: connection)
    client = policy_client.PolicyClient(
        "test",
        8000,
        connect_timeout_s=1.0,
        infer_timeout_s=1.0,
    )
    return client, connection


def test_d457_request_keeps_legacy_observation_wire_format(monkeypatch):
    client, connection = make_client(
        monkeypatch,
        {"a3_observation": {"camera_source": policy_client.CAMERA_SOURCE_ADU_D457}},
    )
    image = np.zeros((3, 224, 224), dtype=np.uint8)
    observation = {"state": np.zeros(37), "images": {"chest_rgb": image}, "prompt": "move cup"}

    client.open()
    result, _, _ = client.infer(policy_client.PolicyRequest(observation))

    assert client.camera_source == policy_client.CAMERA_SOURCE_ADU_D457
    assert connection.sent[-1].keys() == observation.keys()
    np.testing.assert_array_equal(connection.sent[-1]["images"]["chest_rgb"], image)
    assert result["actions"].shape == (50, 37)


def test_missing_camera_metadata_remains_compatible_with_legacy_d457_server(monkeypatch):
    client, connection = make_client(monkeypatch, {"normalization": {"mode": "legacy"}})
    observation = {
        "state": np.zeros(37),
        "images": {"chest_rgb": np.zeros((3, 224, 224), dtype=np.uint8)},
        "prompt": "move cup",
    }

    client.open()
    client.infer(policy_client.PolicyRequest(observation))

    assert client.camera_source == policy_client.CAMERA_SOURCE_ADU_D457
    assert connection.sent[-1].keys() == observation.keys()


def test_external_request_syncs_clock_then_builds_state_only_envelope(monkeypatch):
    client, connection = make_client(
        monkeypatch,
        {
            "a3_observation": {
                "protocol": policy_client.A3_PROTOCOL,
                "camera_source": policy_client.CAMERA_SOURCE_EXTERNAL_D455,
                "clock_sync": {"method": "ntp_min_rtt", "probe_count": 7, "interval_s": 5.0},
            }
        },
    )
    built_after_handshake = []
    window = {"schema": "openpi.a3.source-state-window.v1", "groups": {}, "errors": {}}

    def request_factory():
        built_after_handshake.append([message.get("type") for message in connection.sent])
        return policy_client.PolicyRequest(
            {"state": np.zeros(37), "prompt": "move cup"},
            source_state_window=window,
        )

    client.open()
    client.infer(request_factory)

    assert built_after_handshake
    assert built_after_handshake[0].count("clock_probe") == 7
    assert "clock_sync_result" in built_after_handshake[0]
    request = connection.sent[-1]
    assert request["type"] == "inference"
    assert request["protocol"] == policy_client.A3_PROTOCOL
    assert request["source_state_window"] == window
    assert "images" not in request["observation"]
