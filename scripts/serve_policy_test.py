import pytest
import tyro

from scripts import serve_policy


@pytest.mark.parametrize("normfix", [False, True])
@pytest.mark.parametrize("checkpoint", [False, True])
def test_normfix_cli_is_explicit_and_forwarded(monkeypatch, normfix, checkpoint):
    argv = ["--normfix"] if normfix else []
    argv += ["--default-prompt", "move cup"]
    if checkpoint:
        argv += ["policy:checkpoint", "--policy.config", "pi05_a3_0902_move_cup_both", "--policy.dir", "/test/19999"]
    args = tyro.cli(serve_policy.Args, args=argv)
    calls = []

    def create_policy(*args, **kwargs):
        calls.append((args, kwargs))
        return object()

    monkeypatch.setattr("openpi.training.config.get_config", lambda name: name)
    monkeypatch.setattr("openpi.policies.policy_config.create_trained_policy", create_policy)
    serve_policy.create_policy(args)
    assert len(calls) == 1
    assert calls[0][1]["normfix"] is normfix
    assert calls[0][1]["default_prompt"] == "move cup"
    if checkpoint:
        assert calls[0][0] == ("pi05_a3_0902_move_cup_both", "/test/19999")


def test_active_sides_cli_is_forwarded_for_checkpoint(monkeypatch):
    args = tyro.cli(
        serve_policy.Args,
        args=[
            "--active-sides",
            "right",
            "policy:checkpoint",
            "--policy.config",
            "pi05_a3_sim2real_move_cup",
            "--policy.dir",
            "/test/20000",
        ],
    )
    calls = []
    monkeypatch.setattr("openpi.training.config.get_config", lambda name: name)
    monkeypatch.setattr(
        "openpi.policies.policy_config.create_trained_policy",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )

    serve_policy.create_policy(args)

    assert calls[0][1]["active_sides"] == ("right",)
