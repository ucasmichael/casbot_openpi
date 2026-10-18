from types import SimpleNamespace

import numpy as np
import pytest

from openpi import transforms
from openpi.policies import policy_config


@pytest.mark.parametrize("normfix", [False, True])
@pytest.mark.parametrize("metadata", [None, {"test": "preserved"}])
def test_checkpoint_normalization_option_reaches_both_transforms(monkeypatch, tmp_path, normfix, metadata):
    stats = transforms.NormStats(
        mean=np.asarray([3000, 0], dtype=np.float32),
        std=np.asarray([0, 2], dtype=np.float32),
        q01=np.asarray([3000, 0], dtype=np.float32),
        q99=np.asarray([3000, 2], dtype=np.float32),
    )
    data_config = SimpleNamespace(
        asset_id="a3_test",
        use_quantile_norm=True,
        data_transforms=transforms.Group(),
        model_transforms=transforms.Group(),
    )
    train_config = SimpleNamespace(
        model=SimpleNamespace(load=lambda params: params),
        data=SimpleNamespace(create=lambda *_: data_config),
        assets_dirs=tmp_path,
        policy_metadata=metadata,
    )
    monkeypatch.setattr(policy_config.download, "maybe_download", lambda _: tmp_path)
    monkeypatch.setattr("openpi.models.model.restore_params", lambda *_, **__: object())
    monkeypatch.setattr("openpi.training.checkpoints.load_norm_stats", lambda *_: {"state": stats, "actions": stats})
    monkeypatch.setattr("openpi.policies.policy.Policy", lambda model, **kwargs: kwargs)
    kwargs = {"normfix": True} if normfix else {}
    result = policy_config.create_trained_policy(train_config, tmp_path, **kwargs)
    normalize = next(t for t in result["transforms"] if isinstance(t, transforms.Normalize))
    unnormalize = next(t for t in result["output_transforms"] if isinstance(t, transforms.Unnormalize))
    assert normalize.normfix is normfix
    assert unnormalize.normfix is normfix
    if metadata:
        assert result["metadata"]["test"] == "preserved"
    assert result["metadata"]["normalization"]["mode"] == ("normfix" if normfix else "legacy")
    if normfix:
        assert result["metadata"]["normalization"]["frozen_dims"] == {"state": [0], "actions": [0]}
        value = np.asarray([2800, 10], dtype=np.float32)
        np.testing.assert_array_equal(normalize({"state": value})["state"], [0, 3])
        np.testing.assert_array_equal(unnormalize({"state": value, "actions": value})["actions"], [3000, 4])
