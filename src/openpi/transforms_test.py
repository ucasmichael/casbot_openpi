import numpy as np
import pytest

import openpi.models.tokenizer as _tokenizer
import openpi.transforms as _transforms


def _norm_stats(q01, q99):
    q01 = np.asarray(q01, dtype=np.float32)
    q99 = np.asarray(q99, dtype=np.float32)
    return _transforms.NormStats(mean=q01, std=q99 - q01, q01=q01, q99=q99)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_normfix_quantile_freezes_constant_hand_and_clips_both_directions(dtype):
    stats = {"state": _norm_stats([3000, 0], [3000, 2])}
    normalize = _transforms.Normalize(stats, use_quantiles=True, normfix=True)
    unnormalize = _transforms.Unnormalize(stats, use_quantiles=True, normfix=True)
    values = np.asarray([[2800, 10], [4000, -10]], dtype=dtype)
    normalized = normalize({"state": values})["state"]
    np.testing.assert_array_equal(normalized, [[0, 3], [0, -3]])
    assert normalized.dtype == dtype
    # Model predictions also need clipping before conversion to physical units.
    restored = unnormalize({"state": np.asarray([[99, 99], [-99, -99]], dtype=dtype)})["state"]
    np.testing.assert_array_equal(restored, [[3000, 4], [3000, -2]])


def test_normfix_quantile_threshold_and_nonclipped_roundtrip():
    stats = {"state": _norm_stats([0, 0, 0, 1], [0.000999, 0.001, 0.001001, 3])}
    values = np.asarray([0.000999, 0.001, 0.001001, 2], dtype=np.float32)
    normalized = _transforms.Normalize(stats, use_quantiles=True, normfix=True)({"state": values})
    np.testing.assert_array_equal(normalized["state"], [0, 1, 1, 0])
    restored = _transforms.Unnormalize(stats, use_quantiles=True, normfix=True)(normalized)["state"]
    np.testing.assert_array_equal(restored, np.asarray([0, 0.001, 0.001001, 2], dtype=np.float32))


def test_normfix_quantile_chunk_keeps_padding_and_freezes_noninteger_constant():
    stats = {"actions": _norm_stats([3999.4698, 0], [3999.4698, 2])}
    predictions = np.tile(np.asarray([9, -9, 123], dtype=np.float32), (50, 1))
    restored = _transforms.Unnormalize(stats, use_quantiles=True, normfix=True)({"actions": predictions})
    expected = np.tile(np.asarray([3999.4698, -2, 123], dtype=np.float32), (50, 1))
    np.testing.assert_array_equal(restored["actions"], expected)


def test_normfix_zscore_matches_training_rules():
    stats = {"state": _norm_stats([10, 20], [10, 22])}
    normalized = _transforms.Normalize(stats, normfix=True)({"state": np.asarray([100, 40], dtype=np.float32)})
    np.testing.assert_array_equal(normalized["state"], [0, 3])
    restored = _transforms.Unnormalize(stats, normfix=True)({"state": np.asarray([99, -99], dtype=np.float32)})
    np.testing.assert_array_equal(restored["state"], [10, 14])


@pytest.mark.parametrize("use_quantiles", [False, True])
def test_normalization_default_preserves_legacy_behavior(use_quantiles):
    stats = _norm_stats([3000, 0], [3000, 2])
    values = np.asarray([2800, 10], dtype=np.float32)
    normalized = _transforms.Normalize({"state": stats}, use_quantiles=use_quantiles)({"state": values})
    expected = (values - stats.mean) / (stats.std + 1e-6)
    if use_quantiles:
        expected = expected * 2 - 1
    np.testing.assert_array_equal(normalized["state"], expected)
    predictions = np.asarray([99, -99], dtype=np.float32)
    restored = _transforms.Unnormalize({"state": stats}, use_quantiles=use_quantiles)({"state": predictions})
    expected = ((predictions + 1) / 2 if use_quantiles else predictions) * (stats.std + 1e-6) + stats.mean
    np.testing.assert_array_equal(restored["state"], expected)


def test_repack_transform():
    transform = _transforms.RepackTransform(
        structure={
            "a": {"b": "b/c"},
            "d": "e/f",
        }
    )
    item = {"b": {"c": 1}, "e": {"f": 2}}
    assert transform(item) == {"a": {"b": 1}, "d": 2}


def test_delta_actions():
    item = {"state": np.array([1, 2, 3]), "actions": np.array([[3, 4, 5], [5, 6, 7]])}

    transform = _transforms.DeltaActions(mask=[False, True])
    transformed = transform(item)

    assert np.all(transformed["state"] == np.array([1, 2, 3]))
    assert np.all(transformed["actions"] == np.array([[3, 2, 5], [5, 4, 7]]))


def test_delta_actions_noop():
    item = {"state": np.array([1, 2, 3]), "actions": np.array([[3, 4, 5], [5, 6, 7]])}

    # No-op when the mask is disabled.
    transform = _transforms.DeltaActions(mask=None)
    assert transform(item) is item

    # No-op when there are no actions in the input.
    del item["actions"]
    transform = _transforms.DeltaActions(mask=[True, False])
    assert transform(item) is item


def test_absolute_actions():
    item = {"state": np.array([1, 2, 3]), "actions": np.array([[3, 4, 5], [5, 6, 7]])}

    transform = _transforms.AbsoluteActions(mask=[False, True])
    transformed = transform(item)

    assert np.all(transformed["state"] == np.array([1, 2, 3]))
    assert np.all(transformed["actions"] == np.array([[3, 6, 5], [5, 8, 7]]))


def test_absolute_actions_noop():
    item = {"state": np.array([1, 2, 3]), "actions": np.array([[3, 4, 5], [5, 6, 7]])}

    # No-op when the mask is disabled.
    transform = _transforms.AbsoluteActions(mask=None)
    assert transform(item) is item

    # No-op when there are no actions in the input.
    del item["actions"]
    transform = _transforms.AbsoluteActions(mask=[True, False])
    assert transform(item) is item


def test_make_bool_mask():
    assert _transforms.make_bool_mask(2, -2, 2) == (True, True, False, False, True, True)
    assert _transforms.make_bool_mask(2, 0, 2) == (True, True, True, True)


def test_tokenize_prompt():
    tokenizer = _tokenizer.PaligemmaTokenizer(max_len=12)
    transform = _transforms.TokenizePrompt(tokenizer)

    data = transform({"prompt": "Hello, world!"})

    tok_prompt, tok_mask = tokenizer.tokenize("Hello, world!")
    assert np.allclose(tok_prompt, data["tokenized_prompt"])
    assert np.allclose(tok_mask, data["tokenized_prompt_mask"])


def test_tokenize_no_prompt():
    transform = _transforms.TokenizePrompt(_tokenizer.PaligemmaTokenizer())

    with pytest.raises(ValueError, match="Prompt is required"):
        transform({})


def test_transform_dict():
    # Rename and remove keys.
    input = {"a": {"b": 1, "c": 2}}
    output = _transforms.transform_dict({"a/b": "a/c", "a/c": None}, input)
    assert output == {"a": {"c": 1}}

    # Raises and error since the renamed key conflicts with an existing key.
    with pytest.raises(ValueError, match="Key 'a/c' already exists in output"):
        _transforms.transform_dict({"a/b": "a/c"}, input)

    # Full match is required and so nothing will be removed.
    input = {"a": {"b": 1, "c": 2}}
    output = _transforms.transform_dict({"a": None}, input)
    assert output == input

    # The regex matches the entire key and so the entire input will be removed.
    input = {"a": {"b": 1, "c": 2}}
    output = _transforms.transform_dict({"a.+": None}, input)
    assert output == {}

    # Replace keys using backreferences. All leaves named 'c' are replaced with 'd'.
    input = {"a": {"b": 1, "c": 1}, "b": {"c": 2}}
    output = _transforms.transform_dict({"(.+)/c": r"\1/d"}, input)
    assert output == {"a": {"b": 1, "d": 1}, "b": {"d": 2}}


def test_extract_prompt_from_task():
    transform = _transforms.PromptFromLeRobotTask({1: "Hello, world!"})

    data = transform({"task_index": 1})
    assert data["prompt"] == "Hello, world!"

    with pytest.raises(ValueError, match="task_index=2 not found in task mapping"):
        transform({"task_index": 2})
