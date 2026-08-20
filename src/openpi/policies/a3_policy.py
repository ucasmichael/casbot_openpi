import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model

A3_ONE_SIDE_ACTION_DIM = 17
A3_BOTH_ACTION_DIM = 37
A3_ACTION_DIM = A3_BOTH_ACTION_DIM


def make_a3_example() -> dict:
    return {
        "state": np.zeros((A3_ACTION_DIM,), dtype=np.float32),
        "images": {
            "chest_rgb": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
        },
        "prompt": "do something",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.ndim == 3 and image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class A3Inputs(transforms.DataTransformFn):
    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        chest_image = _parse_image(data["images"]["chest_rgb"])

        match self.model_type:
            case _model.ModelType.PI0 | _model.ModelType.PI05:
                images = {
                    "base_0_rgb": chest_image,
                    "left_wrist_0_rgb": np.zeros_like(chest_image),
                    "right_wrist_0_rgb": np.zeros_like(chest_image),
                }
                image_masks = {
                    "base_0_rgb": np.True_,
                    "left_wrist_0_rgb": np.False_,
                    "right_wrist_0_rgb": np.False_,
                }
            case _model.ModelType.PI0_FAST:
                images = {
                    "base_0_rgb": chest_image,
                    "base_1_rgb": np.zeros_like(chest_image),
                    "wrist_0_rgb": np.zeros_like(chest_image),
                }
                image_masks = {
                    "base_0_rgb": np.True_,
                    "base_1_rgb": np.True_,
                    "wrist_0_rgb": np.True_,
                }
            case _:
                raise ValueError(f"Unsupported model type: {self.model_type}")

        inputs = {
            "state": np.asarray(data["state"], dtype=np.float32),
            "image": images,
            "image_mask": image_masks,
        }
        if "actions" in data:
            inputs["actions"] = np.asarray(data["actions"], dtype=np.float32)
        if "prompt" in data:
            prompt = data["prompt"]
            if isinstance(prompt, bytes):
                prompt = prompt.decode("utf-8")
            inputs["prompt"] = prompt
        return inputs


@dataclasses.dataclass(frozen=True)
class A3Outputs(transforms.DataTransformFn):
    action_dim: int = A3_ACTION_DIM

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, : self.action_dim])}
