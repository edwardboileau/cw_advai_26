from pathlib import Path
from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from lab2_adapted.skip import skip

data_dir = Path(__file__).resolve().parents[1] / "data"


def load_image(path):
    image = Image.open(path).convert("RGB")
    pixels = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(pixels).permute(2, 0, 1).unsqueeze(0)


def display_images(
    images: torch.Tensor | Sequence[torch.Tensor],
    titles: str | Sequence[str] | None = None,
) -> None:
    """Display one image or a sequence of images side by side.

    Each image is a tensor shaped [3, H, W] or [1, 3, H, W],
    with RGB pixel values in [0, 1]. Images may have different sizes.
    Supply one title per image, or omit titles
    """
    if isinstance(images, torch.Tensor):
        images = [images]
    if not images:
        raise ValueError("Provide at least one image")

    if titles is None:
        titles = [""] * len(images)
    elif isinstance(titles, str):
        titles = [titles]
    if len(titles) != len(images):
        raise ValueError("Provide one title per image")

    for image in images:
        if not isinstance(image, torch.Tensor):
            raise TypeError("Each image must be a torch.Tensor")
        if not (
            (image.ndim == 3 and image.shape[0] == 3)
            or (image.ndim == 4 and image.shape[:2] == (1, 3))
        ):
            raise ValueError("Each image must have shape [3, H, W] or [1, 3, H, W]")

    fig, axes = plt.subplots(1, len(images), figsize=(6 * len(images), 5), squeeze=False)
    for ax, image, title in zip(axes[0], images, titles):
        if image.ndim == 4:
            image = image[0]
        # Move colour channels last for plotting.
        ax.imshow(image.detach().cpu().permute(1, 2, 0).numpy())
        ax.set_title(title)
        ax.axis("off")

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    lr_image = load_image(data_dir / "DIV2K_valid_LR_x8/0801x8.png")
    hr_image = load_image(data_dir / "DIV2K_valid_HR/0801.png")

    hr_height, hr_width = hr_image.shape[-2:]

    #create the fixed network input image
    fixed_random_noise = torch.rand(1, 32, hr_height, hr_width) * 0.1

    #test by displaying the low res/high res images and the first three channels of the 32 channel noise image
    display_images([lr_image, hr_image, fixed_random_noise[:, :3] / 0.1], ["LR", "HR", "Noise (first 3 channels)"])

    #create network as in dip paper
    net = skip(
        num_input_channels=32,
        num_output_channels=3,
        num_channels_down=[128] * 5,
        num_channels_up=[128] * 5,
        num_channels_skip=[4] * 5,
        filter_size_down=3,
        filter_size_up=3,
        filter_skip_size=1,
        pad="reflection",
        upsample_mode="bilinear",
        downsample_mode="stride",
        act_fun="LeakyReLU",
        need_sigmoid=True,
        need_bias=True,
        need1x1_up=True,
    )

    
