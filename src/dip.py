from pathlib import Path
from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch.utils.tensorboard import SummaryWriter

from lab2_adapted.skip import skip

data_dir = Path(__file__).resolve().parents[1] / "data"

from time import perf_counter

def timestamp():
    if device.type == "cuda":
        torch.cuda.synchronize()
    return perf_counter()

def load_image(path):
    image = Image.open(path).convert("RGB")
    pixels = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(pixels).permute(2, 0, 1).unsqueeze(0)

#one image or sequence of images side by side. Each image is a tensor shaped [3, H, W] or [1, 3, H, W],
#with RGB pixel values between 0 and 1. either all titles or no titles please
def display_images(
    images: torch.Tensor | Sequence[torch.Tensor],
    titles: str | Sequence[str] | None = None,
) -> None:
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


if __name__ == "__main__":
    output_dir = Path(__file__).resolve().parents[1] / "outputs"
    output_dir.mkdir(exist_ok=True)

    run_number = max(
        (int(path.name[4:]) for path in output_dir.glob("run_*")
         if path.name[4:].isdigit()),
        default=0,
    ) + 1
    output_dir = output_dir / f"run_{run_number:03d}"
    output_dir.mkdir()
    print(f"Saving images to: {output_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")


    lr_image = load_image(
        data_dir / "DIV2K_valid_LR_x8/0801x8.png"
    ).to(device)
    hr_image = load_image(data_dir / "DIV2K_valid_HR/0801.png")

    hr_height, hr_width = hr_image.shape[-2:]

    #create the fixed network input image
    fixed_random_noise = torch.rand(1, 32, hr_height, hr_width, device = device) * 0.1

    #test by adding the low res/high res images and the first three channels 
    #of the 32 channel noise image to the matplotlib plot
    display_images([lr_image, hr_image, fixed_random_noise[:, :3] / 0.1], ["LR", "HR", "Noise (first 3 channels)"])
    
    #plt.show() #use this to have images just appear on the screen
    plt.savefig(output_dir / "plot.png", dpi=150, bbox_inches="tight") #use this to save images

    #create network as in dip paper
    net = skip(
        num_input_channels=32,
        num_output_channels=3,
        num_channels_down=[32] * 5, #in paper, was 128, but reducing
        num_channels_up=[32] * 5, #in paper, was [128] * 5 i think, but reducing
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
    ).to(device)

    optimizer = torch.optim.Adam(net.parameters(), lr=0.0075) #original learning rate = 0.01
    num_steps = 2000
    log_every = 50
    run_summary = (
        f"Device: {device}\n"
        f"Input shape: {tuple(fixed_random_noise.shape)}\n"
        f"LR shape: {tuple(lr_image.shape)}\n"
        f"Steps: {num_steps}\n"
        f"Log every: {log_every} steps\n"
        f"Parameters: {sum(p.numel() for p in net.parameters()):,}\n"
        f"Trainable parameters: {sum(p.numel() for p in net.parameters() if p.requires_grad):,}\n"
        f"Optimizer:\n{optimizer}\n\n"
        f"Model architecture:\n{net}\n\n"
    )

    #summarywriter writes tensorflow logs for the run
    with SummaryWriter(log_dir=str(output_dir)) as writer:
        writer.add_text("Run/settings", "```text\n" + run_summary + "\n```", 0)
        writer.add_image("Images/LR", lr_image[0].detach().cpu(), 0)
        writer.add_image("Images/HR_reference", hr_image[0], 0)

        for step in range(num_steps):
            this_step_is_logged = (step + 1) % log_every == 0
            optimizer.zero_grad()
            if this_step_is_logged:
                t0 = timestamp()

            predicted_hr = net(fixed_random_noise)
            if this_step_is_logged:
                t1 = timestamp()

            predicted_lr = torch.nn.functional.interpolate(
                predicted_hr,
                size=lr_image.shape[-2:],
                mode="bicubic",
                align_corners=False,
                antialias=True,
            )
            loss = torch.nn.functional.mse_loss(predicted_lr, lr_image)
            if this_step_is_logged:
                t2 = timestamp()

            loss.backward()
            optimizer.step()
            if this_step_is_logged:
                t3 = timestamp()

            if this_step_is_logged:
                prediction = predicted_hr[0].detach().cpu()
                image = prediction.permute(1, 2, 0).numpy()
                plt.imsave(output_dir / f"prediction_{step + 1}.png", image)
                t4 = timestamp()

                timing = (
                    f"Step {step + 1} | Forward: {t1-t0:.3f}s | "
                    f"Downsample + loss: {t2-t1:.3f}s | "
                    f"Backward + update: {t3-t2:.3f}s | "
                    f"Save: {t4-t3:.3f}s | Loss: {loss.item():.6f}"
                )
                print(timing)
                writer.add_scalar("Loss/LR_MSE", loss.item(), step + 1)
                writer.add_scalar("Timing/forward_seconds", t1 - t0, step + 1)
                writer.add_scalar("Timing/downsample_loss_seconds", t2 - t1, step + 1)
                writer.add_scalar("Timing/backward_update_seconds", t3 - t2, step + 1)
                writer.add_scalar("Timing/save_png_seconds", t4 - t3, step + 1)
                writer.add_image("Images/prediction", prediction, step + 1)
                writer.flush()
