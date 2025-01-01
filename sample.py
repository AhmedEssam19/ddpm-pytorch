import os
import torch
import typer

from torchvision.utils import save_image
from pl_utils import PLModel


def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr


def main(
    checkpoint_path: str,
    device: str = "cuda",
    num_images: int = 8,
    image_size: int = 256,
    timesteps: int = 1000,
    batch_size: int = 8,
    output_dir: str = "samples",
):
    model = PLModel.load_from_checkpoint(checkpoint_path)
    model.eval()
    model.to(device)

    os.makedirs(output_dir, exist_ok=True)

    for i, img_batch in enumerate(num_to_groups(num_images, batch_size)):
        shape = (img_batch, 3, image_size, image_size)
        x_t = torch.randn(shape, device=device)
        for t in reversed(range(timesteps)):
            x_t = model.diffusion_utils.p_sample(model, x_t, torch.tensor([t] * shape[0], device=device))
        all_images = (x_t + 1) * 0.5
        for j in range(shape[0]):
            save_image(all_images[j], f"{output_dir}/sample_{i * batch_size + j:02d}.png")


if __name__ == "__main__":
    typer.run(main)
