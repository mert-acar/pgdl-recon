import torch
import numpy as np
from typing import Dict, Union
import matplotlib.pyplot as plt
from PIL import Image, ImageFont, ImageDraw
from utils import coil_combine, ifftc, real2complex
from torchvision.transforms import ToTensor, ToPILImage


def overlay_scores(
  tensor: torch.tensor, psnr: float, ssim: float, font_size: int = 12
) -> torch.tensor:
  # Check if the tensor is in the right shape
  if len(tensor.shape) != 3:
    raise ValueError("Tensor should have shape [N, H, W]")

  # Prepare text to overlay
  text = f'PSNR: {psnr:.2f}, SSIM: {ssim:.2f}'
  font = ImageFont.truetype("data/font.ttf", 20)

  to_pil = ToPILImage()
  to_tensor = ToTensor()
  images = []
  for img in tensor:
    image = to_pil(img)
    draw = ImageDraw.Draw(image)
    draw.text((10, tensor.shape[-1] - 90), text, font=font, fill="white")
    images.append(to_tensor(image))
  return torch.stack(images)


def visualize_batch(sample: Dict[str, torch.tensor]):
  batch_size = len(sample["kspace"])
  _, axs = plt.subplots(batch_size, 3, tight_layout=True, squeeze=False)
  fs_image = coil_combine(ifftc(sample["kspace"]), sample["csm"]).abs().numpy()
  us_image = real2complex(sample["us_image"]).abs().numpy()
  for i in range(batch_size):
    axs[i, 0].imshow(fs_image[i], cmap='gray')
    axs[i, 0].set_title("Fullysampled")
    axs[i, 1].imshow(sample["mask"][i, 0].numpy(), cmap='gray')
    axs[i, 1].set_title("Mask")
    axs[i, 2].imshow(us_image[i], cmap='gray')
    axs[i, 2].set_title("Undersampled")
  for ax in axs.ravel():
    ax.axis(False)
  plt.show()


def plot_k_space_magnitude(
  magnitude: Union[torch.tensor, np.ndarray], title: str, vmin: float = 0, vmax: float = 1
):
  """ Function to plot k-space magnitude """
  plt.imshow(magnitude, cmap='gray', vmin=vmin, vmax=vmax)
  plt.title(title)
  plt.colorbar(shrink=0.62, aspect=13)
  plt.axis('off')
