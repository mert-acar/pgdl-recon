import os
import torch
from typing import Union, Dict
import matplotlib.pyplot as plt
from PIL import Image, ImageFont, ImageDraw
from utils import real2complex, mc_kspace_to_image
from torchvision.transforms import ToTensor, ToPILImage


def overlay_scores(
  tensor: torch.Tensor, psnr: float, ssim: float, font_size: int = 12
) -> torch.Tensor:
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


def visualize_results(
  reconstructions: torch.Tensor,
  fs_images: torch.Tensor,
  path: Union[str, os.PathLike, None] = None,
):
  batch_size = reconstructions.shape[0]
  _, axs = plt.subplots(batch_size, 2, tight_layout=True, figsize=(5 * batch_size, 5 * 2))
  for i, (recon, fs) in enumerate(zip(reconstructions, fs_images)):
    axs[i, 0].imshow(recon, cmap='gray')
    axs[i, 0].axis(False)
    axs[i, 1].imshow(fs, cmap='gray')
    axs[i, 1].axis(False)

  if path is not None:
    plt.savefig(path, bbox_inches='tight')
  else:
    plt.show()
    

def visualize_batch(batch: Dict[str, torch.tensor], kspace: torch.tensor):
  keys = list(batch.keys())
  keys.remove("csm")
  batch_size = batch[keys[0]].shape[0]
  fs = mc_kspace_to_image(kspace, batch["csm"]).abs()
  _, axs = plt.subplots(batch_size, len(keys) + 1, tight_layout=True, squeeze=False)
  for j in range(batch_size):
    for i, key in enumerate(keys):
      arr = batch[key]
      if arr.shape[1] == 2:
        arr = real2complex(arr).abs()
      elif arr.shape[1] > 2:
        continue
      elif arr.shape[1] == 1:
        arr = arr[:, 0]
      axs[j, i].imshow(arr[j], cmap='gray') 
      axs[j, i].set_title(key)
      axs[j, i].axis(False)
    axs[j, -1].imshow(fs[j], cmap='gray')
    axs[j, -1].set_title("reference")
    axs[j, -1].axis(False)
  plt.show()
  
