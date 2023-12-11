import torch
import matplotlib.pyplot as plt
from PIL import Image, ImageFont, ImageDraw
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
