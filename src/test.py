import torch
import torch.nn as nn
from tqdm import tqdm
from loss import HFEN
from utils import psnr, ssim, real2complex, image_to_mc_kspace, mc_kspace_to_image


def test_model(
  model: nn.Module,
  dataloader: torch.utils.data.DataLoader,
  device: torch.device,
) -> dict:
  # Make sure to put model in eval mode since some components
  # like BatchNorm2d, Dropout can behave differently during inference
  model.eval()
  running_hfen = 0
  running_psnr = 0
  running_ssim = 0
  criterion = HFEN((1, 1, 320, 336), 23, 3).to(device)
  with torch.no_grad():
    for model_inputs in tqdm(dataloader, total=len(dataloader)):
      for key in model_inputs:
        model_inputs[key] = model_inputs[key].to(device)
      outputs = model(**model_inputs)
      reconstructions = real2complex(outputs)
      loss = criterion(image_to_mc_kspace(reconstructions, model_inputs["csm"]), kspace)
      recon_images = reconstructions.abs().float()
      fs_images = mc_kspace_to_image(model_inputs["kspace"], model_inputs["csm"]).abs().float()
      psnr_score = psnr(recon_images, fs_images)
      ssim_score = ssim(recon_images, fs_images)
      running_hfen += loss.item()
      running_psnr += psnr_score
      running_ssim += ssim_score
  mean_hfen = running_hfen / len(dataloader)
  mean_psnr = running_psnr / len(dataloader)
  mean_ssim = running_ssim / len(dataloader)
  return {"hfen": mean_hfen, "psnr": mean_psnr, "ssim": mean_ssim}

