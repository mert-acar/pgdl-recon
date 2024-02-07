import os
import torch
from typing import Union
from dataset import FastMRIDataset
from models import CascadeNet, load_pretrained_model
from visualize import visualize_results, overlay_scores
from utils import real2complex, psnr, ssim, mc_kspace_to_image


def inference(
  slice_idx: int,
  checkpoint_path: str,
  R: int = 4,
  device: Union[str, torch.device] = "cuda:0",
  plot: bool = True
):
  data_config["acc"] = R

  model, config = load_pretrained_model(checkpoint_path, device, True)
  model.eval()

  config["data"]["acc"] = R
  dataset = FastMRIDataset(**config["data"], split="test")
  sample = dataset[slice_idx]

  for key in sample:
    sample[key] = sample.unsqueeze(0).to(device)

  fs_images = mc_kspace_to_image(sample["kspace"], sample["csm"]).abs().float()
  with torch.inference_mode():
    outputs = model(**sample)
  reconstructions = real2complex(outputs).abs().float()

  psnr_score = psnr(reconstructions, fs_images)
  ssim_score = ssim(reconstructions, fs_images)
  reconstructions = overlay_scores(reconstructions, psnr_score, ssim_score)

  if plot:
    visualize_results(reconstructions, fs_images, f"../figures/{checkpoint_path.split('/')[-1]}.png")
  else:
    return reconstructions, fs_images


if __name__ == "__main__":
  import fire
  fire.Fire(inference)
