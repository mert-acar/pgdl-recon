import os
import torch
from typing import Union
from dataset import FastMRIDataset
from visualize import visualize_results
from models import CascadeNet, load_pretrained_model
from utils import real2complex, load_config, psnr, ssim, mc_kspace_to_image


def inference(
  slice_idx: int,
  checkpoint_path: str,
  R: int = 4,
  device: Union[str, torch.device] = "cuda:0",
  plot: bool = True
):
  data_config, model_config, train_config = load_config(
    os.path.join(checkpoint_path, "ExperimentSummary.yaml")
  )
  data_config["acc"] = R

  model = load_pretrained_model(checkpoint_path, device)
  model.eval()

  dataset = FastMRIDataset(**data_config, split="test", device=device),
  model_inputs, kspace = dataset[slice_idx]
  for key in model_inputs:
    model_inputs[key] = model_inputs.unsqueeze(0).to(device)
  kspace = kspace.unsqueeze(0).to(device)

  fs_images = mc_kspace_to_image(kspace, model_inputs["csm"]).abs().float()
  with torch.inference_mode():
    outputs = model(**model_inputs)
  reconstructions = real2complex(outputs).abs().float()

  if plot:
    visualize_results(reconstructions, fs_images, f"figures/{checkpoint_path.split('/')[-1]}.png")
  return reconstructions, fs_images


if __name__ == "__main__":
  import fire
  fire.Fire(inference)
