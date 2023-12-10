import os
import torch
from typing import Union
from dataset import FastMRIDataset
from visualize import plot_results
from models import CascadeNet, load_pretrained_model
from utils import real2complex, load_config, coil_combine, ifftc, psnr, ssim


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

  dataset = FastMRIDataset(**data_config, split="test")
  slc = dataset[slice_idx - 1]
  for key in slc:
    slc[key] = slc[key].unsqueeze(0).to(device)

  fs = coil_combine(ifftc(slc["kspace"]), slc["csm"]).abs().cpu()
  with torch.inference_mode():
    recon = model(**slc)
  recon = real2complex(recon).detach().abs().cpu()

  if plot:
    plot_results(recon, fs, f"figures/{checkpoint_path.split('/')[-1]}")
  return recon, fs


if __name__ == "__main__":
  import fire
  fire.Fire(inference)
