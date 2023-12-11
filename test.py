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
  criterion = HFEN((1, 1, 320, 368), 23, 3).to(device)
  with torch.no_grad():
    for model_inputs, kspace in tqdm(dataloader, total=len(dataloader)):
      for key in model_inputs:
        model_inputs[key] = model_inputs[key].to(device)
      kspace = kspace.to(device)
      outputs = model(**model_inputs)
      reconstructions = real2complex(outputs)
      loss = criterion(image_to_mc_kspace(reconstructions, model_inputs["csm"]), kspace)
      recon_images = reconstructions.abs().float()
      fs_images = mc_kspace_to_image(kspace, model_inputs["csm"]).abs().float()
      psnr_score = psnr(recon_images, fs_images)
      ssim_score = ssim(recon_images, fs_images)
      running_hfen += loss.item()
      running_psnr += psnr_score
      running_ssim += ssim_score
  mean_hfen = running_hfen / len(dataloader)
  mean_psnr = running_psnr / len(dataloader)
  mean_ssim = running_ssim / len(dataloader)
  return {"hfen": mean_hfen, "psnr": mean_psnr, "ssim": mean_ssim}


if __name__ == "__main__":
  import os
  from models import CascadeNet
  from dataset import FastMRIDataset
  from utils import load_config, dict2yaml

  checkpoint_path = "logs/experiment_multicoil_l1l2/"
  data_config, model_config, _ = load_config(
    os.path.join(checkpoint_path, "ExperimentSummary.yaml")
  )
  device = torch.device("cuda:1")
  dataset = FastMRIDataset(**data_config, split="test", device=device)

  state = torch.load(os.path.join(checkpoint_path, "checkpoint.pt"), map_location=device)
  model = CascadeNet(**model_config).to(device)
  model.load_state_dict(state["model_state"])

  scores = test_model(model, torch.utils.data.DataLoader(dataset, batch_size=16), device)
  dict2yaml(
    dict_to_save=scores,
    yaml_path=os.path.join(checkpoint_path, "scores.yaml"),
  )
  for key in scores:
    print(f"{key}: {scores[key]:.4f}")
