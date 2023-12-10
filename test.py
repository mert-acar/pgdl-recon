import torch
import torch.nn as nn
from tqdm import tqdm
from loss import HFEN
from utils import psnr, ssim, fftc, ifftc, coil_combine, coil_projection, real2complex


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
    for batch in tqdm(dataloader, total=len(dataloader)):
      for key in batch:
        batch[key] = batch[key].to(device)
      outputs, _ = model(**batch)
      loss = criterion(
        # [N, Ch, H, W] complex img -> [N, Ch, H, W] complex kspace
        fftc(
          # [N, H, W] complex img -> [N, Ch, H, W] complex img
          coil_projection(
            # [N, 2, H, W] real img -> [N, H, W] complex img
            real2complex(outputs),
            batch["csm"]
          )
        ),
        batch["kspace"]
      )
      reconstructions = real2complex(outputs).abs()
      fs_images = coil_combine(ifftc(batch["kspace"]), batch["csm"]).abs()
      psnr_score = psnr(reconstructions, fs_images)
      ssim_score = ssim(reconstructions, fs_images)
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
  data_config, model_config, _ = load_config(os.path.join(checkpoint_path,
                                                          "ExperimentSummary.yaml"))
  device = torch.device("cuda:1")
  dataset = FastMRIDataset(**data_config, split="test")

  state = torch.load(os.path.join(checkpoint_path, "checkpoint.pt"), map_location=device)
  model = CascadeNet(**model_config).to(device)
  model.load_state_dict(state["model_state"])

  scores = test_model(model, torch.utils.data.DataLoader(dataset, batch_size=16), device)
  '''
  dict2yaml(
      dict_to_save=scores,
      yaml_path=os.path.join(checkpoint_path, "scores.yaml"),
  )
  '''
  for key in scores:
    print(f"{key}: {scores[key]:.4f}")
 
