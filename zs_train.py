import os
import torch
import loss as Loss
from tqdm import tqdm
from typing import Union
import matplotlib.pyplot as plt
from dataset import FastMRIDataset
from visualize import overlay_scores
from models import CascadeNet, load_pretrained_model
from utils import (
  load_config,
  dict2yaml,
  split_array_by_ratio,
  real2complex,
  complex2real,
  psnr,
  ssim,
  image_to_mc_kspace,
  mc_kspace_to_image,
)


def zs_recon(
  slice_idx: int,
  checkpoint_path: Union[None, str] = None,
  K: int = 10,
  R: int = 4,
  device: Union[str, torch.device] = "cuda:0"
):
  config_path = "config.yaml"
  if checkpoint_path is not None:
    config_path = os.path.join(checkpoint_path, "ExperimentSummary.yaml")

  data_config, model_config, train_config = load_config(yaml_path=config_path)
  data_config["acc"] = R
  dataset = FastMRIDataset(**data_config, split="test")
  model_inputs, kspace = dataset[slice_idx - 1]

  # Don't need it, will create our own
  del model_inputs["us_image"]
  for key in model_inputs:
    model_inputs[key] = model_inputs[key].unsqueeze(0)
  kspace = kspace.unsqueeze(0)

  if checkpoint_path is not None:
    model = load_pretrained_model(checkpoint_path, device)
  else:
    model = CascadeNet(**model_config).to(device)

  optimizer = getattr(torch.optim, train_config["optimizer"])(
    model.parameters(), **train_config["optimizer_args"]
  )
  criterion = getattr(Loss, train_config["criterion"])(**train_config["criterion_args"]).to(device)

  val_mask, train_mask = split_array_by_ratio(model_inputs["mask"], p=0.2)
  masks = [
    tuple(map(lambda x: x.to(device), split_array_by_ratio(train_mask, p=0.4))) for _ in range(K)
  ]
  fs_images = mc_kspace_to_image(kspace, model_inputs["csm"]).abs()

  best_epoch = -1
  best_error = 999999
  epoch = 0
  go = True
  while go:
    print("-" * 20)
    print(f"Epoch {epoch + 1}")
    for phase in ["train", "val"]:
      print(f"{phase}ing...")
      if phase == "train":
        model.train()
      else:
        model.eval()
      running_error = 0
      running_psnr = 0
      running_ssim = 0
      with torch.set_grad_enabled(phase == "train"):
        for (loss_mask, train_mask) in tqdm(masks):
          us_kspace = kspace * train_mask
          us_image = complex2real(mc_kspace_to_image(us_kspace, model_inputs["csm"]), 1).float()
          optimizer.zero_grad()
          outputs = model(us_image, train_mask, model_inputs["csm"])
          if phase == "val":
            loss_mask = val_mask

          reconstructions = real2complex(outputs)
          loss = criterion(
            image_to_mc_kspace(reconstructions, model_inputs["csm"]) * loss_mask,
            kspace * loss_mask
          )
          recon_images = reconstructions.abs().float()
          psnr_score = psnr(recon_images, fs_images)
          ssim_score = ssim(recon_images, fs_images)
          running_error += loss.item()
          running_psnr += psnr_score
          running_ssim += ssim_score
          if phase == "train":
            loss.backward()
            optimizer.step()
      running_error = running_error / K
      running_psnr = running_psnr / K
      running_ssim = running_ssim / K
      print(f"Loss: {running_error:.5f}")
      print(f"PSNR: {running_psnr:.5f}")
      print(f"SSIM: {running_ssim:.5f}")

      if phase == "val":
        if running_error < best_error:
          best_error = running_error
          best_epoch = epoch
        else:
          go = False
          print("Stopping training...")
    epoch += 1

  model.eval()

  us_kspace = kspace * model_inputs["mask"]
  us_image = complex2real(mc_kspace_to_image(us_kspace, model_inputs["csm"]), 1).float()
  outputs = model(us_image, model_inputs["mask"], model_inputs["csm"])
  reconstructions = real2complex(outputs)

  loss = criterion(image_to_mc_kspace(reconstructions, model_inputs["csm"]), kspace)

  reconstructions = reconstructions.abs().float()
  psnr_score = psnr(reconstructions, fs_images)
  ssim_score = ssim(reconstructions, fs_images)
  print(f"Final Loss: {loss.item():.5f}")
  print(f"Final PSNR: {psnr_score:.5f}")
  print(f"Final SSIM: {ssim_score:.5f}")

  err = ((reconstructions - fs_images)**2).detach().cpu().numpy()
  reconstructions = overlay_scores(reconstructions.unsqueeze(1), psnr_score, ssim_score)
  reconstructions = reconstructions.detach().cpu().numpy()[0, 0]
  fs_images = fs_images.cpu().numpy()[0]
  _, axs = plt.subplots(1, 3, tight_layout=True, figsize=(12, 5))
  axs[0].imshow(fs_images, cmap='gray')
  axs[0].set_title("SENSE1")
  axs[1].imshow(reconstructions, cmap='gray')
  axs[1].set_title("Reconstruction")
  axs[2].imshow(err[0], cmap='jet')
  axs[2].set_title("Squared Error")
  for ax in axs:
    ax.axis(False)

  if not os.path.exists("results"):
    os.mkdir("results/")
  filename = f"results/zs_slice_{slice_idx}_recon.png"
  plt.savefig(filename, bbox_inches="tight")
  print(f"Done. Figure saved to {filename}")
  return model, final_recon


if __name__ == "__main__":
  from fire import Fire
  Fire(zs_recon)
