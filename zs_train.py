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
  coil_projection,
  split_array_by_ratio,
  coil_combine,
  real2complex,
  complex2real,
  fftc,
  ifftc,
  psnr,
  ssim,
)


def zs_recon(
  slice_idx: int,
  checkpoint_path: Union[None, str] = None,
  K: int = 10,
  R: int = 4,
  device: Union[str, torch.device] = "cuda:0"
):
  data_config, model_config, train_config = load_config()
  data_config["acc"] = R
  dataset = FastMRIDataset(**data_config, split="test")
  slc = dataset[slice_idx - 1]
  # Don't need it, will create our own
  del slc["us_image"]
  for key in slc:
    slc[key] = slc[key].unsqueeze(0).to(device)

  if checkpoint_path is not None:
    model = load_pretrained_model(checkpoint_path, device)
  else:
    model = CascadeNet(**model_config).to(device)

  optimizer = getattr(torch.optim, train_config["optimizer"])(
    model.parameters(), **train_config["optimizer_args"]
  )
  criterion = getattr(Loss, train_config["criterion"])(**train_config["criterion_args"]).to(device)

  val_mask, train_mask = split_array_by_ratio(slc["mask"], p=0.2)
  masks = [
    tuple(map(lambda x: x.to(device), split_array_by_ratio(train_mask, p=0.4))) for _ in range(K)
  ]
  fs_images = coil_combine(ifftc(slc["kspace"]), slc["csm"]).abs()

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
          us_kspace = slc["kspace"] * train_mask
          us_image = complex2real(coil_combine(ifftc(us_kspace), slc["csm"]), 1).float()
          optimizer.zero_grad()
          outputs = model(us_image, train_mask, slc["csm"])
          if phase == "val":
            loss_mask = val_mask

          # Take multicoil loss in k-space
          loss = criterion(
            # [N, Ch, H, W] complex img -> [N, Ch, H, W] complex kspace
            fftc(
              # [N, H, W] complex img -> [N, Ch, H, W] complex img
              coil_projection(
                # [N, 2, H, W] real img -> [N, H, W] complex img
                real2complex(outputs),
                slc["csm"]
              )
            ) * loss_mask,
            slc["kspace"] * loss_mask
          )
          reconstructions = real2complex(outputs).abs()
          psnr_score = psnr(reconstructions, fs_images)
          ssim_score = ssim(reconstructions, fs_images)
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

  us_kspace = slc["kspace"] * slc["mask"]
  us_image = complex2real(coil_combine(ifftc(us_kspace), slc["csm"]), 1).float()
  final_recon = model(us_image, slc["mask"], slc["csm"])

  loss = criterion(
    # [N, Ch, H, W] complex img -> [N, Ch, H, W] complex kspace
    fftc(
      # [N, H, W] complex img -> [N, Ch, H, W] complex img
      coil_projection(
        # [N, 2, H, W] real img -> [N, H, W] complex img
        real2complex(final_recon),
        slc["csm"]
      )
    ),
    slc["kspace"]
  )

  reconstructions = real2complex(outputs).abs()
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
  filename = f"results/zs_slice_{slice_idx}_recon.png"
  plt.savefig(filename, bbox_inches="tight")
  print(f"Done. Figure saved to {filename}")


if __name__ == "__main__":
  from fire import Fire
  Fire(zs_recon)
