import os
import torch
import loss as Loss
from tqdm import tqdm
from test import test_model
from models import CascadeNet
from dataset import FastMRIDataset
from shutil import rmtree, copyfile
from visualize import overlay_scores
from torch.utils.tensorboard import SummaryWriter
from utils import (
  load_config,
  dict2yaml,
  real2complex,
  psnr,
  ssim,
  image_to_mc_kspace,
  mc_kspace_to_image,
)


def checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, path: str):
  print(f"Saving model to {path}...")
  # Save both the model state and optimizer state incase we need to resume training
  # from a checkpoint
  torch.save(
    {
      "model_state": model.state_dict(),
      "optimizer_state": optimizer.state_dict()
    },
    path,
  )


if __name__ == "__main__":
  data_config, model_config, train_config = load_config()

  # Device can be either ['cpu', 0, 1, ..., num_gpu - 1]
  if isinstance(train_config["device"], int):
    device = torch.device("cuda:" + str(train_config["device"]))
  elif train_config["device"] == "cpu":
    device = torch.device("cpu")
  else:
    raise ValueError("Invalid device selection")

  # Create the checkpoint output path
  if os.path.exists(train_config["output_path"]):
    c = input(
      f"Output path {train_config['output_path']} is not empty! Do you want to delete the folder [y / n]: "
    )
    if "y" == c.lower():
      rmtree(train_config["output_path"], ignore_errors=True)
    else:
      print("Exit!")
      raise SystemExit
  os.mkdir(train_config["output_path"])

  # Save the training recipe to the output path for retrospective inspection
  # copyfile("config.yaml", os.path.join(train_config["output_path"], "ExperimentSummary.yaml"))
  dict2yaml(
    dict_to_save={
      'data_args': data_config,
      "model_args": model_config,
      "train_args": train_config
    },
    yaml_path=os.path.join(train_config["output_path"], "ExperimentSummary.yaml"),
  )

  # Create tensorboard object
  writer = SummaryWriter(log_dir=train_config["output_path"])

  # Create the dataloaders
  dataloaders = {
    "train":
      torch.utils.data.DataLoader(
        FastMRIDataset(**data_config, split="train"),
        # make sure to shuffle the data during training
        shuffle=True,
        # Batch_size & num_workers are specified in training recipe
        **train_config["dataloader_args"],
      ),
    "val":
      torch.utils.data.DataLoader(
        FastMRIDataset(**data_config, split="val"),
        **train_config["dataloader_args"],
      ),
  }

  model = CascadeNet(**model_config).to(device)
  model.train()

  optimizer = getattr(torch.optim, train_config["optimizer"])(
    model.parameters(), **train_config["optimizer_args"]
  )

  criterion = getattr(Loss, train_config["criterion"])(**train_config["criterion_args"]).to(device)

  """
  scheduler = getattr(torch.optim.lr_scheduler,
                      train_config["scheduler"])(optimizer, **train_config["scheduler_args"])
  """

  print(
    f"Training starting with {len(dataloaders['train'].dataset)} training and {len(dataloaders['val'].dataset)} validation data..."
  )

  best_epoch = -1
  best_error = 999999
  for epoch in range(train_config["num_epochs"]):
    print("-" * 20)
    print(f"Epoch {epoch + 1} / {train_config['num_epochs']}")
    for phase in ["train", "val"]:
      if phase == "train":
        model.train()
      else:
        model.eval()
      running_error = 0
      running_psnr = 0
      running_ssim = 0
      pbar = tqdm(dataloaders[phase], total=len(dataloaders[phase]))
      # Grad trick to help with GPU memory management
      # If the model is in eval mode torch will run using torch.no_grad():
      with torch.set_grad_enabled(phase == "train"):
        for model_inputs, kspace in pbar:
          for key in model_inputs:
            model_inputs[key] = model_inputs[key].to(device)
          kspace = kspace.to(device)
          optimizer.zero_grad()
          outputs = model(**model_inputs)
          # Take multicoil loss in k-space
          reconstructions = real2complex(outputs)
          loss = criterion(image_to_mc_kspace(reconstructions, model_inputs["csm"]), kspace)

          recon_images = reconstructions.abs().float()
          fs_images = mc_kspace_to_image(kspace, model_inputs["csm"]).abs().float()

          psnr_score = psnr(recon_images, fs_images)
          ssim_score = ssim(recon_images, fs_images)

          running_error += loss.item()
          running_psnr += psnr_score
          running_ssim += ssim_score
          pbar.set_description(f"{loss.item():.5f}")
          # Since we are using the grad trick above, can only call .backward()
          # during training. Otherwise the loss tensor has no grad and .backward()
          # call returns an error.
          if phase == "train":
            loss.backward()
            optimizer.step()

      running_error = running_error / len(dataloaders[phase])
      running_psnr = running_psnr / len(dataloaders[phase])
      running_ssim = running_ssim / len(dataloaders[phase])
      print(f"Loss: {running_error:.5f}")
      print(f"PSNR: {running_psnr:.5f}")
      print(f"SSIM: {running_ssim:.5f}")

      if phase == "val":
        # LR scheduler step. If the validation error is not improved by a factor
        # specified in the training recipe, the learning rate will be scaled by factor
        # again specified in the training recipe
        # scheduler.step(running_error)

        # Add the current epoch losses to the tensorboard logs
        writer.add_scalars("Losses", {"train": last_train_loss, "test": running_error}, epoch)
        writer.add_scalars("PSNR", {"train": last_train_psnr, "test": running_psnr}, epoch)
        writer.add_scalars("SSIM", {"train": last_train_ssim, "test": running_ssim}, epoch)

        # if the validation error has improved checkpoint the model
        if running_error < best_error:
          best_error = running_error
          best_epoch = epoch

          checkpoint_path = os.path.join(train_config["output_path"], "checkpoint.pt")
          checkpoint(model, optimizer, checkpoint_path)
      else:
        last_train_loss = running_error
        last_train_psnr = running_psnr
        last_train_ssim = running_ssim
        writer.add_scalar("Mu", model.dc.mu.item(), epoch)
        writer.add_images(
          "Reconstructions", overlay_scores(recon_images, psnr_score, ssim_score), epoch
        )
        writer.add_images("Reference Images", fs_images.unsqueeze(1), epoch)

    # If no validation improvement has been recorded for "early_stop" number of epochs
    # stop the training.
    if epoch - best_epoch >= train_config["early_stop"]:
      print(f"No improvements in {train_config['early_stop']} epochs, stop!")
      break

  # Close the tensorboard object
  writer.close()

  # Load the best checkpoint to run the test on.
  best_state = torch.load(
    os.path.join(train_config["output_path"], "checkpoint.pt"),
    map_location=device,
  )
  model.load_state_dict(best_state["model_state"])

  # Create the dataloader that will be used for testing
  test_dataloader = torch.utils.data.DataLoader(
    FastMRIDataset(**data_config, split="test"),
    **train_config["dataloader_args"],
  )

  print("\nTesting model...")
  scores = test_model(model, test_dataloader, device)
  # Write the test scores to the output path
  dict2yaml(
    dict_to_save=scores,
    yaml_path=os.path.join(train_config["output_path"], "scores.yaml"),
  )
  for key in scores:
    print(f"{key}: {scores[key]:.4f}")

