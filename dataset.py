import os
import torch
import pandas as pd
from scipy.io import loadmat
from typing import Union, Dict
from torch.utils.data import Dataset
from utils import cartesian_mask, uniform_mask, mc_kspace_to_image, complex2real


class FastMRIDataset(Dataset):
  def __init__(
    self,
    data_path: Union[str, os.PathLike],
    csv_path: Union[str, os.PathLike],
    acc: int,
    split: Union[str, None] = None,
    acs: int = 24,
    mask: str = "random"
  ):
    # Read the data from disk
    self.data_path = data_path
    self.df = pd.read_csv(csv_path)
    if split is not None:
      self.df = self.df[self.df["split"] == split]

    # Acceleration rate / Undersampling rate
    self.acc = acc
    # Auto Calibration signal size
    self.acs = acs
    # Mask function
    self.mask_func = cartesian_mask if mask == "random" else uniform_mask

  def __len__(self):
    return len(self.df)

  def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
    data = loadmat(os.path.join(self.data_path, self.df.iloc[idx]["filename"]))
    kspace = torch.from_numpy(data["kspace"])
    csm = torch.from_numpy(data["csm"])
    mask = torch.from_numpy(
      self.mask_func((1, kspace.shape[1], kspace.shape[2]), self.acc, self.acs)
    )
    us_kspace = kspace * mask
    us_image = complex2real(mc_kspace_to_image(us_kspace, csm, 0), 0).float()
    model_inputs = {
      "us_image": us_image,
      "csm": csm,
      "mask": mask
    }
    return model_inputs, kspace


if __name__ == "__main__":
  from utils import load_config  
  from visualize import visualize_batch
  data_args = load_config("config.yaml", "data")
  dataloader = torch.utils.data.DataLoader(
    FastMRIDataset(**data_args, split="train"),
    batch_size=1
  )
  sample_input, kspace = next(iter(dataloader))
  for key in sample_input:
    print(f"{key}: {sample_input[key].shape} [{sample_input[key].dtype}]")
  print(f"kspace: {kspace.shape} [{kspace.dtype}]")
  visualize_batch(sample_input, kspace)
