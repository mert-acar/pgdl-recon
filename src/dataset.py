import os
import torch
from glob import glob
from scipy.io import loadmat
from functools import partial
from typing import Union, Dict
from torch.utils.data import Dataset
from utils import random_mask, uniform_mask, mc_kspace_to_image, complex2real


class FastMRIDataset(Dataset):
  def __init__(
    self,
    data_path: Union[str, os.PathLike],
    acc: int,
    acs: int = 25,
    mask: str = "uniform",
    split: Union[str, None] = None,
  ):
    # Data points
    if split is None:
      self.data_list = glob(f"{data_path}/**/*.mat")
    else:
      self.data_list = glob(f"{data_path}/{split}/*.mat")

    # Mask function
    mask_func = random_mask if mask == "random" else uniform_mask
    self.mask_func = partial(mask_func, acceleration_rate=int(acc), sample_n=acs)

  def __len__(self):
    return len(self.df)

  def __getitem__(self, idx: int) -> Dict[str, torch.tensor]:
    data = loadmat(self.data_list[idx])
    kspace = torch.from_numpy(data["k"])
    csm = torch.from_numpy(data["csm"])
    mask = torch.from_numpy(self.mask_func((1, *kspace.shape[-2:])))
    us_kspace = kspace * mask
    us_image = complex2real(mc_kspace_to_image(us_kspace, csm, 0), 0).float()
    return {
      "us_image": us_image,
      "csm": csm,
      "mask": mask,
      "kspace": kspace
    }


if __name__ == "__main__":
  from utils import load_config  

  data_args = load_config("config.yaml", "data")
  dataloader = torch.utils.data.DataLoader(
    FastMRIDataset(**data_args, split="train"),
    batch_size=1
  )
  sample = next(iter(dataloader))
  for key in sample:
    print(f"{key}: {sample[key].shape} [{sample[key].dtype}]")
