import os
import torch
import pandas as pd
from scipy.io import loadmat
from typing import Union, Dict
from torch.utils.data import Dataset
from utils import ifftc, cartesian_mask, complex2real, coil_combine, uniform_mask


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
    us_image = coil_combine(ifftc(us_kspace), csm, 0)
    return {
      "kspace": kspace,
      "us_image": complex2real(us_image).float(),
      "csm": csm,
      "mask": mask.float(),
    }


if __name__ == "__main__":
  from scipy.io import savemat
  from utils import load_config
  from visualize import visualize_batch

  data_config, _, _ = load_config()
  dataset = FastMRIDataset(**data_config)
  dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
  sample = next(iter(dataloader))
  for key in sample:
    print(
      f"{key}: {sample[key].shape} -> [{sample[key].dtype}] -> [{sample[key].abs().min():.4f} - {sample[key].abs().max():.4f}]"
    )
  visualize_batch(sample)

  for key in sample:
    sample[key] = sample[key].numpy()
  # savemat("data/sample_batch.mat", sample)
