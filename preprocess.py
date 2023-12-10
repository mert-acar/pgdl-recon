import os
import torch
import h5py as h5
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.io import savemat
from utils import *

if __name__ == "__main__":
  out_path = "data/mat_files_normalized"
  if not os.path.exists(out_path):
    os.mkdir(out_path)

  df = {"filename": [], "split": []}
  ks = [
      "data/Knee_Coronal_PD_RawData_300Slices_Train.h5",
      "data/Knee_Coronal_PD_RawData_392Slices_Test.h5"
  ]
  csms = [
      "data/Knee_Coronal_PD_CoilMaps_300Slices_Train.h5",
      "data/Knee_Coronal_PD_CoilMaps_392Slices_Test.h5"
  ]

  j = 0
  for kspace_path, csm_path in zip(ks, csms):
    kspace = h5.File(kspace_path, "r")["kspace"]
    if "Train" in kspace_path:
      csm = h5.File(csm_path, "r")["trnCsm"]
    else:
      csm = h5.File(csm_path, "r")["testCsm"]
    for i in tqdm(range(len(kspace))):
      sample_k = torch.from_numpy(kspace[i]).permute(2, 0, 1)
      sample_csm = torch.from_numpy(csm[i]).permute(2, 0, 1)

      # Remove Readout oversampling
      RO = sample_k.shape[1]
      sample_k = sample_k[:, RO // 4:(3 * RO) // 4]
      sample_csm = sample_csm[: ,RO // 4:(3 * RO) // 4]

      img = coil_combine(ifftc(sample_k), sample_csm, 0)
      img = img / img.abs().max()
      sample_k = fftc(coil_projection(img, sample_csm, 0)).numpy()
      sample_csm = sample_csm.numpy()
      j = j + 1

      filename = f"slice_{j}.mat"
      if "Train" in kspace_path:
        split = "train"
      else:
        split = "test" if i <= int(len(kspace) * 0.9) else "val"
      df["filename"].append(filename)
      df["split"].append(split)
      savemat(os.path.join(out_path, filename), {"kspace": sample_k, "csm": sample_csm})

  df = pd.DataFrame(df)
  df.to_csv("data/metadata_normalized.csv", index=False)
  print(df.value_counts("split"))
