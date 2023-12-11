import os
import torch
import numpy as np
import torch.fft as fft
from functools import lru_cache
from yaml import full_load, dump
from numpy.lib.stride_tricks import as_strided
from typing import Union, Tuple, Dict, Any, List
from torchmetrics.functional.image import (
  peak_signal_noise_ratio, structural_similarity_index_measure
)


def real2complex(arr: torch.tensor) -> torch.tensor:
  """ [B, 2, N, M] real tensor -> [B, N, M] complex tensor """
  ch_dim = list(arr.shape).index(2)
  idx = [slice(None)] * arr.ndim
  idx[ch_dim] = 0
  real = arr[idx]
  idx[ch_dim] = 1
  imag = arr[idx]
  return torch.complex(real, imag)


def complex2real(arr: torch.tensor, stack_dim: int = 0) -> torch.tensor:
  """ [N, M] complex tensor -> [2, N, M] real tensor """
  return torch.stack((arr.real, arr.imag), dim=stack_dim)


def fftc(image: torch.tensor, dims: Tuple[int] = [-2, -1], norm: str = "ortho") -> torch.tensor:
  """ Centered FFT [Unitary if norm == 'ortho'] """
  return fft.fftshift(fft.fftn(fft.ifftshift(image, dim=dims), dim=dims, norm=norm), dim=dims)


def ifftc(kspace: torch.tensor, dims: Tuple[int] = [-2, -1], norm: str = "ortho") -> torch.tensor:
  """ Centered Inverse FFT [Unitary if norm == 'ortho'] """
  return fft.ifftshift(fft.ifftn(fft.fftshift(kspace, dim=dims), norm=norm, dim=dims), dim=dims)


def coil_combine(image: torch.tensor, csm: torch.tensor, coil_dim: int = 1) -> torch.tensor:
  """ [N, C, H, W] complex image and [N, C, H, W] csm -> [N, H, W] complex image """
  return (image * torch.conj(csm)).sum(coil_dim)


def coil_project(image: torch.tensor, csm: torch.tensor, coil_dim: int = 1) -> torch.tensor:
  """ [N, H, W] complex image and [N, C, H, W] csm -> [N, C, H, W] complex image """
  return image.unsqueeze(coil_dim) * csm


def psnr(reconstructed: torch.tensor, original: torch.tensor) -> float:
  """ Peak Signal to Noise Ratio """
  return peak_signal_noise_ratio(reconstructed, original).item()


def ssim(reconstructed: torch.tensor, original: torch.tensor) -> float:
  """ 
  Structural Similarity Index Measure between two images. 
  So supplied tensors should be real images.
  """
  if reconstructed.ndim == 3:
    reconstructed = reconstructed.unsqueeze(1)
  if original.ndim == 3:
    original = original.unsqueeze(1)
  return structural_similarity_index_measure(reconstructed, original).item()


def load_config(config_path: Union[str, os.PathLike] = "config.yaml",
                *keys: List[str]) -> Union[Dict[str, Any], Tuple[Dict[str, Any], ...]]:
  """ Load experiment config from an external yaml file """
  with open(config_path, "r") as f:
    config = full_load(f)
  if not keys:
    return tuple(config.values())
  # Fetching the configurations for the given keys
  configs = tuple(config.get(key) for key in keys)

  # Return a single config dictionary if only one key is specified,
  # otherwise return a tuple of config dictionaries.
  return configs if len(keys) > 1 else configs[0]


def dict2yaml(dict_to_save: dict, yaml_path: Union[str, os.PathLike]):
  """ Save a dict object to a yaml file """
  with open(yaml_path, "w") as outfile:
    dump(dict_to_save, outfile)


def split_array_by_ratio(arr: torch.tensor, p: float) -> Tuple[torch.tensor]:
  """
  Split the given array into two disjoint sets with the ratio of non-zero elements p
  Will return two arrays with ratios |arr1| / |arr| = p and |arr2| / |arr| = (1 - p)
  """
  assert 0 <= p <= 1, f"p must be in the range [0, 1], got {p}"
  flattened = arr.flatten()
  nonzero_locations = torch.nonzero(flattened)

  theta_cardinality = round(nonzero_locations.size(0) * p)
  theta_indices = nonzero_locations[torch.randperm(nonzero_locations.size(0))[:theta_cardinality]]
  theta = torch.zeros_like(flattened)
  theta[theta_indices] = flattened[theta_indices]
  theta = theta.reshape(arr.shape)

  lmbda = flattened.clone()
  lmbda[theta_indices] = 0
  lmbda = lmbda.reshape(arr.shape)
  return theta, lmbda


@lru_cache(maxsize=1)
def uniform_mask(shape: Tuple[int], acceleration_rate: int, sample_n: int) -> np.ndarray:
  """ 
  Creates a uniform sampling mask for a given acceleration rate and an ACS zone.
  Samples the k-space in the phase direction (lines are vertical.
  Since the mask is deterministic the result is explicitly cached for efficiency. 
  """
  acs_indx = (shape[-1] - sample_n) // 2 - 1
  mask = np.zeros(shape)
  mask[:, :, 0::acceleration_rate] = 1
  mask[:, :, acs_indx:acs_indx + sample_n] = 1
  return mask.astype(np.float32)


@lru_cache(maxsize=1)
def normal_pdf(length: int, sensitivity: float) -> np.ndarray:
  """ Gaussian PDF """
  return np.exp(-sensitivity * (np.arange(length) - length / 2)**2)


def cartesian_mask(shape: Tuple[int], acceleration_rate: int, sample_n: int) -> np.ndarray:
  """ 
  Create an undersampling mask for the given acceleration rate (R). 
  sample_n dictates the number of ACS lines.
  Samples k-space in the phase direction (lines are vertical)
  """
  N, Nx, Ny = int(np.prod(shape[:-2])), shape[-2], shape[-1]

  pdf_y = normal_pdf(Ny, 0.5 / (Ny / 10.)**2)

  lmda = Ny / (2. * acceleration_rate)
  n_lines = int(Ny / acceleration_rate)

  pdf_y += lmda * 1. / Ny

  if sample_n:
    pdf_y[Ny // 2 - sample_n // 2:Ny // 2 + sample_n // 2] = 0
    pdf_y /= np.sum(pdf_y)
    n_lines -= sample_n

  mask = np.zeros((N, Ny))
  for i in range(N):
    idx = np.random.choice(Ny, n_lines, False, pdf_y)
    mask[i, idx] = 1

  if sample_n:
    mask[:, Ny // 2 - sample_n // 2:Ny // 2 + sample_n // 2] = 1

  size = mask.itemsize
  mask = as_strided(mask, (N, Nx, Ny), (size * Ny, 0, size))
  mask = mask.reshape(shape)
  return mask.astype(np.float32)


def mc_kspace_to_image(
  kspace: torch.tensor, coil_sensitivities: torch.tensor, coil_dim: int = 1
) -> torch.tensor:
  """ 
  Takes in [N, C, H, W] kspace measurements and converts them into [N, H, W] complex images.
  """
  return coil_combine(ifftc(kspace), coil_sensitivities, coil_dim)


def image_to_mc_kspace(
  image: torch.tensor, coil_sensitivities: torch.tensor, coil_dim: int = 1
) -> torch.tensor:
  """ 
  Takes in [N, H, W] complex images and converts them into [N, C, H, W] kspace measurements.
  """
  return fftc(coil_project(image, coil_sensitivities, coil_dim))


def complex_dot_product(input_x: torch.tensor, input_y: torch.tensor) -> torch.tensor:
  """ Complex dot product that preserves the batch dimension """
  assert input_x.shape == input_y.shape, "Inputs must have the same shape!"
  dims = tuple(range(1, input_x.ndim))
  return (torch.conj(input_y) * input_x).sum(dims)


def table(logs_dir: Union[str, os.PathLike] = "logs/", filtr: Union[str, None] = None):
  df = {
    "Acceleration Rate": [],
    "Training Mask": [],
    "Loss Function": [],
    "Laplacian Center": [],
    "Mixing Parameter": [],
    "HFEN": [],
    "PSNR": [],
    "SSIM": [],
  }
  experiments = os.listdir(logs_dir)
  for exp in experiments:
    if not os.path.isdir(os.path.join(logs_dir, exp)):
      continue
    try:
      with open(os.path.join(logs_dir, exp, "scores.yaml"), "r") as f:
        scores = full_load(f)
    except:
      continue

    if filtr is not None and filtr not in exp:
      continue

    features = exp.split("_")
    df["Acceleration Rate"].append(features[0])
    df["Training Mask"].append(features[1])
    if features[2] == "hfen":
      df["Loss Function"].append("L1L2 + HFEN")
      df["Laplacian Center"].append(features[3].replace("c", ""))
      df["Mixing Parameter"].append(features[4].replace("l", ""))
    else:
      df["Loss Function"].append("L1L2")
      df["Laplacian Center"].append("")
      df["Mixing Parameter"].append("")
    df["HFEN"].append(scores["hfen"])
    df["PSNR"].append(scores["psnr"])
    df["SSIM"].append(scores["ssim"])
  df = pd.DataFrame(df)
  return df


if __name__ == "__main__":
  df = table("logs/")
  print(df)
  df.to_csv(f"data/results.csv")
