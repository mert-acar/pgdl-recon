import os
import torch
import torch.nn as nn
from typing import Union
from utils import (
  real2complex, coil_projection, coil_combine, fftc, ifftc, load_config
)


def conv2d(in_filters: int, out_filters: int, kernel_size: int, batch_norm: bool = False):
  res = [nn.Conv2d(in_filters, out_filters, kernel_size, padding=kernel_size // 2, bias=False)]
  if batch_norm:
    res.append(nn.BatchNorm2d(out_filters))
  res.append(nn.ReLU(inplace=True))
  return nn.Sequential(*res)


class ResNetBlock(nn.Module):
  def __init__(
    self,
    num_layers: int = 5,
    num_filters: int = 64,
    kernel_size: int = 3,
    batch_norm: bool = False
  ):

    super().__init__()
    self.layers = [conv2d(2, num_filters, kernel_size, batch_norm)]
    for _ in range(1, num_layers - 1):
      self.layers.append(conv2d(num_filters, num_filters, kernel_size, batch_norm))
    self.layers.append(conv2d(num_filters, 2, kernel_size, batch_norm=False))
    self.layers = nn.Sequential(*self.layers)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    res = x.clone()
    x = self.layers(x)
    return x + res


class DataConsistency(nn.Module):
  def __init__(self, mu: float = 0.5, cg_iter: int = 10):
    super().__init__()
    self.mu = nn.Parameter(torch.tensor(mu), requires_grad=True)
    self.cg_iter = cg_iter

  def forward(
    self, us_image: torch.Tensor, reconstruction: torch.Tensor, mask: torch.Tensor,
    csm: torch.Tensor
  ) -> torch.Tensor:
    curr_r = real2complex(us_image) + self.mu * real2complex(reconstruction)
    curr_p = curr_r.clone()
    img_estimate = torch.zeros_like(curr_r)
    for j in range(self.cg_iter):
      coil_p = coil_projection(curr_p, csm)
      coil_p_k = fftc(coil_p) * mask
      q = coil_combine(ifftc(coil_p_k), csm) + self.mu * curr_p
      alpha = (curr_r * torch.conj(curr_r)).sum() / (q * torch.conj(curr_p)).sum()
      next_b = img_estimate + alpha * curr_p
      next_r = curr_r - alpha * q
      beta = (next_r * torch.conj(next_r)).sum() / (curr_r * torch.conj(curr_r)).sum()
      next_p = next_r + beta * curr_p

      img_estimate = next_b
      curr_p = next_p
      curr_r = next_r
    return torch.stack((img_estimate.real, img_estimate.imag), axis=1).float()


class CascadeNet(nn.Module):
  def __init__(
    self,
    num_cascades: int = 5,
    num_layers: int = 6,
    num_filters: int = 64,
    kernel_size: int = 3,
    batch_norm: bool = False,
    mu: float = 0.5,
    cg_iter: int = 10
  ):
    super().__init__()
    self.blocks = nn.ModuleList(
      [ResNetBlock(num_layers, num_filters, kernel_size, batch_norm) for _ in range(num_cascades)]
    )
    self.dc = DataConsistency(mu, cg_iter)

  def forward(
    self, us_image: torch.Tensor, mask: torch.Tensor, csm: torch.Tensor, **kwargs
  ) -> torch.Tensor:
    zero_filled = us_image.clone()
    for block in self.blocks:
      us_image = block(us_image)
      us_image = self.dc(zero_filled, us_image, mask, csm)
    return us_image


def load_pretrained_model(
  log_path: Union[str, os.PathLike],
  device: Union[str, torch.device] = "cpu",
  return_optimizer: bool = False
) -> torch.nn.Module:
  _, model_args, train_args = load_config(os.path.join(log_path, "ExperimentSummary.yaml"))
  model = CascadeNet(**model_args).to(device)
  states = torch.load(os.path.join(log_path, "checkpoint.pt"), map_location=device)
  model.load_state_dict(states["model_state"])
  if return_optimizer:
    optimizer = getattr(torch.optim, train_config["optimizer"])(
      model.parameters(), **train_config["optimizer_args"]
    )
    optimizer.load_state_dict(states["optimizer_state"])
    return model, optimizer
  return model
