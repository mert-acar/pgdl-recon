import os
import torch
import torch.nn as nn
from typing import Union
from utils import (
  ip, expand, load_config, image_to_mc_kspace, mc_kspace_to_image, real2complex, complex2real
)


def conv2d(in_filters: int, out_filters: int, kernel_size: int, instance_norm: bool = False):
  """ Automatic [Conv2d -> Optional(instanceNorm) -> ReLU] block """
  res = [nn.Conv2d(in_filters, out_filters, kernel_size, padding=kernel_size // 2, bias=False)]
  if instance_norm:
    res.append(nn.InstanceNorm2d(out_filters, affine=True))
  res.append(nn.ReLU(inplace=True))
  return nn.Sequential(*res)


class ResNetBlock(nn.Module):
  def __init__(
    self,
    num_layers: int = 5,
    num_filters: int = 64,
    kernel_size: int = 3,
    instance_norm: bool = False
  ):
    super().__init__()
    self.layers = [conv2d(2, num_filters, kernel_size, instance_norm)]
    for _ in range(1, num_layers - 1):
      self.layers.append(conv2d(num_filters, num_filters, kernel_size))
    self.layers.append(conv2d(num_filters, 2, kernel_size))
    self.layers = nn.Sequential(*self.layers)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return x + self.layers(x)


class VSDC(nn.Module):
  """
  Data consistency (DC) module implemented using Variable Splitting (VS)
  with quadratic penalty method
  """
  def __init__(self, mu: float, cg_iter: int = 10):
    super().__init__()
    self.mu = nn.Parameter(torch.tensor(mu), requires_grad=True)
    self.CG_ITER = cg_iter

  def forward(
    self, zero_filled: torch.tensor, reconstruction: torch.tensor, mask: torch.tensor,
    csm: torch.tensor
  ) -> torch.tensor:
    r = real2complex(zero_filled + self.mu * reconstruction)
    dims = list(range(1, r.ndim))
    p = r.clone()
    x = torch.zeros_like(r)
    for _ in range(self.CG_ITER):
      rTr = expand(ip(r, r, dims), r.ndim)
      Ap = mc_kspace_to_image(image_to_mc_kspace(p, csm) * mask, csm) + self.mu * p
      pTAp = expand(ip(Ap, p, dims), r.ndim)
      alpha = rTr / pTAp
      x = x + alpha * p
      next_r = r - alpha * Ap
      beta = expand(ip(next_r, next_r, dims), r.ndim) / rTr
      next_p = next_r + beta * p
      r = next_r
      p = next_p
    return complex2real(x, 1).float()


class CascadeNet(nn.Module):
  def __init__(
    self,
    num_cascades: int = 5,
    num_layers: int = 6,
    num_filters: int = 64,
    kernel_size: int = 3,
    instance_norm: bool = False,
    mu: float = 0.5,
    cg_iter: int = 10
  ):
    super().__init__()
    self.blocks = nn.ModuleList(
      [
        ResNetBlock(num_layers, num_filters, kernel_size, instance_norm)
        for _ in range(num_cascades)
      ]
    )
    self.dc = VSDC(mu, cg_iter)

  def forward(
    self, us_image: torch.Tensor, mask: torch.Tensor, csm: torch.Tensor, **kwargs
  ) -> torch.Tensor:
    zero_filled = us_image.clone()
    for block in self.blocks:
      # R Block
      us_image = block(us_image)
      # DC Block
      us_image = self.dc(zero_filled, us_image, mask, csm)
    return us_image


def load_pretrained_model(
  experiment_path: Union[str, os.PathLike],
  device: Union[str, torch.device] = "cpu",
  return_config: bool = False
) -> nn.Module:
  try:
    config = load_config(os.path.join(experiment_path, "ExperimentSummary.yaml"))
  except FileNotFoundError:
    config = load_config()
  model = CascadeNet(**config["model"]).to(device)
  state = torch.load(os.path.join(experiment_path, "checkpoint.pt"), map_location=device)
  model.load_state_dict(state)
  if return_config:
    return model, config
  return model
