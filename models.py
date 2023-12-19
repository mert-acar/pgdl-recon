import os
import torch
import torch.nn as nn
from typing import Union, Dict, Any, Tuple
from utils import (
  load_config, image_to_mc_kspace, mc_kspace_to_image, complex_dot_product, real2complex,
  complex2real, psnr, ssim
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


class VanillaDC(nn.Module):
  """
  Data consistency layer with vanilla replacement.
  Essentially does x = sampled_k * mask + (1 - mask) * recon_k
  """
  def forward(
    self, us_image: torch.tensor, reconstruction: torch.tensor, mask: torch.tensor,
    csm: torch.tensor
  ) -> torch.tensor:
    # Decompose the images into multi coil k-spaces
    sampled_k = image_to_mc_kspace(real2complex(us_image), csm)
    recon_k = image_to_mc_kspace(real2complex(reconstruction), csm)

    # Actual DC step
    k = mask * sampled_k + (1 - mask) * recon_k

    # Reconstruct the kspace into the image
    recon = complex2real(mc_kspace_to_image(k, csm), 1).float()
    return recon


class PGDDC(nn.Module):
  """
  Data consistency layer with Proxial Gradient Descent
  """
  def __init__(self, mu: float = 0.1, *args):
    super().__init__()
    self.mu = nn.Parameter(torch.tensor(mu), requires_grad=True)

  def forward(
    self, us_image: torch.tensor, reconstruction: torch.tensor, mask: torch.tensor,
    csm: torch.tensor
  ) -> torch.tensor:
    pass
    


class VSDC(nn.Module):
  def __init__(self, mu: float = 0.5, cg_iter: int = 10):
    """
    Data consistency layer with variable splitting method.
    The k-space merge is done using Conjugate Gradient Descent
    """
    super().__init__()
    self.mu = nn.Parameter(torch.tensor(mu), requires_grad=True)
    self.CG_ITER = cg_iter

  def forward(
    self, us_image: torch.tensor, reconstruction: torch.tensor, mask: torch.tensor,
    csm: torch.tensor
  ) -> torch.tensor:
    N = us_image.shape[0]
    # r: [N, H, W] complex
    r = real2complex(us_image + self.mu * reconstruction)
    # p: [N, H, W] complex
    p = r.clone()
    # x: [N, H, W] complex
    x = torch.zeros_like(r)
    # rdotr: [N] real
    rdotr = complex_dot_product(r, r)
    for _ in range(self.CG_ITER):
      # (E^hE + mu*I)x -> Ap: [N, H, W]
      Ap = mc_kspace_to_image(image_to_mc_kspace(p, csm) * mask, csm) + self.mu * p
      # pAp for alpha denominator -> pAp: [N]
      pAp = complex_dot_product(Ap, p)
      # alpha: [N, 1, 1]
      alpha = (rdotr / pAp).view(N, 1, 1)
      # x: [N, H, W]
      x = x + alpha * p
      # r: [N, H, W]
      r = r - alpha * Ap
      # rrnew: [N]
      rrnew = complex_dot_product(r, r)
      # beta: [N, 1, 1]
      beta = (rdotr / rrnew).view(N, 1, 1)
      # rdotr: [N]
      rdotr = rrnew
      # p: [N, H, W]
      p = beta * p + r
    # return [N, 2, H, W]
    return complex2real(x, 1).float()


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
    # self.dc = VCDC(mu, cg_iter)
    self.dc = VanillaDC()

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
    optimizer = getattr(torch.optim, train_args["optimizer"])(
      model.parameters(), **train_args["optimizer_args"]
    )
    optimizer.load_state_dict(states["optimizer_state"])
    return model, optimizer
  return model
