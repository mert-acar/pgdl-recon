import torch
from torch.nn import *
from typing import Tuple
from utils import fftc, complex2real


class L1L2Loss:
  def __init__(self):
    self.l1 = L1Loss()
    self.l2 = self.complex_mse

  def to(self, device: torch.device):
    return self

  def complex_mse(self, reconstruction: torch.Tensor, original: torch.Tensor) -> torch.Tensor:
    return ((reconstruction - original).abs()**2).mean()

  def __call__(self, reconstruction: torch.Tensor, original: torch.Tensor) -> torch.Tensor:
    l1_loss = self.l1(reconstruction, original) / original.abs().mean()
    l2_loss = self.l2(reconstruction, original) / (original.abs()**2).mean()
    return (l1_loss + l2_loss) / 2


class HFEN:
  def __init__(self, img_shape: Tuple[int], kernel_size: int = 5, sigma: float = 1.0):
    self.l1l2 = L1L2Loss()

    r = torch.arange(-kernel_size // 2 + 1., kernel_size // 2 + 1.)
    x, y = torch.meshgrid(r, r, indexing="ij")
    normal = 1 / (2.0 * torch.pi * sigma ** 2)
    exp_component = torch.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
    log_filter = - normal * exp_component * (1 - (x ** 2 + y ** 2) / (2 * sigma ** 2))
    log_filter = log_filter / log_filter.abs().sum()

    H = torch.zeros(img_shape)
    start_row = img_shape[-2] // 2 - 1
    start_col = img_shape[-1] // 2 - 1
    H[..., start_row:start_row + kernel_size, start_col:start_col + kernel_size] = log_filter
    self.H = fftc(H)

  def to(self, device: torch.device):
    self.H = self.H.to(device)
    return self

  def __call__(self, reconstruction: torch.Tensor, original: torch.Tensor) -> torch.Tensor:
    return self.l1l2(self.H * reconstruction, self.H * original)


class CompositeLoss:
  def __init__(
    self, img_shape: Tuple[int], lmbda: float, laplacian_center: float, kernel_size: int
  ):
    self.l1l2 = L1L2Loss()
    self.hfen = HFEN(img_shape, laplacian_center, kernel_size)
    self.lmbda = lmbda

  def to(self, device: torch.device):
    self.hfen = self.hfen.to(device)
    self.l1l2 = self.l1l2.to(device)
    return self

  def __call__(self, reconstruction: torch.Tensor, original: torch.Tensor) -> torch.Tensor:
    l1l2 = self.l1l2(reconstruction, original)
    hfen = self.hfen(reconstruction, original)
    return (1 - self.lmbda) * l1l2 + self.lmbda * hfen


if __name__ == "__main__":
  import matplotlib.pyplot as plt

  ls = [3, 5, 7, 11]
  sigmas = [0.5, 1, 3]
  ref = HFEN((1, 320, 368), sorted(ls)[len(ls) // 2]).H.abs()
  min_window, max_window = ref.min(), ref.max()
  _, axs = plt.subplots(len(sigmas), len(ls), tight_layout=True, squeeze=False)
  for s, row in zip(sigmas, axs):
    for l, ax in zip(ls, row):
      h = HFEN((1, 320, 368), l, s).H.abs()[0]
      ax.imshow(h, cmap='gray', vmin=min_window, vmax=max_window)
      ax.axis(False)
      ax.set_title(f"Gaussian Kernel Size: {l},\nSigma: {s}")
  plt.show()
