# burgers2_data.py
import torch
import numpy as np
import scipy.io
from scipy.interpolate import RegularGridInterpolator

# ---------- 基础均匀采样器 ----------
class Sampler:
    def __init__(self, dim, coords, device="cpu"):
        self.dim = dim
        self.coords = coords.to(device)   # shape (2, dim)
        self.device = device

    def sample(self, N):
        rand = torch.rand(N, self.dim, device=self.device)
        return self.coords[0:1] + (self.coords[1:2] - self.coords[0:1]) * rand

# ---------- RAD 自适应采样器 ----------
class AdaptiveSampler:
    def __init__(self, dim, coords, device, k=1, c=1,
                 num_candidate=20000, num_points=700, init_uniform=True):
        self.dim = dim
        self.coords = coords.to(device)
        self.device = device
        self.k = k
        self.c = c
        self.num_candidate = num_candidate
        self.num_points = num_points
        self.points = None
        if init_uniform:
            self._init_uniform_points()

    def _init_uniform_points(self):
        rand = torch.rand(self.num_points, self.dim, device=self.device)
        self.points = self.coords[0:1] + (self.coords[1:2] - self.coords[0:1]) * rand
        self.points.requires_grad_(False)

    def sample(self, N):
        idx = torch.randint(0, self.num_points, (N,), device=self.device)
        return self.points[idx]

    def resample(self, model, operator_func, nu):
        """根据当前 PDE 残差重新采样"""
        cand = torch.rand(self.num_candidate, self.dim, device=self.device)
        cand = self.coords[0:1] + (self.coords[1:2] - self.coords[0:1]) * cand
        t = cand[:, 0:1].detach().requires_grad_(True)
        x = cand[:, 1:2].detach().requires_grad_(True)
        _, r_pred = operator_func(model, t, x, nu)
        residual_abs = torch.abs(r_pred).detach().flatten()
        mean_k = torch.mean(residual_abs ** self.k).clamp_min(1e-12)
        weights = (residual_abs ** self.k) / mean_k + self.c
        probs = weights / weights.sum()
        indices = torch.multinomial(probs, self.num_points, replacement=False)
        self.points = cand[indices].detach()

# ---------- 加载参考解（用于测试集） ----------
def load_reference_solution(mat_path, device="cpu"):
    """从 burgers_shock.mat 加载并返回插值函数"""
    data = scipy.io.loadmat(mat_path)
    x_ref = data['x'].flatten()
    t_ref = data['t'].flatten()
    usol = data['usol'].squeeze()
    if usol.ndim > 2:
        usol = usol.squeeze()
    if usol.shape != (len(t_ref), len(x_ref)):
        usol = usol.T   # 确保形状为 (t, x)
    interp = RegularGridInterpolator((t_ref, x_ref), usol,bounds_error=False, fill_value=None)
    return interp