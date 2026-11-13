import torch

class Sampler:
    def __init__(self, dim, coords, name=None, device='cpu'):
        self.dim = dim
        self.coords = coords.to(device)
        self.name = name
        self.device = device

    def sample(self, N):
        rand = torch.rand(N, self.dim, device=self.device)
        return self.coords[0:1] + (self.coords[1:2] - self.coords[0:1]) * rand

class AdaptiveSampler:
    def __init__(self, dim, coords, device, k=1, c=1,
                 num_candidate=20000, num_points=1000, init_uniform=True):
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

    def resample(self, model, operator_func, D=0.01, v_x=1.0, v_y=1.0):
        cand = torch.rand(self.num_candidate, self.dim, device=self.device)
        cand = self.coords[0:1] + (self.coords[1:2] - self.coords[0:1]) * cand
        t = cand[:, 0:1].detach().requires_grad_(True)
        x = cand[:, 1:2].detach().requires_grad_(True)
        y = cand[:, 2:3].detach().requires_grad_(True)
        _, r_pred = operator_func(model, t, x, y, D=D, v_x=v_x, v_y=v_y)
        residual_abs = torch.abs(r_pred).detach().flatten()
        mean_k = torch.mean(residual_abs ** self.k).clamp_min(1e-12)
        weights = (residual_abs ** self.k) / mean_k + self.c
        probs = weights / weights.sum()
        indices = torch.multinomial(probs, self.num_points, replacement=False)
        self.points = cand[indices].detach()