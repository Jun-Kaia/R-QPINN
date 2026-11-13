import torch
import torch.nn as nn
import os
from src.nn.DVQuantumLayer import DVQuantumLayer

class ClassicalDiffusionNet(nn.Module):
    def __init__(self, args, logger, device):
        super().__init__()
        self.device = device
        self.args = args
        self.logger = logger
        self.loss_history = []
        self.error_history = []
        self._initialize_logging()

        self.net = nn.Sequential(
            nn.Linear(3, 100), nn.Tanh(),
            nn.Linear(100, 100), nn.Tanh(),
            nn.Linear(100, 100), nn.Tanh(),
            nn.Linear(100, 100), nn.Tanh(),
            nn.Linear(100, 100), nn.Tanh(),
            nn.Linear(100, 1)
        )
        self._init_weights()

    def _initialize_logging(self):
        self.log_path = self.logger.get_output_dir()
        self.logger.print(f"Checkpoint path: {self.log_path}")

    def _init_weights(self):
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def output_transform(self, t, x, y, u_raw):
        # 边界条件（两边均为 exp(-25 - 100*(y-0.5)^2 - t)）
        L = torch.exp(-25 - 100*(y-0.5)**2 - t)
        R = torch.exp(-25 - 100*(y-0.5)**2 - t)   # 与 L 相同，因为两边一致
        u_bc = (1-x)*L + x*R
        I = torch.exp(-100*((x-0.5)**2 + (y-0.5)**2))
        u_bc0 = torch.exp(-25 - 100*(y-0.5)**2)   # t=0 时边界值
        correction = (1-t) * (I - u_bc0)
        factor = t * x * (1-x)
        return u_bc + correction + factor * u_raw

    def forward(self, X):
        t, x, y = X[:, 0:1], X[:, 1:2], X[:, 2:3]
        u_raw = self.net(X)
        return self.output_transform(t, x, y, u_raw)

    def save_state(self, path=None):
        state = {
            'args': self.args,
            'model_state_dict': self.state_dict(),
            'loss_history': self.loss_history,
            'error_history': self.error_history,
        }
        if path is None:
            path = os.path.join(self.log_path, 'model.pth')
        torch.save(state, path)
        self.logger.print(f"Model saved to {path}")

class HybridDiffusionNet(nn.Module):
    def __init__(self, args, logger, device):
        super().__init__()
        self.device = device
        self.args = args
        self.logger = logger
        self.loss_history = []
        self.error_history = []
        self.num_qubits = args['num_qubits']
        self._initialize_logging()

        self.preprocessor = nn.Sequential(
            nn.Linear(3, 100), nn.Tanh(),
            nn.Linear(100, 100), nn.Tanh(),
            nn.Linear(100, self.num_qubits), nn.Tanh()
        )
        self.quantum_layer = DVQuantumLayer(args)
        self.quantum_feature_dim = self.quantum_layer.output_dim
        self.postprocessor = nn.Sequential(
            nn.Linear(self.quantum_feature_dim, 100), nn.Tanh(),
            nn.Linear(100, 100), nn.Tanh(),
            nn.Linear(100, 1)
        )
        self._init_weights()

    def _initialize_logging(self):
        self.log_path = self.logger.get_output_dir()
        self.logger.print(f"Checkpoint path: {self.log_path}")

    def _init_weights(self):
        for module in [self.preprocessor, self.postprocessor]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)

    def output_transform(self, t, x, y, u_raw):
        L = torch.exp(-25 - 100*(y-0.5)**2 - t)
        R = torch.exp(-25 - 100*(y-0.5)**2 - t)   # 与 L 相同
        u_bc = (1-x)*L + x*R
        I = torch.exp(-100*((x-0.5)**2 + (y-0.5)**2))
        u_bc0 = torch.exp(-25 - 100*(y-0.5)**2)
        correction = (1-t) * (I - u_bc0)
        factor = t * x * (1-x)
        return u_bc + correction + factor * u_raw

    def forward(self, X):
        t, x, y = X[:, 0:1], X[:, 1:2], X[:, 2:3]
        q_in = self.preprocessor(X)
        q_features = self.quantum_layer(q_in)
        q_features = q_features.to(self.postprocessor[0].weight.dtype)
        u_raw = self.postprocessor(q_features)
        return self.output_transform(t, x, y, u_raw)

    def save_state(self, path=None):
        state = {
            'args': self.args,
            'model_state_dict': self.state_dict(),
            'loss_history': self.loss_history,
            'error_history': self.error_history,
        }
        if path is None:
            path = os.path.join(self.log_path, 'model.pth')
        torch.save(state, path)
        self.logger.print(f"Model saved to {path}")