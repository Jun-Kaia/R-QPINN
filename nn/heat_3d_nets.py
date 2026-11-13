import torch
import torch.nn as nn
import os
from src.nn.DVQuantumLayer import DVQuantumLayer

class ClassicalHeat3DNet(nn.Module):
    def __init__(self, args, logger, device):
        super().__init__()
        self.device = device
        self.args = args
        self.logger = logger
        self.loss_history = []
        self.error_history = []
        self._initialize_logging()

        self.net = nn.Sequential(
            nn.Linear(4, 100), nn.Tanh(),
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

    def output_transform(self, t, x, y, z, u_raw):
        # 硬约束：u(0)=sinπx sinπy sinπz，边界为零
        pi = torch.tensor(torch.pi, device=t.device)
        u0 = torch.sin(pi*x) * torch.sin(pi*y) * torch.sin(pi*z)
        factor = t * x * (1 - x) * y * (1 - y) * z * (1 - z)
        return u0 + factor * u_raw

    def forward(self, X):
        t, x, y, z = X[:, 0:1], X[:, 1:2], X[:, 2:3], X[:, 3:4]
        u_raw = self.net(X)
        return self.output_transform(t, x, y, z, u_raw)

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

class HybridHeat3DNet(nn.Module):
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
            nn.Linear(4, 100), nn.Tanh(),
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

    def output_transform(self, t, x, y, z, u_raw):
        pi = torch.tensor(torch.pi, device=t.device)
        u0 = torch.sin(pi*x) * torch.sin(pi*y) * torch.sin(pi*z)
        factor = t * x * (1 - x) * y * (1 - y) * z * (1 - z)
        return u0 + factor * u_raw

    def forward(self, X):
        t, x, y, z = X[:, 0:1], X[:, 1:2], X[:, 2:3], X[:, 3:4]
        q_in = self.preprocessor(X)
        q_features = self.quantum_layer(q_in)
        q_features = q_features.to(self.postprocessor[0].weight.dtype)
        u_raw = self.postprocessor(q_features)
        return self.output_transform(t, x, y, z, u_raw)

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