# burgers2_nets.py (修正版)
import torch
import torch.nn as nn
import os
from src.nn.DVQuantumLayer import DVQuantumLayer

class ClassicalBurgersNet(nn.Module):
    def __init__(self, args, logger, device):
        super().__init__()
        self.device = device
        self.args = args
        self.logger = logger
        self.loss_history = []
        self.error_history = []

        # ---------- 新增：初始化日志路径 ----------
        self._initialize_logging()

        self.net = nn.Sequential(
            nn.Linear(2, 100), nn.Tanh(),
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

    def output_transform(self, t, x, u_raw):
        pi = torch.tensor(torch.pi, device=t.device)
        u0 = -torch.sin(pi * x)
        factor = t * (1 - x**2)
        return u0 + factor * u_raw

    def forward(self, X):
        t, x = X[:, 0:1], X[:, 1:2]
        u_raw = self.net(X)
        return self.output_transform(t, x, u_raw)

    def save_state(self, path=None):
        """保存模型参数和训练状态"""
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

class HybridBurgersNet(nn.Module):
    def __init__(self, args, logger, device):
        super().__init__()
        self.device = device
        self.args = args
        self.logger = logger
        self.loss_history = []
        self.error_history = []
        self.num_qubits = args['num_qubits']

        # ---------- 新增：初始化日志路径 ----------
        self._initialize_logging()

        self.preprocessor = nn.Sequential(
            nn.Linear(2, 100), nn.Tanh(),
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

    def output_transform(self, t, x, u_raw):
        pi = torch.tensor(torch.pi, device=t.device)
        u0 = -torch.sin(pi * x)
        factor = t * (1 - x**2)
        return u0 + factor * u_raw

    def forward(self, X):
        t, x = X[:, 0:1], X[:, 1:2]
        q_in = self.preprocessor(X)
        q_features = self.quantum_layer(q_in)
        q_features = q_features.to(self.postprocessor[0].weight.dtype)
        u_raw = self.postprocessor(q_features)
        return self.output_transform(t, x, u_raw)

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