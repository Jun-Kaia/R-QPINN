import pennylane as qml
import torch
import torch.nn as nn


class DVQuantumLayer(nn.Module):
    def __init__(self, args, diff_method="backprop"):
        super().__init__()

        """
        使用给定的量子比特数和参数初始化量子层
        Args:
            args (dict): Additional arguments for the quantum circuit (e.g., hyperparameters).
            diff_method (str): Differentiation method for the QNode (default: "backprop").
        """
        self.num_qubits = args["num_qubits"]
        self.num_quantum_layers = args["num_quantum_layers"]
        self.shots = args["shots"]
        self.q_ansatz = args["q_ansatz"]
        self.problem = args["problem"]
        self.encoding = args.get("encoding", "angle")

        # 仅支持 cascade3
        if self.q_ansatz != "cascade3":
            raise ValueError(
                f"Only 'cascade3' ansatz is supported, got: {self.q_ansatz}"
            )

        self.params = nn.Parameter(
            torch.empty(
                self.num_quantum_layers,
                self.num_qubits * 2,
                requires_grad=True,
                dtype=torch.float32,
            )
        )
        self._initialize_weights()

        # 由计算基态概率恢复每个量子比特的 <Z>
        self.output_dim = 2 ** self.num_qubits
        self.angle_scale = nn.Parameter(
            torch.full((self.num_qubits,), float(torch.pi))
        )
        self.angle_shift = nn.Parameter(torch.zeros(self.num_qubits))

        self.dev = qml.device("default.qubit", wires=self.num_qubits, shots=None)

        self.circuit = qml.QNode(
            self._quantum_circuit,
            self.dev,
            interface="torch",
            diff_method=diff_method,
            max_diff=3,
            cache=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        angles = self.angle_scale * torch.tanh(x) + self.angle_shift

        probs = self.circuit(angles)

        # 兼容 PennyLane 0.29 的批量输出形状
        if probs.ndim == 2 and probs.shape == (self.output_dim, batch_size):
            probs = probs.transpose(0, 1)

        if probs.numel() != batch_size * self.output_dim:
            raise RuntimeError(
                f"Unexpected probability shape: {tuple(probs.shape)}"
            )

        probs = probs.reshape(batch_size, self.output_dim)

        # 居中、缩放，便于后面的经典层训练
        return (probs - 1.0 / self.output_dim) * (self.output_dim ** 0.5)

    # 量子电路
    def _quantum_circuit(self, x):
        if self.encoding == "amplitude":
            qml.templates.AmplitudeEmbedding(
                x, wires=range(self.num_qubits), normalize=True, pad_with=0.0
            )
        else:
            qml.templates.AngleEmbedding(
                x, wires=range(self.num_qubits), rotation="Y"
            )

        for layer in range(self.num_quantum_layers):
            self.cascade3(self.params[layer])

        # 测量：每个量子比特的计算基态概率
        return qml.probs(wires=range(self.num_qubits))

    def _initialize_weights(self):
        """Apply Xavier initialization."""
        torch.nn.init.xavier_normal_(
            self.params.view(self.num_quantum_layers, self.num_qubits * 2)
        )

    def cascade3(self, params):
        param_idx = 0
        # 对每个量子比特应用 RZ 和 RX 门
        for qubit_id in range(self.num_qubits):
            qml.RZ(params[param_idx], wires=qubit_id)
            param_idx += 1
            qml.RX(params[param_idx], wires=qubit_id)
            param_idx += 1

        # 应用 CNOT 门创建纠缠
        for qubit_id in range(self.num_qubits):
            qml.CNOT(wires=[qubit_id, (qubit_id + 1) % self.num_qubits])
