import pennylane as qml
import torch
import numpy as np
import torch.nn as nn

# 定义离散变量量子层类，继承自PyTorch的nn.Module
class DVQuantumLayer(nn.Module):
    def __init__(self, args, diff_method="backprop"): #反向传播
        super().__init__()

        """
       使用给定的量子比特数和参数初始化量子层
        Args:
            num_qubits (int): Number of qubits in the quantum circuit.
            args (dict): Additional arguments for the quantum circuit (e.g., hyperparameters).
            diff_method (str): Differentiation method for the QNode (default: "backprop").
            QNode的微分方法（默认："backprop"）
        """
        self.num_qubits = args["num_qubits"]
        self.num_quantum_layers = args["num_quantum_layers"]
        self.shots = args["shots"] # 测量次数（None表示使用解析梯度）
        self.q_ansatz = args["q_ansatz"] # 量子ansatz类型
        self.problem = args["problem"]  # 问题类型（例如"diffusion"）
        self.encoding = args.get("encoding", "angle") # 编码方式，默认为"angle"

        if self.q_ansatz == "layered": #第四种电路
            self.params = nn.Parameter(
                torch.empty(
                    self.num_quantum_layers,
                    self.num_qubits * 4, # 每层参数数：4个参数*量子比特
                    requires_grad=True,
                    dtype=torch.float32,
                )
            )

        elif self.q_ansatz == "alternate": #第一种电路
            self.params = nn.Parameter(
                torch.empty(
                    self.num_quantum_layers,
                    (self.num_qubits * 4) - 4, # 每层参数数：4*量子比特数-4
                    requires_grad=True,
                    dtype=torch.float32,
                )
            )
        elif self.q_ansatz == "cascade": #第二种电路
            self.params = nn.Parameter( # 这些参数是量子门的旋转角度，只有权重（旋转角度）
                torch.empty(
                    self.num_quantum_layers,
                    self.num_qubits * 3, # 每层参数数：3个参数*量子比特
                    requires_grad=True,
                    dtype=torch.float32,
                )
            )

        elif self.q_ansatz == "cascade2": #自创1电路
            self.params = nn.Parameter( # 这些参数是量子门的旋转角度，只有权重（旋转角度）
                torch.empty(
                    self.num_quantum_layers,
                    self.num_qubits * 3, # 每层参数数：3个参数*量子比特
                    requires_grad=True,
                    dtype=torch.float32,
                )
            )

        elif self.q_ansatz == "cascade3": #自创2电路
            self.params = nn.Parameter( # 这些参数是量子门的旋转角度，只有权重（旋转角度）
                torch.empty(
                    self.num_quantum_layers,
                    self.num_qubits * 2, # 每层参数数：3个参数*量子比特
                    requires_grad=True,
                    dtype=torch.float32,
                )
            )

        elif self.q_ansatz == "farhi": #第？种电路
            self.params = nn.Parameter(
                torch.empty(
                    self.num_quantum_layers,
                    (2 * self.num_qubits - 2),
                    requires_grad=True,
                    dtype=torch.float32,
                )
            )

        elif self.q_ansatz == "sim_circ_15": #第？种电路
            self.params = nn.Parameter(
                torch.empty(
                    self.num_quantum_layers,
                    self.num_qubits * 2,
                    requires_grad=True,
                    dtype=torch.float32,
                )
            )

        elif self.q_ansatz == "cross_mesh": #第三种电路
            self.params = nn.Parameter(
                torch.empty(
                    self.num_quantum_layers,
                    (4 * self.num_qubits) +  self.num_qubits *(self.num_qubits - 1),
                    requires_grad=True,
                    dtype=torch.float32,
                )
            )
        else:
            self.params = None
        # 如果没有初始化参数，抛出错误
        if not hasattr(self, "params") or self.params is None:
            raise ValueError(
                "Parameters are not initialized. Check the q_ansatz value."
            )
        self._initialize_weights()
        # ===== 新增：由计算基态概率恢复每个量子比特的 <Z> =====
        self.output_dim = 2 ** self.num_qubits
        self.angle_scale = nn.Parameter(
            torch.full((self.num_qubits,), float(torch.pi))
        )
        self.angle_shift = nn.Parameter(torch.zeros(self.num_qubits))
        # ========================================================
        # 创建PennyLane量子设备，使用默认的量子比特模拟器
        self.dev = qml.device("default.qubit", wires=self.num_qubits, shots=None)

        self.circuit = qml.QNode( # 创建量子节点（QNode），将量子电路转换为可自动微分的Torch函数
            self._quantum_circuit, self.dev, interface="torch", diff_method=diff_method, max_diff=3, cache=False
        )
        '''
        self.circuit = qml.QNode(
            self._quantum_circuit,
            self.dev,
            interface="torch",
            diff_method="adjoint",  # 改为 adjoint
            cache=False
        )
        '''
    '''
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # return torch.stack([self.circuit(sample) for sample in x]) # this line does the same but very slow since it requires running the quantum circuit for each input instance
        return self.circuit(x)   # vectorized input, no Python loop.. transpose in the main function or here is important
    '''

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
        if self.encoding == "amplitude": # 幅度编码
            qml.templates.AmplitudeEmbedding(
                x, wires=range(self.num_qubits), normalize=True, pad_with=0.0# 归一化
            )
        else:
            qml.templates.AngleEmbedding(x, wires=range(self.num_qubits), rotation="Y")#rx旋转门

        if self.q_ansatz == "layered":
            for layer in range(self.num_quantum_layers):
                self.layered(self.params[layer])

        elif self.q_ansatz == "alternate":
            for layer in range(self.num_quantum_layers):
                self.alternate(self.params[layer])
        elif self.q_ansatz == "cascade":
            for layer in range(self.num_quantum_layers):
                self.cascade(self.params[layer])
        elif self.q_ansatz == "cascade2":
            for layer in range(self.num_quantum_layers):
                self.cascade2(self.params[layer])
        elif self.q_ansatz == "cascade3":
            for layer in range(self.num_quantum_layers):
                self.cascade3(self.params[layer])
        elif self.q_ansatz == "farhi":
            for layer in range(self.num_quantum_layers):
                self.farhi_ansatz(self.params[layer])

        elif self.q_ansatz == "sim_circ_15":
            for layer in range(self.num_quantum_layers):
                self.create_sim_circuit_15(self.params[layer])

        elif self.q_ansatz == "cross_mesh":
            for layer in range(self.num_quantum_layers):
                self.create_cross_mesh(self.params[layer])
        # 测量：测量每个量子比特的PauliZ期望值
        return qml.probs(wires=range(self.num_qubits))

    def _initialize_weights(self):
        """Apply Xavier initialization to all layers."""
        if self.q_ansatz == "farhi":
            torch.nn.init.xavier_normal_(
                self.params.view(self.num_quantum_layers, (2 * self.num_qubits - 2))
            )
        elif self.q_ansatz in ["sim_circ_15"]:
            torch.nn.init.xavier_normal_(
                self.params.view(self.num_quantum_layers, self.num_qubits * 2)
            )
        elif self.q_ansatz in [ "alternate"]:
            torch.nn.init.xavier_normal_(
                self.params.view(self.num_quantum_layers, (self.num_qubits * 4) - 4)
            )

        elif self.q_ansatz in ["layered"]:
            torch.nn.init.xavier_normal_(
                self.params.view(self.num_quantum_layers, (self.num_qubits * 4))
            )
        elif self.q_ansatz == "cascade":
            torch.nn.init.xavier_normal_(
                self.params.view(self.num_quantum_layers, self.num_qubits * 3)
            )
        elif self.q_ansatz == "cascade2":
            torch.nn.init.xavier_normal_(
                self.params.view(self.num_quantum_layers, self.num_qubits * 3)
            )
        elif self.q_ansatz == "cascade3":
            torch.nn.init.xavier_normal_(
                self.params.view(self.num_quantum_layers, self.num_qubits * 2)
            )
        elif self.q_ansatz == "cross_mesh":
            torch.nn.init.xavier_normal_(
                self.params.view(
                    self.num_quantum_layers, (4 * self.num_qubits) +  self.num_qubits *(self.num_qubits - 1)
                )
            )
        else:
            raise ValueError("Invalid q_ansatz value.", self.q_ansatz)


    def layered(self, params):
        """
        创建一个具有num_layers * num_qubits参数的量子电路
        Args:
            params (list or tensor): 长度为num_layers * num_qubits的平坦参数列表或张量.
            num_qubits (int): The number of qubits in the circuit.
            num_layers (int): The number of layers in the circuit.

        Returns:
            None: Constructs the quantum circuit.
        """
        # 确保参数数量和格式正确
        assert params is not None and len(params) == self.num_qubits * 4, (
            "The number of parameters must be equal to 4* num_qubits."
        )

        # track the parameter index索引
        param_idx = 0

        # print(f"{len(params)=}")

        # 对每个量子比特应用RZ和RX门
        for qubit_id in range(self.num_qubits):
            # print(f"layer: {layer} ,{param_idx=}")
            qml.RZ(params[param_idx], wires=qubit_id)
            param_idx += 1
            qml.RX(params[param_idx], wires=qubit_id)
            param_idx += 1

        # qml.Barrier(wires=range(self.num_qubits)) ### barriers用于清晰绘制量子电路，但会减慢计算并阻碍优化
        # 应用CNOT门创建纠缠
        for qubit_id in range(self.num_qubits):
            qml.CNOT(wires=[qubit_id, (qubit_id + 1) % self.num_qubits])

        # qml.Barrier(wires=range(self.num_qubits))
        # 再次对每个量子比特应用RX和RZ门
        for qubit_id in range(self.num_qubits):
            # print(f"layer: {layer} ,{param_idx=}")
            qml.RX(params[param_idx], wires=qubit_id)
            param_idx += 1
            qml.RZ(params[param_idx], wires=qubit_id)
            param_idx += 1

    def alternate(self, params):
        """
        构建具有交替薄层CNOT门的变分电路。

        Args:
            params (list or np.ndarray): Parameters for the circuit. Should have a size of
                                        `num_layers * num_qubits * 4` (4 parameters per thinly dressed CNOT gate).
        """
        # 确保参数数量和格式正确
        assert params is not None and len(params) == (self.num_qubits * 4) - 4, (
            "The number of parameters must be equal to  num_qubits * 4."
        )

        param_idx = 0  # Initialize the parameter index

        def build_tdcnot(ctrl, tgt):
            """Build a thinly dressed CNOT gate with the required parameters."""
            nonlocal param_idx  # 允许修改外部变量
            qml.RY(params[param_idx], wires=ctrl) # 控制比特的RY旋转
            param_idx += 1
            qml.RY(params[param_idx], wires=tgt) # 目标比特的RY旋转
            param_idx += 1
            qml.CNOT(wires=[ctrl, tgt])
            qml.RZ(params[param_idx], wires=ctrl) # 控制比特的RZ旋转
            param_idx += 1
            qml.RZ(params[param_idx], wires=tgt) # 目标比特的RZ旋转
            param_idx += 1

        # 添加ansatz的层
        for i in range(self.num_qubits - 1)[::2]:
            ctrl, tgt = i, ((i + 1) % self.num_qubits)
            build_tdcnot(ctrl, tgt)

        # barrier after entanglement
        # qml.Barrier(wires=range(self.num_qubits))

        for i in range(self.num_qubits)[1::2]:
            ctrl, tgt = i, ((i + 1) % self.num_qubits)
            build_tdcnot(ctrl, tgt)

    def cascade(self, params):
        def add_rotations(): #旋转门
            param_counter = 0
            for i in range(0, self.num_qubits):
                qml.RX(params[param_counter], wires=i)
                param_counter += 1

            for i in range(0, self.num_qubits):
                qml.RZ(params[param_counter], wires=i)
                param_counter += 1

            # barrier after entanglement
            # qml.Barrier(wires=range(self.num_qubits))

        def add_entangling_gates(): #纠缠门
            param_counter = 0
            qml.CRX(params[param_counter], wires=[self.num_qubits - 1, 0])
            param_counter += 1
            for i in reversed(range(1, self.num_qubits)): #reversed反转
                qml.CRX(params[param_counter], wires=[i - 1, i])
                param_counter += 1

        # add layers of the ansatz
        add_rotations()
        add_entangling_gates()

    def cascade2(self, params):
        def add_rotations2(): #旋转门
            param_counter = 0
            for i in range(0, self.num_qubits):
                qml.RX(params[param_counter], wires=i)
                param_counter += 1

            for i in range(0, self.num_qubits):
                qml.RZ(params[param_counter], wires=i)
                param_counter += 1

            # barrier after entanglement
            # qml.Barrier(wires=range(self.num_qubits))

        def add_entangling_gates2(): #纠缠门
            param_counter = 0
            for i in range(0, self.num_qubits - 1):
                qml.CRZ(params[param_counter], wires=[i, i + 1])
                param_counter += 1
            qml.CRZ(params[param_counter], wires=[self.num_qubits - 1, 0])
            param_counter += 1

        # add layers of the ansatz
        add_rotations2()
        add_entangling_gates2()

    def cascade3(self, params):
        param_idx = 0
        # 对每个量子比特应用RZ和RX门
        for qubit_id in range(self.num_qubits):
            # print(f"layer: {layer} ,{param_idx=}")
            qml.RZ(params[param_idx], wires=qubit_id)
            param_idx += 1
            qml.RX(params[param_idx], wires=qubit_id)
            param_idx += 1

            # barrier after entanglement
            # qml.Barrier(wires=range(self.num_qubits))
        # 应用CNOT门创建纠缠
        for qubit_id in range(self.num_qubits):
            qml.CNOT(wires=[qubit_id, (qubit_id + 1) % self.num_qubits])


    def farhi_ansatz(self, params):
        param_counter = 0

        # 确保有足够的参数用于RXX和RZX门
        if len(params) != (2 * self.num_qubits - 2):
            raise ValueError("Insufficient parameters for RXX and RZX gates")

        # custom RXX and RZX gate definitions
        def RXX(theta, wires):
            qml.CNOT(wires=wires)
            qml.RX(theta, wires=wires[0])
            qml.CNOT(wires=wires)

        def RZX(theta, wires):
            qml.CNOT(wires=wires)
            qml.RZ(theta, wires=wires[0])
            qml.CNOT(wires=wires)

        # RXX gates
        for i in range(self.num_qubits - 1):
            RXX(params[param_counter], wires=[self.num_qubits - 1, i])
            param_counter += 1

        # RZX gates
        for i in range(self.num_qubits - 1):
            RZX(params[param_counter], wires=[self.num_qubits - 1, i])
            param_counter += 1

    def create_sim_circuit_15(self, params):
        """
        基于arXiv:1905.10876中的电路15创建变分电路。
        Args:
            n_data_qubits (int): Number of qubits in the circuit
            layers (int): Number of layers in the circuit
            sweeps_per_layer (int): Number of sweeps per layer
            activation_function (callable, optional): Activation function to apply between layers
            层之间应用的激活函数
        Returns:
            callable: A function that constructs the quantum circuit with given parameters
        """
        if params is None or len(params) != 2 * self.num_qubits:
            raise ValueError("Insufficient parameters for RXX and RZX gates")

        param_index = 0

        # apply rotations
        def apply_rotations(): # 应用旋转门
            nonlocal param_index
            for i in range(self.num_qubits):
                qml.RY(params[param_index], wires=i)
                param_index += 1

        # 应用纠缠块1
        def apply_entangling_block1():
            for i in reversed(range(self.num_qubits)):
                qml.CNOT(wires=[i, (i + 1) % self.num_qubits])

        # 应用纠缠块2
        def apply_entangling_block2():
            for i in range(self.num_qubits):
                control_qubit = (i + self.num_qubits - 1) % self.num_qubits
                target_qubit = (control_qubit + 3) % self.num_qubits
                qml.CNOT(wires=[control_qubit, target_qubit])

        # 主电路构建
        apply_rotations()
        # barrier after entanglement
        # qml.Barrier(wires=range(self.num_qubits))
        apply_entangling_block1()
        # barrier after entanglement
        # qml.Barrier(wires=range(self.num_qubits))
        apply_rotations()
        # barrier after entanglement
        # qml.Barrier(wires=range(self.num_qubits))
        apply_entangling_block2()
        # barrier after entanglement
        # qml.Barrier(wires=range(self.num_qubits))

    def create_cross_mesh(self, params):
        """
        创建电路5的通用版本，其中控制线和目标线被适当分离的CRZ门.

        Args:
            params (np.ndarray): Array of parameters for the rotation gates
        """
        param_idx = 0
        # 验证参数计数
        expected_params = (4 * self.num_qubits) +  self.num_qubits *(self.num_qubits - 1)

        if params is None or len(params) != expected_params:
            raise ValueError(
                f"Expected {expected_params} parameters but got {params.shape}"
            )

        # 所有量子比特上的初始RX门
        for i in range(self.num_qubits):
            qml.RX(params[param_idx], wires=i)
            param_idx += 1
        # 所有量子比特上的RZ门
        for i in range(self.num_qubits):
            qml.RZ(params[param_idx], wires=i)
            param_idx += 1

        # barrier after entanglement
        # qml.Barrier(wires=range(self.num_qubits))
        # 额外的CRZ门（除了量子比特自身外，每个量子比特与其他量子比特之间）
        for i in range(self.num_qubits - 1, -1, -1):
            for j in range(self.num_qubits - 1, -1, -1):
                if j != i:
                    qml.CRZ(params[param_idx], wires=[i, j])
                    param_idx += 1

        # barrier after entanglement
        # qml.Barrier(wires=range(self.num_qubits))

        # 中间层（使用RX而不是原始电路中的CNOT）
        for i in range(self.num_qubits):
            qml.RX(params[param_idx], wires=i)
            param_idx += 1
        # 中间层的RZ门
        for i in range(self.num_qubits):
            qml.RZ(params[param_idx], wires=i)
            param_idx += 1

        # barrier after entanglement
        # qml.Barrier(wires=range(self.num_qubits))

    def quantum_tanh_n_qubits(self, params, scale=1.0):
        """
        具有跨量子比特相互作用的增强型非线性量子tanh激活

        Args:
            scale (float): 激活的缩放因子
            params (list): List of trainable parameters for the rotations
        """
        if self.num_qubits is None:
            raise ValueError("Wires cannot be None.")

        if params is None:
            # 为直接和交叉相互作用创建参数
            n_params = self.num_qubits * (self.num_qubits - 1) // 2
            params = [scale * np.pi / 2.0 * index for index in range(n_params)]

        # add nonlinear phase shifts 添加非线性相移
        for index in range(self.num_qubits):
            qml.PhaseShift(np.sin(params[index]) * np.pi, wires=index)
