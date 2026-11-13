# burgers2_main.py
import torch
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from src.nn.pde import burgers_operator
from src.utils.logger import Logging
from src.data.burgers2_dataset import load_reference_solution
from src.utils.plot_prediction import plt_prediction1
from src.nn.burgers2_nets import ClassicalBurgersNet, HybridBurgersNet
from src.trainer.burgers2_train import train

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------- 参数配置 ----------
args = {
    "batch_size": 128,
    "lr": 0.001,
    "seed": 1,
    "print_every": 100,
    "log_path": "./checkpoints/burgers2",
    "input_dim": 2,
    "output_dim": 1,
    "problem": "burgers",
    "solver": "DV",          # 或 "Hybrid"
    "device": DEVICE,
    "rad_enabled": True,            # 是否启用 RAD
    "rad_cycles": 10,
    # 量子相关（仅 Hybrid 使用）
    "num_qubits": 5,
    "num_quantum_layers": 1,
    "q_ansatz": "cascade3",
    "shots": None,
    "encoding": "angle",
}

log_path = args["log_path"]
logger = Logging(log_path)

# ---------- 模型选择 ----------
if args["solver"] == "Classical":
    model = ClassicalBurgersNet(args, logger, DEVICE)
    logger.print("Using Classical Solver (6x100, hard constraint)")
else:
    model = HybridBurgersNet(args, logger, DEVICE)
    logger.print("Using Hybrid Solver (classical + quantum + classical, hard constraint)")

logger.print("Settings:")
for k, v in args.items():
    logger.print(f"  {k}: {v}")

total_params = sum(p.numel() for p in model.parameters())
logger.print(f"Total number of parameters: {total_params}")

# ---------- 加载参考解（测试集） ----------
mat_path = r"D:\python\《量子机器学习》资源\QCPINN\QCPINN-main\src\data\burgers_shock.mat"
interp = load_reference_solution(mat_path, DEVICE)

# 生成评估网格 [0,1] × [-1,1]
NUM = 100
t = torch.linspace(0, 1, NUM, device=DEVICE)
x = torch.linspace(-1, 1, NUM, device=DEVICE)
T, X = torch.meshgrid(t, x, indexing='ij')
X_star = torch.stack([T.flatten(), X.flatten()], dim=1)
u_star_np = interp(X_star.cpu().numpy())
u_star = torch.tensor(u_star_np, dtype=torch.float32, device=DEVICE).unsqueeze(1)

# ---------- 开始训练 ----------
train(model, eval_data=(X_star, u_star), eval_every=100)

# ---------- 保存模型 ----------
model.save_state()
logger.print("Training completed successfully!")

# ---------- 最终预测与误差 ----------
with torch.no_grad():
    u_pred = model(X_star)
error_final = torch.norm(u_pred - u_star, 2) / torch.norm(u_star, 2) * 100
logger.print(f"Final Relative L2 error: {error_final:.4f}%")

# ---------- 绘制损失曲线 ----------
plt.figure()
plt.plot(model.loss_history)
plt.yscale('log')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig(os.path.join(model.log_path, "loss_history.pdf"), bbox_inches="tight")
plt.show()

# ---------- 绘制误差曲线 ----------
if hasattr(model, 'error_history') and len(model.error_history) > 0:
    steps = [e[0] for e in model.error_history]
    errors = [e[1] for e in model.error_history]
    plt.figure()
    plt.plot(steps, errors)
    plt.yscale('log')
    ax = plt.gca()
    # 固定刻度位置（数值）和标签
    fixed_ticks = [0.1, 1, 10, 100]
    fixed_labels = ['$10^{-1}$', '$10^0$', '$10^1$', '$10^2$']
    ax.set_yticks(fixed_ticks)
    ax.set_yticklabels(fixed_labels)
    ax.yaxis.set_minor_formatter(ticker.NullFormatter())
    plt.xlabel("Epochs")
    plt.ylabel("Relative L2 Error (%)")
    plt.title("Relative L2 Error Over Epochs")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(model.log_path, "error_vs_steps.pdf"), bbox_inches="tight")
    plt.show()
    logger.print("Error history plot saved.")
else:
    logger.print("No error history recorded.")

# 调用 plot_prediction 绘制 u 的对比图（不再传入 f 数据）
plt_prediction1(
    logger,
    X_star.cpu().numpy(),
    u_star.cpu().numpy(),
    u_pred.cpu().detach().numpy()
)