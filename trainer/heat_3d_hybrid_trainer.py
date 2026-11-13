import torch
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from src.utils.logger import Logging
from src.data.heat_3d_dataset import u_exact
from src.nn.heat_3d_nets import ClassicalHeat3DNet, HybridHeat3DNet
from src.trainer.heat_3d_train import train
from src.utils.ContourPlotter import ContourPlotter

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
D = 0.1   # 热扩散系数

args = {
    "batch_size": 128,
    "lr": 0.001,
    "seed": 1,
    "print_every": 100,
    "log_path": "./checkpoints/heat_3d",
    "input_dim": 4,
    "output_dim": 1,
    "problem": "heat_3d",
    "solver": "DV",          # 或 "Hybrid"
    "device": DEVICE,
    "rad_enabled": True,
    "rad_cycles": 10,
    "num_qubits": 5,
    "num_quantum_layers": 1,
    "q_ansatz": "cascade3",
    "shots": None,
    "encoding": "angle",
}

logger = Logging(args["log_path"])

if args["solver"] == "Classical":
    model = ClassicalHeat3DNet(args, logger, DEVICE)
    logger.print("Using Classical Solver")
else:
    model = HybridHeat3DNet(args, logger, DEVICE)
    logger.print("Using Hybrid Solver")

logger.print("Settings:")
for k, v in args.items():
    logger.print(f"  {k}: {v}")

total_params = sum(p.numel() for p in model.parameters())
logger.print(f"Total number of parameters: {total_params}")

# ================== 生成测试网格（用于全局误差评估） ==================
NUM_T = 20      # 时间点数（0 到 2）
NUM_XYZ = 10    # 每个空间维度点数（0 到 1，为节省内存可减小）

t_vals = torch.linspace(0, 2, NUM_T, device=DEVICE)
x_vals = torch.linspace(0, 1, NUM_XYZ, device=DEVICE)
y_vals = torch.linspace(0, 1, NUM_XYZ, device=DEVICE)
z_vals = torch.linspace(0, 1, NUM_XYZ, device=DEVICE)

# 创建完整网格（用于计算全局误差）
T, X, Y, Z = torch.meshgrid(t_vals, x_vals, y_vals, z_vals, indexing='ij')
X_star = torch.stack([T.flatten(), X.flatten(), Y.flatten(), Z.flatten()], dim=1)
u_star = u_exact(X_star, D=D)

# ================== 训练 ==================
train(model, eval_data=(X_star, u_star), eval_every=100, D=D)

model.save_state()
logger.print("Training completed successfully!")

with torch.no_grad():
    u_pred = model(X_star)

# 全局相对 L2 误差
error_final = torch.norm(u_pred - u_star, 2) / torch.norm(u_star, 2) * 100
logger.print(f"Final Relative L2 error: {error_final:.4f}%")

# ================== 绘制损失曲线 ==================
plt.figure()
plt.plot(model.loss_history)
plt.yscale('log')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig(os.path.join(model.log_path, "loss_history.pdf"), bbox_inches="tight")
plt.show()

# ================== 绘制误差曲线 ==================
if hasattr(model, 'error_history') and len(model.error_history) > 0:
    steps = [e[0] for e in model.error_history]
    errors = [e[1] for e in model.error_history]
    plt.figure()
    plt.plot(steps, errors)
    plt.yscale('log')
    ax = plt.gca()
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.yaxis.set_minor_formatter(ticker.NullFormatter())
    plt.xlabel("Epochs")
    plt.ylabel("Relative L2 Error (%)")
    plt.title("Relative L2 Error Over Epochs")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(model.log_path, "error_vs_steps.pdf"), bbox_inches="tight")
    plt.show()
    logger.print("Error history plot saved.")

# ================== 热力图绘制（使用 ContourPlotter） ==================
logger.print("Generating heatmaps using ContourPlotter...")

# 选择 t=1 和 z=0.5 切片
t_idx = (torch.abs(t_vals - 1.0)).argmin().item()
z_idx = (torch.abs(z_vals - 0.5)).argmin().item()

# 提取数据并重塑为 (时间, x, y) —— 此处时间维只有 1
u_exact_4d = u_star.cpu().numpy().reshape(NUM_T, NUM_XYZ, NUM_XYZ, NUM_XYZ)
u_pred_4d = u_pred.cpu().numpy().reshape(NUM_T, NUM_XYZ, NUM_XYZ, NUM_XYZ)

# 切片：取 t=t_idx, z=z_idx -> 形状 (NUM_XYZ, NUM_XYZ)
exact_slice = u_exact_4d[t_idx, :, :, z_idx]
pred_slice = u_pred_4d[t_idx, :, :, z_idx]
error_slice = np.abs(exact_slice - pred_slice)

# 扩展维度为 (1, NUM_XYZ, NUM_XYZ) 以适应 ContourPlotter
visualization_data = [
    np.expand_dims(exact_slice, axis=0),
    np.expand_dims(pred_slice, axis=0),
    np.expand_dims(error_slice, axis=0),
]

titles = [
    "exact_u (t=1, z=0.5)",
    "pred_u (t=1, z=0.5)",
    "abs_error_u (t=1, z=0.5)",
]

nrows_ncols = (1, 3)          # 一行三列
time_steps = [0]              # 因为时间维只有 1 个

# 准备坐标数组（用于绘图）
xf = x_vals.cpu().numpy()     # 形状 (NUM_XYZ,)
yf = y_vals.cpu().numpy()
tf = np.array([0.0])          # 占位，实际不使用

plotter = ContourPlotter(fontsize=7, labelsize=7, axes_pad=0.5)
plotter.draw_contourf_regular_2D(
    tf=tf,
    xf=xf,
    yf=yf,
    data=visualization_data,
    titles=titles,
    nrows_ncols=nrows_ncols,
    time_steps=time_steps,
    xref=1,
    yref=1,
    model_dirname=model.log_path,
    img_width=10,
    img_height=10,        # 只有一行，高度减小
    ticks=3,
)

logger.print("Heatmap plots saved to {}".format(model.log_path))