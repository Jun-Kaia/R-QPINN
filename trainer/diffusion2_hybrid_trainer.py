import torch
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from src.utils.logger import Logging
from src.nn.diffusion_nets import ClassicalDiffusionNet, HybridDiffusionNet
from src.trainer.diffusion2_train import train
from src.utils.ContourPlotter import ContourPlotter   # 新增导入

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
D, v_x, v_y = 0.01, 1.0, 1.0

args = {
    "batch_size": 128,
    "lr": 0.001,
    "seed": 1,
    "print_every": 100,
    "log_path": "./checkpoints/diffusion",
    "input_dim": 3,
    "output_dim": 1,
    "problem": "diffusion",
    "solver":"DV",          # 或 "Hybrid" DV Classical
    "device": DEVICE,
    "rad_enabled": True,
    "rad_cycles": 50,
    "num_qubits": 5,
    "num_quantum_layers": 1,
    "q_ansatz": "cascade3",
    "shots": None,
    "encoding": "angle",
}

logger = Logging(args["log_path"])

if args["solver"] == "Classical":
    model = ClassicalDiffusionNet(args, logger, DEVICE)
    logger.print("Using Classical Solver")
else:
    model = HybridDiffusionNet(args, logger, DEVICE)
    logger.print("Using Hybrid Solver")

logger.print("Settings:")
for k, v in args.items():
    logger.print(f"  {k}: {v}")

total_params = sum(p.numel() for p in model.parameters())
logger.print(f"Total number of parameters: {total_params}")

# 生成测试网格（用于评估）
NUM = 20
t_vals = torch.linspace(0, 1, NUM, device=DEVICE)
x_vals = torch.linspace(0, 1, NUM, device=DEVICE)
y_vals = torch.linspace(0, 1, NUM, device=DEVICE)
T, X, Y = torch.meshgrid(t_vals, x_vals, y_vals, indexing='ij')
X_star = torch.stack([T.flatten(), X.flatten(), Y.flatten()], dim=1)

# 精确解（与源项匹配）
def u_exact(txy, D=0.01, v_x=1.0, v_y=1.0):
    t = txy[:, 0:1]
    x = txy[:, 1:2]
    y = txy[:, 2:3]
    return torch.exp(-100 * ((x - 0.5)**2 + (y - 0.5)**2) - t)

u_star = u_exact(X_star, D, v_x, v_y)

# 训练
train(model, eval_data=(X_star, u_star), eval_every=100,
      D=D, v_x=v_x, v_y=v_y)

model.save_state()
logger.print("Training completed successfully!")

with torch.no_grad():
    u_pred = model(X_star)
error_final = torch.norm(u_pred - u_star, 2) / torch.norm(u_star, 2) * 100
logger.print(f"Final Relative L2 error: {error_final:.4f}%")

# 绘制损失曲线
plt.figure()
plt.plot(model.loss_history)
plt.yscale('log')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig(os.path.join(model.log_path, "loss_history.pdf"), bbox_inches="tight")
plt.show()

# 绘制误差曲线
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

# ================== 使用 ContourPlotter 绘制热力图 ==================
logger.print("Generating heatmaps using ContourPlotter...")

# 重塑为三维数组 (时间, x, y)
u_exact_3d = u_star.cpu().numpy().reshape(NUM, NUM, NUM)
u_pred_3d = u_pred.cpu().numpy().reshape(NUM, NUM, NUM)

# 准备可视化数据（只包含 u，不包含 f）
visualization_data = [
    u_exact_3d,          # exact_u
    u_pred_3d,           # pred_u
    np.abs(u_exact_3d - u_pred_3d),  # abs error u
]

titles = [
    "exact_u",
    "pred_u",
    "abs_error_u",
]

nrows_ncols = (1, 3)                # 一行三列
time_steps = [10]             # 选择中间时间切片 (t ≈ 0.5)
# 准备一维坐标（用于绘图）
tf = t_vals.cpu().numpy()
xf = x_vals.cpu().numpy()
yf = y_vals.cpu().numpy()

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
    img_height=10,       # 只有一行，高度适当减小
    ticks=3,
)

logger.print("Heatmap plots saved.")