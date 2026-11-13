import torch
import os
import sys
import re
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import csv

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.nn.heat_3d_nets import ClassicalHeat3DNet, HybridHeat3DNet
from src.data.heat_3d_dataset import u_exact
from src.utils.ContourPlotter import ContourPlotter

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
D = 0.1


class DummyLogger:
    def __init__(self):
        self.output_dir = str(Path(__file__).parent / "dummy_logs")
        os.makedirs(self.output_dir, exist_ok=True)

    def get_output_dir(self):
        return self.output_dir

    def print(self, *args):
        pass


dummy_logger = DummyLogger()

# ========== 用户配置 ==========
BASE_DIR = Path(__file__).parent / "src" / "trainer" / "checkpoints" / "h3rq"
NUM_T = 20
NUM_XYZ = 10
T_SLICE = 1.0
Y_SLICE = 0.5           # 新增：y 切片位置
# Z_SLICE = 0.5          # 不再使用，可删除
OUTPUT_SUBDIR = "average_plots"
# ===============================


def load_model_from_checkpoint(ckpt_path, device):
    checkpoint = torch.load(ckpt_path, map_location=device)
    args = checkpoint['args']

    if args.get("solver") == "Classical":
        model = ClassicalHeat3DNet(args, dummy_logger, device)
    else:
        model = HybridHeat3DNet(args, dummy_logger, device)

    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print("  使用完整 model_state_dict 加载成功")
    else:
        # 兼容旧版（如果有）
        try:
            model.preprocessor.load_state_dict(checkpoint['preprocessor'])
            model.postprocessor.load_state_dict(checkpoint['postprocessor'])
            if 'quantum_layer' in checkpoint:
                model.quantum_layer.load_state_dict(checkpoint['quantum_layer'])
            elif 'hidden_network' in checkpoint:
                model.hidden.load_state_dict(checkpoint['hidden_network'])
        except AttributeError:
            raise RuntimeError("检查点格式不兼容")
    model.to(device)
    model.eval()
    return model, args


def parse_error_history_from_log(log_path):
    """
    从 output.log 中提取每个误差记录对应的真实全局迭代步数。
    支持 Adam、普通 L-BFGS 以及 RAD 循环中的 L-BFGS。
    """
    if not log_path.exists():
        return [], []

    with open(log_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    iters = []
    errors = []
    adam_steps = 10000          # Adam 总步数（固定）
    in_rad = False
    cycle_num = 0
    cycle_steps = 0             # 当前 RAD cycle 的 L-BFGS 总步数

    for line in lines:
        # 检测 RAD cycle 行，获取 cycle 编号和该 cycle 的总步数
        rad_match = re.search(r'RAD cycle (\d+)/\d+ \(L-BFGS (\d+) steps\)', line)
        if rad_match:
            cycle_num = int(rad_match.group(1))
            cycle_steps = int(rad_match.group(2))
            in_rad = True
            continue

        # 检测 Adam 步数
        adam_match = re.search(r'Adam \[(\d+)/\d+\]', line)
        if adam_match:
            step = int(adam_match.group(1))
            err_match = re.search(r'L2 error\s*=\s*([\d.]+)%', line)
            if err_match:
                iters.append(step)
                errors.append(float(err_match.group(1)))
            continue

        # 检测 L-BFGS 步数
        lbfgs_match = re.search(r'L-BFGS \[(\d+)/\d+\]', line)
        if lbfgs_match:
            step = int(lbfgs_match.group(1))
            err_match = re.search(r'L2 error\s*=\s*([\d.]+)%', line)
            if err_match:
                if in_rad and cycle_num > 0:
                    # RAD 循环内：全局步数 = Adam步数 + (cycle-1)*cycle_steps + 当前局部步数
                    global_step = adam_steps + (cycle_num - 1) * cycle_steps + step
                else:
                    # 普通 L-BFGS（非 RAD）
                    global_step = adam_steps + step
                iters.append(global_step)
                errors.append(float(err_match.group(1)))
            continue

    return iters, errors


def main():
    if not BASE_DIR.exists():
        raise FileNotFoundError(f"目录不存在: {BASE_DIR}")

    experiment_dirs = [d for d in BASE_DIR.iterdir() if d.is_dir() and (d / "model.pth").exists()]
    print(f"找到 {len(experiment_dirs)} 个实验目录")
    if not experiment_dirs:
        raise FileNotFoundError("未找到包含 model.pth 的实验目录")

    output_dir = BASE_DIR / OUTPUT_SUBDIR
    output_dir.mkdir(exist_ok=True)

    # 生成测试网格
    t_vals = torch.linspace(0, 2, NUM_T, device=DEVICE)
    x_vals = torch.linspace(0, 1, NUM_XYZ, device=DEVICE)
    y_vals = torch.linspace(0, 1, NUM_XYZ, device=DEVICE)
    z_vals = torch.linspace(0, 1, NUM_XYZ, device=DEVICE)

    T, X, Y, Z = torch.meshgrid(t_vals, x_vals, y_vals, z_vals, indexing='ij')
    X_star = torch.stack([T.flatten(), X.flatten(), Y.flatten(), Z.flatten()], dim=1)

    u_exact_flat = u_exact(X_star, D=D)
    u_exact_4d = u_exact_flat.cpu().numpy().reshape(NUM_T, NUM_XYZ, NUM_XYZ, NUM_XYZ)

    # 累积变量
    sum_u_pred = np.zeros((NUM_T, NUM_XYZ, NUM_XYZ, NUM_XYZ))
    all_u_preds = []
    loss_histories = []
    error_histories = []      # (iters, errors)
    experiment_info = []

    for i, exp_dir in enumerate(experiment_dirs):
        print(f"处理实验 {i+1}/{len(experiment_dirs)}: {exp_dir.name}")
        model_path = exp_dir / "model.pth"
        try:
            model, args = load_model_from_checkpoint(model_path, DEVICE)
        except Exception as e:
            print(f"  加载失败: {e}，跳过")
            continue

        with torch.no_grad():
            u_pred_flat = model(X_star)
        u_pred_4d = u_pred_flat.cpu().numpy().reshape(NUM_T, NUM_XYZ, NUM_XYZ, NUM_XYZ)
        sum_u_pred += u_pred_4d
        all_u_preds.append(u_pred_4d)

        error_u = (np.linalg.norm(u_exact_4d - u_pred_4d) / np.linalg.norm(u_exact_4d)) * 100.0

        checkpoint = torch.load(model_path, map_location='cpu')
        loss_hist = checkpoint.get('loss_history')
        if loss_hist is not None:
            if torch.is_tensor(loss_hist):
                loss_hist = loss_hist.cpu().numpy()
            loss_histories.append(loss_hist)
        else:
            print(f"  实验 {exp_dir.name}: 未找到 loss_history")

        # 解析误差历史
        log_path = exp_dir / "output.log"
        iters_err, errors_err = parse_error_history_from_log(log_path)
        if iters_err:
            error_histories.append((iters_err, errors_err))
        else:
            print(f"  实验 {exp_dir.name}: 未能解析 error history")

        final_loss = loss_hist[-1] if loss_hist is not None else None
        experiment_info.append({
            'experiment_name': exp_dir.name,
            'solver_type': args.get('solver', 'Hybrid'),
            'actual_epochs': len(loss_hist) if loss_hist is not None else args.get('epochs', None),
            'final_loss': final_loss,
            'error_u_percent': error_u,
        })

    num_exp = len(all_u_preds)
    if num_exp == 0:
        raise RuntimeError("没有成功加载任何实验模型")

    avg_u_pred = sum_u_pred / num_exp
    all_u_preds = np.array(all_u_preds)
    std_u_per_point = np.std(all_u_preds, axis=0)
    global_std_u = np.mean(std_u_per_point)

    # 保存实验汇总
    csv_path = output_dir / "experiment_summary.csv"
    with open(csv_path, 'w', newline='') as f:
        fieldnames = ['experiment_name', 'solver_type', 'actual_epochs', 'final_loss', 'error_u_percent']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for info in experiment_info:
            writer.writerow(info)
        writer.writerow({
            'experiment_name': 'AVERAGE/STD',
            'solver_type': '',
            'actual_epochs': np.mean([info['actual_epochs'] for info in experiment_info if info['actual_epochs'] is not None]),
            'final_loss': np.mean([info['final_loss'] for info in experiment_info if info['final_loss'] is not None]),
            'error_u_percent': np.mean([info['error_u_percent'] for info in experiment_info]),
        })
    print(f"实验信息已保存到: {csv_path}")
    print(f"全局平均预测标准差 (u): {global_std_u:.4e}")

    # ---------- 绘制平均损失曲线 ----------
    if len(loss_histories) > 0:
        min_len = min(len(h) for h in loss_histories)
        if any(len(h) != min_len for h in loss_histories):
            print(f"损失历史长度不一致，截断到最小长度 {min_len}")
            loss_histories = [h[:min_len] for h in loss_histories]

        avg_loss = np.mean(loss_histories, axis=0)
        std_loss = np.std(loss_histories, axis=0)

        avg_loss_csv = output_dir / "average_loss.csv"
        with open(avg_loss_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'avg_loss', 'std_loss'])
            for epoch_idx in range(min_len):
                writer.writerow([epoch_idx + 1, avg_loss[epoch_idx], std_loss[epoch_idx]])
        print(f"平均损失已保存到: {avg_loss_csv}")

        plt.figure(figsize=(10, 6))
        epochs = np.arange(1, min_len + 1)
        plt.plot(epochs, avg_loss, label='Average Loss', linewidth=2)
        if len(loss_histories) > 1:
            plt.fill_between(epochs, avg_loss - std_loss, avg_loss + std_loss, alpha=0.3, label='± std')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'Average Training Loss over {len(loss_histories)} Experiments')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
        plt.tight_layout()
        loss_plot_path = output_dir / "average_loss_curve.png"
        plt.savefig(loss_plot_path, dpi=300, bbox_inches='tight')
        print(f"平均损失曲线已保存到: {loss_plot_path}")
        plt.close()

    # ---------- 绘制平均相对 L2 误差曲线 ----------
    if len(error_histories) > 0:
        min_len_err = min(len(err) for _, err in error_histories)
        common_iters = error_histories[0][0][:min_len_err]
        error_matrix = []
        for iters, errs in error_histories:
            if len(errs) >= min_len_err:
                error_matrix.append(errs[:min_len_err])
        if error_matrix:
            error_matrix = np.array(error_matrix)
            avg_error = np.mean(error_matrix, axis=0)
            std_error = np.std(error_matrix, axis=0)

            err_csv = output_dir / "average_error.csv"
            with open(err_csv, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['iteration', 'avg_error_percent', 'std_error_percent'])
                for idx in range(min_len_err):
                    writer.writerow([common_iters[idx], avg_error[idx], std_error[idx]])
            print(f"平均误差已保存到: {err_csv}")

            plt.figure(figsize=(10, 6))
            plt.plot(common_iters, avg_error, label='Average Relative L2 Error', linewidth=2)
            if len(error_matrix) > 1:
                plt.fill_between(common_iters, avg_error - std_error, avg_error + std_error,
                                 alpha=0.3, label='± std')
            plt.xlabel('Iteration')
            plt.ylabel('Relative L2 Error (%)')
            plt.title(f'Average Relative L2 Error over {len(error_matrix)} Experiments')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.yscale('log')
            plt.tight_layout()
            err_plot_path = output_dir / "average_error_curve.png"
            plt.savefig(err_plot_path, dpi=300, bbox_inches='tight')
            print(f"平均误差曲线已保存到: {err_plot_path}")
            plt.close()
    else:
        print("未解析到任何误差历史，跳过平均误差曲线绘制")

    # ---------- 绘制切片热力图 (y 切片) ----------
    t_idx = (torch.abs(t_vals - T_SLICE)).argmin().item()
    y_idx = (torch.abs(y_vals - Y_SLICE)).argmin().item()   # 新：固定 y

    # 提取 (x, z) 平面切片
    exact_slice = u_exact_4d[t_idx, :, y_idx, :]        # 形状 (x, z)
    avg_slice   = avg_u_pred[t_idx, :, y_idx, :]
    error_slice = np.abs(exact_slice - avg_slice)

    visualization_data = [
        np.expand_dims(exact_slice, axis=0),
        np.expand_dims(avg_slice, axis=0),
        np.expand_dims(error_slice, axis=0),
    ]
    titles = [
        f"Exact u ",
        f"Predicted u ",
        f"Absolute Error u ",
    ]
    nrows_ncols = (1, 3)
    time_steps = [0]

    # 横轴为 x，纵轴为 z
    xf = x_vals.cpu().numpy()
    yf = z_vals.cpu().numpy()
    tf = np.array([0.0])

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
        model_dirname=str(output_dir),
        img_width=10,
        img_height=10,
        ticks=3,
    )
    print(f"平均切片图 (y={Y_SLICE}) 已保存到: {output_dir}")


if __name__ == "__main__":
    main()