import torch
import os
import sys
import re
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import csv

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.nn.burgers2_nets import ClassicalBurgersNet, HybridBurgersNet
from src.data.burgers2_dataset import load_reference_solution
from src.utils.plot_prediction import plt_prediction1

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------- 虚拟 Logger ----------
class DummyLogger:
    def __init__(self):
        self.output_dir = str(Path(__file__).parent / "dummy_logs")
        os.makedirs(self.output_dir, exist_ok=True)

    def get_output_dir(self):
        return self.output_dir

    def print(self, *args):
        pass

dummy_logger = DummyLogger()

# ---------- 配置 ----------
BASE_DIR = Path(__file__).parent / "src" / "trainer" / "checkpoints" / "b2rq"

NUM = 100
MAT_PATH = r"D:\python\《量子机器学习》资源\QCPINN\QCPINN-main\src\data\burgers_shock.mat"
OUTPUT_SUBDIR = "average_plots"

# ---------- 加载参考解插值 ----------
def get_reference_interpolator():
    interp = load_reference_solution(MAT_PATH, DEVICE)
    return interp

# ---------- 模型加载 ----------
def load_model_from_checkpoint(ckpt_path, device):
    checkpoint = torch.load(ckpt_path, map_location=device)
    args = checkpoint['args']

    if args.get("solver") == "Classical":
        model = ClassicalBurgersNet(args, dummy_logger, device)
    else:
        model = HybridBurgersNet(args, dummy_logger, device)

    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print("  使用完整 model_state_dict 加载成功")
    else:
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

# ---------- 通用误差解析函数 ----------
def parse_error_history_from_log(log_path):
    """
    从 output.log 中提取所有 L2 error 记录及其对应的全局迭代步数。
    支持三种阶段：
      1. Adam: 步数直接为 [0..10000]
      2. 普通 L-BFGS (无 RAD): 步数为 10000 + step (step 从 1 开始)
      3. RAD 循环内的 L-BFGS: 步数为 10000 + (cycle-1)*cycle_steps + step
    返回 (iters, errors)
    """
    if not log_path.exists():
        return [], []

    with open(log_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    iters = []
    errors = []
    adam_steps = 10000          # 固定值，可根据日志中 Adam 的总步数动态读取，但为简化固定
    in_rad = False
    cycle_num = 0
    lbfgs_per_cycle = 1000       # 默认值，后续从 RAD 声明行读取

    for line in lines:
        # 检测 RAD cycle 声明行，获取 cycle 编号和该 cycle 的步数
        rad_match = re.search(r'RAD cycle (\d+)/\d+ \(L-BFGS (\d+) steps\)', line)
        if rad_match:
            cycle_num = int(rad_match.group(1))
            lbfgs_per_cycle = int(rad_match.group(2))
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

        # 检测 L-BFGS 步数（可能出现在普通阶段或 RAD 内）
        lbfgs_match = re.search(r'L-BFGS \[(\d+)/\d+\]', line)
        if lbfgs_match:
            step = int(lbfgs_match.group(1))
            err_match = re.search(r'L2 error\s*=\s*([\d.]+)%', line)
            if err_match:
                if in_rad and cycle_num > 0:
                    # RAD 循环内
                    global_step = adam_steps + (cycle_num - 1) * lbfgs_per_cycle + step
                else:
                    # 普通 L-BFGS（无 RAD）
                    global_step = adam_steps + step
                iters.append(global_step)
                errors.append(float(err_match.group(1)))
            continue

        # 若遇到 "Stage 2: L-BFGS" 行，表示进入普通 L-BFGS 阶段，重置 RAD 标志（如果有）
        if re.search(r'Stage 2: L-BFGS', line):
            in_rad = False
            cycle_num = 0

    return iters, errors

# ---------- 主函数 ----------
def main():
    if not BASE_DIR.exists():
        raise FileNotFoundError(f"目录不存在: {BASE_DIR}")

    experiment_dirs = [d for d in BASE_DIR.iterdir() if d.is_dir() and (d / "model.pth").exists()]
    print(f"找到 {len(experiment_dirs)} 个实验目录")
    if not experiment_dirs:
        raise FileNotFoundError("未找到包含 model.pth 的实验目录")

    output_dir = BASE_DIR / OUTPUT_SUBDIR
    output_dir.mkdir(exist_ok=True)

    # ---------- 测试网格 ----------
    t_vals = torch.linspace(0, 1, NUM, device=DEVICE)
    x_vals = torch.linspace(-1, 1, NUM, device=DEVICE)
    T, X = torch.meshgrid(t_vals, x_vals, indexing='ij')
    X_star = torch.stack([T.flatten(), X.flatten()], dim=1)

    interp = get_reference_interpolator()
    u_star_np = interp(X_star.cpu().numpy())
    u_star = torch.tensor(u_star_np, dtype=torch.float32, device=DEVICE).unsqueeze(1)
    u_star_2d = u_star.cpu().numpy().reshape(NUM, NUM)

    # ---------- 遍历 ----------
    sum_u_pred = np.zeros((NUM, NUM))
    all_u_preds = []
    loss_histories = []
    error_histories = []
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
        u_pred_2d = u_pred_flat.cpu().numpy().reshape(NUM, NUM)
        sum_u_pred += u_pred_2d
        all_u_preds.append(u_pred_2d)

        error_u = (np.linalg.norm(u_star_2d - u_pred_2d) / np.linalg.norm(u_star_2d)) * 100.0

        checkpoint = torch.load(model_path, map_location='cpu')
        loss_hist = checkpoint.get('loss_history')
        if loss_hist is not None:
            if torch.is_tensor(loss_hist):
                loss_hist = loss_hist.cpu().numpy()
            loss_histories.append(loss_hist)
        else:
            print(f"  实验 {exp_dir.name}: 未找到 loss_history")

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

    # ---------- 保存汇总 CSV ----------
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

    # ---------- 平均损失曲线 ----------
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

    # ---------- 平均误差曲线 ----------
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

    # ---------- 热力图 ----------
    class PlotLogger:
        def get_output_dir(self):
            return str(output_dir)

    plot_logger = PlotLogger()
    plt_prediction1(
        logger=plot_logger,
        X_star=X_star.cpu().numpy(),
        u_star=u_star.cpu().numpy(),
        u_pred=torch.tensor(avg_u_pred, dtype=torch.float32, device='cpu').numpy().reshape(-1, 1),
        f_star=None,
        f_pred=None
    )
    print(f"平均预测热力图已保存到: {output_dir}")

if __name__ == "__main__":
    main()