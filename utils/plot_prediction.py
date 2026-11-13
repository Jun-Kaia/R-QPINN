import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter

# 综合对比图
def plt_prediction(logger, X_star, u_star, u_pred, f_star, f_pred):
    # Set global font sizes
    plt.rcParams.update(
        {
            "font.size": 14,
            "axes.titlesize": 16,
            "axes.labelsize": 14,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
        }
    )

    # Data dictionary for u(x) and f(x) to be used in loops
    data = {
        "u": {
            "exact": u_star,
            "predicted": u_pred,
            "error": np.abs(u_star - u_pred),
            "title": r"$u(x)$",
        },
        "f": {
            "exact": f_star,
            "predicted": f_pred,
            "error": np.abs(f_star - f_pred),
            "title": r"$f(x)$",
        },
    }

    fig, axs = plt.subplots(2, 3, figsize=(18, 10))  # 2 rows, 3 columns
    content = ["exact", "predicted", "error"]

    x_unique = np.unique(X_star[:, 0])
    y_unique = np.unique(X_star[:, 1])
    X, Y = np.meshgrid(x_unique, y_unique)

    for row, (key, value) in enumerate(data.items()):
        for col, field in enumerate(content):
            Z = value[field].reshape(len(y_unique), len(x_unique))

            contour = axs[row, col].contourf(
                X,
                Y,
                Z,
                levels=20,  # Number of contour levels
                cmap="coolwarm",
            )

            # Only show y-axis labels and ticks for leftmost column
            if (col == 0) and (
                row == (len(data.items()) - 1)
            ):  # If not leftmost column
                axs[row, col].set_ylabel(r"$x_2$ →", fontsize=14)
                axs[row, col].set_xlabel(r"$x_1$ →", fontsize=14)
            else:
                axs[row, col].set_yticklabels([])
                axs[row, col].set_xticklabels([])
                axs[row, col].set_ylabel("")
                axs[row, col].set_xlabel("")

            axs[row, col].set_title(
                f"{field.capitalize()} {value['title']}", fontsize=16
            )

            # Add colorbar with larger font size
            cbar = fig.colorbar(contour, ax=axs[row, col])
            cbar.ax.tick_params(labelsize=12)

    # Adjust layout and save the figure
    plt.tight_layout()
    path = os.path.join(logger.get_output_dir(), "prediction.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()

# 单一等高线图
def plot_contour(
    X_star, u_star, img_name, plot_xy=False, xy_labels=[r"$x_1$", r"$x_2$"]
):
    """ """
    fig, axs = plt.subplots(1, 1, figsize=(5, 4))

    min_val = np.min(u_star)
    max_val = np.max(u_star)
    if min_val == max_val == 0:
        min_val += -1e-16
        max_val += 1e-6

    levels = np.linspace(min_val, max_val, 60)

    contour = axs.contourf(
        X_star[..., 0],
        X_star[..., 1],
        u_star,
        levels=levels,
        cmap="jet",
        vmin=min_val,
        vmax=max_val,
    )

    if plot_xy:
        axs.set_xlabel(xy_labels[0], fontsize=18, color="grey")
        axs.set_ylabel(xy_labels[1], fontsize=18, color="grey")

    axs.tick_params(axis="both", which="major", labelsize=14, colors="grey")

    cbar = fig.colorbar(
        contour, ax=axs, ticks=np.linspace(min_val, max_val, 4), pad=0.01
    )
    cbar.ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    cbar.ax.tick_params(labelsize=14, colors="grey")
    cbar.outline.set_visible(False)

    axs.spines["top"].set_visible(False)
    axs.spines["right"].set_visible(False)
    axs.spines["left"].set_visible(False)
    axs.spines["bottom"].set_visible(False)

    # Adjust layout and save the figure with reduced gap and dark grey border
    # plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    plt.savefig(img_name, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.show()
    plt.close()

# 自定义网格图
def grid_one_contour_plots_regular(
    data,
    x,
    y,
    dirname,
    plot_xy=False,
    xy_labels=[r"$t$", r"$x$"],
    img_width=4,
    img_height=5,
    ticks=4,
    fontsize=3,
    labelsize=3,
):
    fig, ax = plt.subplots(figsize=(img_width, img_height))

    min_ = np.min(data)
    max_ = np.max(data)
    if min_ == max_ == 0:
        min_ += -1e-16
        max_ += 1e-6

    levels = np.linspace(min_, max_, 60)
    contour = ax.contourf(x, y, data, levels=levels, cmap="jet", vmin=min_, vmax=max_)

    cbar = fig.colorbar(
        contour, ax=ax, ticks=np.linspace(min_, max_, ticks), format="%.1e"
    )
    cbar.ax.tick_params(labelsize=labelsize)

    if plot_xy:
        ax.set_xlabel(xy_labels[0], fontsize=fontsize)
        ax.set_ylabel(xy_labels[1], fontsize=fontsize)

    ax.tick_params(labelsize=labelsize)
    plt.savefig(dirname, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)

    return fig

def plt_prediction1(logger, X_star, u_star, u_pred, f_star=None, f_pred=None):
    """
    绘制预测与真实值的对比等高线图。
    如果 f_star 和 f_pred 均为 None，则只绘制 u 的对比（1行3列）；
    否则绘制 u 和 f 两行（2行3列）。
    """
    # 设置全局字体
    plt.rcParams.update({
        "font.size": 14,
        "axes.titlesize": 16,
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
    })

    # 准备数据字典：u 总是存在，f 可选
    data = {
        "u": {
            "exact": u_star,
            "predicted": u_pred,
            "error": np.abs(u_star - u_pred),
            "title": r"$u(x)$",
        }
    }
    if f_star is not None and f_pred is not None:
        data["f"] = {
            "exact": f_star,
            "predicted": f_pred,
            "error": np.abs(f_star - f_pred),
            "title": r"$f(x)$",
        }

    n_rows = len(data)
    n_cols = 3
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(18, 5 * n_rows))

    # 如果只有一行，将 axs 转换为二维数组以便统一处理
    if n_rows == 1:
        axs = axs.reshape(1, -1)

    # 获取网格坐标
    x_unique = np.unique(X_star[:, 0])
    y_unique = np.unique(X_star[:, 1])
    X, Y = np.meshgrid(x_unique, y_unique)

    for row, (key, value) in enumerate(data.items()):
        for col, field in enumerate(["exact", "predicted", "error"]):
            Z = value[field].reshape(len(y_unique), len(x_unique))
            contour = axs[row, col].contourf(
                X, Y, Z, levels=20, cmap="coolwarm"
            )

            # 只在最左侧列显示 y 标签，最下行显示 x 标签
            if col == 0:
                axs[row, col].set_ylabel(r"$x_2$ →", fontsize=14)
            else:
                axs[row, col].set_yticklabels([])
                axs[row, col].set_ylabel("")

            if row == n_rows - 1:
                axs[row, col].set_xlabel(r"$x_1$ →", fontsize=14)
            else:
                axs[row, col].set_xticklabels([])
                axs[row, col].set_xlabel("")

            axs[row, col].set_title(
                f"{field.capitalize()} {value['title']}", fontsize=16
            )
            cbar = fig.colorbar(contour, ax=axs[row, col])
            cbar.ax.tick_params(labelsize=12)

    plt.tight_layout()
    path = os.path.join(logger.get_output_dir(), "prediction.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()

