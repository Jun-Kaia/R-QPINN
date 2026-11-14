import torch
import numpy as np

def diffusion_operator(
    model, t, x, y, sigma_t=1.0, sigma_x=1.0, sigma_y=1.0, D=0.01, v_x=1.0, v_y=1.0
):
    """
    Operator to compute residuals for the 2D convection-diffusion equation
    """

    t.requires_grad = True
    x.requires_grad = True
    y.requires_grad = True

    # forward pass through the model
    u = model(torch.cat((t, x, y), 1))

    # compute derivatives
    u_t = torch.autograd.grad(u, t, torch.ones_like(u), create_graph=True)[0] / sigma_t
    u_x = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0] / sigma_x
    u_y = torch.autograd.grad(u, y, torch.ones_like(u), create_graph=True)[0] / sigma_y

    u_xx = (
        torch.autograd.grad(u_x, x, torch.ones_like(u_x), create_graph=True)[0]
        / sigma_x
    )
    u_yy = (
        torch.autograd.grad(u_y, y, torch.ones_like(u_y), create_graph=True)[0]
        / sigma_y
    )

    # convection-diffusion equation residual
    residual = u_t + v_x * u_x + v_y * u_y - D * (u_xx + u_yy)

    return u, residual

def diffusion2_operator(model, t, x, y, sigma_t=1.0, sigma_x=1.0, sigma_y=1.0,
                       D=0.01, v_x=1.0, v_y=1.0):
    device = next(model.parameters()).device
    t = t.to(device).requires_grad_(True)
    x = x.to(device).requires_grad_(True)
    y = y.to(device).requires_grad_(True)

    u = model(torch.cat([t, x, y], dim=1))

    grad_u = torch.autograd.grad(
        outputs=u,
        inputs=(t, x, y),
        grad_outputs=torch.ones_like(u),
        create_graph=True,
        retain_graph=True
    )
    u_t, u_x, u_y = grad_u[0], grad_u[1], grad_u[2]

    u_xx = torch.autograd.grad(
        u_x, x,
        grad_outputs=torch.ones_like(u_x),
        create_graph=True,
        retain_graph=True
    )[0] / sigma_x

    u_yy = torch.autograd.grad(
        u_y, y,
        grad_outputs=torch.ones_like(u_y),
        create_graph=True,
        retain_graph=True
    )[0] / sigma_y

    # 源项（包含 -200*(y-0.5)）
    exp_part = torch.exp(-100 * ((x - 0.5)**2 + (y - 0.5)**2) - t)
    f = exp_part * (3 - 200*(x-0.5) - 200*(y-0.5) - 400*((x-0.5)**2 + (y-0.5)**2))

    residual = u_t + v_x * u_x + v_y * u_y - D * (u_xx + u_yy) - f

    residual = torch.clamp(residual, -1e8, 1e8)
    residual = torch.nan_to_num(residual, nan=0.0, posinf=1e8, neginf=-1e8)

    return u, residual


def heat_3d_operator(model, t, x, y, z, sigma_t=1.0, sigma_x=1.0, sigma_y=1.0, sigma_z=1.0, D=0.1):
    """
    计算三维热传导方程残差：r = u_t - D*(u_xx + u_yy + u_zz)
    返回 (u, residual)
    """
    device = next(model.parameters()).device
    t = t.to(device).requires_grad_(True)
    x = x.to(device).requires_grad_(True)
    y = y.to(device).requires_grad_(True)
    z = z.to(device).requires_grad_(True)

    u = model(torch.cat([t, x, y, z], dim=1))

    # 一次性计算一阶导数
    grad_u = torch.autograd.grad(
        outputs=u,
        inputs=(t, x, y, z),
        grad_outputs=torch.ones_like(u),
        create_graph=True,
        retain_graph=True
    )
    u_t, u_x, u_y, u_z = grad_u[0], grad_u[1], grad_u[2], grad_u[3]

    # 二阶导数
    u_xx = torch.autograd.grad(
        u_x, x,
        grad_outputs=torch.ones_like(u_x),
        create_graph=True,
        retain_graph=True
    )[0] / sigma_x

    u_yy = torch.autograd.grad(
        u_y, y,
        grad_outputs=torch.ones_like(u_y),
        create_graph=True,
        retain_graph=True
    )[0] / sigma_y

    u_zz = torch.autograd.grad(
        u_z, z,
        grad_outputs=torch.ones_like(u_z),
        create_graph=True,
        retain_graph=True
    )[0] / sigma_z

    residual = u_t - D * (u_xx + u_yy + u_zz)

    # 数值稳定性处理
    residual = torch.clamp(residual, -1e6, 1e6)
    residual = torch.nan_to_num(residual, nan=0.0, posinf=1e8, neginf=-1e8)

    return u, residual

def burgers_operator1(model, t, x, nu = 0.01):
    """
    输入: t, x 均为 (N,1) 张量
    返回: (u_pred, r_pred)
    r_pred = u_t + u*u_x - nu*u_xx
    """
    X = torch.cat([t, x], dim=1)
    X.requires_grad_(True)
    u = model(X)

    # 一阶导数
    u_t = torch.autograd.grad(u, X, grad_outputs=torch.ones_like(u),
                              create_graph=True, retain_graph=True)[0][:, 0:1]
    u_x = torch.autograd.grad(u, X, grad_outputs=torch.ones_like(u),
                              create_graph=True, retain_graph=True)[0][:, 1:2]

    # 二阶导数
    u_xx = torch.autograd.grad(u_x, X, grad_outputs=torch.ones_like(u_x),
                               create_graph=True, retain_graph=True)[0][:, 1:2]

    r = u_t + u * u_x - nu * u_xx
    return u, r

def burgers_operator(model, t, x, nu=0.01/np.pi, sigma_t=1.0, sigma_x=1.0):
    """
    计算 Burgers 方程残差：r = u_t + u*u_x - nu*u_xx
    参数：
        model: 神经网络
        t, x: 张量 (batch, 1)，必须允许梯度
        nu: 粘性系数（默认 0.01/π）
        sigma_t, sigma_x: 坐标缩放因子（用于归一化导数）
    返回：
        u: 预测值
        residual: PDE 残差
    """
    device = next(model.parameters()).device
    t = t.to(device).requires_grad_(True)
    x = x.to(device).requires_grad_(True)

    u = model(torch.cat([t, x], dim=1))

    # 一次性计算一阶导数
    grad_u = torch.autograd.grad(
        outputs=u,
        inputs=(t, x),
        grad_outputs=torch.ones_like(u),
        create_graph=True,
        retain_graph=True
    )
    u_t, u_x = grad_u[0], grad_u[1]

    # 二阶导数（仅对 x 求导）
    u_xx = torch.autograd.grad(
        u_x, x,
        grad_outputs=torch.ones_like(u_x),
        create_graph=True,
        retain_graph=True
    )[0] / sigma_x   # 缩放因子

    # 残差（注意一阶导数不除缩放因子，与 wave_operator 一致）
    residual = u_t + u * u_x - nu * u_xx

    # 数值稳定性
    residual = torch.clamp(residual, -1e6, 1e6)
    residual = torch.nan_to_num(residual, nan=0.0, posinf=1e6, neginf=-1e6)

    return u, residual

