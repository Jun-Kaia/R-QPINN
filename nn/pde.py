import torch
import numpy as np

def navier_stokes_2D_operator(model, t, x, y, min_x=0, max_x=1):
    """
    Operator to compute residuals for the 2D Navier-Stokes equation
    """

    mu = 0.00345
    DENSITY = 1056.0

    t.requires_grad = True
    x.requires_grad = True
    y.requires_grad = True

    uvp = model(torch.concatenate((t, x, y), 1))

    u = uvp[:, 0:1]
    v = uvp[:, 1:2]
    p = uvp[:, 2:3]

    u_t = torch.autograd.grad(u, t, torch.ones_like(u), create_graph=True)[0]
    u_x = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
    u_y = torch.autograd.grad(u, y, torch.ones_like(u), create_graph=True)[0]

    v_t = torch.autograd.grad(v, t, torch.ones_like(v), create_graph=True)[0]
    v_x = torch.autograd.grad(v, x, torch.ones_like(v), create_graph=True)[0]
    v_y = torch.autograd.grad(v, y, torch.ones_like(v), create_graph=True)[0]
    p_y = torch.autograd.grad(p, y, torch.ones_like(p), create_graph=True)[0]
    p_x = torch.autograd.grad(p, x, torch.ones_like(p), create_graph=True)[0]

    u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u_x), create_graph=True)[0]
    u_yy = torch.autograd.grad(u_y, y, torch.ones_like(u_y), create_graph=True)[0]
    v_xx = torch.autograd.grad(v_x, x, torch.ones_like(v_x), create_graph=True)[0]
    v_yy = torch.autograd.grad(v_y, y, torch.ones_like(v_y), create_graph=True)[0]

    continuity = u_x + v_y
    f_u = u_t + (u * u_x + v * u_y) + 1.0 / DENSITY * p_x - mu * (u_xx + u_yy)
    f_v = v_t + (u * v_x + v * v_y) + 1.0 / DENSITY * p_y - mu * (v_xx + v_yy)

    return [continuity, f_u, f_v]


def klein_gordon_operator(fluid_model, t, x, x_min=0.0, x_max=1.0):
    """
    Operator to compute residuals for the 1D Klein-Gordon equation
    """

    alpha = -1.0
    beta = 0.0
    gamma = 1.0
    k = 3

    t.requires_grad = True
    x.requires_grad = True

    u = fluid_model(torch.concatenate((t, x), 1))

    u_t = torch.autograd.grad(u, t, torch.ones_like(u), create_graph=True)[0]
    u_x = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
    u_tt = torch.autograd.grad(u_t, t, torch.ones_like(u_t), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u_x), create_graph=True)[0]

    f_val = -25 * torch.pi ** 2 * x * torch.cos(5 * torch.pi * t) + 6 * t * x ** 3 - (6 * t ** 3 * x) + (
                x * torch.cos(5 * torch.pi * t) + (t * x) ** 3) ** 3

    residual = u_tt + alpha * u_xx + beta * u + gamma * u**k - f_val
    return u, residual


def wave_operator(model, t, x, sigma_t=1.0, sigma_x=1.0):
    """
    Operator to compute residuals for the 1D wave equation
    """
    c = 2

    device = next(model.parameters()).device
    t = t.to(device)
    x = x.to(device)

    u = model(torch.cat((t, x), 1))

    # 一次性计算 u_t 和 u_x
    # grad_outputs: 每个输出对应的梯度系数（这里为全1向量）
    # create_graph=True 保留计算图以便继续求二阶导
    grad_u = torch.autograd.grad(
        outputs=u,
        inputs=(t, x),
        grad_outputs=torch.ones_like(u),
        create_graph=True,
        retain_graph=True  # 保留计算图，以便后续再次求导
    )
    u_t, u_x = grad_u[0], grad_u[1]

    u_tt = torch.autograd.grad(u_t, t, torch.ones_like(u_t), create_graph=True)[0]/ sigma_t
    u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u_x), create_graph=True)[0]/ sigma_x
    residual = u_tt - c**2 * u_xx

    # 裁剪残差，防止 nan 和 inf
    residual = torch.clamp(residual, -1e6, 1e6)
    residual = torch.nan_to_num(residual, nan=0.0, posinf=1e6, neginf=-1e6)

    return u, residual


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

def helmholtz_operator(model, x, y, a_1=1, a_2=4, LAMBDA = 1.0, sigma_x=1.0, sigma_y=1.0):
    """
    Operator to compute residuals for the 2D helmholtz equation
    """

    x.requires_grad_(True)
    y.requires_grad_(True)
    u = model(torch.concatenate((x, y), 1))
    # compute gradients with respect to x1 and x2
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]/ sigma_x
    u_y = torch.autograd.grad(u, y, grad_outputs=torch.ones_like(u), create_graph=True)[0]/ sigma_y
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]/ sigma_x
    u_yy = torch.autograd.grad(u_y, y, grad_outputs=torch.ones_like(u_y), create_graph=True)[0]/ sigma_y
    residual = u_xx + u_yy + LAMBDA * u

    return [u, residual]


def wave_2d_operator(model, t, x, y, sigma_t=1.0, sigma_x=1.0, sigma_y=1.0, a=1.0):
    # 设置需要梯度
    t.requires_grad = True
    x.requires_grad = True
    y.requires_grad = True

    # 前向传播通过模型
    u = model(torch.cat((t, x, y), 1))

    # 计算一阶导数
    u_t = torch.autograd.grad(u, t, torch.ones_like(u), create_graph=True)[0] / sigma_t
    u_x = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0] / sigma_x
    u_y = torch.autograd.grad(u, y, torch.ones_like(u), create_graph=True)[0] / sigma_y

    # 计算二阶导数
    u_tt = torch.autograd.grad(u_t, t, torch.ones_like(u_t), create_graph=True)[0] / sigma_t
    u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u_x), create_graph=True)[0] / sigma_x
    u_yy = torch.autograd.grad(u_y, y, torch.ones_like(u_y), create_graph=True)[0] / sigma_y

    # 计算波动方程残差
    residual = u_tt - a ** 2 * (u_xx + u_yy)

    return u, residual

def wave2_2d_operator(model, t, x, y, sigma_t=1.0, sigma_x=1.0, sigma_y=1.0, a=1.0):
    """
    计算二维波动方程残差：r = u_tt - a^2 (u_xx + u_yy)
    返回 (u, residual)
    """
    device = next(model.parameters()).device
    t = t.to(device).requires_grad_(True)
    x = x.to(device).requires_grad_(True)
    y = y.to(device).requires_grad_(True)

    u = model(torch.cat([t, x, y], dim=1))

    # 一次性计算一阶导数
    grad_u = torch.autograd.grad(
        outputs=u,
        inputs=(t, x, y),
        grad_outputs=torch.ones_like(u),
        create_graph=True,
        retain_graph=True
    )
    u_t, u_x, u_y = grad_u[0], grad_u[1], grad_u[2]

    # 二阶导数
    u_tt = torch.autograd.grad(
        u_t, t,
        grad_outputs=torch.ones_like(u_t),
        create_graph=True,
        retain_graph=True
    )[0] / sigma_t

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

    residual = u_tt - a**2 * (u_xx + u_yy)

    # 数值稳定性处理
    residual = torch.clamp(residual, -1e8, 1e8)
    residual = torch.nan_to_num(residual, nan=0.0, posinf=1e8, neginf=-1e8)

    return u, residual

def advection_operator(model, t, x, c=10.0, sigma_t=1.0, sigma_x=1.0):
    # 设置需要梯度
    t.requires_grad = True
    x.requires_grad = True

    # 前向传播通过模型
    u = model(torch.cat((t, x), 1))

    # 计算一阶导数
    u_t = torch.autograd.grad(u, t, torch.ones_like(u), create_graph=True)[0] / sigma_t
    u_x = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0] / sigma_x

    # 计算平流方程残差: u_t + c * u_x
    residual = u_t + c * u_x

    return u, residual


def helmholtz_inverse_operator(model, x1, x2):
    # 设置需要梯度
    x1.requires_grad_(True)
    x2.requires_grad_(True)

    # 前向传播获取u的预测值
    u = model(torch.cat((x1, x2), 1))

    # 计算一阶导数
    u_x1 = torch.autograd.grad(u, x1, torch.ones_like(u),create_graph=True)[0]
    u_x2 = torch.autograd.grad(u, x2, torch.ones_like(u),create_graph=True,)[0]

    # 计算二阶导数
    u_x1x1 = torch.autograd.grad(u_x1, x1, torch.ones_like(u_x1),create_graph=True,)[0]
    u_x2x2 = torch.autograd.grad(u_x2, x2, torch.ones_like(u_x2),create_graph=True,)[0]

    # 获取模型的k参数
    k = model.k
    epsilon = 1e-8
    # 计算亥姆霍兹方程的残差: u_xx + u_yy + k^2 * u
    residual = u_x1x1 + u_x2x2 + (k ** 2 + epsilon)  * u

    return u, residual


def heat_2d_operator(model, t, x, y, sigma_t=1.0, sigma_x=1.0, sigma_y=1.0, kappa=1.0):
    # 设置需要梯度
    t.requires_grad = True
    x.requires_grad = True
    y.requires_grad = True

    # 前向传播通过模型
    u = model(torch.cat((t, x, y), 1))
    u_t = torch.autograd.grad(u, t, torch.ones_like(u), create_graph=True)[0] / sigma_t
    u_x = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0] / sigma_x
    u_y = torch.autograd.grad(u, y, torch.ones_like(u), create_graph=True)[0] / sigma_y

    u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u_x), create_graph=True)[0] / sigma_x
    u_yy = torch.autograd.grad(u_y, y, torch.ones_like(u_y), create_graph=True)[0] / sigma_y


    # 计算残差：κ(∂²u/∂x² + ∂²u/∂y²) - ∂u/∂t
    residual = kappa * (u_xx + u_yy) - u_t

    return u, residual

def heat2_2d_operator(model, t, x, y, sigma_t=1.0, sigma_x=1.0, sigma_y=1.0, kappa=1.0):
    """
    计算二维热传导方程残差：r = κ(∂²u/∂x² + ∂²u/∂y²) - ∂u/∂t
    参数：
        model: 神经网络
        t, x, y: 张量 (batch, 1)，需要梯度
        sigma_t, sigma_x, sigma_y: 坐标缩放因子（用于归一化导数）
        kappa: 热扩散系数
    返回：
        u: 预测值
        residual: PDE 残差
    """
    device = next(model.parameters()).device
    t = t.to(device).requires_grad_(True)
    x = x.to(device).requires_grad_(True)
    y = y.to(device).requires_grad_(True)

    u = model(torch.cat([t, x, y], dim=1))

    # 一次性计算一阶导数
    grad_u = torch.autograd.grad(
        outputs=u,
        inputs=(t, x, y),
        grad_outputs=torch.ones_like(u),
        create_graph=True,
        retain_graph=True
    )
    u_t, u_x, u_y = grad_u[0], grad_u[1], grad_u[2]

    # 二阶导数（仅对空间坐标）
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

    # 残差：κ(∂²u/∂x² + ∂²u/∂y²) - ∂u/∂t
    residual = kappa * (u_xx + u_yy) - u_t

    # 数值稳定性处理（与 wave_operator 保持一致）
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

def allen_cahn_operator(model, t, x, nu=0.0001):
    """
    计算 Allen-Cahn 方程的残差: u_t - nu*u_xx + 5*u^3 - 5*u
    输入:
        model: 神经网络模型
        t, x: 张量 (batch, 1)
        nu: 扩散系数 (默认 0.0001)
    返回:
        u: 预测值
        f: 残差 (PDE 左端)
    """
    u = model(torch.cat([t, x], dim=1))
    u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
    f = u_t - nu * u_xx + 5 * u**3 - 5 * u
    return u, f

def kdv_operator(model, t, x, epsilon=0.0025):
    """
    Korteweg–De Vries (KdV) 方程残差
    u_t + u*u_x + epsilon * u_xxx = 0
    输入:
        model: 神经网络
        t, x: 张量 (batch, 1)
        epsilon: 色散系数，默认 0.0025
    返回:
        u: 预测值
        residual: PDE 残差
    """
    t.requires_grad_(True)
    x.requires_grad_(True)

    u = model(torch.cat([t, x], dim=1))

    u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u),
                               create_graph=True, retain_graph=True)[0]
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u),
                              create_graph=True, retain_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x),
                               create_graph=True, retain_graph=True)[0]
    u_xxx = torch.autograd.grad(u_xx, x, grad_outputs=torch.ones_like(u_xx),
                                create_graph=True, retain_graph=True)[0]

    residual = u_t + u * u_x + epsilon * u_xxx
    return u, residual