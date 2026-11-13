# burgers2_train.py
import time
import torch
from src.nn.pde import burgers_operator
from src.data.burgers2_dataset import Sampler, AdaptiveSampler

def train(model, eval_data=None, eval_every=100):
    device = model.device
    args = model.args
    nu = 0.01 / torch.pi   # 粘性系数

    # ---------- 超参数 ----------
    batch_size = 128
    adam_lr = 0.001
    adam_steps = 10000
    lbfgs_steps = 10000
    lbfgs_per_cycle = 1000
    rad_cycles = 10
    dom_coords = torch.tensor([[0.0, -1.0], [1.0, 1.0]], device=device)

    # ---------- 损失计算函数 ----------
    def compute_loss(sampler, batch_size):
        if hasattr(sampler, 'points') and sampler.points is not None:
            all_pts = sampler.points
            n = all_pts.shape[0]
            if batch_size is None or batch_size >= n:
                X = all_pts.detach().requires_grad_(True)
            else:
                perm = torch.randperm(n, device=device)
                X = all_pts[perm[:batch_size]].detach().requires_grad_(True)
        else:
            X = sampler.sample(batch_size).detach().requires_grad_(True)
        t, x = X[:, 0:1], X[:, 1:2]
        _, r_pred = burgers_operator(model, t, x, nu)
        return torch.mean(r_pred ** 2)

    # ---------- 阶段1：Adam 预热 ----------
    uniform_sampler = Sampler(2, dom_coords, device)
    optimizer_adam = torch.optim.Adam(model.parameters(), lr=adam_lr)
    global_step = 0

    for step in range(adam_steps):
        step_start = time.time()
        optimizer_adam.zero_grad()
        loss = compute_loss(uniform_sampler, batch_size)
        loss.backward()
        optimizer_adam.step()
        step_time = time.time() - step_start
        global_step += 1
        model.loss_history.append(loss.item())

        if step % 100 == 0 or step == adam_steps - 1:
            msg = f"Adam [{step}/{adam_steps}]: loss = {loss.item():.3e}"
            if eval_data is not None and step % eval_every == 0:
                with torch.no_grad():
                    X_star, u_star = eval_data
                    u_pred = model(X_star)
                    err = torch.norm(u_pred - u_star, 2) / torch.norm(u_star, 2) * 100
                    model.error_history.append((global_step, err.item(), step_time))
                    msg += f", L2 error = {err:.4f}%, step_time = {step_time:.4f}s"
            model.logger.print(msg)
    '''
    # ---------- 阶段2：L‑BFGS 精细优化 ----------
    model.logger.print(f"Stage 2: L-BFGS training for {lbfgs_steps} steps")
    optimizer_lbfgs = torch.optim.LBFGS(
        model.parameters(), lr=1.0, max_iter=1,
        history_size=100, line_search_fn='strong_wolfe'
    )

    def lbfgs_closure():
        optimizer_lbfgs.zero_grad()
        loss = compute_loss(uniform_sampler, batch_size)
        loss.backward()
        return loss

    for step in range(lbfgs_steps):
        step_start = time.time()
        loss = optimizer_lbfgs.step(lbfgs_closure)
        step_time = time.time() - step_start
        global_step += 1
        model.loss_history.append(loss.item())

        if step % 100 == 0 or step == lbfgs_steps - 1:
            msg = f"L-BFGS [{step+1}/{lbfgs_steps}]: loss = {loss:.3e}"
            if eval_data is not None and (step % eval_every == 0 or step == lbfgs_steps - 1):
                with torch.no_grad():
                    X_star, u_star = eval_data
                    u_pred = model(X_star)
                    err = torch.norm(u_pred - u_star, 2) / torch.norm(u_star, 2) * 100
                    model.error_history.append((global_step, err.item(), step_time))
                    msg += f", L2 error = {err:.4f}%, step_time = {step_time:.4f}s"
            model.logger.print(msg)
    '''
    # ---------- 阶段3：RAD 自适应循环 ----------
    if rad_cycles > 0:
        rad_sampler = AdaptiveSampler(
            dim=2, coords=dom_coords, device=device,
            k=1, c=1, num_candidate=20000, num_points=700
        )
        rad_sampler.resample(model, burgers_operator, nu)
        current_sampler = rad_sampler

        for cycle in range(1, rad_cycles + 1):
            model.logger.print(f"RAD cycle {cycle}/{rad_cycles} (L-BFGS {lbfgs_per_cycle} steps)")
            optimizer_lbfgs = torch.optim.LBFGS(
                model.parameters(), lr=1.0, max_iter=1,
                history_size=100, line_search_fn='strong_wolfe'
            )

            for step in range(lbfgs_per_cycle):
                step_start = time.time()
                def closure():
                    optimizer_lbfgs.zero_grad()
                    loss = compute_loss(current_sampler, None)   # 全批量
                    loss.backward()
                    return loss
                loss = optimizer_lbfgs.step(closure)
                step_time = time.time() - step_start
                global_step += 1
                model.loss_history.append(loss.item())

                if step % 100 == 0 or step == lbfgs_per_cycle - 1:
                    msg = f"  L-BFGS [{step+1}/{lbfgs_per_cycle}]: loss = {loss:.3e}"
                    if eval_data is not None and (step % eval_every == 0 or step == lbfgs_per_cycle - 1):
                        with torch.no_grad():
                            X_star, u_star = eval_data
                            u_pred = model(X_star)
                            err = torch.norm(u_pred - u_star, 2) / torch.norm(u_star, 2) * 100
                            model.error_history.append((global_step, err.item(), step_time))
                            msg += f", L2 error = {err:.4f}%, step_time = {step_time:.4f}s"
                    model.logger.print(msg)

            if cycle < rad_cycles:
                model.logger.print("  Resampling points (RAD)...")
                rad_sampler.resample(model, burgers_operator, nu)

    model.logger.print("Training completed.")
