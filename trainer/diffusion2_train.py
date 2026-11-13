import time
import torch
from src.nn.pde import diffusion2_operator
from src.data.diffusion2_dataset import Sampler, AdaptiveSampler

def train(model, eval_data=None, eval_every=100,
          D=0.01, v_x=1.0, v_y=1.0):
    device = model.device
    args = model.args

    batch_size = 128
    adam_lr = 0.001
    adam_steps = 10000
    lbfgs_steps = 10000
    lbfgs_per_cycle = 200
    rad_cycles = 50
    dom_coords = torch.tensor([[0.0, 0.0, 0.0],
                               [1.0, 1.0, 1.0]], device=device)

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
        t, x, y = X[:, 0:1], X[:, 1:2], X[:, 2:3]
        _, r_pred = diffusion2_operator(model, t, x, y, D=D, v_x=v_x, v_y=v_y)
        return torch.mean(r_pred ** 2)

    uniform_sampler = Sampler(3, dom_coords, device=device)
    optimizer_adam = torch.optim.Adam(model.parameters(), lr=adam_lr)
    global_step = 0

    # Adam 预热
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
    # L-BFGS
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
    # RAD 自适应循环
    if rad_cycles > 0:
        rad_sampler = AdaptiveSampler(
            dim=3, coords=dom_coords, device=device,
            k=1, c=1, num_candidate=20000, num_points=500
        )
        rad_sampler.resample(model, diffusion2_operator, D=D, v_x=v_x, v_y=v_y)
        current_sampler = rad_sampler

        for cycle in range(1, rad_cycles + 1):
            model.logger.print(f"RAD cycle {cycle}/{rad_cycles} (L-BFGS {lbfgs_per_cycle} steps) (num_points:500)")
            optimizer_lbfgs = torch.optim.LBFGS(
                model.parameters(), lr=1.0, max_iter=1,
                history_size=100, line_search_fn='strong_wolfe'
            )

            for step in range(lbfgs_per_cycle):
                step_start = time.time()
                def closure():
                    optimizer_lbfgs.zero_grad()
                    loss = compute_loss(current_sampler, None)
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
                rad_sampler.resample(model, diffusion2_operator, D=D, v_x=v_x, v_y=v_y)

    model.logger.print("Training completed.")
