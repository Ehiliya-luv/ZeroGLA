import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from zoo_attack import RandomGradientEstimator, RandomGradEstimateMethod
from utils import make_dir_if_not_exist, setup_seed, progress_bar, show_imgs


def zgla_algorithm(grad, y, mean_var_list, model, input_size, max_iteration, 
                  device, seed: int, num_pert: int = 12, dummy_x = None):
    # If a good approximation of the target image x is already available, 
    # it can be used as an initialization to enhance reconstruction performance.
    is_min = True
    grad = [g.to(device) for g in grad]
    model = model.to(device)
    y = y.to(device)
    if dummy_x is None:
        dummy_x = torch.randn(input_size).to(device).requires_grad_(True)
    else:
        dummy_x = dummy_x.to(device).requires_grad_(True)

    hook = BNStatisticsHook(model, train=True)
    optim = torch.optim.Adam([dummy_x], lr=0.1)
    scheduler = get_warmup_cosine_scheduler(optim, warm_up_iter=50, T_max=20000, lr_max=0.1, lr_min=1e-5)
    criterion = nn.CrossEntropyLoss()
    zo_estimator = RandomGradientEstimator(
        parameters=model.parameters(),
        mu=0.001,
        num_pert=num_pert,
        grad_estimate_method=RandomGradEstimateMethod.rge_central,
        normalize_perturbation=True,
        device=device,
        torch_dtype=torch.float32,
        paramwise_perturb=False,
    )

    min_total_loss = 1e10
    min_dummy_x = None
    for iteration in range(max_iteration):
        hook.clear()
        optim.zero_grad()
        model.zero_grad()
        
        # loss_fn = lambda x, y: criterion(model(x), y)
        # dummy_grad = zo_estimator.compute_grad_with_graph(dummy_x, y, loss_fn, seed=seed)
        # dummy_loss = criterion(model(dummy_x), y)
        # dummy_grad = torch.autograd.grad(dummy_loss, model.parameters(), create_graph=True)
        dummy_loss = criterion(model(dummy_x), y)
        generated_grad = torch.autograd.grad(dummy_loss, model.parameters(), create_graph=True)
        gen_grad_flat = torch.cat([g.view(-1) for g in generated_grad])
        dummy_grad = []
        for i in range(num_pert):
            rng = zo_estimator.get_rng(seed, i)
            pb_norm = zo_estimator.generate_perturbation_norm(rng)
            
            dot_value = torch.dot(gen_grad_flat, pb_norm)
            dummy_grad += [dot_value]

        dummy_mean_var_list = hook.mean_var_list
        grad_loss = 0
        for g1, g2 in zip(dummy_grad, grad):
            grad_loss += 1 * F.mse_loss(g1, g2, reduction="sum")
            
        bn_loss = 0
        for (m1, v1), (m2, v2) in zip(dummy_mean_var_list, mean_var_list):
            bn_loss += 0.1 * F.mse_loss(m1, m2, reduction="sum")
            bn_loss += 0.1 * F.mse_loss(v1, v2, reduction="sum")

        tv_loss = 1e-4 * tv_loss_fn(dummy_x)
        l2_loss = 1e-6 * l2_loss_fn(dummy_x)

        total_loss = grad_loss + bn_loss + tv_loss + l2_loss
        total_loss.backward()
        optim.step()
        scheduler.step()

        cur_lr = optim.state_dict()['param_groups'][0]['lr']
        with torch.no_grad():
            dummy_x += 0.2 * cur_lr * torch.randn(dummy_x.shape).to(device)
        if is_min:
            if total_loss < min_total_loss:
                min_total_loss = total_loss
                min_dummy_x = dummy_x.detach().clone()
        if iteration % 10 == 0:
            print(
                f"\riter:{iteration}, "
                f"total:{total_loss:.8f}, "
                f"grad:{grad_loss:.8f}, "
                f"bn:{bn_loss:.8f}, "
                f"tv:{tv_loss:.8f}, "
                f"l2:{l2_loss:.8f}, "
                f"lr:{cur_lr:.8f}",
                end="")
    print("\nfinish gradient inversion!")
    hook.close()
    if is_min:
        return min_dummy_x.detach()
    else:
        return dummy_x.detach()


def tv_loss_fn(x):
    bz = x.size(0)
    dx = torch.sum(torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:]))
    dy = torch.sum(torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :]))
    return (dx + dy) / bz


def l2_loss_fn(x):
    return torch.norm(x, p=2)


def get_warmup_cosine_scheduler(optim, warm_up_iter, T_max, lr_max, lr_min):
    lr_lambda = lambda cur_iter: cur_iter / warm_up_iter if cur_iter < warm_up_iter else (lr_min + 0.5 * (
            lr_max - lr_min) * (1.0 + math.cos((cur_iter - warm_up_iter) / (T_max - warm_up_iter) * math.pi))) / 0.1
    return torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lr_lambda)


class BNStatisticsHook:
    def __init__(self, model, train=True):
        self.train = train
        self.mean_var_list = []
        self.hook_list = []
        for name, module in model.named_modules():
            if isinstance(module, nn.BatchNorm2d):
                hook = module.register_forward_pre_hook(self.hook_fn)
                self.hook_list.append(hook)

    def hook_fn(self, _, input_data):
        mean = input_data[0].mean(dim=[0, 2, 3])
        var = input_data[0].var(dim=[0, 2, 3])
        if not self.train:
            mean = mean.detach().clone()
            var = var.detach().clone()
        self.mean_var_list.append([mean, var])

    def close(self):
        self.mean_var_list.clear()
        for hook in self.hook_list:
            hook.remove()

    def clear(self):
        self.mean_var_list.clear()
