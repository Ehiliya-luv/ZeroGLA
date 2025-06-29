import copy
import torch
import torch.nn as nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from zoo_attack import RandomGradientEstimator, RandomGradEstimateMethod

from hook import BNStatisticsHook

transform = transforms.Compose([
    transforms.Resize([224, 224]),
    transforms.ToTensor()]
)


def get_imagenet_exp_dl(batch_size):
    imagenet_dir = "~/data/dataset/imagenet/"
    train_ds = datasets.ImageNet(imagenet_dir, split="train", transform=transform)
    val_ds = datasets.ImageNet(imagenet_dir, split="val", transform=transform)
    train_dl = DataLoader(train_ds, batch_size, num_workers=8, pin_memory=True, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size, num_workers=8, pin_memory=True, shuffle=False)
    return train_dl, val_dl


def get_dataloader(name, batch_size, shuffle=False, train=False):
    if name == "imagenet":
        split = "train" if train else "val"
        dataset = datasets.ImageNet("~/data/dataset/imagenet/", split=split, transform=transform)
    elif name == "caltech256":
        dataset = datasets.Caltech256("~/data/dataset/", transform=transform, download=False)
    elif name == "cifar100":
        dataset = datasets.CIFAR100("~/data/dataset/cifar100", transform=transform, download=False, train=train)
    return DataLoader(dataset, batch_size, drop_last=True, shuffle=shuffle)


def get_grad_dl(model: nn.Module, dataloader: DataLoader, device, seed:int, num_pert:int = 12, usegz: bool = False):
    model = copy.deepcopy(model).to(device)
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
    hook = BNStatisticsHook(model, train=False)
    for i, (x, y) in enumerate(dataloader):
        if i == 0:
            continue
        model.zero_grad()
        hook.clear()
        x, y = x.to(device), y.to(device)

        loss_fn = lambda x, y: criterion(model(x), y)
        if usegz:
            _ = zo_estimator.compute_grad(x, y, loss_fn, seed=seed)
            grad = [p.grad.detach().clone() for p in model.parameters() if p.grad is not None]
        else:
            grad = zo_estimator.compute_grad(x, y, loss_fn, seed=seed)
        mean_var_list = hook.mean_var_list
        yield x, y, grad, mean_var_list
