import os

import torch
import torch.distributed as dist
import torch.nn as nn


def zeropower_via_svd(G, steps=None):
    U, S, V = G.svd()
    return U @ V.T


@torch.compile
def zeropower_via_newtonschulz5(G, steps=10, eps=1e-7):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G.
    """
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    X /= (X.norm() + eps)  # ensure top singular value <= 1
    if G.size(0) > G.size(1):
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = A @ X
        X = a * X + b * B + c * A @ B
    if G.size(0) > G.size(1):
        X = X.T
    return X


zeropower_backends = dict(svd=zeropower_via_svd, newtonschulz5=zeropower_via_newtonschulz5)


class Muon(torch.optim.Optimizer):
    """
    Muon - MomentUm Orthogonalized by Newton-Schulz.
    """

    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True,
                 backend='newtonschulz5', backend_steps=5, weight_decay=0.0):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov,
                        backend=backend, backend_steps=backend_steps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    def step(self):
        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            zeropower_backend = zeropower_backends[group['backend']]
            weight_decay = getattr(group, 'weight_decay', 0)

            total_params = sum(p.numel() for p in group['params'])
            updates_flat = torch.zeros(total_params, device='cuda', dtype=torch.bfloat16)
            curr_idx = 0
            for i, p in enumerate(group['params']):
                if i % int(os.environ['WORLD_SIZE']) == int(os.environ['RANK']):
                    g = p.grad
                    assert g is not None
                    state = self.state[p]
                    if 'momentum_buffer' not in state:
                        state['momentum_buffer'] = torch.zeros_like(g)
                    buf = state['momentum_buffer']
                    buf.mul_(momentum).add_(g)
                    if group['nesterov']:
                        g = g.add(buf, alpha=momentum)
                    g = zeropower_backend(g, steps=group['backend_steps'])
                    g *= max(1, g.size(0) / g.size(1)) ** 0.5
                    updates_flat[curr_idx:curr_idx + p.numel()] = g.flatten()
                curr_idx += p.numel()

            dist.all_reduce(updates_flat, op=dist.ReduceOp.SUM)

            curr_idx = 0
            for p in group['params']:
                if weight_decay != 0.0:
                    p.data.mul_(1.0 - lr * weight_decay)
                g = updates_flat[curr_idx:curr_idx + p.numel()].view_as(p.data).type_as(p.data)
                p.data.add_(g, alpha=-lr)
                curr_idx += p.numel()


class MuonClip(torch.optim.Optimizer):
    """
    MuonClip - Muon with consistent update RMS scaling and decoupled weight decay.
    """

    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True,
                 backend='newtonschulz5', backend_steps=5, weight_decay=0.0,
                 update_scale=0.2):
        defaults = dict(
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            backend=backend,
            backend_steps=backend_steps,
            weight_decay=weight_decay,
            update_scale=update_scale,
        )
        super().__init__(params, defaults)

    def step(self):
        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            zeropower_backend = zeropower_backends[group['backend']]
            weight_decay = group['weight_decay']
            update_scale = group['update_scale']

            total_params = sum(p.numel() for p in group['params'])
            updates_flat = torch.zeros(total_params, device='cuda', dtype=torch.bfloat16)
            curr_idx = 0
            for i, p in enumerate(group['params']):
                if i % int(os.environ['WORLD_SIZE']) == int(os.environ['RANK']):
                    g = p.grad
                    assert g is not None
                    state = self.state[p]
                    if 'momentum_buffer' not in state:
                        state['momentum_buffer'] = torch.zeros_like(g)
                    buf = state['momentum_buffer']
                    buf.mul_(momentum).add_(g)
                    if group['nesterov']:
                        g = g.add(buf, alpha=momentum)
                    g = zeropower_backend(g, steps=group['backend_steps'])
                    scale = (max(g.size(0), g.size(1)) ** 0.5) * update_scale
                    g *= scale
                    updates_flat[curr_idx:curr_idx + p.numel()] = g.flatten()
                curr_idx += p.numel()

            dist.all_reduce(updates_flat, op=dist.ReduceOp.SUM)

            curr_idx = 0
            for p in group['params']:
                if weight_decay != 0.0:
                    p.data.mul_(1.0 - lr * weight_decay)
                g = updates_flat[curr_idx:curr_idx + p.numel()].view_as(p.data).type_as(p.data)
                p.data.add_(g, alpha=-lr)
                curr_idx += p.numel()

def _collect_router_temperature_params(raw_model):
    temperature_params = []
    if not getattr(raw_model.config, "use_router_temperature", False):
        return temperature_params
    for block in raw_model.transformer.h:
        temp_param = getattr(block.mlp, 'router_temperature_log', None)
        if isinstance(temp_param, nn.Parameter):
            temperature_params.append(temp_param)
    return temperature_params


def _collect_router_params(raw_model):
    router_params = []
    for block in raw_model.transformer.h:
        router_module = getattr(block.mlp, 'router', None)
        if router_module is not None:
            router_params.extend(list(router_module.parameters()))
    return router_params


def _collect_theta_params(raw_model):
    theta_params = []
    for block in raw_model.transformer.h:
        theta_param = getattr(block.mlp, 'load_balance_theta', None)
        if isinstance(theta_param, nn.Parameter):
            theta_params.append(theta_param)
    return theta_params


def get_optimizers(raw_model, args):
    """
    Builds and returns optimizer instances for the current model configuration.

    Returns:
        optimizers: List[torch.optim.Optimizer]
        router_optimizer: Optional[torch.optim.Optimizer]
        router_temperature_optimizer: Optional[torch.optim.Optimizer]
    """
    all_h_params = list(raw_model.transformer.h.parameters())
    adamw_betas = tuple(args.adamw_betas)
    adamw_fused = args.adamw_fused

    optimizer1 = torch.optim.AdamW(
        [raw_model.transformer.wte.weight],
        lr=args.lr_embed,
        betas=adamw_betas,
        # weight_decay=args.weight_decay,
        weight_decay=0,
        fused=adamw_fused,
    )
    optimizer2 = torch.optim.AdamW(
        [raw_model.lm_head.weight],
        lr=args.lr_head,
        betas=adamw_betas,
        # weight_decay=args.weight_decay,
        weight_decay=0,
        fused=adamw_fused,
    )

    router_optimizer = None
    router_temperature_optimizer = None

    router_temperature_params = _collect_router_temperature_params(raw_model)
    if router_temperature_params:
        temp_param_ids = {id(p) for p in router_temperature_params}
        all_h_params = [p for p in all_h_params if id(p) not in temp_param_ids]
    theta_params = _collect_theta_params(raw_model)
    if theta_params:
        theta_param_ids = {id(p) for p in theta_params}
        all_h_params = [p for p in all_h_params if id(p) not in theta_param_ids]

    muon_cls = MuonClip if getattr(args, "use_muon_clip", False) else Muon
    muon_kwargs = dict(
        lr=args.lr_muon,
        momentum=args.momentum,
        backend=args.muon_svd_backend,
        nesterov=args.muon_nesterov,
        backend_steps=args.muon_backend_steps,
        weight_decay=args.weight_decay
    )
    if muon_cls is MuonClip:
        muon_kwargs["update_scale"] = args.muon_update_scale

    if args.only_router_muon:
        router_params = _collect_router_params(raw_model)
        router_optimizer = muon_cls(
            router_params,
            **muon_kwargs,
        )
        router_param_ids = {id(p) for p in router_params}
        blocks_params = [p for p in all_h_params if id(p) not in router_param_ids]
        optimizer3 = torch.optim.AdamW(
            blocks_params,
            lr=6e-4,
            betas=adamw_betas,
            weight_decay=args.weight_decay,
            fused=adamw_fused,
        )
    elif args.use_adamw_opt3:
        optimizer3 = torch.optim.AdamW(
            all_h_params,
            lr=6e-4,
            betas=adamw_betas,
            weight_decay=args.weight_decay,
            fused=adamw_fused,
        )
    elif args.use_adamw_router:
        router_params = _collect_router_params(raw_model)
        router_optimizer = torch.optim.AdamW(
            router_params,
            lr=args.lr_muon,
            betas=adamw_betas,
            weight_decay=args.weight_decay,
            fused=adamw_fused,
        )
        router_param_ids = {id(p) for p in router_params}
        muon_params = [p for p in all_h_params if id(p) not in router_param_ids]
        optimizer3 = muon_cls(
            muon_params,
            **muon_kwargs,
        )
    else:
        optimizer3 = muon_cls(
            all_h_params,
            **muon_kwargs,
        )

    theta_optimizer = None
    if theta_params:
        theta_optimizer = torch.optim.AdamW(
            theta_params,
            lr=args.lr_theta,
            betas=adamw_betas,
            weight_decay=0.0,
            fused=adamw_fused,
        )
    if router_temperature_params:
        router_temperature_optimizer = torch.optim.AdamW(
            router_temperature_params,
            lr=args.lr_muon,
            betas=adamw_betas,
            weight_decay=args.weight_decay,
            fused=adamw_fused,
        )

    optimizers = [optimizer1, optimizer2, optimizer3]
    if theta_optimizer is not None:
        optimizers.append(theta_optimizer)
    if router_optimizer is not None:
        optimizers.append(router_optimizer)
    if router_temperature_optimizer is not None:
        optimizers.append(router_temperature_optimizer)

    return optimizers, router_optimizer, router_temperature_optimizer


__all__ = ["Muon", "MuonClip", "get_optimizers"]
