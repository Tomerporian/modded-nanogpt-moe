import os

import torch
import wandb


def _prepare_balance_tensor(balance_tensor: torch.Tensor) -> torch.Tensor:
    if balance_tensor.ndim == 1:
        return balance_tensor.unsqueeze(0)
    return balance_tensor


def _expected_frac(balance_tensor: torch.Tensor) -> torch.Tensor:
    return balance_tensor.new_tensor(1.0 / balance_tensor.size(-1))


def _log_per_layer_metric(per_layer_values: torch.Tensor, aggregate_key: str, per_layer_key_base: str):
    metric_cpu = per_layer_values.detach().cpu()
    log = {aggregate_key: float(metric_cpu.mean().item())}
    for li in range(metric_cpu.numel()):
        log[f'{per_layer_key_base}/Layer {li}'] = float(metric_cpu[li].item())
    return log


def maxvio_per_layer(balance_tensor: torch.Tensor) -> torch.Tensor:
    """
    Compute per-layer MaxVio (maximal violation) as defined in Eq. (4) of
    Wang et al. (2024), i.e., (max_i Load_i - Load_bar) / Load_bar.
    """
    balance_tensor = _prepare_balance_tensor(balance_tensor)
    expected_frac = _expected_frac(balance_tensor)
    per_layer_max = balance_tensor.max(dim=-1).values
    return (per_layer_max - expected_frac) / expected_frac


def minvio_per_layer(balance_tensor: torch.Tensor) -> torch.Tensor:
    balance_tensor = _prepare_balance_tensor(balance_tensor)
    expected_frac = _expected_frac(balance_tensor)
    per_layer_min = balance_tensor.min(dim=-1).values
    return (expected_frac - per_layer_min) / expected_frac


def totalvio_per_layer(balance_tensor: torch.Tensor) -> torch.Tensor:
    balance_tensor = _prepare_balance_tensor(balance_tensor)
    expected_frac = _expected_frac(balance_tensor)
    return torch.abs(balance_tensor - expected_frac).sum(dim=-1) / expected_frac


def log_max_vio(layer_expert_balance_avg, aggregate_key, per_layer_key_base):
    return _log_per_layer_metric(
        maxvio_per_layer(layer_expert_balance_avg),
        aggregate_key,
        per_layer_key_base,
    )


def log_min_vio(layer_expert_balance_avg, aggregate_key, per_layer_key_base):
    return _log_per_layer_metric(
        minvio_per_layer(layer_expert_balance_avg),
        aggregate_key,
        per_layer_key_base,
    )


def log_total_vio(layer_expert_balance_avg, aggregate_key, per_layer_key_base):
    return _log_per_layer_metric(
        totalvio_per_layer(layer_expert_balance_avg),
        aggregate_key,
        per_layer_key_base,
    )


def log_entropy(layer_router_entropy_avg):
    log = {}
    for li in range(layer_router_entropy_avg.size(0)):
        log[f'Router Entropy/Layer {li}'] = float(layer_router_entropy_avg[li].item())
    return log


def log_expert_balance(expert_balance_avg, layer_expert_balance_avg, num_experts, prefix):
    log = {}
    for i_exp in range(num_experts):
        log[f'{prefix}/expert_balance/{i_exp}'] = float(expert_balance_avg[i_exp].item())
    n_layers = layer_expert_balance_avg.size(0)
    for li in range(n_layers):
        for ei in range(num_experts):
            log[f'Expert Balance/Layer {li}/{ei}'] = float(layer_expert_balance_avg[li, ei].item())
    return log


def log_router_values(layer_router_values_avg, total_router_values_avg, router_value_keys):
    log = {}
    n_layers = next(iter(layer_router_values_avg.values())).size(0)
    for key in router_value_keys:
        if key.startswith('ste_lb_'):
            metric_name = key[len('ste_lb_'):]
            for li in range(n_layers):
                log[f'ste_lb/layer_{li}/{metric_name}'] = float(layer_router_values_avg[key][li].item())
            log[f'ste_lb/all_layers/{metric_name}'] = float(total_router_values_avg[key].item())
            continue
        if key.startswith('ste_'):
            metric_name = key[len('ste_'):]
            for li in range(n_layers):
                log[f'ste/layer_{li}/{metric_name}'] = float(layer_router_values_avg[key][li].item())
            log[f'ste/all_layers/{metric_name}'] = float(total_router_values_avg[key].item())
            continue
        for li in range(n_layers):
            log[f'router_values/layer_{li}_{key}'] = float(layer_router_values_avg[key][li].item())
        log[f'router_values/total_{key}'] = float(total_router_values_avg[key].item())
    return log


def log_attention_logits(attn_logit_max):
    log = {}
    if attn_logit_max is None:
        return log
    per_layer_max = attn_logit_max.max(dim=-1).values
    per_layer_mean = attn_logit_max.mean(dim=-1)
    for li in range(per_layer_max.numel()):
        log[f'attn_logits/max/Layer {li}'] = float(per_layer_max[li].item())
        log[f'attn_logits/mean/Layer {li}'] = float(per_layer_mean[li].item())
    log['attn_logits/max'] = float(per_layer_max.max().item())
    log['attn_logits/mean'] = float(per_layer_mean.mean().item())
    return log


def log_router_weight_norms(raw_model, prefix):
    log = {}
    layer_means = []
    layer_maxs = []
    for li, block in enumerate(raw_model.transformer.h):
        norms = []
        router = getattr(block.mlp, 'router', None)
        if router is not None:
            for param in router.parameters():
                norms.append(param.detach().float().norm(2))
        scaler = getattr(block.mlp, 'router_scaler', None)
        if scaler is not None:
            for param in scaler.parameters():
                norms.append(param.detach().float().norm(2))
        if norms:
            norms_tensor = torch.stack(norms)
            mean_val = norms_tensor.mean()
            max_val = norms_tensor.max()
            layer_means.append(mean_val)
            layer_maxs.append(max_val)
            log[f'{prefix}/mean/Layer {li}'] = float(mean_val.item())
            log[f'{prefix}/max/Layer {li}'] = float(max_val.item())
    if layer_means:
        log[f'{prefix}/mean'] = float(torch.stack(layer_means).mean().item())
        log[f'{prefix}/max'] = float(torch.stack(layer_maxs).max().item())
    return log


def log_loss_free_bias(raw_model, prefix):
    log = {}
    for li, block in enumerate(raw_model.transformer.h):
        mlp = block.mlp
        bias_vec = mlp._loss_free_bias_vector()
        if bias_vec is None:
            continue
        bias_vals = bias_vec.detach().float().cpu()
        for ei in range(mlp.num_experts):
            log[f'{prefix}_loss_free_bias/Layer {li}/expert_{ei}'] = float(bias_vals[ei].item())
    return log

def log_theta_stats(raw_model, prefix):
    log = {}
    if getattr(raw_model.config, "theta_load_balance_coeff", 0) == 0:
        return log
    theta_mins = []
    theta_maxs = []
    for li, block in enumerate(raw_model.transformer.h):
        theta_param = getattr(block.mlp, 'load_balance_theta', None)
        if isinstance(theta_param, torch.nn.Parameter):
            theta_detached = theta_param.detach().float()
            theta_mins.append(theta_detached.min())
            theta_maxs.append(theta_detached.max())
            log[f'{prefix}/theta/min/Layer {li}'] = float(theta_detached.min().item())
            log[f'{prefix}/theta/max/Layer {li}'] = float(theta_detached.max().item())
    if theta_mins:
        log[f'{prefix}/theta/min'] = float(torch.stack(theta_mins).min().item())
        log[f'{prefix}/theta/max'] = float(torch.stack(theta_maxs).max().item())
    return log

def build_router_grad_norm_log(n_layers, ce_grad_norms, aux_grad_norms):
    log = {}
    for li in range(n_layers):
        log[f'Router Grad Norms (CE)/Layer {li}'] = float(ce_grad_norms[li].item())
        log[f'Router Grad Norms (AUX)/Layer {li}'] = float(aux_grad_norms[li].item())
    return log


def build_track_tokens_log(n_layers, topk_change_percentages, any_topk_changed_percentages):
    log = {}
    for li in range(n_layers):
        for k, percentages in topk_change_percentages.items():
            log[f'track_tokens/layer_{li}/top{k}_change'] = float(percentages[li])
        log[f'track_tokens/layer_{li}/chosen_changed'] = float(any_topk_changed_percentages[li])
    return log


def wandb_train_log(
    log_step,
    train_loss,
    train_diff_topk_reg,
    train_theta_lb_loss,
    router_entropy_avg,
    grad_norm,
    approx_time,
    timed_steps,
    optimizers,
    router_optimizer,
    layer_expert_balance_avg,
    layer_router_entropy_avg,
    expert_balance_avg,
    layer_router_values_avg,
    total_router_values_avg,
    raw_model,
    num_experts,
    router_value_keys,
    diff_topk_reg_coeff,
    attn_logit_max,
):
    log = {
        'train/loss': float(train_loss.item()),
        'train/diff_topk_reg': float(train_diff_topk_reg.item()),
        'train/theta_lb_loss': float(train_theta_lb_loss.item()),
        'train/router_entropy': float(router_entropy_avg.item()),
        'train/grad_norm': float(grad_norm.item()),
        'train/step_time_ms': float(approx_time),
        'train/step_avg_ms': float(approx_time / timed_steps),
        'lr/embed': float(optimizers[0].param_groups[0]['lr']),
        'lr/head': float(optimizers[1].param_groups[0]['lr']),
        'lr/blocks': float(optimizers[2].param_groups[0]['lr']),
        'diff_topk_reg/coeff': float(diff_topk_reg_coeff),
    }
    if router_optimizer is not None:
        log['lr/router'] = float(router_optimizer.param_groups[0]['lr'])
    log.update(log_max_vio(layer_expert_balance_avg, 'train/MaxViobatch', 'train/MaxViobatch'))
    log.update(log_min_vio(layer_expert_balance_avg, 'train/MinViobatch', 'train/MinViobatch'))
    log.update(log_total_vio(layer_expert_balance_avg, 'train/TotalViobatch', 'train/TotalViobatch'))
    log.update(log_entropy(layer_router_entropy_avg))
    log.update(log_expert_balance(expert_balance_avg, layer_expert_balance_avg, num_experts, 'train'))
    log.update(log_router_values(layer_router_values_avg, total_router_values_avg, router_value_keys))
    log.update(log_router_weight_norms(raw_model, 'router_weight_norm'))
    log.update(log_loss_free_bias(raw_model, 'train'))
    log.update(log_theta_stats(raw_model, 'thetas'))
    log.update(_collect_router_temperatures(raw_model))
    log.update(log_attention_logits(attn_logit_max))
    wandb.log(log, step=log_step)


def wandb_val_log(
    log_step,
    val_loss,
    val_ce_loss,
    val_aux_loss,
    val_theta_lb_loss,
    val_diff_topk_reg,
    val_router_entropy,
    training_time_ms,
    timed_steps,
    val_layer_expert_balance,
    val_layer_router_entropy,
    val_expert_balance,
    num_experts,
    val_layer_router_values,
    val_total_router_values,
    raw_model,
    ce_router_layer_grad_norms,
    aux_router_layer_grad_norms,
    topk_change_percentages,
    any_topk_changed_percentages,
    router_value_keys,
    diff_weight,
):
    def _to_float(value):
        return float(value.item() if isinstance(value, torch.Tensor) else value)

    log = {}
    n_layers = raw_model.config.n_layer
    log.update(build_router_grad_norm_log(
        n_layers,
        ce_router_layer_grad_norms,
        aux_router_layer_grad_norms,
    ))
    log.update(build_track_tokens_log(
        n_layers,
        topk_change_percentages,
        any_topk_changed_percentages,
    ))
    log.update({
        'val/loss': _to_float(val_loss),
        'val/ce_loss': _to_float(val_ce_loss),
        'val/aux_loss': _to_float(val_aux_loss),
        'val/theta_lb_loss': _to_float(val_theta_lb_loss),
        'val/diff_topk_reg': _to_float(val_diff_topk_reg),
        'val/router_entropy': float(val_router_entropy.item()),
        'train/time_ms': float(training_time_ms),
        'train/step_avg_ms': float(training_time_ms / max(timed_steps - 1, 1)),
        'transition/diff_weight': diff_weight,
    })
    log.update(log_max_vio(val_layer_expert_balance, 'val/MaxVioglobal', 'val/MaxVioglobal'))
    log.update(log_min_vio(val_layer_expert_balance, 'val/MinVioglobal', 'val/MinVioglobal'))
    log.update(log_total_vio(val_layer_expert_balance, 'val/TotalVioglobal', 'val/TotalVioglobal'))
    log.update(log_entropy(val_layer_router_entropy))
    log.update(log_expert_balance(val_expert_balance, val_layer_expert_balance, num_experts, 'val'))
    log.update(log_router_values(val_layer_router_values, val_total_router_values, router_value_keys))
    log.update(log_loss_free_bias(raw_model, 'val'))
    wandb.log(log, step=log_step)


def init_wandb(args,
               optimizer3,
               train_accumulation_steps,
               val_steps,
               ddp_world_size,
               ddp_rank,
               ddp_local_rank):
    config = vars(args).copy()
    blocks_group = optimizer3.param_groups[0]
    config.update({
        'train_accumulation_steps': train_accumulation_steps,
        'val_steps': val_steps,
        'ddp_world_size': ddp_world_size,
        'ddp_rank': ddp_rank,
        'ddp_local_rank': ddp_local_rank,
        'optimizer_blocks_lr': blocks_group['lr'],
        'amp_dtype': 'bfloat16',
        'torch_compile': True,
        'attention_backend': 'cudnn_sdp',
    })

    run_name = os.path.basename(args.output)
    wandb.init(project=args.wandb_project,
               name=run_name,
               config=config)


def _collect_router_temperatures(raw_model):
    log = {}
    if not raw_model.config.use_router_temperature:
        return log
    for li, block in enumerate(raw_model.transformer.h):
        temp_param = getattr(block.mlp, 'router_temperature_log', None)
        if temp_param is None:
            continue
        temp_value = torch.exp(temp_param.detach().float().cpu()).item()
        log[f'router_temperature/Layer {li}'] = float(temp_value)
    return log
