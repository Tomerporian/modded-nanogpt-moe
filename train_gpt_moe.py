import os
import sys
with open(sys.argv[0]) as f:
    code = f.read() # read the code of this file ASAP, for logging
import uuid
import glob
import time
from dataclasses import dataclass
import math
import gc
import argparse
import yaml
import logging
import warnings
import random
import re

import numpy as np
import torch

# Suppress specific PyTorch Inductor warnings
warnings.filterwarnings("ignore", message="Online softmax is disabled on the fly since Inductor decides to split the reduction")
warnings.filterwarnings("ignore", category=UserWarning, module="torch._inductor.lowering")
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist
import torch._inductor.config as config
import torch._dynamo as dynamo
from torch.nn.parallel import DistributedDataParallel as DDP
import wandb

# Import Megatron dataloader for indexed datasets
from megatron_indexed_dataset import MegatronDataLoader

# -----------------------------------------------------------------------------
# Muon optimizer

def zeropower_via_svd(G, steps=None):
    U, S, V = G.svd()
    return U @ V.T

@torch.compile
def zeropower_via_newtonschulz5(G, steps=10, eps=1e-7):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' \\ sim Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750,  2.0315)
    X = G.bfloat16()
    X /= (X.norm() + eps) # ensure top singular value <= 1
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
    Muon - MomentUm Orthogonalized by Newton-schulz

    Muon internally runs standard SGD-momentum, and then performs an orthogonalization post-
    processing step, in which each 2D parameter's update is replaced with the nearest orthogonal
    matrix. To efficiently orthogonalize each update, we use a Newton-Schulz iteration, which has
    the advantage that it can be stably run in bfloat16 on the GPU.

    Some warnings:
    - This optimizer assumes that all parameters passed in are 2D.
    - It should not be used for the embedding layer, the final fully connected layer, or any {0,1}-D
    parameters; those should all be optimized by a standard method (e.g., AdamW).
    - To use it with 4D convolutional filters, it works well to just flatten their last 3 dimensions.
    - We believe it is unlikely to work well for training with small batch size.
    - We believe it may not work well for finetuning pretrained models, but we haven't tested this.
    - We have not yet tried this optimizer for training scenarios larger than NanoGPT (124M).

    Arguments:
        lr: The learning rate used by the internal SGD.
        momentum: The momentum used by the internal SGD.
        nesterov: Whether to use Nesterov-style momentum in the internal SGD. (recommended)
        backend: The chosen backend for the orthogonalization step. (recommended: 'newtonschulz5')
        backend_steps: The number of iteration steps to use in the backend, if it is iterative.
    """
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True,
                 backend='newtonschulz5', backend_steps=5):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, backend=backend, backend_steps=backend_steps)
        super().__init__(params, defaults)

    def step(self):

        for group in self.param_groups:

            lr = group['lr']
            momentum = group['momentum']
            zeropower_backend = zeropower_backends[group['backend']]

            # generate weight updates in distributed fashion
            total_params = sum(p.numel() for p in group['params'])
            updates_flat = torch.zeros(total_params, device='cuda', dtype=torch.bfloat16)
            curr_idx = 0
            for i, p in enumerate(group['params']):
                # luckily this will perfectly distribute a transformer with multiple of 4 layers to 8 GPUs
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
                    g *= max(1, g.size(0)/g.size(1))**0.5
                    updates_flat[curr_idx:curr_idx+p.numel()] = g.flatten()
                curr_idx += p.numel()

            # sync updates across devices. we are not memory-constrained so can do this simple deserialization
            dist.all_reduce(updates_flat, op=dist.ReduceOp.SUM)

            # deserialize and apply updates
            curr_idx = 0
            for p in group['params']:
                g = updates_flat[curr_idx:curr_idx+p.numel()].view_as(p.data).type_as(p.data)
                p.data.add_(g, alpha=-lr)
                curr_idx += p.numel()

# -----------------------------------------------------------------------------
# PyTorch nn.Module definitions for the GPT-2 model

class Rotary(torch.nn.Module):

    def __init__(self, dim, base=10000):
        super().__init__()
        self.inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, x):
        seq_len = x.shape[1]
        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
            freqs = torch.outer(t, self.inv_freq).to(x.device)
            self.cos_cached = freqs.cos().bfloat16()
            self.sin_cached = freqs.sin().bfloat16()
        return self.cos_cached[None, :, None, :], self.sin_cached[None, :, None, :]

def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4 # multihead attention
    d = x.shape[3]//2
    x1 = x[..., :d]
    x2 = x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3).type_as(x)

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        self.c_q = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_embd, bias=False)
        # output projection
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.c_proj.weight.data.zero_() # zero init suggested by @Grad62304977
        self.rotary = Rotary(self.head_dim)

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_head, self.head_dim)
        cos, sin = self.rotary(q)
        q, k = F.rms_norm(q, (q.size(-1),)), F.rms_norm(k, (k.size(-1),)) # QK norm suggested by @Grad62304977
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        y = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True)
        y = y.transpose(1, 2).contiguous().view_as(x) # re-assemble all head outputs side by side
        y = self.c_proj(y)
        return y

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)
        self.c_proj.weight.data.zero_() # zero init suggested by @Grad62304977

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square() # https://arxiv.org/abs/2109.08668v2; ~1-2% better than GELU; suggested by @SKYLINEZ007 and @Grad62304977
        x = self.c_proj(x)
        return x

def switch_topk(logits, k, null_expert_bias=0.0, weight_logits=None, probs_override=None):
    """Switch/Top‑k. Returns (indices, probs, expert output weights)."""
    probs_from_logits = logits.softmax(dim=-1)
    probs = probs_override if probs_override is not None else probs_from_logits
    gate_vals, topk_idx = torch.topk(probs_from_logits, k, dim=-1)
    if weight_logits is None:
        gate = gate_vals / (gate_vals.sum(dim=-1, keepdim=True) + null_expert_bias)
    else:
        topk_logits = torch.gather(weight_logits, dim=-1, index=topk_idx)
        gate = F.softmax(topk_logits, dim=-1)
    return topk_idx, probs, gate


def hash_select(token_ids, num_experts, null_expert_bias=0.0):
    expert_idx = (token_ids[..., None].float() % num_experts).to(token_ids.dtype)
    routing_weights = torch.nn.functional.one_hot(expert_idx, num_classes=num_experts)
    selected_prob, selected_expert = torch.max(routing_weights, dim=-1, keepdim=True)
    expert_mask = torch.nn.functional.one_hot(torch.argmax(routing_weights, dim=-1), num_classes=num_experts)
    if null_expert_bias > 0:
        selected_prob = selected_prob / (selected_prob + null_expert_bias)
    return selected_expert, routing_weights, selected_prob

def diff_routing(logits, k):
    probs = logits.softmax(dim=-1)
    z_max = torch.max(logits, dim=-1, keepdim=True).values
    exp_z = torch.exp(logits - z_max)
    num_experts = exp_z.shape[-1]
    kth_smallest_idx = num_experts - k
    m_exp_z = torch.kthvalue(exp_z, k=kth_smallest_idx, dim=-1, keepdim=True).values
    topk_exp_z = F.relu(exp_z - m_exp_z)
    topk_weights = topk_exp_z / (topk_exp_z.sum(dim=-1, keepdim=True) + 1e-8)
    # Get indices of top-k for compatibility
    gate, topk_idx = torch.topk(topk_weights, k, dim=-1)
    return topk_idx, probs, gate


def maxvio_per_layer(balance_tensor: torch.Tensor) -> torch.Tensor:
    """
    Compute per-layer MaxVio (maximal violation) as defined in Eq. (4) of
    Wang et al. (2024), i.e., (max_i Load_i - Load_bar) / Load_bar.
    """
    if balance_tensor.ndim == 1:
        balance_tensor = balance_tensor.unsqueeze(0)
    expected_frac = balance_tensor.new_tensor(1.0 / balance_tensor.size(-1))
    per_layer_max = balance_tensor.max(dim=-1).values
    return (per_layer_max - expected_frac) / expected_frac


class ReLUSquared(nn.Module):
    def forward(self, x):
        return torch.relu(x).square()

class MoE(nn.Module):
    def __init__(
        self,
        config,
        num_experts=8,
        top_k=2,
        router_type='diff',
        router_depth=1,
        router_activation='gelu',
        global_load_balance=False,
        layer_idx=0,
        loss_free_mode='none',
        loss_free_decay=0.99,
        loss_free_strength=1.0,
        loss_free_update_rate=0.001,
        loss_free_bias_rule='ema',
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k       = top_k
        assert 1 <= self.top_k <= self.num_experts, "`k` must be in [1, #experts]"
        self.router_type  = router_type
        self.router_depth = router_depth
        self.router_activation = router_activation
        self.global_load_balance = global_load_balance
        self.layer_idx = layer_idx
        self.loss_free_mode = loss_free_mode
        self.loss_free_decay = loss_free_decay
        self.loss_free_strength = loss_free_strength
        self.loss_free_update_rate = loss_free_update_rate
        self.loss_free_bias_rule = loss_free_bias_rule

        assert self.router_type in ('hash', 'switch', 'diff')
        assert self.router_activation in ('gelu', 'relu', 'relu_squared')
        assert self.loss_free_mode in ('none', 'deepseek', 'stopgrad')
        assert self.loss_free_bias_rule in ('ema', 'sign')
        if self.router_type == 'hash' and self.loss_free_mode != 'none':
            raise ValueError("Loss-free load balancing requires a learned router (switch or diff).")
        if self.loss_free_mode == 'deepseek' and self.router_type != 'switch':
            raise ValueError("DeepSeek-style loss-free routing is only defined for switch routers.")

        self.experts = nn.ModuleList([MLP(config) for _ in range(self.num_experts)])
        init_frac = torch.full((self.num_experts,), 1.0 / self.num_experts, dtype=torch.float32)
        self.register_buffer('loss_free_ema', init_frac, persistent=True)
        self.register_buffer('loss_free_bias_state', torch.zeros(self.num_experts, dtype=torch.float32), persistent=True)
        self.register_buffer('loss_free_tokens_accum', torch.zeros(self.num_experts, dtype=torch.float32), persistent=False)
        self.register_buffer('loss_free_total_accum', torch.tensor(0.0, dtype=torch.float32), persistent=False)
        self.loss_free_override_frac = None
        
        if self.router_type != 'hash':
            layers = []
            for _ in range(router_depth - 1):
                layers.append(nn.Linear(config.n_embd, config.n_embd, bias=False))
                layers.append(self._build_router_activation())
            layers.append(nn.Linear(config.n_embd, self.num_experts, bias=False))
            self.router = nn.Sequential(*layers)
        else:
            self.router = None

    def _build_router_activation(self):
        if self.router_activation == 'gelu':
            return nn.GELU()
        elif self.router_activation == 'relu':
            return nn.ReLU()
        elif self.router_activation == 'relu_squared':
            return ReLUSquared()
        else:
            raise ValueError(f"Unsupported router activation: {self.router_activation}")

    @property
    def loss_free_enabled(self):
        return self.loss_free_mode != 'none'

    def _loss_free_bias_vector(self):
        if not self.loss_free_enabled:
            return None
        if self.loss_free_bias_rule == 'sign':
            bias_vec = self.loss_free_bias_state * self.loss_free_strength
        else:
            bias_vec = (1.0 / self.num_experts - self.loss_free_ema) * self.loss_free_strength
        return bias_vec - bias_vec.mean()

    def _loss_free_bias(self, logits):
        bias_vec = self._loss_free_bias_vector()
        if bias_vec is None:
            return None
        bias_vec = bias_vec.to(dtype=logits.dtype, device=logits.device)
        view_shape = [1] * (logits.dim() - 1) + [self.num_experts]
        return bias_vec.view(*view_shape)

    def _update_loss_free_state(self, frac_tensor):
        if not self.loss_free_enabled or not self.training:
            return
        with torch.no_grad():
            decay = self.loss_free_decay
            update = frac_tensor.detach().to(self.loss_free_ema.dtype)
            self.loss_free_ema.mul_(decay).add_(update * (1.0 - decay))

    def _update_sign_bias(self, frac_tensor):
        if not self.loss_free_enabled or not self.training:
            return
        if frac_tensor is None:
            return
        with torch.no_grad():
            target = 1.0 / self.num_experts
            update = frac_tensor.detach().to(dtype=self.loss_free_bias_state.dtype, device=self.loss_free_bias_state.device)
            violation = update - target
            step = torch.sign(violation) * self.loss_free_update_rate
            self.loss_free_bias_state.add_(step)
            self.loss_free_bias_state.add_(-self.loss_free_bias_state.mean())

    def _accumulate_loss_free_tokens(self, tokens_per_expert):
        if not self.loss_free_enabled or not self.training:
            return
        with torch.no_grad():
            self.loss_free_tokens_accum.add_(tokens_per_expert.to(self.loss_free_tokens_accum.dtype))
            self.loss_free_total_accum.add_(tokens_per_expert.sum().to(self.loss_free_total_accum.dtype))

    def _set_loss_free_override(self, frac_tensor):
        if not self.loss_free_enabled or not self.training:
            return
        self.loss_free_override_frac = frac_tensor.detach().to(self.loss_free_ema.dtype, device=self.loss_free_ema.device).clone()

    def _reset_loss_free_accumulators(self):
        self.loss_free_tokens_accum.zero_()
        self.loss_free_total_accum.zero_()
        self.loss_free_override_frac = None

    def finalize_loss_free_update(self):
        if not self.loss_free_enabled:
            self._reset_loss_free_accumulators()
            return
        frac = None
        if self.loss_free_override_frac is not None:
            frac = self.loss_free_override_frac
        else:
            tokens = self.loss_free_tokens_accum.clone()
            total = self.loss_free_total_accum.clone()
            if dist.is_available() and dist.is_initialized():
                dist.all_reduce(tokens, op=dist.ReduceOp.SUM)
                dist.all_reduce(total, op=dist.ReduceOp.SUM)
            total_val = float(total.item())
            if total_val > 0:
                frac = tokens / total_val
        if frac is not None:
            if self.loss_free_bias_rule == 'sign':
                self._update_sign_bias(frac)
            elif self.loss_free_bias_rule == 'ema':
                self._update_loss_free_state(frac)
        self._reset_loss_free_accumulators()

    def forward(self, x, token_idx=None, return_expert_assignments=False, router_context=None):

        ctx_mode = router_context.get('mode', None) if router_context is not None else None
        logits = None
        logits_for_selection = None
        logits_for_weights = None
        logits_for_stats = None
        if self.router_type in ("switch", "diff"):
            logits = self.router(x)
            bias = self._loss_free_bias(logits)
            if self.loss_free_mode == 'deepseek' and bias is not None:
                logits_for_selection = logits + bias
                logits_for_weights = logits
            elif self.loss_free_mode == 'stopgrad' and bias is not None:
                bias_detached = bias.detach()
                logits_for_selection = logits + bias_detached
                logits_for_weights = logits_for_selection
            else:
                logits_for_selection = logits
                logits_for_weights = logits
            logits_for_stats = logits_for_weights
        if self.router_type == "switch":
            router_probs = logits_for_stats.softmax(dim=-1)
            topk_idx, _, gate = switch_topk(
                logits_for_selection,
                self.top_k,
                weight_logits=logits_for_weights,
                probs_override=router_probs,
            )
            probs = router_probs
        elif self.router_type == "diff":
            topk_idx, _, gate = diff_routing(logits_for_selection, self.top_k)
            probs = logits_for_weights.softmax(dim=-1)
        elif self.router_type == "hash":
            topk_idx, probs, gate = hash_select(token_idx, self.num_experts)
            gate = gate.to(x.dtype)
            probs = probs.to(x.dtype)
        else:
            raise ValueError(f"unknown routing type: {self.router_type}")
        B, T, C = x.shape
        BT      = B * T
        x_flat  = x.reshape(BT, C)
        y_flat  = torch.zeros_like(x_flat)

        probs_flat = probs.reshape(BT, -1)
        gate_flat = gate.reshape(BT, self.top_k)  # Now always (BT, k)
        idx_flat = topk_idx.reshape(BT, self.top_k)        
        for expert_id in range(self.num_experts):
            sel_mask = (idx_flat == expert_id)
            token_rows, which_k = torch.nonzero(sel_mask, as_tuple=True)
            inp   = x_flat.index_select(0, token_rows)
            out   = self.experts[expert_id](inp)
            coeff = gate_flat[token_rows, which_k].unsqueeze(1)
            y_flat.index_add_(0, token_rows, out * coeff)

        y = y_flat.view_as(x)

        # aux loss and router statistics
        with torch.autocast(device_type="cpu", enabled=False):
            tokens_per_expert = torch.bincount(
                idx_flat.flatten(), minlength=self.num_experts
            ).float()
            local_total_tokens = tokens_per_expert.sum()
            denom_local = torch.clamp(local_total_tokens, min=1.0)

            frac = tokens_per_expert / denom_local
            global_frac_override = None

            if self.global_load_balance and router_context is not None:
                if ctx_mode == 'collect':
                    router_context['tokens_accum'][self.layer_idx] += tokens_per_expert
                    router_context['totals_accum'][self.layer_idx] += local_total_tokens
                elif ctx_mode == 'use':
                    global_frac_tensor = router_context.get('global_frac', None)
                    if global_frac_tensor is not None:
                        global_frac_override = global_frac_tensor[self.layer_idx]

            if self.loss_free_enabled and self.training and ctx_mode != 'collect':
                if global_frac_override is not None:
                    self._set_loss_free_override(global_frac_override)
                else:
                    self._accumulate_loss_free_tokens(tokens_per_expert)

            # token-wise entropy of router distribution (normalized to [0,1])
            eps = 1e-9
            with torch.no_grad():
                token_H = -(probs_flat * (probs_flat + eps).log()).sum(-1)
                router_entropy = token_H.mean() / math.log(float(self.num_experts))
            # Switch paper:  L_aux = E * <load,prob>
            if self.router_type != "hash":
                probs_full = logits_for_stats.softmax(dim=-1).reshape(-1, self.num_experts)
                frac_for_aux = global_frac_override if global_frac_override is not None else frac
                probs_mean = probs_full.mean(0)
                aux = self.num_experts * (frac_for_aux * probs_mean).sum()
            elif self.router_type == "hash":
                aux = torch.tensor(0.0, device=x.device, requires_grad=self.training)
            else:
                raise ValueError(f"unknown routing type: {self.router_type}")

        if return_expert_assignments:
            return y, aux, router_entropy, frac, topk_idx.view_as(x[:, :, :self.top_k])
        else:
            return y, aux, router_entropy, frac


class Block(nn.Module):

    def __init__(
        self,
        config,
        num_experts=8,
        top_k=2,
        router_type='diff',
        router_depth=1,
        router_activation='gelu',
        global_load_balance=False,
        layer_idx=0,
        loss_free_mode='none',
        loss_free_decay=0.99,
        loss_free_strength=1.0,
        loss_free_update_rate=0.001,
        loss_free_bias_rule='ema',
    ):
        super().__init__()
        self.attn = CausalSelfAttention(config)
        self.mlp = MoE(
            config,
            num_experts=num_experts,
            top_k=top_k,
            router_type=router_type,
            router_depth=router_depth,
            router_activation=router_activation,
            global_load_balance=global_load_balance,
            layer_idx=layer_idx,
            loss_free_mode=loss_free_mode,
            loss_free_decay=loss_free_decay,
            loss_free_strength=loss_free_strength,
            loss_free_update_rate=loss_free_update_rate,
            loss_free_bias_rule=loss_free_bias_rule,
        )

    def forward(self, x, token_idx=None, return_expert_assignments=False, router_context=None):
        x = x + self.attn(F.rms_norm(x, (x.size(-1),)))
        if return_expert_assignments:
            mlp_out, aux, router_entropy, expert_balance, expert_assignments = self.mlp(F.rms_norm(x, (x.size(-1),)), token_idx, return_expert_assignments=True, router_context=router_context)
            x = x + mlp_out
            return x, aux, router_entropy, expert_balance, expert_assignments
        else:
            mlp_out, aux, router_entropy, expert_balance = self.mlp(F.rms_norm(x, (x.size(-1),)), token_idx, router_context=router_context)
            x = x + mlp_out
            return x, aux, router_entropy, expert_balance

# -----------------------------------------------------------------------------
# The main GPT-2 model

@dataclass
class GPTConfig:
    vocab_size : int = 50304
    n_layer : int = 12
    n_head : int = 6 # head dim 128 suggested by @Grad62304977
    n_embd : int = 768
    num_experts : int = 8
    top_k : int = 2
    router_type : str = 'diff'
    router_depth : int = 1
    router_activation : str = 'gelu'
    global_load_balance : bool = False
    loss_free_mode : str = 'none'
    loss_free_decay : float = 0.99
    loss_free_strength : float = 1.0
    loss_free_update_rate : float = 0.001
    loss_free_bias_rule : str = 'ema'

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            h = nn.ModuleList([
                Block(
                    config,
                    num_experts=config.num_experts,
                    top_k=config.top_k,
                    router_type=config.router_type,
                    router_depth=config.router_depth,
                    router_activation=config.router_activation,
                    global_load_balance=config.global_load_balance,
                    layer_idx=layer_idx,
                    loss_free_mode=config.loss_free_mode,
                    loss_free_decay=config.loss_free_decay,
                    loss_free_strength=config.loss_free_strength,
                    loss_free_update_rate=config.loss_free_update_rate,
                    loss_free_bias_rule=config.loss_free_bias_rule,
                )
                for layer_idx in range(config.n_layer)
            ]),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.weight.data.zero_()

    def forward(self, idx, targets=None, return_logits=True, aux_coeff=0.0, return_expert_assignments=False, router_context=None):

        # forward the GPT model itself
        x = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        x = F.rms_norm(x, (x.size(-1),))
        total_aux = 0
        total_router_entropy = 0
        total_expert_balance = None
        per_layer_router_entropy = []
        per_layer_expert_balance = []
        all_layer_expert_assignments = []
        for block in self.transformer.h:
            if return_expert_assignments:
                x, aux, router_entropy, expert_balance, expert_assignments = block(x, idx, return_expert_assignments=True, router_context=router_context)
                all_layer_expert_assignments.append(expert_assignments)
            else:
                x, aux, router_entropy, expert_balance = block(x, idx, router_context=router_context)
            total_aux = total_aux + aux
            total_router_entropy = total_router_entropy + router_entropy
            if total_expert_balance is None:
                total_expert_balance = expert_balance
            else:
                total_expert_balance = total_expert_balance + expert_balance
            per_layer_router_entropy.append(router_entropy)
            per_layer_expert_balance.append(expert_balance)
        x = F.rms_norm(x, (x.size(-1),))

        # average stats across blocks
        num_blocks = len(self.transformer.h)
        avg_router_entropy = total_router_entropy / num_blocks
        avg_expert_balance = total_expert_balance / num_blocks
        layer_router_entropy = torch.stack(per_layer_router_entropy)
        layer_expert_balance = torch.stack(per_layer_expert_balance, dim=0)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            logits = logits.float() # use tf32/fp32 for logits
            ce_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
            loss = ce_loss + total_aux * aux_coeff
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            logits = logits.float() # use tf32/fp32 for logits
            loss = None
            ce_loss = None

        # there are performance reasons why not returning logits is prudent, if not needed
        if not return_logits:
            logits = None

        if return_expert_assignments:
            # Stack assignments: shape (n_layers, batch, seq_len, top_k)
            stacked_assignments = torch.stack(all_layer_expert_assignments, dim=0)
            return logits, loss, ce_loss, total_aux, avg_router_entropy, avg_expert_balance, layer_router_entropy, layer_expert_balance, stacked_assignments
        else:
            return logits, loss, ce_loss, total_aux, avg_router_entropy, avg_expert_balance, layer_router_entropy, layer_expert_balance

    def finalize_loss_free_updates(self):
        for block in self.transformer.h:
            block.mlp.finalize_loss_free_update()

# -----------------------------------------------------------------------------
# Our own simple Distributed Data Loader

def _peek_data_shard(filename):
    # only reads the header, returns header data
    with open(filename, "rb") as f:
        # first read the header, which is 256 int32 integers (4 bytes each)
        header = np.frombuffer(f.read(256*4), dtype=np.int32)
    if header[0] != 20240520:
        logging.info("ERROR: magic number mismatch in the data .bin file!")
        logging.info("---> HINT: Are you passing in a correct file with --input_bin?")
        logging.info("---> HINT: Dataset encoding changed recently, re-run data prepro or refer again to README")
        logging.info("---> HINT: For example re-run: `python dev/data/tinyshakespeare.py`, then re-try")
        exit(1)
    assert header[1] == 1, "unsupported version"
    ntok = header[2] # number of tokens (claimed)
    return ntok # for now just return the number of tokens

def _load_data_shard(filename):
    with open(filename, "rb") as f:
        # first read the header, which is 256 int32 integers (4 bytes each)
        header = np.frombuffer(f.read(256*4), dtype=np.int32)
        assert header[0] == 20240520, "magic number mismatch in the data .bin file"
        assert header[1] == 1, "unsupported version"
        ntok = header[2] # number of tokens (claimed)
        # the rest of it are tokens, stored as uint16
        tokens = np.frombuffer(f.read(), dtype=np.uint16)
    assert len(tokens) == ntok, "number of tokens read does not match header?"
    return tokens

class DistributedDataLoader:
    def __init__(self, filename_pattern, B, T, process_rank, num_processes):
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.B = B
        self.T = T

        # glob files that match the pattern
        self.files = sorted(glob.glob(filename_pattern))
        assert len(self.files) > 0, f"did not find any files that match the pattern {filename_pattern}"

        # determine file format based on path
        if 'fineweb10B' in filename_pattern:
            self.file_format = 'fineweb'
            self.header_size = 256 * 4  # 256 int32 values
            self.dtype = np.uint16
        elif 'tokenized_owt' in filename_pattern:
            self.file_format = 'openwebtext'
            self.header_size = 0  # no header
            self.dtype = np.uint16
        elif 'tokenized_c4' in filename_pattern:
            self.file_format = 'c4'
            self.header_size = 0  # no header
            self.dtype = np.uint8
        else:
            raise ValueError(f"Unknown dataset format for pattern: {filename_pattern}")

        # validate all data shards and get lengths
        self.shard_lengths = []
        ntok_total = 0
        
        for fname in self.files:
            if self.file_format == 'fineweb':
                # peek to get the number of tokens and validate format
                shard_ntok = _peek_data_shard(fname)
            else:  # openwebtext or c4
                # calculate tokens from file size
                import os
                file_size = os.path.getsize(fname)
                # token size depends on dtype
                token_size = np.dtype(self.dtype).itemsize
                shard_ntok = file_size // token_size

            self.shard_lengths.append(shard_ntok)
            ntok_total += int(shard_ntok)
        
        self.ntok_total = ntok_total

        # create cumulative lengths for sampling across shards (use int64 to avoid overflow)
        # subtract T from each shard length to ensure we never sample too close to the end
        self.cumulative_lengths = []
        cumsum = 0
        for length in self.shard_lengths:
            # Ensure we have at least T+1 tokens available for sampling
            effective_length = max(0, int(length) - self.T)
            cumsum += effective_length
            self.cumulative_lengths.append(cumsum)

        # Update total tokens to reflect the effective sampling space
        self.ntok_total = cumsum

    def next_batch(self):
        B = self.B
        T = self.T
        
        # Sample all random positions at once (like Diff_topK_nanoMoE)
        random_positions = torch.randint(0, self.ntok_total - T, (B,))
        
        # Map positions to (shard_idx, pos_in_shard) for each sequence
        shard_info = []
        for pos in random_positions:
            pos = pos.item()
            # Find which shard contains this position
            shard_idx = 0
            for i, cum_len in enumerate(self.cumulative_lengths):
                if pos < cum_len:
                    shard_idx = i
                    break
            
            # Calculate position within the shard
            if shard_idx == 0:
                pos_in_shard = pos
            else:
                pos_in_shard = pos - self.cumulative_lengths[shard_idx - 1]

            # Note: pos_in_shard is now guaranteed to be valid since we excluded
            # the last T tokens from each shard during cumulative length calculation
            
            shard_info.append((shard_idx, pos_in_shard))
        
        # Extract sequences (similar to Diff_topK_nanoMoE list comprehension style)
        x_list = []
        y_list = []
        
        for shard_idx, pos_in_shard in shard_info:
            # Recreate memmap to avoid memory leak (like Diff_topK_nanoMoE)
            tokens = np.memmap(self.files[shard_idx], dtype=self.dtype, mode='r', offset=self.header_size)
            seq = tokens[pos_in_shard:pos_in_shard + T + 1]
            x_list.append(torch.from_numpy(seq[:T].astype(np.int64)))      # inputs
            y_list.append(torch.from_numpy(seq[1:T+1].astype(np.int64)))   # targets
        
        x = torch.stack(x_list)
        y = torch.stack(y_list)

        return x.cuda(), y.cuda()


def is_megatron_dataset(path_pattern):
    """
    Detect if the path refers to a Megatron indexed dataset.

    Returns True if:
    - Path has no wildcards (not a glob pattern)
    - Corresponding .idx file exists
    """
    # If there are wildcards, it's a multi-file dataset (fineweb/owt style)
    if '*' in path_pattern or '?' in path_pattern:
        return False

    # Check if .idx file exists (Megatron format)
    # Handle both cases: path ends with .bin or not
    if path_pattern.endswith('.bin'):
        idx_path = path_pattern[:-4] + '.idx'
    else:
        idx_path = path_pattern + '.idx'

    return os.path.exists(idx_path)


def create_dataloader(path_pattern, B, T, ddp_rank, ddp_world_size, split='train'):
    """
    Create appropriate dataloader based on dataset format.

    - Megatron indexed datasets (.idx/.bin): Use MegatronDataLoader with split support
    - Multi-file datasets (*.bin): Use DistributedDataLoader

    Args:
        path_pattern: Path to dataset (with wildcards for multi-file, or path for indexed)
        B: Batch size
        T: Sequence length
        ddp_rank: DDP rank
        ddp_world_size: Number of DDP processes
        split: For Megatron datasets, which split to use ('train', 'val', 'test')
    """
    if is_megatron_dataset(path_pattern):
        # Remove .bin extension if present for Megatron loader
        if path_pattern.endswith('.bin'):
            path_pattern = path_pattern[:-4]

        if ddp_rank == 0:
            logging.info(f"Using MegatronDataLoader for indexed dataset: {path_pattern}, split: {split}")

        return MegatronDataLoader(
            dataset_path=path_pattern,
            B=B,
            T=T,
            process_rank=ddp_rank,
            num_processes=ddp_world_size,
            split=split
        )
    else:
        if ddp_rank == 0:
            logging.info(f"Using DistributedDataLoader for multi-file dataset: {path_pattern}")

        return DistributedDataLoader(
            filename_pattern=path_pattern,
            B=B,
            T=T,
            process_rank=ddp_rank,
            num_processes=ddp_world_size
        )


def setup_default_logging(default_level=logging.INFO, log_path=''):
    formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s', datefmt='%Y-%m-%d,%H:%M:%S')
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logging.root.addHandler(console_handler)
    logging.root.setLevel(default_level)
    if log_path:
        file_handler = logging.handlers.RotatingFileHandler(log_path, maxBytes=(1024 ** 2 * 2), backupCount=3)
        file_formatter = logging.Formatter("%(asctime)s - %(name)20s: [%(levelname)8s] - %(message)s")
        file_handler.setFormatter(file_formatter)
        logging.root.addHandler(file_handler)

setup_default_logging()

_CKPT_REGEX = re.compile(r'state_step(\d+)\.pt')

def find_latest_checkpoint(output_dir):
    """
    Returns the path to the checkpoint with the highest step number in `output_dir`,
    or None if no checkpoints are found.
    """
    pattern = os.path.join(output_dir, 'state_step*.pt')
    candidates = glob.glob(pattern)
    if not candidates:
        return None

    def _ckpt_key(path):
        match = _CKPT_REGEX.search(os.path.basename(path))
        return int(match.group(1)) if match else -1

    latest = max(candidates, key=_ckpt_key)
    if _ckpt_key(latest) < 0:
        return None
    return latest

# -----------------------------------------------------------------------------
# Argument parsing

# The first arg parser parses out only the --config argument, this argument is used to
# load a yaml file containing key-values that override the defaults for the main parser below
config_parser = parser = argparse.ArgumentParser(description='Training Config', add_help=False)
parser.add_argument('-c', '--config', default='', type=str, metavar='FILE',
                    help='YAML config file specifying default arguments')

parser = argparse.ArgumentParser(description='NanoGPT MoE Training')

# Data parameters
group = parser.add_argument_group('Data parameters')
group.add_argument('--input-bin', default='data/fineweb10B/fineweb_train_*.bin', type=str,
                   help='input .bin to train on')
group.add_argument('--input-val-bin', default='data/fineweb10B/fineweb_val_*.bin', type=str,
                   help='input .bin to eval validation loss on')

# Model parameters
group = parser.add_argument_group('Model parameters')
group.add_argument('--vocab-size', default=50304, type=int,
                   help='vocabulary size')
group.add_argument('--n-layer', default=12, type=int,
                   help='number of transformer layers')
group.add_argument('--n-head', default=6, type=int,
                   help='number of attention heads')
group.add_argument('--n-embd', default=768, type=int,
                   help='embedding dimension')
group.add_argument('--num-experts', default=8, type=int,
                   help='number of MoE experts')
group.add_argument('--top-k', default=2, type=int,
                   help='top-k experts to use')
group.add_argument('--router-type', default='diff', type=str, choices=['switch', 'diff', 'hash'],
                   help='router type for MoE')
group.add_argument('--router-depth', default=1, type=int,
                   help='number of layers in the router MLP for non-hash routing (hidden dim == input dim)')
group.add_argument('--router-activation', default='gelu', type=str, choices=['gelu', 'relu', 'relu_squared'],
                   help='activation to use between router MLP layers (if depth > 1)')
group.add_argument('--global-load-balance', action='store_true', default=False,
                   help='enable global batch load balancing for auxiliary router loss')
group.add_argument('--loss-free-mode', default='none', type=str, choices=['none', 'deepseek', 'stopgrad'],
                   help='loss-free router biasing strategy (deepseek for switch only, stopgrad supports switch/diff)')
group.add_argument('--loss-free-decay', default=0.99, type=float,
                   help='EMA decay for tracking per-layer expert usage in loss-free routing')
group.add_argument('--loss-free-strength', default=1.0, type=float,
                   help='scale factor applied to the loss-free routing bias')
group.add_argument('--loss-free-update-rate', default=0.001, type=float,
                   help='per-expert bias update rate for sign-based loss-free routing')
group.add_argument('--loss-free-bias-rule', default='ema', type=str, choices=['ema', 'sign'],
                   help='controls whether router bias uses EMA or sign-step updates')

# Optimization parameters
group = parser.add_argument_group('Optimization parameters')
group.add_argument('--batch-size', default=8*64, type=int,
                   help='batch size, in sequences, across all devices')
group.add_argument('--device-batch-size', default=16, type=int,
                   help='batch size, in sequences, per device')
group.add_argument('--sequence-length', default=1024, type=int,
                   help='sequence length, in tokens')
group.add_argument('--num-iterations', default=4578, type=int,
                   help='number of iterations to run')
group.add_argument('--warmup-iters', default=0, type=int,
                   help='number of warmup iterations')
group.add_argument('--warmdown-iters', default=1308, type=int,
                   help='number of iterations of linear warmup/warmdown for triangular or trapezoidal schedule')
group.add_argument('--weight-decay', default=0.0, type=float,
                   help='weight decay')
group.add_argument('--use_adamw_opt3', action='store_true', default=False,
                   help='use AdamW instead of Muon for transformer blocks')
group.add_argument('--use_adamw_router', action='store_true', default=False,
                   help='optimize router parameters with AdamW instead of Muon (requires learned routers)')

# Learning rate parameters
group = parser.add_argument_group('Learning rate parameters')
group.add_argument('--lr-embed', default=0.3, type=float,
                   help='learning rate for embedding layer')
group.add_argument('--lr-head', default=0.002, type=float,
                   help='learning rate for head layer')
group.add_argument('--lr-muon', default=0.02, type=float,
                   help='learning rate for muon optimizer (transformer blocks)')
group.add_argument('--momentum', default=0.95, type=float,
                   help='momentum for muon optimizer')

# Evaluation and logging parameters
group = parser.add_argument_group('Evaluation and logging parameters')
group.add_argument('--val-loss-every', default=125, type=int,
                   help='every how many steps to evaluate val loss? 0 for only at the end')
group.add_argument('--val-tokens', default=10485760, type=int,
                   help='how many tokens of validation data? it\'s important to keep this fixed for consistent comparisons')
group.add_argument('--save-every', default=0, type=int,
                   help='every how many steps to save the checkpoint? 0 for only at the end')
group.add_argument('--n-tracked-seq', default=100, type=int,
                   help='number of sequences to track for expert assignment changes (0 to disable tracking)')
group.add_argument('--wandb-project', default='modded-nanogpt-moe', type=str,
                   help='wandb project name')
group.add_argument('--output', default='logs', type=str,
                   help='output directory for logs and checkpoints')

# Loss parameters
group = parser.add_argument_group('Loss parameters')
group.add_argument('--aux-coeff-train', default=0.0, type=float,
                   help='auxiliary loss coefficient for training')
group.add_argument('--aux-coeff-val', default=0.0, type=float,
                   help='auxiliary loss coefficient for validation')

# Misc:
group = parser.add_argument_group('Run config')
group.add_argument('--device_0', action='store_true', default=False,
                   help='Always use device=0')
group.add_argument('--seed', type=int, default=42,
                   help='random seed (default: 42)')
group.add_argument('--resume', default='auto', type=str,
                   help="checkpoint path to resume from, or 'auto' to load the newest checkpoint in --output")

def _parse_args():
    # Do we have a config file to parse?
    args_config, remaining = config_parser.parse_known_args()
    if args_config.config:
        with open(args_config.config, 'r') as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)

    # The main arg parser parses the rest of the args, the usual
    # defaults will have been overridden if config file specified.
    args = parser.parse_args(remaining)

    # Cache the args as a text string to save them in the output dir later
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)

    return args, args_text

# -----------------------------------------------------------------------------
# int main

# Parse command line arguments and config file  
args, args_text = _parse_args()

# set up DDP (distributed data parallel). torchrun sets this env variable
assert torch.cuda.is_available()
dist.init_process_group(backend='nccl')
ddp_rank = int(os.environ['RANK'])
ddp_local_rank = int(os.environ['LOCAL_RANK'])
ddp_world_size = int(os.environ['WORLD_SIZE'])
if args.device_0:
    device = 'cuda:0'  # Each process only sees one GPU with SLURM
else:
    device = f'cuda:{ddp_local_rank}'

torch.cuda.set_device(device)
logging.info(f"using device: {device}")

torch.manual_seed(args.seed + ddp_rank)
np.random.seed(args.seed + ddp_rank)
random.seed(args.seed + ddp_rank)

# TODO consider making it more deterministic - but make it slower
# torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
# torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn

# torch.use_deterministic_algorithms(True)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False


master_process = (ddp_rank == 0) # this process will do logging, checkpointing etc.

# convenience variables
B, T = args.device_batch_size, args.sequence_length
# calculate the number of steps to take in the val loop.
assert args.val_tokens % (B * T * ddp_world_size) == 0
val_steps = args.val_tokens // (B * T * ddp_world_size)
# calculate the steps of gradient accumulation required to attain the desired global batch size.
assert args.batch_size % (B * ddp_world_size) == 0
train_accumulation_steps = args.batch_size // (B * ddp_world_size)

# load tokens
train_loader = create_dataloader(args.input_bin, B, T, ddp_rank, ddp_world_size, split='train')
val_loader = create_dataloader(args.input_val_bin, B, T, ddp_rank, ddp_world_size, split='val')
if master_process:
    # Log dataset info - handle both loader types
    if hasattr(train_loader, 'ntok_total'):
        # DistributedDataLoader
        logging.info(f"Training DataLoader: total number of tokens: {train_loader.ntok_total} across {len(train_loader.files)} files")
        logging.info(f"Validation DataLoader: total number of tokens: {val_loader.ntok_total} across {len(val_loader.files)} files")
    else:
        # MegatronDataLoader
        logging.info(f"Training DataLoader: total number of tokens: {train_loader.total_tokens}")
        logging.info(f"Validation DataLoader: total number of tokens: {val_loader.total_tokens}")
x, y = train_loader.next_batch()

# create model using parsed arguments
model = GPT(GPTConfig(
    vocab_size=args.vocab_size, 
    n_layer=args.n_layer, 
    n_head=args.n_head, 
    n_embd=args.n_embd,
    num_experts=args.num_experts,
    top_k=args.top_k,
    router_type=args.router_type,
    router_depth=args.router_depth,
    router_activation=args.router_activation,
    global_load_balance=args.global_load_balance,
    loss_free_mode=args.loss_free_mode,
    loss_free_decay=args.loss_free_decay,
    loss_free_strength=args.loss_free_strength,
    loss_free_update_rate=args.loss_free_update_rate,
    loss_free_bias_rule=args.loss_free_bias_rule,
))
model = model.cuda()
if hasattr(config, "coordinate_descent_tuning"):
    config.coordinate_descent_tuning = True # suggested by @Chillee
model = torch.compile(model)
# here we wrap model into DDP container
if args.device_0:
    model = DDP(model, device_ids=[0], find_unused_parameters=True)
else:
    model = DDP(model, device_ids=[ddp_local_rank], find_unused_parameters=True)
    
raw_model = model.module # always contains the "raw" unwrapped model
num_experts = raw_model.transformer.h[0].mlp.num_experts
ctx = torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16)

# CUDNN attention is ~4ms faster than Flash, but doesn't get selected by default in PyTorch 2.5.1
from torch.backends.cuda import enable_cudnn_sdp, enable_flash_sdp, enable_math_sdp, enable_mem_efficient_sdp
enable_cudnn_sdp(True)
enable_flash_sdp(False)
enable_mem_efficient_sdp(False)
enable_math_sdp(False)

# init the optimizer(s)
all_h_params = list(raw_model.transformer.h.parameters())
optimizer1 = torch.optim.AdamW([raw_model.transformer.wte.weight], lr=args.lr_embed, betas=(0.9, 0.95), weight_decay=args.weight_decay, fused=True)
optimizer2 = torch.optim.AdamW([raw_model.lm_head.weight], lr=args.lr_head, betas=(0.9, 0.95), weight_decay=args.weight_decay, fused=True)
router_optimizer = None

if args.use_adamw_opt3:
    optimizer3 = torch.optim.AdamW(all_h_params, lr=6e-4, betas=(0.9, 0.95), weight_decay=args.weight_decay, fused=True)
else:
    muon_params = all_h_params
    if args.use_adamw_router:
        router_params = []
        for block in raw_model.transformer.h:
            router_module = getattr(block.mlp, 'router', None)
            if router_module is not None:
                router_params.extend(list(router_module.parameters()))
        if router_params:
            router_param_ids = {id(p) for p in router_params}
            muon_params = [p for p in all_h_params if id(p) not in router_param_ids]
            router_optimizer = torch.optim.AdamW(router_params, lr=args.lr_muon, betas=(0.9, 0.95), weight_decay=args.weight_decay, fused=True)
        elif master_process:
            logging.warning("AdamW router optimization requested, but no router parameters were found.")
    optimizer3 = Muon(muon_params, lr=args.lr_muon, momentum=args.momentum)
optimizers = [optimizer1, optimizer2, optimizer3]
if router_optimizer is not None:
    optimizers.append(router_optimizer)
# learning rate decay scheduler (linear warmup and warmdown)
def get_lr(it):
    assert it <= args.num_iterations
    # 1) linear warmup for warmup_iters steps
    if it < args.warmup_iters:
        return (it+1) / args.warmup_iters
    # 2) constant lr for a while
    elif it < args.num_iterations - args.warmdown_iters:
        return 1.0
    # 3) linear warmdown
    else:
        decay_ratio = (args.num_iterations - it) / args.warmdown_iters
        return decay_ratio
schedulers = [torch.optim.lr_scheduler.LambdaLR(opt, get_lr) for opt in optimizers]

# handle resume-from-checkpoint
start_step = 0
resume_training_time_ms = 0.0
resolved_resume_path = None
if args.resume:
    if args.resume == 'auto':
        resolved_resume_path = find_latest_checkpoint(args.output)
        if resolved_resume_path is None and master_process:
            logging.info(f"Auto-resume requested but no checkpoints found under {args.output}, starting fresh.")
    else:
        resolved_resume_path = args.resume

    if resolved_resume_path and os.path.isfile(resolved_resume_path):
        checkpoint = torch.load(resolved_resume_path, map_location='cpu')
        raw_model.load_state_dict(checkpoint['model'])
        checkpoint_opts = checkpoint.get('optimizers', [])
        for opt, state in zip(optimizers, checkpoint_opts):
            opt.load_state_dict(state)
        checkpoint_schedulers = checkpoint.get('schedulers', [])
        for sched, state in zip(schedulers, checkpoint_schedulers):
            sched.load_state_dict(state)
        start_step = checkpoint.get('step', 0) + 1
        start_step = min(start_step, args.num_iterations)
        resume_training_time_ms = checkpoint.get('training_time_ms', 0.0)
        args.resume = resolved_resume_path
        if master_process:
            logging.info(f"Resumed from checkpoint {resolved_resume_path} at step {start_step}.")
    elif args.resume and master_process:
        logging.info(f"Resume requested but checkpoint {args.resume} not found. Starting from scratch.")

# begin logging
if master_process:
    run_id = str(uuid.uuid4())
    os.makedirs(args.output, exist_ok=True)
    logfile = os.path.join(args.output, f'{run_id}.txt')
    # create the log file
    with open(logfile, "w") as f:
        # begin the log by printing this file (the Python code)
        f.write('='*100 + '\n')
        f.write(code)
        f.write('='*100 + '\n')
        # log information about the hardware/software environment this is running on
        # and print the full `nvidia-smi` to file
        f.write(f"Running pytorch {torch.version.__version__} compiled for CUDA {torch.version.cuda}\nnvidia-smi:\n")
        import subprocess
        result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        f.write(f'{result.stdout}\n')
        f.write('='*100 + '\n')
    # save args to file
    with open(os.path.join(args.output, 'args.yaml'), 'w') as f:
        f.write(args_text)
    # init wandb
    
    # Create config from args and add computed/runtime information
    config = vars(args).copy()  # Convert args to dict
    config.update({
        'train_accumulation_steps': train_accumulation_steps,
        'val_steps': val_steps,
        'num_experts': num_experts,
        'ddp_world_size': ddp_world_size,
        'ddp_rank': ddp_rank,
        'ddp_local_rank': ddp_local_rank,
        'model_n_layer': raw_model.config.n_layer,
        'model_n_head': raw_model.config.n_head,
        'model_n_embd': raw_model.config.n_embd,
        'model_router_type': raw_model.transformer.h[0].mlp.router_type,
        'model_router_top_k': raw_model.transformer.h[0].mlp.top_k,
        'model_router_depth': raw_model.transformer.h[0].mlp.router_depth,
        'model_router_activation': raw_model.transformer.h[0].mlp.router_activation,
        'global_load_balance': args.global_load_balance,
        'model_global_load_balance': raw_model.config.global_load_balance,
        'optimizer_embed_lr': optimizer1.param_groups[0]['lr'],
        'optimizer_embed_betas': tuple(optimizer1.param_groups[0]['betas']),
        'optimizer_embed_fused': bool(optimizer1.param_groups[0].get('fused', False)),
        'optimizer_head_lr': optimizer2.param_groups[0]['lr'],
        'optimizer_head_betas': tuple(optimizer2.param_groups[0]['betas']),
        'optimizer_head_fused': bool(optimizer2.param_groups[0].get('fused', False)),
        'optimizer_muon_lr': optimizer3.param_groups[0]['lr'],
        'optimizer_muon_momentum': optimizer3.defaults.get('momentum', None),
        'optimizer_muon_nesterov': optimizer3.defaults.get('nesterov', None),
        'optimizer_muon_backend': optimizer3.defaults.get('backend', None),
        'optimizer_muon_backend_steps': optimizer3.defaults.get('backend_steps', None),
        'amp_dtype': 'bfloat16',
        'torch_compile': True,
        'attention_backend': 'cudnn_sdp',
    })
    if router_optimizer is not None:
        config.update({
            'optimizer_router_lr': router_optimizer.param_groups[0]['lr'],
            'optimizer_router_betas': tuple(router_optimizer.param_groups[0]['betas']),
            'optimizer_router_fused': bool(router_optimizer.param_groups[0].get('fused', False)),
        })
    
    run_name = os.path.basename(args.output)
    # group seeds
    # group_args = "+".join([x for x in run_name.split("+")[1:] if 'see=' not in x])
    # group_base = run_name.split("+")[0][13:]
    # group = group_base + group_args
    
    # tags = [x for x in os.path.basename(args.output).split("+")[1:] if len(x) > 0]
    
    wandb_run = wandb.init(project=args.wandb_project, 
                           name=run_name, 
                        #    tags=tags,
                           config=config)

# Sample fixed sequences for expert assignment tracking
if master_process and args.n_tracked_seq > 0:
    # Sample sequences for tracking expert assignments over time
    tracking_sequences = []
    for _ in range(args.n_tracked_seq):
        x_sample, _ = val_loader.next_batch()
        tracking_sequences.append(x_sample[0:1])  # Take first sequence from batch
    tracking_x = torch.cat(tracking_sequences, dim=0).cuda()  # Shape: (n_tracked_seq, T)
    # Store previous expert assignments for comparison
    prev_expert_assignments = None  # Will be set on first validation
else:
    tracking_x = None
    prev_expert_assignments = None

# Sample specific tokens for matrix visualization
if master_process and tracking_x is not None:
    # Sample 5 random token positions from the tracked sequences
    torch.manual_seed(args.seed)  # For reproducibility
    random.seed(args.seed)
    
    tracked_token_positions = []
    for i in range(5):
        seq_idx = random.randint(0, tracking_x.shape[0] - 1)
        token_idx = random.randint(0, tracking_x.shape[1] - 1)
        tracked_token_positions.append((seq_idx, token_idx))
        token_id = tracking_x[seq_idx, token_idx].item()
        logging.info(f"Tracking token #{i}: seq={seq_idx}, pos={token_idx}, token_id={token_id}")
else:
    tracked_token_positions = None

training_time_ms = resume_training_time_ms
# start the clock
torch.cuda.synchronize()
t0 = time.time()
# begin training
for step in range(start_step, args.num_iterations + 1):
    last_step = (step == args.num_iterations)
    # This effectively ignores timing first 10 steps, which are slower for weird reasons.
    # Alternately, and slightly more correctly in terms of benchmarking, we could do 10
    # steps with dummy data first, and then re-initialize the model and reset the loader.
    if step == 10:
        training_time_ms = 0
        t0 = time.time()
    timed_steps = float('nan') if step <= 11 else (step - 10) + 1 # <= 11 to avoid bug in val

    # once in a while evaluate the validation dataset
    if (last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0)):
        # stop the clock
        torch.cuda.synchronize()
        training_time_ms += 1000 * (time.time() - t0)
        # run validation batches
        model.eval()
        val_loss = 0.0
        val_ce_loss = 0.0
        val_aux_loss = 0.0
        val_router_entropy = torch.tensor(0.0, device=device)
        val_expert_balance = torch.zeros(num_experts, device=device)
        # per-layer
        n_layers = raw_model.config.n_layer
        val_layer_router_entropy = torch.zeros(n_layers, device=device)
        val_layer_expert_balance = torch.zeros(n_layers, num_experts, device=device)
        for _ in range(val_steps):
            x_val, y_val = val_loader.next_batch()
            with torch.no_grad():
                with ctx:
                    _, loss, ce_loss, total_aux, router_entropy, expert_balance, layer_router_entropy, layer_expert_balance = model(x_val, y_val, return_logits=False, aux_coeff=args.aux_coeff_val)
                    val_loss += loss.detach()
                    val_ce_loss += ce_loss.detach()
                    val_aux_loss += total_aux.detach()
                    val_router_entropy = val_router_entropy + router_entropy.detach()
                    val_expert_balance = val_expert_balance + expert_balance.detach()
                    val_layer_router_entropy = val_layer_router_entropy + layer_router_entropy.detach()
                    val_layer_expert_balance = val_layer_expert_balance + layer_expert_balance.detach()
                    del loss, ce_loss
        dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)
        dist.all_reduce(val_ce_loss, op=dist.ReduceOp.AVG)
        dist.all_reduce(val_aux_loss, op=dist.ReduceOp.AVG)
        val_loss /= val_steps
        val_ce_loss /= val_steps
        val_aux_loss /= val_steps
        # average and all-reduce router stats
        val_router_entropy = val_router_entropy / val_steps
        val_expert_balance = val_expert_balance / val_steps
        val_layer_router_entropy = val_layer_router_entropy / val_steps
        val_layer_expert_balance = val_layer_expert_balance / val_steps
        dist.all_reduce(val_router_entropy, op=dist.ReduceOp.AVG)
        dist.all_reduce(val_expert_balance, op=dist.ReduceOp.AVG)
        dist.all_reduce(val_layer_router_entropy, op=dist.ReduceOp.AVG)
        dist.all_reduce(val_layer_expert_balance, op=dist.ReduceOp.AVG)
        # log val loss to console and to logfile
        if master_process:
            logging.info(f'step:{step}/{args.num_iterations} val_loss:{val_loss:.4f} train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms/(timed_steps-1):.2f}ms')
            with open(logfile, "a") as f:
                f.write(f'step:{step}/{args.num_iterations} val_loss:{val_loss:.4f} train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms/(timed_steps-1):.2f}ms\n')
        # compute router grad norms (CE and AUX separately) occasionally at validation interval
        # Use a single micro-batch to avoid heavy cost
        model.train()
        # grab a fresh batch (train or val doesn't matter for grads inspection)
        x_probe, y_probe = val_loader.next_batch()
        # 1) CE-only
        model.zero_grad(set_to_none=True)
        gc.collect()
        with ctx:
            _, loss_ce, ce_loss_probe, total_aux_probe, _, _, _, _ = model(x_probe, y_probe, return_logits=False, aux_coeff=0.0)
        loss_ce.backward()
        ce_router_layer_grad_norms = []
        for li in range(raw_model.config.n_layer):
            if raw_model.transformer.h[li].mlp.router_type != 'hash':
                p = raw_model.transformer.h[li].mlp.router[-1].weight
                gnorm = p.grad.detach().float().norm(2) if p.grad is not None else torch.tensor(0.0, device=device)
                ce_router_layer_grad_norms.append(gnorm)
            else:
                ce_router_layer_grad_norms.append(torch.tensor(0.0, device=device))
        ce_router_layer_grad_norms = torch.stack(ce_router_layer_grad_norms)
        dist.all_reduce(ce_router_layer_grad_norms, op=dist.ReduceOp.AVG)
        # 2) AUX-only
        model.zero_grad(set_to_none=True)
        gc.collect()
        with ctx:
            _, _, _, total_aux_probe, _, _, _, _ = model(x_probe, y_probe, return_logits=False, aux_coeff=0.0)
        # Backprop aux explicitly
        total_aux_probe.backward()
        aux_router_layer_grad_norms = []
        for li in range(raw_model.config.n_layer):
            if raw_model.transformer.h[li].mlp.router_type != 'hash':
                p = raw_model.transformer.h[li].mlp.router[-1].weight
                gnorm = p.grad.detach().float().norm(2) if p.grad is not None else torch.tensor(0.0, device=device)
                aux_router_layer_grad_norms.append(gnorm)
            else:
                aux_router_layer_grad_norms.append(torch.tensor(0.0, device=device))
        aux_router_layer_grad_norms = torch.stack(aux_router_layer_grad_norms)
        dist.all_reduce(aux_router_layer_grad_norms, op=dist.ReduceOp.AVG)
        # zero out any probe grads
        model.zero_grad(set_to_none=True)
        gc.collect()
        
        # Expert assignment tracking
        topk_change_percentages = {}  # Dict to store changes for each k value
        any_topk_changed_percentages = []
        if master_process and tracking_x is not None:
            model.eval()
            with torch.no_grad():
                with ctx:
                    # Get current expert assignments for tracking sequences
                    _, _, _, _, _, _, _, _, current_assignments = model(tracking_x, return_logits=False, aux_coeff=0.0, return_expert_assignments=True)
                    # current_assignments shape: (n_layers, 100, seq_len, top_k)
                    sorted_curr_assignments = current_assignments.clone().sort(dim=-1)[0]
                    
                    if prev_expert_assignments is not None:
                        # Compare with previous assignments
                        for layer_idx in range(current_assignments.shape[0]):
                            # Loop over expert positions (1st, 2nd, ..., top_k-th)
                            for pos in range(current_assignments.shape[3]):
                                k = pos + 1  # Convert 0-indexed to 1-indexed for logging
                                if k not in topk_change_percentages:
                                    topk_change_percentages[k] = []
                                
                                # Check if the expert at position 'pos' changed
                                curr_expert_at_pos = current_assignments[layer_idx, :, :, pos]  # (100, seq_len)
                                prev_expert_at_pos = prev_expert_assignments[layer_idx, :, :, pos]
                                pos_changes = (curr_expert_at_pos != prev_expert_at_pos).float()
                                
                                pos_change_pct = pos_changes.mean().item()
                                topk_change_percentages[k].append(pos_change_pct)
                            
                            any_topk_changed = (sorted_prev_assignments[layer_idx, :, :, :] != sorted_curr_assignments[layer_idx, :, :, :]).sum(dim=-1).type(torch.bool)
                            any_topk_changed_percentages.append(any_topk_changed.float().mean().item())
                    else:
                        # First validation - initialize with zeros
                        for k in range(1, current_assignments.shape[3] + 1):
                            topk_change_percentages[k] = [0.0] * current_assignments.shape[0]
                        any_topk_changed_percentages = [0.0] * current_assignments.shape[0]
                    
                    # Store current assignments for next comparison
                    prev_expert_assignments = current_assignments.clone()
                    sorted_prev_assignments = sorted_curr_assignments.clone()
        
        # Single token matrix tracking for wandb visualization
        if master_process and tracked_token_positions is not None and tracking_x is not None:
            model.eval()
            with torch.no_grad():
                with ctx:
                    # Get expert assignments for all tracked sequences
                    _, _, _, _, _, _, _, _, all_assignments = model(tracking_x, return_logits=False, aux_coeff=0.0, return_expert_assignments=True)
                    # all_assignments shape: (n_layers, n_tracked_seq, seq_len, top_k)
                    
                    # Create matrices for each tracked token
                    for token_idx, (seq_idx, pos_idx) in enumerate(tracked_token_positions):
                        # Create matrix: rows = experts (0-7), cols = layers
                        n_layers = all_assignments.shape[0]
                        matrix = torch.zeros((num_experts, n_layers), dtype=torch.float32)
                        
                        # Fill matrix with expert assignments for this specific token
                        for layer_idx in range(n_layers):
                            # Get top-k experts chosen for this token in this layer
                            chosen_experts = all_assignments[layer_idx, seq_idx, pos_idx, :]  # shape: (top_k,)
                            # Mark chosen experts as 1
                            for expert_id in chosen_experts:
                                matrix[expert_id.item(), layer_idx] = 1.0
                        
                        # Log matrix as wandb Table for slider visualization
                        token_id = tracking_x[seq_idx, pos_idx].item()
                        matrix_data = matrix.cpu().numpy()
                        
                        # Create wandb Table with proper column/row labels
                        columns = [f"Layer_{i}" for i in range(n_layers)]
                        rows = [f"Expert_{i}" for i in range(num_experts)]
                        
                        # Convert matrix to list of lists for wandb
                        matrix_list = [[float(matrix_data[row, col]) for col in range(n_layers)] for row in range(num_experts)]
                        
                        # Create wandb Image from matrix for visualization
                        import matplotlib.pyplot as plt
                        import matplotlib
                        matplotlib.use('Agg')  # Non-interactive backend
                        
                        fig, ax = plt.subplots(figsize=(n_layers, num_experts))
                        im = ax.imshow(matrix_data, cmap='Blues', aspect='auto')
                        
                        # Set labels
                        ax.set_xticks(range(n_layers))
                        ax.set_xticklabels(columns)
                        ax.set_yticks(range(num_experts))
                        ax.set_yticklabels(rows)
                        
                        # Add text annotations
                        for i in range(num_experts):
                            for j in range(n_layers):
                                text = ax.text(j, i, f'{matrix_data[i, j]:.0f}',
                                             ha="center", va="center", color="red" if matrix_data[i, j] > 0.5 else "black")
                        
                        ax.set_title(f'Token {token_idx} (ID: {token_id}) Expert Selection')
                        ax.set_xlabel('Layers')
                        ax.set_ylabel('Experts')
                        plt.tight_layout()
                        
                        wandb.log({
                            f"token_matrix/token_{token_idx}_id_{token_id}": wandb.Image(fig)
                        }, step=step)
                        
                        plt.close(fig)
        
        # log to wandb
        if master_process:
            wandb_log_extra = {}
            for li in range(raw_model.config.n_layer):
                wandb_log_extra[f'Router Grad Norms (CE)/Layer {li}'] = float(ce_router_layer_grad_norms[li].item())
                wandb_log_extra[f'Router Grad Norms (AUX)/Layer {li}'] = float(aux_router_layer_grad_norms[li].item())
                # Add expert assignment change percentages for all k values
                for k in topk_change_percentages:
                    wandb_log_extra[f'track_tokens/layer_{li}/top{k}_change'] = float(topk_change_percentages[k][li])
                    
                wandb_log_extra[f'track_tokens/layer_{li}/chosen_changed'] = float(any_topk_changed_percentages[li])
                
            wandb.log(wandb_log_extra, step=step)
        # now also log the earlier val metrics
        if master_process:
            wandb_log = {
                'val/loss': float(val_loss.item() if isinstance(val_loss, torch.Tensor) else val_loss),
                'val/ce_loss': float(val_ce_loss.item() if isinstance(val_ce_loss, torch.Tensor) else val_ce_loss),
                'val/aux_loss': float(val_aux_loss.item() if isinstance(val_aux_loss, torch.Tensor) else val_aux_loss),
                'val/router_entropy': float(val_router_entropy.item()),
            }
            val_maxvio_layers = maxvio_per_layer(val_layer_expert_balance.detach())
            val_maxvio_layers_cpu = val_maxvio_layers.cpu()
            wandb_log['val/MaxVioglobal'] = float(val_maxvio_layers_cpu.mean().item())
            for i in range(num_experts):
                wandb_log[f'val/expert_balance/{i}'] = float(val_expert_balance[i].item())
            for li in range(n_layers):
                wandb_log[f'Router Entropy/Layer {li}'] = float(val_layer_router_entropy[li].item())
                wandb_log[f'val/MaxVioglobal/Layer {li}'] = float(val_maxvio_layers_cpu[li].item())
                for ei in range(num_experts):
                    wandb_log[f'Expert Balance/Layer {li}/{ei}'] = float(val_layer_expert_balance[li, ei].item())
                mlp = raw_model.transformer.h[li].mlp
                bias_vec = mlp._loss_free_bias_vector()
                if bias_vec is not None:
                    bias_vals = bias_vec.detach().float().cpu()
                    for ei in range(mlp.num_experts):
                        wandb_log[f'val_loss_free_bias/Layer {li}/expert_{ei}'] = float(bias_vals[ei].item())
            wandb_log['train/time_ms'] = float(training_time_ms)
            wandb_log['train/step_avg_ms'] = float(training_time_ms/(timed_steps-1))
            wandb.log(wandb_log, step=step)

        # start the clock again
        torch.cuda.synchronize()
        t0 = time.time()

    if master_process and (last_step or (args.save_every > 0 and step % args.save_every == 0)):
        # stop the clock
        torch.cuda.synchronize()
        training_time_ms += 1000 * (time.time() - t0)
        # save the state of the training process
        log = dict(
            step=step,
            code=code,
            model=raw_model.state_dict(),
            optimizers=[opt.state_dict() for opt in optimizers],
            schedulers=[sched.state_dict() for sched in schedulers],
            training_time_ms=training_time_ms,
        )
        torch.save(log, os.path.join(args.output, f'state_step{step:06d}.pt'))
        # start the clock again
        torch.cuda.synchronize()
        t0 = time.time()

    # bit confusing: we want to make sure to eval on 0th iteration
    # but also after the very last iteration. so we loop for step <= num_iterations
    # instead of just < num_iterations (one extra due to <=), only to do
    # the validation/sampling one last time, and then we break right here as we're done.
    if last_step:
        break

    # --------------- TRAINING SECTION BEGIN -----------------
    model.train()
    n_layers = raw_model.config.n_layer
    use_global_lb = raw_model.config.global_load_balance
    router_context_use = None
    cached_batches = None

    if use_global_lb:
        cached_batches = []
        tokens_accum = torch.zeros(n_layers, num_experts, device=device)
        totals_accum = torch.zeros(n_layers, device=device)
        collect_context = {
            'mode': 'collect',
            'tokens_accum': tokens_accum,
            'totals_accum': totals_accum,
        }
        for _ in range(train_accumulation_steps):
            cached_batches.append((x.clone(), y.clone()))
            with torch.no_grad():
                with ctx:
                    model(x, y, return_logits=False, aux_coeff=0.0, router_context=collect_context)
            x, y = train_loader.next_batch()
        dist.all_reduce(tokens_accum, op=dist.ReduceOp.SUM)
        dist.all_reduce(totals_accum, op=dist.ReduceOp.SUM)
        denom = torch.clamp(totals_accum.unsqueeze(1), min=1.0)
        global_frac = tokens_accum / denom
        router_context_use = {
            'mode': 'use',
            'global_frac': global_frac,
        }

    router_entropy_sum = torch.tensor(0.0, device=device)
    expert_balance_sum = torch.zeros(num_experts, device=device)
    layer_router_entropy_sum = torch.zeros(n_layers, device=device)
    layer_expert_balance_sum = torch.zeros(n_layers, num_experts, device=device)
    for i in range(1, train_accumulation_steps+1):
        if use_global_lb:
            x_batch, y_batch = cached_batches[i-1]
        else:
            x_batch, y_batch = x, y
        # forward pass
        with ctx:
            _, loss, ce_loss, total_aux, router_entropy, expert_balance, layer_router_entropy, layer_expert_balance = model(
                x_batch, y_batch, return_logits=False, aux_coeff=args.aux_coeff_train, router_context=router_context_use
            )
            train_loss = loss.detach()
            router_entropy_sum = router_entropy_sum + router_entropy.detach()
            expert_balance_sum = expert_balance_sum + expert_balance.detach()
            layer_router_entropy_sum = layer_router_entropy_sum + layer_router_entropy.detach()
            layer_expert_balance_sum = layer_expert_balance_sum + layer_expert_balance.detach()
        if not use_global_lb:
            # advance the dataset for the next batch
            x, y = train_loader.next_batch()
        # backward pass
        if i < train_accumulation_steps:
            with model.no_sync(): # there's no need to sync gradients every accumulation step
                loss.backward()
        else:
            loss.backward() # just sync on the last step
    for n, p in model.named_parameters():
        if p.grad is None:
            logging.info(n)
    for p in model.parameters():
        p.grad /= train_accumulation_steps

    # compute gradient norm (after accumulation average, before optimizer step)
    grad_norm = torch.tensor(0.0, device=device)
    grads_norms = []
    for p in model.parameters():
        if p.grad is not None:
            grads_norms.append(p.grad.detach().float().norm(2))
    if len(grads_norms) > 0:
        grad_norm = torch.norm(torch.stack(grads_norms), 2)

    # average and all-reduce router stats across accumulation steps and processes
    router_entropy_avg = router_entropy_sum / train_accumulation_steps
    expert_balance_avg = expert_balance_sum / train_accumulation_steps
    layer_router_entropy_avg = layer_router_entropy_sum / train_accumulation_steps
    layer_expert_balance_avg = layer_expert_balance_sum / train_accumulation_steps
    dist.all_reduce(router_entropy_avg, op=dist.ReduceOp.AVG)
    dist.all_reduce(expert_balance_avg, op=dist.ReduceOp.AVG)
    dist.all_reduce(layer_router_entropy_avg, op=dist.ReduceOp.AVG)
    dist.all_reduce(layer_expert_balance_avg, op=dist.ReduceOp.AVG)

    # step the optimizers and schedulers
    for opt, sched in zip(optimizers, schedulers):
        opt.step()
        sched.step()
    # capture the current learning rates after scheduler updates
    current_embed_lr = optimizers[0].param_groups[0]['lr']
    current_head_lr = optimizers[1].param_groups[0]['lr']
    current_blocks_lr = optimizers[2].param_groups[0]['lr']
    current_router_lr = router_optimizer.param_groups[0]['lr'] if router_optimizer is not None else None
    # null the gradients
    model.zero_grad(set_to_none=True)
    raw_model.finalize_loss_free_updates()
    # --------------- TRAINING SECTION END -------------------
    # everything that follows now is just diagnostics, prints, logging, etc.

    #dist.all_reduce(train_loss, op=dist.ReduceOp.AVG) # all-reducing the training loss would be more correct in terms of logging, but slower
    if master_process:
        approx_time = training_time_ms + 1000 * (time.time() - t0)
        logging.info(f"step:{step+1}/{args.num_iterations} train_loss:{train_loss.item():.4f} train_time:{approx_time:.0f}ms step_avg:{approx_time/timed_steps:.2f}ms")
        with open(logfile, "a") as f:
            f.write(f"step:{step+1}/{args.num_iterations} train_loss:{train_loss.item():.4f} train_time:{approx_time:.0f}ms step_avg:{approx_time/timed_steps:.2f}ms\n")
        # wandb logging
        wandb_log = {
            'train/loss': float(train_loss.item()),
            'train/router_entropy': float(router_entropy_avg.item()),
            'train/grad_norm': float(grad_norm.item()),
            'train/step_time_ms': float(approx_time),
            'train/step_avg_ms': float(approx_time/timed_steps),
            'lr/embed': float(current_embed_lr),
            'lr/head': float(current_head_lr),
            'lr/blocks': float(current_blocks_lr),
        }
        if current_router_lr is not None:
            wandb_log['lr/router'] = float(current_router_lr)
        train_maxvio_layers = maxvio_per_layer(layer_expert_balance_avg.detach())
        train_maxvio_layers_cpu = train_maxvio_layers.cpu()
        wandb_log['train/MaxViobatch'] = float(train_maxvio_layers_cpu.mean().item())
        for i_exp in range(num_experts):
            wandb_log[f'train/expert_balance/{i_exp}'] = float(expert_balance_avg[i_exp].item())
        # per-layer router stats (no per-step grad norms anymore)
        for li in range(n_layers):
            wandb_log[f'Router Entropy/Layer {li}'] = float(layer_router_entropy_avg[li].item())
            wandb_log[f'train/MaxViobatch/Layer {li}'] = float(train_maxvio_layers_cpu[li].item())
            for ei in range(num_experts):
                wandb_log[f'Expert Balance/Layer {li}/{ei}'] = float(layer_expert_balance_avg[li, ei].item())
            mlp = raw_model.transformer.h[li].mlp
            bias_vec = mlp._loss_free_bias_vector()
            if bias_vec is not None:
                bias_vals = bias_vec.detach().float().cpu()
                for ei in range(mlp.num_experts):
                    wandb_log[f'train_loss_free_bias/Layer {li}/expert_{ei}'] = float(bias_vals[ei].item())
        wandb.log(wandb_log, step=step+1)

if master_process:
    logging.info(f"peak memory consumption: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB")
    try:
        wandb.finish()
    except Exception:
        pass

# -------------------------------------------------------------------------
# clean up nice
dist.destroy_process_group()
