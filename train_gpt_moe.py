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
from wandb_logging import init_wandb, wandb_train_log, wandb_val_log
from params import parse_args
from optimizers import Muon
from logger import setup_default_logging

# Import Megatron dataloader for indexed datasets
from megatron_indexed_dataset import MegatronDataLoader

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

    @staticmethod
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
        q = Rotary.apply_rotary_emb(q, cos, sin)
        k = Rotary.apply_rotary_emb(k, cos, sin)
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



ROUTER_VALUE_KEYS = (
    'top1_logit',
    'top2_logit',
    'logit_diff',
    'top1_coef',
    'top2_coef',
    'coef_diff',
)


def init_layer_router_value_tensors(n_layers, device):
    return {key: torch.zeros(n_layers, device=device, dtype=torch.float32) for key in ROUTER_VALUE_KEYS}


def init_total_router_value_tensors(device):
    return {key: torch.tensor(0.0, device=device, dtype=torch.float32) for key in ROUTER_VALUE_KEYS}


class ReLUSquared(nn.Module):
    def forward(self, x):
        return torch.relu(x).square()

class MoE(nn.Module):
    def __init__(self, config, layer_idx=0):
        super().__init__()
        self.num_experts = config.num_experts
        self.top_k = config.top_k
        assert 1 <= self.top_k <= self.num_experts, "`k` must be in [1, #experts]"
        self.router_type = config.router_type
        self.router_depth = config.router_depth
        self.router_activation = config.router_activation
        self.global_load_balance = config.global_load_balance
        self.layer_idx = layer_idx
        self.loss_free_mode = config.loss_free_mode
        self.loss_free_decay = config.loss_free_decay
        self.loss_free_strength = config.loss_free_strength
        self.loss_free_update_rate = config.loss_free_update_rate
        self.loss_free_bias_rule = config.loss_free_bias_rule
        self.router_logit_jitter = config.router_logit_jitter
        self.use_router_temperature = config.use_router_temperature
        if self.use_router_temperature:
            self.router_temperature_log = nn.Parameter(torch.zeros(1, dtype=torch.float32))
        else:
            self.register_parameter('router_temperature_log', None)

        assert self.router_type in ('hash', 'switch', 'diff')
        assert self.router_activation in ('gelu', 'relu', 'relu_squared')
        assert self.loss_free_mode in ('none', 'deepseek', 'stopgrad')
        assert self.loss_free_bias_rule in ('ema', 'sign')
        assert self.router_logit_jitter >= 0.0, "router_logit_jitter must be non-negative"
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
            for _ in range(self.router_depth - 1):
                layers.append(nn.Linear(config.n_embd, config.n_embd, bias=False))
                layers.append(self._build_router_activation())
            layers.append(nn.Linear(config.n_embd, self.num_experts, bias=False))
            self.router = nn.Sequential(*layers)
        else:
            self.router = None

    def _get_router_temperature(self, reference_tensor):
        if not self.use_router_temperature or self.router_temperature_log is None:
            return None
        temperature = torch.exp(self.router_temperature_log)
        if reference_tensor is not None:
            temperature = temperature.to(dtype=reference_tensor.dtype, device=reference_tensor.device)
        return temperature

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
            violation = target - update
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
            if self.training and self.router_logit_jitter > 0.0:
                noise = torch.empty_like(logits).uniform_(
                    1.0 - self.router_logit_jitter,
                    1.0 + self.router_logit_jitter,
                )
                logits = logits * noise
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

            if self.use_router_temperature:
                temperature = self._get_router_temperature(logits_for_selection)
                logits_for_selection = logits_for_selection / temperature
                logits_for_weights = logits_for_weights / temperature
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

        with torch.no_grad():
            gate_stats = gate_flat.float()
            top1_coef_vals = gate_stats[:, 0]
            if gate_stats.size(1) >= 2:
                top2_coef_vals = gate_stats[:, 1]
            else:
                top2_coef_vals = torch.zeros_like(top1_coef_vals)

            if logits_for_selection is not None:
                logits_flat = logits_for_selection.reshape(BT, self.num_experts).float()
                num_top_logits = 2 if logits_flat.size(1) >= 2 else 1
                top_logits = torch.topk(logits_flat, k=num_top_logits, dim=-1).values
                top1_logits_vals = top_logits[:, 0]
                if num_top_logits == 2:
                    top2_logits_vals = top_logits[:, 1]
                else:
                    top2_logits_vals = torch.zeros_like(top1_logits_vals)
            else:
                zero_vals = torch.zeros(BT, device=x.device, dtype=torch.float32)
                top1_logits_vals = zero_vals
                top2_logits_vals = zero_vals

            router_value_stats = {
                'top1_logit': top1_logits_vals.mean(),
                'top2_logit': top2_logits_vals.mean(),
                'logit_diff': (top1_logits_vals - top2_logits_vals).mean(),
                'top1_coef': top1_coef_vals.mean(),
                'top2_coef': top2_coef_vals.mean(),
                'coef_diff': (top1_coef_vals - top2_coef_vals).mean(),
            }

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
            return y, aux, router_entropy, frac, router_value_stats, topk_idx.view_as(x[:, :, :self.top_k])
        else:
            return y, aux, router_entropy, frac, router_value_stats


class Block(nn.Module):
    def __init__(self, config, layer_idx=0):
        super().__init__()
        self.attn = CausalSelfAttention(config)
        self.mlp = MoE(
            config,
            layer_idx=layer_idx,
        )

    def forward(self, x, token_idx=None, return_expert_assignments=False, router_context=None):
        x = x + self.attn(F.rms_norm(x, (x.size(-1),)))
        if return_expert_assignments:
            mlp_out, aux, router_entropy, expert_balance, router_value_stats, expert_assignments = self.mlp(
                F.rms_norm(x, (x.size(-1),)), token_idx, return_expert_assignments=True, router_context=router_context
            )
            x = x + mlp_out
            return x, aux, router_entropy, expert_balance, router_value_stats, expert_assignments
        else:
            mlp_out, aux, router_entropy, expert_balance, router_value_stats = self.mlp(
                F.rms_norm(x, (x.size(-1),)), token_idx, router_context=router_context
            )
            x = x + mlp_out
            return x, aux, router_entropy, expert_balance, router_value_stats

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
    router_logit_jitter : float = 0.0
    use_router_temperature : bool = False

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            h = nn.ModuleList([
                Block(config, layer_idx=layer_idx)
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
        per_layer_router_values = []
        for block in self.transformer.h:
            if return_expert_assignments:
                x, aux, router_entropy, expert_balance, router_value_stats, expert_assignments = block(
                    x, idx, return_expert_assignments=True, router_context=router_context
                )
                all_layer_expert_assignments.append(expert_assignments)
            else:
                x, aux, router_entropy, expert_balance, router_value_stats = block(
                    x, idx, router_context=router_context
                )
            total_aux = total_aux + aux
            total_router_entropy = total_router_entropy + router_entropy
            if total_expert_balance is None:
                total_expert_balance = expert_balance
            else:
                total_expert_balance = total_expert_balance + expert_balance
            per_layer_router_entropy.append(router_entropy)
            per_layer_expert_balance.append(expert_balance)
            per_layer_router_values.append(router_value_stats)
        x = F.rms_norm(x, (x.size(-1),))

        # average stats across blocks
        num_blocks = len(self.transformer.h)
        avg_router_entropy = total_router_entropy / num_blocks
        avg_expert_balance = total_expert_balance / num_blocks
        layer_router_entropy = torch.stack(per_layer_router_entropy)
        layer_expert_balance = torch.stack(per_layer_expert_balance, dim=0)
        layer_router_values = {
            key: torch.stack([layer_stats[key] for layer_stats in per_layer_router_values])
            for key in ROUTER_VALUE_KEYS
        }
        avg_router_values = {key: layer_router_values[key].mean() for key in ROUTER_VALUE_KEYS}

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
            return (
                logits,
                loss,
                ce_loss,
                total_aux,
                avg_router_entropy,
                avg_expert_balance,
                layer_router_entropy,
                layer_expert_balance,
                layer_router_values,
                avg_router_values,
                stacked_assignments,
            )
        else:
            return (
                logits,
                loss,
                ce_loss,
                total_aux,
                avg_router_entropy,
                avg_expert_balance,
                layer_router_entropy,
                layer_expert_balance,
                layer_router_values,
                avg_router_values,
            )

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


setup_default_logging()


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
        checkpoint_regex = re.compile(r'state_step(\d+)\.pt')
        match = checkpoint_regex.search(os.path.basename(path))
        return int(match.group(1)) if match else -1

    latest = max(candidates, key=_ckpt_key)
    if _ckpt_key(latest) < 0:
        return None
    return latest


# -----------------------------------------------------------------------------
# int main

# Parse command line arguments and config file  
args, args_text = parse_args()

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
    router_logit_jitter=args.router_logit_jitter,
    use_router_temperature=args.use_router_temperature,
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
adamw_betas = tuple(args.adamw_betas)
adamw_fused = args.adamw_fused
optimizer1 = torch.optim.AdamW([raw_model.transformer.wte.weight], lr=args.lr_embed, betas=adamw_betas, weight_decay=args.weight_decay, fused=adamw_fused)
optimizer2 = torch.optim.AdamW([raw_model.lm_head.weight], lr=args.lr_head, betas=adamw_betas, weight_decay=args.weight_decay, fused=adamw_fused)
router_optimizer = None
router_temperature_optimizer = None

def _collect_router_temperature_params():
    temperature_params = []
    if not raw_model.config.use_router_temperature:
        return temperature_params
    for block in raw_model.transformer.h:
        temp_param = getattr(block.mlp, 'router_temperature_log', None)
        if isinstance(temp_param, nn.Parameter):
            temperature_params.append(temp_param)
    return temperature_params

router_temperature_params = _collect_router_temperature_params()
if router_temperature_params:
    temp_param_ids = {id(p) for p in router_temperature_params}
    all_h_params = [p for p in all_h_params if id(p) not in temp_param_ids]

def _collect_router_params():
    router_params_local = []
    for block in raw_model.transformer.h:
        router_module = getattr(block.mlp, 'router', None)
        if router_module is not None:
            router_params_local.extend(list(router_module.parameters()))
    return router_params_local

if args.only_router_muon:
    router_params = _collect_router_params()
    router_optimizer = Muon(router_params, lr=args.lr_muon, momentum=args.momentum, backend=args.muon_svd_backend, nesterov=args.muon_nesterov, backend_steps=args.muon_backend_steps)
    router_param_ids = {id(p) for p in router_params}
    blocks_params = [p for p in all_h_params if id(p) not in router_param_ids]

    optimizer3 = torch.optim.AdamW(blocks_params, lr=6e-4, betas=adamw_betas, weight_decay=args.weight_decay, fused=adamw_fused)
elif args.use_adamw_opt3:
    optimizer3 = torch.optim.AdamW(all_h_params, lr=6e-4, betas=adamw_betas, weight_decay=args.weight_decay, fused=adamw_fused)
elif args.use_adamw_router:
    router_params = _collect_router_params()
    router_optimizer = torch.optim.AdamW(router_params, lr=args.lr_muon, betas=adamw_betas, weight_decay=args.weight_decay, fused=adamw_fused)
    router_param_ids = {id(p) for p in router_params}
    muon_params = [p for p in all_h_params if id(p) not in router_param_ids]
    
    optimizer3 = Muon(muon_params, lr=args.lr_muon, momentum=args.momentum, backend=args.muon_svd_backend, nesterov=args.muon_nesterov, backend_steps=args.muon_backend_steps)
else:
    optimizer3 = Muon(all_h_params, lr=args.lr_muon, momentum=args.momentum, backend=args.muon_svd_backend, nesterov=args.muon_nesterov, backend_steps=args.muon_backend_steps)
if router_temperature_params:
    router_temperature_optimizer = torch.optim.AdamW(
        router_temperature_params,
        lr=args.lr_muon,
        betas=adamw_betas,
        weight_decay=args.weight_decay,
        fused=adamw_fused,
    )

optimizers = [optimizer1, optimizer2, optimizer3]
if router_optimizer is not None:
    optimizers.append(router_optimizer)
if router_temperature_optimizer is not None:
    optimizers.append(router_temperature_optimizer)
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
    init_wandb(
        args,
        optimizer3,
        train_accumulation_steps,
        val_steps,
        ddp_world_size,
        ddp_rank,
        ddp_local_rank,
    )

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
        val_layer_router_values = init_layer_router_value_tensors(n_layers, device=device)
        val_total_router_values = init_total_router_value_tensors(device=device)
        for _ in range(val_steps):
            x_val, y_val = val_loader.next_batch()
            with torch.no_grad():
                with ctx:
                    (
                        _,
                        loss,
                        ce_loss,
                        total_aux,
                        router_entropy,
                        expert_balance,
                        layer_router_entropy,
                        layer_expert_balance,
                        layer_router_values,
                        total_router_values,
                    ) = model(x_val, y_val, return_logits=False, aux_coeff=args.aux_coeff_val)
                    val_loss += loss.detach()
                    val_ce_loss += ce_loss.detach()
                    val_aux_loss += total_aux.detach()
                    val_router_entropy = val_router_entropy + router_entropy.detach()
                    val_expert_balance = val_expert_balance + expert_balance.detach()
                    val_layer_router_entropy = val_layer_router_entropy + layer_router_entropy.detach()
                    val_layer_expert_balance = val_layer_expert_balance + layer_expert_balance.detach()
                    for key in ROUTER_VALUE_KEYS:
                        val_layer_router_values[key] = val_layer_router_values[key] + layer_router_values[key].detach()
                        val_total_router_values[key] = val_total_router_values[key] + total_router_values[key].detach()
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
        for key in ROUTER_VALUE_KEYS:
            val_layer_router_values[key] = val_layer_router_values[key] / val_steps
            val_total_router_values[key] = val_total_router_values[key] / val_steps
        dist.all_reduce(val_router_entropy, op=dist.ReduceOp.AVG)
        dist.all_reduce(val_expert_balance, op=dist.ReduceOp.AVG)
        dist.all_reduce(val_layer_router_entropy, op=dist.ReduceOp.AVG)
        dist.all_reduce(val_layer_expert_balance, op=dist.ReduceOp.AVG)
        for key in ROUTER_VALUE_KEYS:
            dist.all_reduce(val_layer_router_values[key], op=dist.ReduceOp.AVG)
            dist.all_reduce(val_total_router_values[key], op=dist.ReduceOp.AVG)
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
            _, loss_ce, ce_loss_probe, total_aux_probe, _, _, _, _, _, _ = model(
                x_probe, y_probe, return_logits=False, aux_coeff=0.0
            )
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
            _, _, _, total_aux_probe, _, _, _, _, _, _ = model(x_probe, y_probe, return_logits=False, aux_coeff=0.0)
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
                    _, _, _, _, _, _, _, _, _, _, current_assignments = model(
                        tracking_x, return_logits=False, aux_coeff=0.0, return_expert_assignments=True
                    )
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
        
        # log to wandb
        if master_process:
            wandb_val_log(
                step,
                val_loss,
                val_ce_loss,
                val_aux_loss,
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
                ROUTER_VALUE_KEYS,
            )

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
    layer_router_values_sum = init_layer_router_value_tensors(n_layers, device=device)
    total_router_values_sum = init_total_router_value_tensors(device=device)
    for i in range(1, train_accumulation_steps+1):
        if use_global_lb:
            x_batch, y_batch = cached_batches[i-1]
        else:
            x_batch, y_batch = x, y
        # forward pass
        with ctx:
            (
                _,
                loss,
                ce_loss,
                total_aux,
                router_entropy,
                expert_balance,
                layer_router_entropy,
                layer_expert_balance,
                layer_router_values,
                total_router_values,
            ) = model(
                x_batch, y_batch, return_logits=False, aux_coeff=args.aux_coeff_train, router_context=router_context_use
            )
            train_loss = loss.detach()
            router_entropy_sum = router_entropy_sum + router_entropy.detach()
            expert_balance_sum = expert_balance_sum + expert_balance.detach()
            layer_router_entropy_sum = layer_router_entropy_sum + layer_router_entropy.detach()
            layer_expert_balance_sum = layer_expert_balance_sum + layer_expert_balance.detach()
            for key in ROUTER_VALUE_KEYS:
                layer_router_values_sum[key] = layer_router_values_sum[key] + layer_router_values[key].detach()
                total_router_values_sum[key] = total_router_values_sum[key] + total_router_values[key].detach()
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
    layer_router_values_avg = {
        key: layer_router_values_sum[key] / train_accumulation_steps for key in ROUTER_VALUE_KEYS
    }
    total_router_values_avg = {
        key: total_router_values_sum[key] / train_accumulation_steps for key in ROUTER_VALUE_KEYS
    }
    dist.all_reduce(router_entropy_avg, op=dist.ReduceOp.AVG)
    dist.all_reduce(expert_balance_avg, op=dist.ReduceOp.AVG)
    dist.all_reduce(layer_router_entropy_avg, op=dist.ReduceOp.AVG)
    dist.all_reduce(layer_expert_balance_avg, op=dist.ReduceOp.AVG)
    for key in ROUTER_VALUE_KEYS:
        dist.all_reduce(layer_router_values_avg[key], op=dist.ReduceOp.AVG)
        dist.all_reduce(total_router_values_avg[key], op=dist.ReduceOp.AVG)

    # step the optimizers and schedulers
    for opt, sched in zip(optimizers, schedulers):
        opt.step()
        sched.step()

    # null the gradients
    model.zero_grad(set_to_none=True)
    raw_model.finalize_loss_free_updates()
    # --------------- TRAINING SECTION END -------------------

    if master_process:
        approx_time = training_time_ms + 1000 * (time.time() - t0)
        logging.info(f"step:{step+1}/{args.num_iterations} train_loss:{train_loss.item():.4f} train_time:{approx_time:.0f}ms step_avg:{approx_time/timed_steps:.2f}ms")
        with open(logfile, "a") as f:
            f.write(f"step:{step+1}/{args.num_iterations} train_loss:{train_loss.item():.4f} train_time:{approx_time:.0f}ms step_avg:{approx_time/timed_steps:.2f}ms\n")
        # wandb logging
        # capture the current learning rates after scheduler updates
        current_embed_lr = optimizers[0].param_groups[0]['lr']
        current_head_lr = optimizers[1].param_groups[0]['lr']
        current_blocks_lr = optimizers[2].param_groups[0]['lr']
        current_router_lr = router_optimizer.param_groups[0]['lr'] if router_optimizer is not None else None
        
        wandb_train_log(
            step + 1,
            train_loss,
            router_entropy_avg,
            grad_norm,
            approx_time,
            timed_steps,
            current_embed_lr,
            current_head_lr,
            current_blocks_lr,
            current_router_lr,
            layer_expert_balance_avg,
            layer_router_entropy_avg,
            expert_balance_avg,
            layer_router_values_avg,
            total_router_values_avg,
            raw_model,
            num_experts,
            ROUTER_VALUE_KEYS,
        )

if master_process:
    logging.info(f"peak memory consumption: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB")
    try:
        wandb.finish()
    except Exception:
        pass

# -------------------------------------------------------------------------
# clean up nice
dist.destroy_process_group()
