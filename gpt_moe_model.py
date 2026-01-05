import math
from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist


ROUTER_VALUE_KEYS = (
    'top1_logit',
    'top2_logit',
    'logit_diff',
    'top1_coef',
    'top2_coef',
    'coef_diff',
    'max_scaler',
    'min_scaler'
)


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
            self.cos_cached = freqs.cos() #.bfloat16()
            self.sin_cached = freqs.sin() #.bfloat16()
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
        hidden_dim = int(round(config.n_embd * config.hidden_dim_scale_factor))
        self.c_fc    = nn.Linear(config.n_embd, hidden_dim, bias=False)
        self.c_proj  = nn.Linear(hidden_dim, config.n_embd, bias=False)
        self.c_proj.weight.data.zero_() # zero init suggested by @Grad62304977

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square() # https://arxiv.org/abs/2109.08668v2; ~1-2% better than GELU; suggested by @SKYLINEZ007 and @Grad62304977
        x = self.c_proj(x)
        return x


def switch_topk(logits, k, null_expert_bias=0.0, weight_logits=None, probs_override=None):
    """Switch/Top-k. Returns (indices, probs, expert output weights)."""
    probs_from_logits = logits.softmax(dim=-1)
    probs = probs_override if probs_override is not None else probs_from_logits
    gate_vals, topk_idx = torch.topk(probs_from_logits, k, dim=-1)
    if weight_logits is None:
        gate = gate_vals / (gate_vals.sum(dim=-1, keepdim=True) + null_expert_bias)
    else:
        topk_logits = torch.gather(weight_logits, dim=-1, index=topk_idx)
        gate = F.softmax(topk_logits, dim=-1)
    # Routed probabilities reflect the actual mixture used in the forward pass
    routed_probs = torch.zeros_like(probs)
    routed_probs.scatter_(dim=-1, index=topk_idx, src=gate)
    return topk_idx, routed_probs, gate


def hash_select(token_ids, num_experts, null_expert_bias=0.0):
    expert_idx = (token_ids[..., None].float() % num_experts).to(token_ids.dtype)
    routing_weights = torch.nn.functional.one_hot(expert_idx, num_classes=num_experts)
    selected_prob, selected_expert = torch.max(routing_weights, dim=-1, keepdim=True)
    expert_mask = torch.nn.functional.one_hot(torch.argmax(routing_weights, dim=-1), num_classes=num_experts)
    if null_expert_bias > 0:
        selected_prob = selected_prob / (selected_prob + null_expert_bias)
    return selected_expert, routing_weights, selected_prob


def diff_routing(logits, k):
    z_max = torch.max(logits, dim=-1, keepdim=True).values
    exp_z = torch.exp(logits - z_max)
    num_experts = exp_z.shape[-1]
    kth_smallest_idx = num_experts - k
    m_exp_z = torch.kthvalue(exp_z, k=kth_smallest_idx, dim=-1, keepdim=True).values
    topk_exp_z = F.relu(exp_z - m_exp_z)
    topk_weights = topk_exp_z / (topk_exp_z.sum(dim=-1, keepdim=True) + 1e-8)
    # Get indices of top-k for compatibility
    gate, topk_idx = torch.topk(topk_weights, k, dim=-1)
    return topk_idx, topk_weights, gate


def diff_no_softmax(logits, k):
    num_experts = logits.shape[-1]
    kth_smallest_idx = num_experts - k
    m = torch.kthvalue(logits, k=kth_smallest_idx, dim=-1, keepdim=True).values
    topk = F.relu(logits - m)
    topk_weights = topk / (topk.sum(dim=-1, keepdim=True) + 1e-8)
    gate, topk_idx = torch.topk(topk_weights, k, dim=-1)
    return topk_idx, topk_weights, gate


def scaled_diff_no_softmax(logits, k, scalers):
    exp_scalers = torch.exp(scalers)
    num_experts = logits.shape[-1]
    kth_smallest_idx = num_experts - k
    m = torch.kthvalue(logits, k=kth_smallest_idx, dim=-1, keepdim=True).values
    topk = F.relu(logits - m)
    scaled_topk = exp_scalers * topk
    topk_weights = scaled_topk / (scaled_topk.sum(dim=-1, keepdim=True) + 1e-8)
    gate, topk_idx = torch.topk(topk_weights, k, dim=-1)
    return topk_idx, topk_weights, gate


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
        self.loss_free_strength = config.loss_free_strength
        self.loss_free_update_rate = config.loss_free_update_rate
        self.router_logit_jitter = config.router_logit_jitter
        self.use_router_temperature = config.use_router_temperature
        self.aux_use_routed_prob = config.aux_use_routed_prob
        self.diff_topk_reg_fp32 = config.diff_topk_reg_fp32
        self.diff_topk_reg_enabled = config.diff_topk_reg_max_coeff > 0.0
        self.theta_load_balance_coeff = config.theta_load_balance_coeff
        self.theta_lb_detach_theta = config.theta_lb_detach_theta
        self.theta_lb_detach_logits = config.theta_lb_detach_logits
        if self.use_router_temperature:
            self.router_temperature_log = nn.Parameter(torch.zeros(1, dtype=torch.float32))
        else:
            self.register_parameter('router_temperature_log', None)
        if self.theta_load_balance_coeff > 0.0:
            self.load_balance_theta = nn.Parameter(torch.zeros(self.num_experts, dtype=torch.float32))
        else:
            self.register_parameter('load_balance_theta', None)

        assert self.router_type in ('hash', 'switch', 'diff', 'diff_no_softmax', 'scaled_diff_no_softmax')
        assert self.router_activation in ('gelu', 'relu', 'relu_squared')
        assert self.loss_free_mode in ('none', 'deepseek', 'stopgrad')
        assert self.router_logit_jitter >= 0.0, "router_logit_jitter must be non-negative"
        if self.router_type == 'hash' and self.loss_free_mode != 'none':
            raise ValueError("Loss-free load balancing requires a learned router (switch or diff).")
        if self.loss_free_mode == 'deepseek' and self.router_type != 'switch':
            raise ValueError("DeepSeek-style loss-free routing is only defined for switch routers.")

        self.experts = nn.ModuleList([MLP(config) for _ in range(self.num_experts)])
        init_frac = torch.full((self.num_experts,), 1.0 / self.num_experts, dtype=torch.float32)
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
        
        if self.router_type == 'scaled_diff_no_softmax':
            self.router_scaler = nn.Linear(config.n_embd, self.num_experts, bias=False)
            with torch.no_grad():
                self.router_scaler.weight.zero_()

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
        bias_vec = self.loss_free_bias_state * self.loss_free_strength
        return bias_vec - bias_vec.mean()

    def _loss_free_bias(self, logits):
        bias_vec = self._loss_free_bias_vector()
        if bias_vec is None:
            return None
        bias_vec = bias_vec.to(dtype=logits.dtype, device=logits.device)
        view_shape = [1] * (logits.dim() - 1) + [self.num_experts]
        return bias_vec.view(*view_shape)

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

    def _compute_diff_topk_regularizer(self, logits):
        if not self.diff_topk_reg_enabled:
            return torch.tensor(0.0, device=logits.device, dtype=torch.float32)
        
        if self.diff_topk_reg_fp32:
            logits = logits.float()
        
        num_experts = logits.size(-1)
        kth_smallest_idx = num_experts - self.top_k
        z_max = logits.max(dim=-1, keepdim=True).values
        exp_z = torch.exp(logits - z_max)
        m_exp_z = torch.kthvalue(exp_z, k=kth_smallest_idx, dim=-1, keepdim=True).values
        topk_exp_z = torch.relu(exp_z - m_exp_z)
        trunc_mass = topk_exp_z.sum(dim=-1)
        total_mass = exp_z.sum(dim=-1)
        ratio = trunc_mass / (total_mass + 1e-8)
        reg = -torch.log(torch.clamp(ratio, min=1e-8))
        return reg.mean()
    
    def _get_detached_lb_theta(self):
        if self.load_balance_theta is None:
            return torch.zeros(self.num_experts, device=self.load_balance_theta.device, dtype=torch.float32)
        if self.theta_lb_detach_theta:
            return self.load_balance_theta.detach()
        return self.load_balance_theta

    def _compute_theta_load_balance_loss(self, logits):
        num_experts = logits.size(-1)

        # Detach router logits so gradients flow only into theta.
        logits_detached = logits.detach() if self.theta_lb_detach_logits else logits
        reshaped_logits = logits_detached.reshape(-1, num_experts).float()
        
        x_tilde = reshaped_logits + self.load_balance_theta
        kth_smallest_idx = num_experts - self.top_k
        m_xt = torch.kthvalue(x_tilde, k=kth_smallest_idx, dim=1).values.unsqueeze(1)
        at_top = torch.relu(x_tilde - m_xt)
        at_bottom = torch.relu(m_xt - x_tilde)
        scale = (num_experts - self.top_k) / self.top_k
        balanced = at_top * scale + at_bottom
        
        norm_factor = reshaped_logits.shape[0] * num_experts
        return balanced.sum() / float(norm_factor)

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
            self._update_sign_bias(frac)
        self._reset_loss_free_accumulators()
        
    def run_hash_routing(self, token_idx, x):
        topk_idx, probs, gate = hash_select(token_idx, self.num_experts)
        gate = gate.to(x.dtype)
        probs = probs.to(x.dtype)
        B, T, C = x.shape
        BT      = B * T
        x_flat  = x.reshape(BT, C)
        y_flat  = torch.zeros_like(x_flat)

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
        return y, None, None, None, None, None, None

    def forward(self, x, token_idx=None, return_expert_assignments=False, router_context=None):
        if self.router_type == "hash":
            return self.run_hash_routing(token_idx, x)

        ctx_mode = router_context.get('mode', None) if router_context is not None else None
        logits_for_selection = None
        theta_lb_loss = torch.tensor(0.0, device=x.device, dtype=torch.float32)

        logits = self.router(x)
        if self.training and self.router_logit_jitter > 0.0:
            noise = torch.empty_like(logits).uniform_(
                1.0 - self.router_logit_jitter,
                1.0 + self.router_logit_jitter,
            )
            logits = logits * noise
        bias = self._loss_free_bias(logits)
        if self.loss_free_mode == 'deepseek':
            logits_for_selection = logits + bias
        elif self.loss_free_mode == 'stopgrad':
            bias_detached = bias.detach()
            logits_for_selection = logits + bias_detached
            logits = logits_for_selection
        else:
            logits_for_selection = logits
        if self.use_router_temperature:
            temperature = self._get_router_temperature(logits_for_selection)
            logits_for_selection = logits_for_selection / temperature
            logits = logits / temperature
        
        if self.load_balance_theta is None:
            theta_lb_loss = torch.tensor(0.0, device=logits.device, dtype=torch.float32)
        else:
            theta_lb_loss = self._compute_theta_load_balance_loss(logits)
            lb_theta = self._get_detached_lb_theta()
            logits_for_selection = logits_for_selection + lb_theta
            logits = logits_for_selection

        if self.router_type == "switch":
            router_probs = logits.softmax(dim=-1)
            topk_idx, routed_probs_for_aux, gate = switch_topk(
                logits_for_selection,
                self.top_k,
                weight_logits=logits,
                probs_override=router_probs,
            )
            probs = router_probs
        elif self.router_type == "diff":
            topk_idx, routed_probs_for_aux, gate = diff_routing(logits_for_selection, self.top_k)
            probs = logits.softmax(dim=-1)
        elif self.router_type == "diff_no_softmax":
            topk_idx, routed_probs_for_aux, gate = diff_no_softmax(logits_for_selection, self.top_k)
            probs = logits.softmax(dim=-1)
        elif self.router_type == 'scaled_diff_no_softmax':
            scalers = self.router_scaler(x)
            topk_idx, routed_probs_for_aux, gate = scaled_diff_no_softmax(logits_for_selection, self.top_k, scalers)
            probs = logits.softmax(dim=-1)
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

            logits_flat = logits_for_selection.reshape(BT, self.num_experts).float()
            num_top_logits = 2 if logits_flat.size(1) >= 2 else 1
            top_logits = torch.topk(logits_flat, k=num_top_logits, dim=-1).values
            top1_logits_vals = top_logits[:, 0]
            if num_top_logits == 2:
                top2_logits_vals = top_logits[:, 1]
            else:
                top2_logits_vals = torch.zeros_like(top1_logits_vals)

            router_value_stats = {
                'top1_logit': top1_logits_vals.mean(),
                'top2_logit': top2_logits_vals.mean(),
                'logit_diff': (top1_logits_vals - top2_logits_vals).mean(),
                'top1_coef': top1_coef_vals.mean(),
                'top2_coef': top2_coef_vals.mean(),
                'coef_diff': (top1_coef_vals - top2_coef_vals).mean(),
                'max_scaler': torch.tensor(0, dtype=torch.float, device=x.device),
                'min_scaler': torch.tensor(0, dtype=torch.float, device=x.device)
            }
            
            if self.router_type == 'scaled_diff_no_softmax':
                exp_scalers = torch.exp(scalers)
                router_value_stats['max_scaler'] = exp_scalers.max()
                router_value_stats['min_scaler'] = exp_scalers.min()

        diff_topk_reg = self._compute_diff_topk_regularizer(logits_for_selection)

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
            if self.aux_use_routed_prob:
                probs_full = routed_probs_for_aux.reshape(-1, self.num_experts)
            else:
                probs_full = logits.softmax(dim=-1).reshape(-1, self.num_experts)
            frac_for_aux = global_frac_override if global_frac_override is not None else frac
            probs_mean = probs_full.mean(0)
            aux = self.num_experts * (frac_for_aux * probs_mean).sum()

        if return_expert_assignments:
            return (
                y,
                aux,
                router_entropy,
                frac,
                router_value_stats,
                diff_topk_reg,
                theta_lb_loss,
                topk_idx.view_as(x[:, :, :self.top_k]),
            )
        else:
            return y, aux, router_entropy, frac, router_value_stats, diff_topk_reg, theta_lb_loss


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
            (
                mlp_out,
                aux,
                router_entropy,
                expert_balance,
                router_value_stats,
                diff_topk_reg,
                theta_lb_loss,
                expert_assignments,
            ) = self.mlp(
                F.rms_norm(x, (x.size(-1),)), token_idx, return_expert_assignments=True, router_context=router_context
            )
            x = x + mlp_out
            return (
                x,
                aux,
                router_entropy,
                expert_balance,
                router_value_stats,
                diff_topk_reg,
                theta_lb_loss,
                expert_assignments,
            )
        else:
            mlp_out, aux, router_entropy, expert_balance, router_value_stats, diff_topk_reg, theta_lb_loss = self.mlp(
                F.rms_norm(x, (x.size(-1),)), token_idx, router_context=router_context
            )
            x = x + mlp_out
            return x, aux, router_entropy, expert_balance, router_value_stats, diff_topk_reg, theta_lb_loss


@dataclass
class GPTConfig:
    vocab_size : int = 50304
    n_layer : int = 12
    n_head : int = 6 # head dim 128 suggested by @Grad62304977
    n_embd : int = 768
    hidden_dim_scale_factor : float = 4.0
    num_experts : int = 8
    top_k : int = 2
    router_type : str = 'diff'
    router_depth : int = 1
    router_activation : str = 'gelu'
    global_load_balance : bool = False
    aux_use_routed_prob : bool = False
    loss_free_mode : str = 'none'
    loss_free_strength : float = 1.0
    loss_free_update_rate : float = 0.001
    router_logit_jitter : float = 0.0
    use_router_temperature : bool = False
    diff_topk_reg_max_coeff : float = 0.0
    diff_topk_reg_fp32 : bool = False
    theta_load_balance_coeff : float = 0.0
    theta_lb_detach_theta : bool = True
    theta_lb_detach_logits : bool = True


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

    def forward(
        self,
        idx,
        targets=None,
        return_logits=True,
        aux_coeff=0.0,
        diff_topk_reg_coeff=0.0,
        return_expert_assignments=False,
        router_context=None,
    ):

        # forward the GPT model itself
        x = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        x = F.rms_norm(x, (x.size(-1),))
        total_aux = 0
        total_diff_topk_reg = torch.tensor(0.0, device=x.device, dtype=torch.float32)
        total_theta_lb = torch.tensor(0.0, device=x.device, dtype=torch.float32)
        total_router_entropy = 0
        total_expert_balance = None
        per_layer_router_entropy = []
        per_layer_expert_balance = []
        all_layer_expert_assignments = []
        per_layer_router_values = []
        for block in self.transformer.h:
            if return_expert_assignments:
                (
                    x,
                    aux,
                    router_entropy,
                    expert_balance,
                    router_value_stats,
                    diff_topk_reg,
                    theta_lb_loss,
                    expert_assignments,
                ) = block(
                    x, idx, return_expert_assignments=True, router_context=router_context
                )
                all_layer_expert_assignments.append(expert_assignments)
            else:
                (
                    x,
                    aux,
                    router_entropy,
                    expert_balance,
                    router_value_stats,
                    diff_topk_reg,
                    theta_lb_loss,
                ) = block(
                    x, idx, router_context=router_context
                )
            total_aux = total_aux + aux
            total_diff_topk_reg = total_diff_topk_reg + diff_topk_reg
            total_theta_lb = total_theta_lb + theta_lb_loss
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
            theta_coeff = getattr(self.config, 'theta_load_balance_coeff', 0.0)
            loss = (
                ce_loss
                + total_aux * aux_coeff
                + total_diff_topk_reg * diff_topk_reg_coeff
                + total_theta_lb * theta_coeff
            )
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
                total_diff_topk_reg,
                total_theta_lb,
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
                total_diff_topk_reg,
                total_theta_lb,
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
