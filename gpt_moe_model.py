import math
from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist
import torch._dynamo as dynamo


ROUTER_VALUE_KEYS = (
    'top1_logit',
    'top2_logit',
    'logit_diff',
    'top1_coef',
    'top2_coef',
    'coef_diff',
    'ste_active_token_frac',
    'ste_extra_experts_per_token',
    'ste_boundary_gap',
    'ste_support_prob_mass',
    'ste_lb_active_token_frac',
    'ste_lb_extra_experts_per_token',
    'ste_lb_boundary_gap',
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
        self.qk_clip_tau = getattr(config, "qk_clip_tau", 0.0)
        self.qk_clip_block_size = getattr(config, "qk_clip_block_size", 128)
        self.log_attn_logits = getattr(config, "log_attn_logits", False)
        self.register_buffer("qk_clip_max", torch.zeros(self.n_head, dtype=torch.float32), persistent=False)
        self.register_buffer("attn_logit_max", torch.zeros(self.n_head, dtype=torch.float32), persistent=False)

    def _compute_qk_clip_max(self, q, k):
        q = q.float().transpose(1, 2)  # B, H, T, D
        k = k.float().transpose(1, 2)  # B, H, T, D
        _, _, seq_len, head_dim = q.shape
        scale = 1.0 / math.sqrt(head_dim)
        block = self.qk_clip_block_size if self.qk_clip_block_size > 0 else seq_len
        k_t = k.transpose(-1, -2)
        max_per_head = None
        for start in range(0, seq_len, block):
            q_chunk = q[:, :, start:start + block, :]
            scores = torch.matmul(q_chunk, k_t) * scale
            chunk_max = scores.amax(dim=-1).amax(dim=-1)
            max_per_head = chunk_max if max_per_head is None else torch.maximum(max_per_head, chunk_max)
        if max_per_head is None:
            return None
        return max_per_head.amax(dim=0)

    @dynamo.disable
    def _update_qk_clip_max(self, q, k):
        if self.qk_clip_tau <= 0.0 or not self.training:
            return
        with torch.no_grad():
            qk_max = self._compute_qk_clip_max(q.detach(), k.detach())
            if qk_max is None:
                return
            self.qk_clip_max.copy_(torch.maximum(self.qk_clip_max, qk_max))

    def _update_attn_logit_max(self, q, k):
        if not self.log_attn_logits or not self.training:
            return
        with torch.no_grad():
            qk_max = self._compute_qk_clip_max(q.detach(), k.detach())
            if qk_max is None:
                return
            self.attn_logit_max.copy_(torch.maximum(self.attn_logit_max, qk_max))

    def apply_qk_clip(self, head_scale):
        if head_scale is None or head_scale.numel() != self.n_head:
            return
        with torch.no_grad():
            scale = head_scale.to(dtype=self.c_q.weight.dtype, device=self.c_q.weight.device).view(self.n_head, 1, 1)
            self.c_q.weight.view(self.n_head, self.head_dim, self.n_embd).mul_(scale)
            self.c_k.weight.view(self.n_head, self.head_dim, self.n_embd).mul_(scale)

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_head, self.head_dim)
        cos, sin = self.rotary(q)
        q, k = F.rms_norm(q, (q.size(-1),)), F.rms_norm(k, (k.size(-1),)) # QK norm suggested by @Grad62304977
        q = Rotary.apply_rotary_emb(q, cos, sin)
        k = Rotary.apply_rotary_emb(k, cos, sin)
        if self.qk_clip_tau > 0.0 and self.training:
            self._update_qk_clip_max(q, k)
        if self.log_attn_logits and self.training:
            self._update_attn_logit_max(q, k)
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


def _apply_switch_activation(values, activation, dim=-1):
    if activation == 'softmax':
        return F.softmax(values, dim=dim)
    if activation == 'sigmoid':
        return torch.sigmoid(values)
    raise ValueError(f"Unsupported top-k activation: {activation}")


def _normalize_gate(weights, eps=1e-9):
    denom = torch.clamp(weights.sum(dim=-1, keepdim=True), min=eps)
    return weights / denom


def _maxvio_from_load(load_frac):
    expected_frac = load_frac.new_tensor(1.0 / load_frac.numel())
    return (torch.amax(load_frac) - expected_frac) / expected_frac


class RectIndicatorSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, margin, bandwidth):
        ctx.bandwidth = float(bandwidth)
        ctx.save_for_backward(margin)
        return (margin >= 0).to(dtype=margin.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        (margin,) = ctx.saved_tensors
        half_width = ctx.bandwidth * 0.5
        window = (margin.float().abs() < half_width).to(dtype=torch.float32)
        grad_margin = grad_output.float() * window / ctx.bandwidth
        return grad_margin.to(dtype=margin.dtype), None


def _hard_topk_mask(reference, topk_idx):
    mask = torch.zeros_like(reference)
    mask.scatter_(dim=-1, index=topk_idx, value=1.0)
    return mask


def _ste_support_mask(logits, topk_idx, bandwidth):
    threshold = torch.gather(logits, dim=-1, index=topk_idx).min(dim=-1, keepdim=True).values
    return (logits - threshold).abs() < (bandwidth * 0.5)


def _topk_threshold(logits, topk_idx):
    return torch.gather(logits, dim=-1, index=topk_idx).min(dim=-1, keepdim=True).values


def _switch_topk_common(logits, topk_idx, weight_logits=None, activation='softmax', mask=None):
    weight_probs = _apply_switch_activation(weight_logits, activation) if weight_logits is not None else _apply_switch_activation(logits, activation)
    selected_weights = weight_probs * mask.to(weight_probs.dtype)
    routed_probs = _normalize_gate(selected_weights)
    gate = torch.gather(routed_probs, dim=-1, index=topk_idx)
    return topk_idx, routed_probs, gate


def _soft_support_routed_probs(logits, topk_idx, weight_logits=None, activation='softmax', bandwidth=0.0):
    weight_probs = _apply_switch_activation(weight_logits, activation) if weight_logits is not None else _apply_switch_activation(logits, activation)
    threshold = _topk_threshold(logits, topk_idx)
    soft_mask = ((logits - threshold).abs() < (bandwidth * 0.5)).to(dtype=weight_probs.dtype)
    return _normalize_gate(weight_probs * soft_mask)


def switch_topk(logits, k, weight_logits=None, activation='softmax', ste_width=0.0):
    """Switch/Top-k with optional rectangular STE around the top-k boundary."""
    probs_from_logits = _apply_switch_activation(logits, activation)
    topk_idx = torch.topk(probs_from_logits, k, dim=-1).indices
    hard_mask = _hard_topk_mask(logits, topk_idx)

    if 0.0 < ste_width:
        threshold = _topk_threshold(logits, topk_idx)
        margin = logits - threshold
        soft_mask = RectIndicatorSTE.apply(margin, ste_width)
        mask = hard_mask + soft_mask - soft_mask.detach()
    else:
        mask = hard_mask

    return _switch_topk_common(
        logits,
        topk_idx,
        weight_logits=weight_logits,
        activation=activation,
        mask=mask,
    )


def hash_select(token_ids, num_experts):
    expert_idx = (token_ids[..., None].float() % num_experts).to(token_ids.dtype)
    routing_weights = torch.nn.functional.one_hot(expert_idx, num_classes=num_experts)
    selected_prob, selected_expert = torch.max(routing_weights, dim=-1, keepdim=True)
    expert_mask = torch.nn.functional.one_hot(torch.argmax(routing_weights, dim=-1), num_classes=num_experts)
    return selected_expert, routing_weights, selected_prob


def diff_routing(logits, k, activation='softmax'):
    num_experts = logits.shape[-1]
    kth_smallest_idx = num_experts - k
    
    if activation == 'softmax':
        z_max = torch.max(logits, dim=-1, keepdim=True).values
        exp_z = torch.exp(logits - z_max)
        m_exp_z = torch.kthvalue(exp_z, k=kth_smallest_idx, dim=-1, keepdim=True).values
        topk_exp_z = F.relu(exp_z - m_exp_z)
        activated = topk_exp_z
    elif activation == 'sigmoid':
        m_logits = torch.kthvalue(logits, k=kth_smallest_idx, dim=-1, keepdim=True).values
        activated = F.relu(2 * torch.sigmoid(logits - m_logits) - 1)
    else:
        raise ValueError(f"Unsupported top-k activation for diff routing: {activation}")
    topk_weights = activated / (activated.sum(dim=-1, keepdim=True) + 1e-8)
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

def diff_no_softmax_to_switch_transistion(
    logits,
    k,
    diff_weight,
    weight_logits=None,
    topk_activation='softmax',
    ste_width=0.0,
):
    if diff_weight == 0:
        return switch_topk(
            logits,
            k,
            weight_logits=weight_logits,
            activation=topk_activation,
            ste_width=ste_width,
        )
    if diff_weight == 1:
        return diff_no_softmax(logits, k)

    topk_idx, switch_routed_probs, switch_gate = switch_topk(
        logits,
        k,
        weight_logits=weight_logits,
        activation=topk_activation,
        ste_width=ste_width,
    )
    topk_idx, diff_routed_probs, diff_gate = diff_no_softmax(logits, k)
    routed_probs = diff_weight * diff_routed_probs + (1 - diff_weight) * switch_routed_probs
    gate = diff_weight * diff_gate + (1 - diff_weight) * switch_gate
    return topk_idx, routed_probs, gate


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

class GLUVariant(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, variant):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.variant = variant

        act_dict = {
            'swiglu': F.silu,
            'geglu': F.gelu,
            'reglu': F.relu
        }
        
        self.w_act = nn.Linear(self.input_dim, self.hidden_dim, bias=False)
        self.w_lin = nn.Linear(self.input_dim, self.hidden_dim, bias=False)
        self.w_out = nn.Linear(self.hidden_dim, self.output_dim, bias=False)
        self.act = act_dict[variant]
    
    def forward(self, x):
        act = self.act(self.w_act(x))
        lin = self.w_lin(x)
        return self.w_out(act * lin)

class MoE(nn.Module):
    def __init__(self, config, layer_idx=0):
        super().__init__()
        self.num_experts = config.num_experts
        self.top_k = config.top_k
        assert 1 <= self.top_k <= self.num_experts, "`k` must be in [1, #experts]"
        self.router_type = config.router_type
        self.router_depth = config.router_depth
        self.router_layer_type = config.router_layer_type
        self.router_activation = config.router_activation
        self.topk_activation = config.topk_activation
        self.topk_ste_width = config.topk_ste_width
        self.load_balance_ste_width = config.load_balance_ste_width
        self.global_load_balance = config.global_load_balance
        self.layer_idx = layer_idx
        self.loss_free_mode = config.loss_free_mode
        self.loss_free_strength = config.loss_free_strength
        self.loss_free_update_rate = config.loss_free_update_rate
        self.router_logit_jitter = config.router_logit_jitter
        self.use_router_temperature = config.use_router_temperature
        self.aux_use_routed_prob = config.aux_use_routed_prob
        self.maxvio_load_balance = config.maxvio_load_balance
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
        assert self.topk_activation  in ('softmax', 'sigmoid'), "Unsupported top-k activation: {self.topk_activation}"
        assert self.router_logit_jitter >= 0.0, "router_logit_jitter must be non-negative"
        assert self.load_balance_ste_width >= 0.0, "load_balance_ste_width must be non-negative"
        if self.maxvio_load_balance and self.theta_load_balance_coeff > 0.0:
            raise ValueError("Direct MaxVio load balancing cannot be combined with theta load balancing.")
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
            if self.router_layer_type is None or self.router_layer_type == 'linear':
                layers = []
                for _ in range(self.router_depth - 1):
                    layers.append(nn.Linear(config.n_embd, config.n_embd, bias=False))
                    layers.append(self._build_router_activation())
                layers.append(nn.Linear(config.n_embd, self.num_experts, bias=False))
                self.router = nn.Sequential(*layers)
            else:
                router_hidden_dim = self.num_experts
                self.router = nn.Sequential(GLUVariant(
                    input_dim=config.n_embd, 
                    hidden_dim=router_hidden_dim,
                    output_dim=self.num_experts,
                    variant=self.router_layer_type))
                
        else:
            self.router = None
        
        if self.router_type == 'scaled_diff_no_softmax':
            self.router_scaler = nn.Linear(config.n_embd, self.num_experts, bias=False)
            with torch.no_grad():
                self.router_scaler.weight.zero_()

    def _get_router_temperature(self, reference_tensor):
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

    def forward(self, x, token_idx=None, return_expert_assignments=False, router_context=None, diff_weight=1):
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
            topk_idx, routed_probs_for_aux, gate = switch_topk(
                logits_for_selection,
                self.top_k,
                weight_logits=logits,
                activation=self.topk_activation,
                ste_width=self.topk_ste_width,
            )
            
            unnorm_probs = _apply_switch_activation(logits_for_selection, self.topk_activation)
            probs = _normalize_gate(unnorm_probs)
            # probs = logits.softmax(dim=-1)
            
        elif self.router_type == "diff":
            topk_idx, routed_probs_for_aux, gate = diff_routing(
                logits_for_selection,
                self.top_k,
                activation=self.topk_activation,
            )
            probs = logits.softmax(dim=-1)
        elif self.router_type == "diff_no_softmax":
            # topk_idx, routed_probs_for_aux, gate = diff_no_softmax(logits_for_selection, self.top_k)
            topk_idx, routed_probs_for_aux, gate = diff_no_softmax_to_switch_transistion(
                logits_for_selection,
                self.top_k,
                diff_weight=diff_weight,
                topk_activation=self.topk_activation,
                ste_width=self.topk_ste_width,
            )
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
        router_ste_y_flat = torch.zeros_like(x_flat)

        probs_flat = probs.reshape(BT, -1)
        routed_probs_flat = routed_probs_for_aux.reshape(BT, self.num_experts)
        gate_flat = gate.reshape(BT, self.top_k)  # Now always (BT, k)
        idx_flat = topk_idx.reshape(BT, self.top_k)
        logits_flat_for_stats = logits_for_selection.reshape(BT, self.num_experts).float()
        hard_mask_flat = None
        extra_support_flat = None
        lb_extra_support_flat = None
        ste_soft_routed_probs_flat = None
        if self.topk_ste_width > 0.0 or self.load_balance_ste_width > 0.0:
            hard_mask_flat = _hard_topk_mask(logits_flat_for_stats, idx_flat).bool()
        if self.topk_ste_width > 0.0:
            support_mask_flat = _ste_support_mask(logits_flat_for_stats, idx_flat, self.topk_ste_width)
            extra_support_flat = support_mask_flat & ~hard_mask_flat
            ste_soft_routed_probs_flat = _soft_support_routed_probs(
                logits_for_selection.reshape(BT, self.num_experts),
                idx_flat,
                weight_logits=logits.reshape(BT, self.num_experts),
                activation=self.topk_activation,
                bandwidth=self.topk_ste_width,
            )
        if self.load_balance_ste_width > 0.0:
            lb_support_mask_flat = _ste_support_mask(logits_flat_for_stats, idx_flat, self.load_balance_ste_width)
            lb_extra_support_flat = lb_support_mask_flat & ~hard_mask_flat

        for expert_id in range(self.num_experts):
            sel_mask = (idx_flat == expert_id)
            token_rows, which_k = torch.nonzero(sel_mask, as_tuple=True)
            inp   = x_flat.index_select(0, token_rows)
            out   = self.experts[expert_id](inp)
            coeff = gate_flat[token_rows, which_k].unsqueeze(1)
            y_flat.index_add_(0, token_rows, out * coeff)
            if extra_support_flat is not None:
                extra_rows = torch.nonzero(extra_support_flat[:, expert_id], as_tuple=False).flatten()
                if extra_rows.numel() > 0:
                    extra_inp = x_flat.index_select(0, extra_rows)
                    extra_out = self.experts[expert_id](extra_inp).detach()
                    extra_coeff = routed_probs_flat[extra_rows, expert_id].unsqueeze(1)
                    router_ste_y_flat.index_add_(0, extra_rows, extra_out * extra_coeff)

        y = (y_flat + router_ste_y_flat - router_ste_y_flat.detach()).view_as(x)

        with torch.no_grad():
            gate_stats = gate_flat.float()
            top1_coef_vals = gate_stats[:, 0]
            if gate_stats.size(1) >= 2:
                top2_coef_vals = gate_stats[:, 1]
            else:
                top2_coef_vals = torch.zeros_like(top1_coef_vals)

            num_top_logits = 2 if logits_flat_for_stats.size(1) >= 2 else 1
            top_logits = torch.topk(logits_flat_for_stats, k=num_top_logits, dim=-1).values
            top1_logits_vals = top_logits[:, 0]
            if num_top_logits == 2:
                top2_logits_vals = top_logits[:, 1]
            else:
                top2_logits_vals = torch.zeros_like(top1_logits_vals)

            if extra_support_flat is None:
                ste_active_token_frac = torch.tensor(0.0, device=x.device)
                ste_extra_experts_per_token = torch.tensor(0.0, device=x.device)
                ste_support_prob_mass = torch.tensor(0.0, device=x.device)
            else:
                ste_active_token_frac = extra_support_flat.any(dim=-1).float().mean()
                ste_extra_experts_per_token = extra_support_flat.float().sum(dim=-1).mean()
                ste_support_prob_mass = (ste_soft_routed_probs_flat * extra_support_flat.float()).sum(dim=-1).mean()

            if self.num_experts > self.top_k:
                topk_plus_one = torch.topk(logits_flat_for_stats, k=self.top_k + 1, dim=-1).values
                ste_boundary_gap = (topk_plus_one[:, self.top_k - 1] - topk_plus_one[:, self.top_k]).mean()
            else:
                ste_boundary_gap = torch.tensor(0.0, device=x.device)
            if lb_extra_support_flat is None:
                ste_lb_active_token_frac = torch.tensor(0.0, device=x.device)
                ste_lb_extra_experts_per_token = torch.tensor(0.0, device=x.device)
                ste_lb_boundary_gap = torch.tensor(0.0, device=x.device)
            else:
                ste_lb_active_token_frac = lb_extra_support_flat.any(dim=-1).float().mean()
                ste_lb_extra_experts_per_token = lb_extra_support_flat.float().sum(dim=-1).mean()
                ste_lb_boundary_gap = ste_boundary_gap

            router_value_stats = {
                'top1_logit': top1_logits_vals.mean(),
                'top2_logit': top2_logits_vals.mean(),
                'logit_diff': (top1_logits_vals - top2_logits_vals).mean(),
                'top1_coef': top1_coef_vals.mean(),
                'top2_coef': top2_coef_vals.mean(),
                'coef_diff': (top1_coef_vals - top2_coef_vals).mean(),
                'ste_active_token_frac': ste_active_token_frac,
                'ste_extra_experts_per_token': ste_extra_experts_per_token,
                'ste_boundary_gap': ste_boundary_gap,
                'ste_support_prob_mass': ste_support_prob_mass,
                'ste_lb_active_token_frac': ste_lb_active_token_frac,
                'ste_lb_extra_experts_per_token': ste_lb_extra_experts_per_token,
                'ste_lb_boundary_gap': ste_lb_boundary_gap,
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
            frac_base = global_frac_override if global_frac_override is not None else frac
            if self.load_balance_ste_width > 0.0:
                logits_flat_for_lb = logits_for_selection.reshape(BT, self.num_experts)
                lb_threshold = _topk_threshold(logits_flat_for_lb, idx_flat)
                lb_margin = logits_flat_for_lb - lb_threshold
                lb_soft_mask = RectIndicatorSTE.apply(lb_margin, self.load_balance_ste_width)
                lb_soft_frac = lb_soft_mask.float().mean(0) / float(self.top_k)
                frac_for_aux = frac_base + lb_soft_frac - lb_soft_frac.detach()
            else:
                frac_for_aux = frac_base

            if self.maxvio_load_balance:
                aux = _maxvio_from_load(frac_for_aux)
            else:
                # Switch paper:  L_aux = E * <load,prob>
                if self.aux_use_routed_prob:
                    probs_full = routed_probs_for_aux.reshape(-1, self.num_experts)
                else:
                    probs_full = probs.reshape(-1, self.num_experts)
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

    def forward(self, x, token_idx=None, return_expert_assignments=False, router_context=None, diff_weight=1):
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
                F.rms_norm(x, (x.size(-1),)), token_idx, return_expert_assignments=True, router_context=router_context, diff_weight=diff_weight
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
                F.rms_norm(x, (x.size(-1),)), token_idx, router_context=router_context, diff_weight=diff_weight
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
    router_layer_type : str = None
    router_activation : str = 'gelu'
    topk_activation : str = 'softmax'
    global_load_balance : bool = False
    aux_use_routed_prob : bool = False
    maxvio_load_balance : bool = False
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
    topk_ste_width : float = 0.0
    load_balance_ste_width : float = 0.0
    qk_clip_tau : float = 0.0
    qk_clip_block_size : int = 128
    log_attn_logits : bool = False


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

    def apply_qk_clip(self, tau=None):
        tau = self.config.qk_clip_tau if tau is None else tau
        if tau is None or tau <= 0.0:
            return
        if not self.transformer.h:
            return
        qk_max = torch.stack([block.attn.qk_clip_max for block in self.transformer.h])
        if dist.is_initialized() and dist.get_world_size() > 1:
            dist.all_reduce(qk_max, op=dist.ReduceOp.MAX)
        if torch.all(qk_max <= 0):
            return
        gamma = torch.clamp(tau / (qk_max + 1e-6), max=1.0)
        scale = torch.sqrt(gamma)
        for layer_idx, block in enumerate(self.transformer.h):
            block.attn.apply_qk_clip(scale[layer_idx])
        for block in self.transformer.h:
            block.attn.qk_clip_max.zero_()

    def collect_attn_logit_max(self):
        if not self.transformer.h or not self.transformer.h[0].attn.log_attn_logits:
            return None
        attn_max = torch.stack([block.attn.attn_logit_max for block in self.transformer.h])
        if dist.is_initialized() and dist.get_world_size() > 1:
            dist.all_reduce(attn_max, op=dist.ReduceOp.MAX)
        return attn_max

    def reset_attn_logit_max(self):
        for block in self.transformer.h:
            block.attn.attn_logit_max.zero_()

    def forward(
        self,
        idx,
        targets=None,
        return_logits=True,
        aux_coeff=0.0,
        diff_topk_reg_coeff=0.0,
        return_expert_assignments=False,
        router_context=None,
        diff_weight=1,
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
                    x, idx, return_expert_assignments=True, router_context=router_context, diff_weight=diff_weight
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
                    x, idx, router_context=router_context, diff_weight=diff_weight
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
