# W&B Log Reference

This file explains the W&B metrics emitted by `train_gpt_moe.py`, `wandb_logging.py`, and the router statistics returned from `gpt_moe_model.py`.

## Naming Conventions

- `train/...`: training metrics logged once per train step.
- `val/...`: validation metrics logged at validation time.
- `swa/val/...`: validation metrics from the SWA model, if SWA eval is enabled.
- `.../Layer {li}` or `layer_{li}`: per-layer metric.
- `train/expert_balance/{i}` or `val/expert_balance/{i}`: per-expert metric averaged across layers.
- `Expert Balance/Layer {li}/{ei}`: per-layer, per-expert routing load.
- `router_values/total_*`: average across layers of the corresponding router statistic.
- `ste/...`: diagnostics for the routing/top-k STE window.
- `ste_lb/...`: diagnostics for the auxiliary load-balance STE window.

Important implementation detail:

- `train/expert_balance/{i}` and `val/expert_balance/{i}` are load fractions, not counts.
- Load fractions sum to `1`, because they are normalized over all selected assignments in the batch.
- The balanced target per expert is therefore `1 / num_experts`.

## Core Optimization Metrics

### Losses

- `train/loss`: full training objective used for optimization.
- `val/loss`: full validation objective.
- `val/ce_loss`: validation cross-entropy only.
- `train/diff_topk_reg`, `val/diff_topk_reg`: differentiable top-k regularizer term before multiplying by its coefficient.
- `train/theta_lb_loss`, `val/theta_lb_loss`: theta-based load-balancing term before multiplying by `theta_load_balance_coeff`.
- `val/aux_loss`: validation auxiliary load-balance loss.
- `swa/val/loss`, `swa/val/ce_loss`, `swa/val/aux_loss`, `swa/val/theta_lb_loss`, `swa/val/diff_topk_reg`: same metrics for the SWA model.

Note:

- There is no separate `train/ce_loss` log at the moment.
- `train/loss` already includes CE, aux, diff-topk regularization, and theta load balance according to the active config.

### Optimization State

- `train/grad_norm`: total gradient norm used for training diagnostics.
- `lr/embed`, `lr/head`, `lr/blocks`: learning rates for the three main optimizer groups.
- `lr/router`: router optimizer learning rate, if a separate router optimizer is enabled.
- `diff_topk_reg/coeff`: current coefficient multiplying `train/diff_topk_reg`.
- `transition/diff_weight`: current switch-vs-diff interpolation weight during transition experiments.

### Runtime

- `train/step_time_ms`: measured time for the most recent logged train interval.
- `train/step_avg_ms`: average step time. This key is logged in both train and validation code paths.
- `train/time_ms`: cumulative training time in milliseconds, logged at validation time.

## Load Balance And Router Entropy

### Router Entropy

- `train/router_entropy`, `val/router_entropy`: average normalized entropy of the router distribution.
- `Router Entropy/Layer {li}`: per-layer normalized router entropy.

Interpretation:

- Higher entropy means routing is more diffuse.
- Lower entropy means routing is more confident or specialized.
- Lower entropy is not automatically bad if balance metrics remain good.

### Balance Metrics

These are computed from expert load fractions.

- `train/MaxViobatch`, `val/MaxVioglobal`: maximal overload relative to the balanced target.
- `train/MaxViobatchWorstLayer`, `val/MaxVioglobalWorstLayer`: worst-layer max violation.
- `train/MinViobatch`, `val/MinVioglobal`: maximal underload relative to the balanced target.
- `train/TotalViobatch`, `val/TotalVioglobal`: total absolute imbalance relative to the balanced target.

Formulas:

- Let `target = 1 / num_experts`.
- `MaxVio = (max_i load_i - target) / target`
- `MinVio = (target - min_i load_i) / target`
- `TotalVio = sum_i |load_i - target| / target`

### Per-Expert Balance

- `train/expert_balance/{i}`, `val/expert_balance/{i}`: load fraction of expert `i`, averaged across layers.
- `Expert Balance/Layer {li}/{ei}`: per-layer load fraction of expert `ei`.

Use these when you want to know whether the imbalance is concentrated in a few experts or spread out.

## Router Value Logs

These come from `ROUTER_VALUE_KEYS` in `gpt_moe_model.py`.

### Basic Router Sharpness

- `router_values/total_top1_logit`, `router_values/layer_{li}_top1_logit`: average winning logit.
- `router_values/total_top2_logit`, `router_values/layer_{li}_top2_logit`: average second-place logit.
- `router_values/total_logit_diff`, `router_values/layer_{li}_logit_diff`: average top-1 minus top-2 logit gap.
- `router_values/total_top1_coef`, `router_values/layer_{li}_top1_coef`: average top-1 routed coefficient.
- `router_values/total_top2_coef`, `router_values/layer_{li}_top2_coef`: average top-2 routed coefficient.
- `router_values/total_coef_diff`, `router_values/layer_{li}_coef_diff`: average top-1 minus top-2 coefficient gap.

Interpretation:

- Larger logit and coefficient gaps usually mean more confident routing.
- In rect-STE experiments, increasing gap together with stable balance often indicates more confident expert specialization.

## Top-k STE Diagnostics

These refer to the router STE window used for the main routing/top-k path, not the auxiliary load-balance loss.

- `ste/all_layers/active_token_frac`, `ste/layer_{li}/active_token_frac`: fraction of tokens with at least one extra unselected expert inside the top-k STE window.
- `ste/all_layers/extra_experts_per_token`, `ste/layer_{li}/extra_experts_per_token`: average number of extra unselected experts inside the top-k STE window.
- `ste/all_layers/boundary_gap`, `ste/layer_{li}/boundary_gap`: average gap between the kth selected and the next excluded expert.
- `ste/all_layers/support_prob_mass`, `ste/layer_{li}/support_prob_mass`: average soft routed mass that sits in the extra top-k STE support region.

Interpretation:

- Larger `boundary_gap` means the routing boundary is cleaner and farther from ambiguity.
- Smaller `active_token_frac` usually means fewer tokens are close enough to the boundary to receive STE correction.

## Load-Balance STE Diagnostics

These refer to the auxiliary load-balance STE window, controlled by `load_balance_ste_width`.

### Coverage Metrics

- `ste_lb/all_layers/active_token_frac`, `ste_lb/layer_{li}/active_token_frac`: fraction of tokens with at least one extra unselected expert inside the aux STE window.
- `ste_lb/all_layers/extra_experts_per_token`, `ste_lb/layer_{li}/extra_experts_per_token`: average number of extra unselected experts inside the aux STE window.
- `ste_lb/all_layers/boundary_gap`, `ste_lb/layer_{li}/boundary_gap`: same kth-vs-next gap, reused for the aux window.
- `ste_lb/all_layers/selected_support_frac`, `ste_lb/layer_{li}/selected_support_frac`: fraction of selected assignments that are still inside the aux STE window.

Interpretation:

- `selected_support_frac` is the movable routed mass: selected assignments that still receive aux-loss STE gradient.
- Low `selected_support_frac` means selected assignments are mostly far from the aux boundary.

### Failure-Mode Metrics

These are the overload/underload diagnostics for rect-STE experiments.

Definitions, using load fractions:

- `load_i`: current load fraction of expert `i`
- `target = 1 / num_experts`
- `excess_i = max(0, load_i - target)`
- `deficit_i = max(0, target - load_i)`
- `active_selected_i`: selected routed mass for expert `i` that is inside the aux STE window
- `active_unselected_i`: unselected near-threshold mass for expert `i` inside the aux STE window
- `dead_excess_i = max(0, excess_i - active_selected_i)`
- `dead_deficit_i = max(0, deficit_i - active_unselected_i)`

Logged metrics:

- `ste_lb/all_layers/excess_frac`, `ste_lb/layer_{li}/excess_frac`: total overload mass, `sum_i excess_i`.
- `ste_lb/all_layers/deficit_frac`, `ste_lb/layer_{li}/deficit_frac`: total underload mass, `sum_i deficit_i`.
- `ste_lb/all_layers/dead_excess_frac`, `ste_lb/layer_{li}/dead_excess_frac`: fraction of overload mass that is outside the aux STE-correctable selected support.
- `ste_lb/all_layers/dead_deficit_frac`, `ste_lb/layer_{li}/dead_deficit_frac`: fraction of underload mass that lacks enough near-threshold incoming candidates.
- `ste_lb/all_layers/dead_excess_max_frac`, `ste_lb/layer_{li}/dead_excess_max_frac`: worst per-expert dead-overload fraction.
- `ste_lb/all_layers/dead_deficit_max_frac`, `ste_lb/layer_{li}/dead_deficit_max_frac`: worst per-expert dead-underload fraction.

Interpretation:

- High `dead_excess_frac` is the main rect-STE overload failure signal.
- High `dead_deficit_frac` is the mirror underload failure signal.
- The `*_max_frac` versions tell you whether the problem is concentrated in a single bad expert.

### Specialization-Side Metrics

These quantify the positive side of “dead” selected assignments: stable expert specialization that is not part of frozen overload.

Definitions:

- `selected_dead_i = max(0, load_i - active_selected_i)`
- `anchored_selected_i = max(0, selected_dead_i - dead_excess_i)`

So `anchored_selected_i` is selected mass that is outside the aux STE window but is not part of irreducible overload.

Logged metrics:

- `ste_lb/all_layers/anchored_selected_frac`, `ste_lb/layer_{li}/anchored_selected_frac`: total anchored selected mass.
- `ste_lb/all_layers/anchored_balanced_frac`, `ste_lb/layer_{li}/anchored_balanced_frac`: anchored selected mass normalized by balanced-capacity load, `sum_i min(load_i, target)`.

Interpretation:

- High `anchored_balanced_frac` means a large fraction of the balanced load is confidently assigned outside the aux STE window. This is a useful specialization signal.
- High `anchored_balanced_frac` with low `dead_excess_frac` is the good regime: confident specialization without frozen overload.
- High `anchored_balanced_frac` with high `dead_excess_frac` means specialization exists, but some of it is trapped in overload.

### Practical Rect-STE Reading Guide

When comparing standard aux to rect-STE:

- Lower entropy with similar `MaxVio` and `TotalVio` can be good if `anchored_balanced_frac` rises while `dead_excess_frac` stays low.
- Lower entropy with rising `dead_excess_frac` is more suspicious: the router is becoming rigid in a way the aux loss can no longer correct.
- A falling `selected_support_frac` is not automatically bad. It may reflect specialization. Use it together with `anchored_balanced_frac` and `dead_excess_frac`.

## Validation-Only Diagnostics

### Gradient Probes

These are measured on a probe batch during validation.

- `Router Grad Norms (CE)/Layer {li}`: router gradient norm from cross-entropy only.
- `Router Grad Norms (AUX)/Layer {li}`: router gradient norm from auxiliary loss only.

Interpretation:

- If AUX grad norms collapse while imbalance remains high, the aux signal may be too weak or too dead.
- Comparing CE and AUX grad scales helps diagnose whether the router is dominated by task loss or balancing loss.

### Assignment Stability Tracking

These are computed from a fixed set of tracked validation sequences.

- `track_tokens/layer_{li}/top{k}_change`: fraction of tracked token positions whose expert at rank `k` changed since the previous validation.
- `track_tokens/layer_{li}/chosen_changed`: fraction of tracked token positions whose chosen expert set changed, ignoring order within top-k.

Interpretation:

- High values mean the routing pattern is still moving.
- Low values mean routing assignments are stabilizing.
- Low assignment change together with high `anchored_balanced_frac` is consistent with stable specialization.

## Optional Metrics

These appear only when the corresponding feature is enabled.

### Router Parameter Scale

- `router_weight_norm/mean`, `router_weight_norm/max`
- `router_weight_norm/mean/Layer {li}`, `router_weight_norm/max/Layer {li}`

Use these to catch exploding or collapsing router parameters.

### Loss-Free Biasing

- `train_loss_free_bias/Layer {li}/expert_{ei}`
- `val_loss_free_bias/Layer {li}/expert_{ei}`

These are the learned or accumulated per-expert bias values used by loss-free balancing modes.

### Theta Load Balancing

- `thetas/theta/min`, `thetas/theta/max`
- `thetas/theta/min/Layer {li}`, `thetas/theta/max/Layer {li}`

These summarize the range of the theta offsets used in theta-based load balancing.

### Router Temperature

- `router_temperature/Layer {li}`

Only logged when `use_router_temperature` is enabled.

### Attention Logit Monitoring

- `attn_logits/max`, `attn_logits/mean`
- `attn_logits/max/Layer {li}`, `attn_logits/mean/Layer {li}`

These are useful for diagnosing attention-logit blowup and for QK clipping experiments.

### Scaled Diff Router

- `router_values/total_max_scaler`, `router_values/layer_{li}_max_scaler`
- `router_values/total_min_scaler`, `router_values/layer_{li}_min_scaler`

These are only meaningful for `scaled_diff_no_softmax`.

## Notes On Historical Runs

- Older runs will not contain the newer `ste_lb/...dead_*...` or `ste_lb/...anchored_*...` metrics.
- Those newer metrics cannot be reconstructed exactly from the older aggregate W&B logs alone because the old logs did not preserve the per-expert overlap between imbalance and STE support.

## Suggested Default Dashboard For Rect-STE Experiments

If you want a compact dashboard, start with:

- `val/MaxVioglobal`
- `val/TotalVioglobal`
- `val/router_entropy`
- `ste_lb/all_layers/dead_excess_frac`
- `ste_lb/all_layers/dead_deficit_frac`
- `ste_lb/all_layers/anchored_balanced_frac`
- `ste_lb/all_layers/selected_support_frac`
- `Router Grad Norms (AUX)/Layer 0` and one or two deeper layers
- `track_tokens/layer_0/chosen_changed` and one or two deeper layers

That set gives you balance, entropy, aux-correctability, specialization, aux gradient strength, and assignment stability.
