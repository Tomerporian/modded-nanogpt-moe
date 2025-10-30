# Manual Analysis of Differential vs Switch Routing

## Issue Summary
User reports that differential routing and switch routing produce extremely similar results, which is unexpected given they should use different routing strategies.

## Current Implementations

### Switch Router (`switch_topk`, lines 210-215)
```python
def switch_topk(logits, k, null_expert_bias=0.0):
    probs      = logits.softmax(dim=-1)
    gate, topk_idx = torch.topk(probs, k, dim=-1)
    gate = gate / (gate.sum(dim=-1, keepdim=True) + null_expert_bias)
    return topk_idx, probs, gate
```

**Process:**
1. Compute softmax of logits → probabilities for all experts
2. Select top-k experts based on probability values
3. Renormalize the k selected probabilities
4. Return: indices (k), full probs (num_experts), gate weights (k)

### Differential Router (`diff_routing`, lines 227-238)
```python
def diff_routing(logits, k):
    probs = logits.softmax(dim=-1)
    z_max = torch.max(logits, dim=-1, keepdim=True).values
    exp_z = torch.exp(logits - z_max)
    num_experts = exp_z.shape[-1]
    kth_smallest_idx = num_experts - k
    m_exp_z = torch.kthvalue(exp_z, k=kth_smallest_idx, dim=-1, keepdim=True).values
    topk_exp_z = F.relu(exp_z - m_exp_z)
    topk_weights = topk_exp_z / (topk_exp_z.sum(dim=-1, keepdim=True) + 1e-8)
    _, topk_idx = torch.topk(topk_weights, k, dim=-1)
    return topk_idx, probs, topk_weights
```

**Process:**
1. Compute softmax (for logging only, not used in routing!)
2. Compute exp(logits - max) for numerical stability
3. Find threshold: kth_smallest_idx = num_experts - k
4. Apply relu(exp_z - threshold) to zero out small values
5. Normalize the thresholded values
6. Use torch.topk to get indices
7. Return: indices (k), full probs (num_experts), all weights (num_experts)

## Threshold Calculation Analysis

### Example: 8 experts, top-2

**Current implementation:**
```
kth_smallest_idx = num_experts - k = 8 - 2 = 6
threshold = torch.kthvalue(exp_z, k=6)  # Gets 6th smallest value
```

**Values (sorted ascending):** [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]

- 6th smallest = 6.0
- This is the 3rd largest (since 8-6+1 = 3)
- relu(values - 6.0) = [0, 0, 0, 0, 0, 0, 1.0, 2.0]
- **Result: 2 non-zero values** ✓ CORRECT

The threshold calculation appears to be correct!

## Concrete Example Comparison

### Input logits: [0.5, 0.3, 0.1, 0.05, 1.5, 0.8, 1.0, 0.3]
### Want: top-2

### Switch Routing:
1. **Softmax:**
   - exp(logits) / sum(exp) ≈ [0.103, 0.085, 0.070, 0.066, 0.281, 0.140, 0.171, 0.085]
2. **Top-2:** Indices [4, 6] with probs [0.281, 0.171]
3. **Renormalized:** [0.622, 0.378]
4. **Output gate:** [0.622, 0.378]

### Differential Routing:
1. **exp(logits - max):** ≈ [0.368, 0.301, 0.247, 0.235, 1.0, 0.497, 0.607, 0.301]
2. **Threshold (6th smallest):** 0.497
3. **After relu:** [0, 0, 0, 0, 0.503, 0, 0.11, 0]
4. **Normalized:** [0, 0, 0, 0, 0.82, 0, 0.18, 0]
5. **Top-2 indices:** [4, 6]
6. **Output gate (after gather):** [0.82, 0.18]

### Comparison:
- **Same indices?** YES (both select experts 4 and 6)
- **Same weights?** NO!
  - Switch: [0.622, 0.378]
  - Diff: [0.82, 0.18]
  - **Mean absolute difference: ~0.26** (26%!)

## 🔍 POTENTIAL BUG FOUND!

### Issue: Off-by-one error in threshold calculation?

The current implementation uses:
```python
kth_smallest_idx = num_experts - k
```

This gets the `(num_experts - k)`-th smallest value, which is the `(k+1)`-th largest.

For top-2 of 8:
- Current: 6th smallest = 3rd largest
- relu(values - 3rd_largest) keeps values > 3rd largest = top 2 ✓

**Actually, this is CORRECT!** The threshold works properly.

## 🔍 SECOND POTENTIAL ISSUE

### The "probs" variable in diff_routing is misleading!

Line 228 in diff_routing:
```python
probs = logits.softmax(dim=-1)
```

This computes softmax probabilities but **never uses them** in the actual routing logic!
The routing uses `exp_z` (unnormalized exp) instead.

This `probs` is only returned for aux loss calculation, which is the same for both methods.

## 🚨 CRITICAL FINDING: Both use discrete torch.topk!

Both implementations use `torch.topk` to select expert indices:
- Switch: `torch.topk(probs, k)`
- Diff: `torch.topk(topk_weights, k)`

`torch.topk` is **not differentiable** with respect to which indices are selected. Both methods have the same non-differentiable expert selection process!

The "differential" aspect only affects:
1. **Weight computation:** How the gate weights are calculated
2. **NOT index selection:** Both use discrete topk

This means:
- Gradients flow through the weights differently
- But the expert selection process is identical (discrete)
- If the weight differences don't significantly affect training, results could be similar

## MoE Forward Integration (lines 258-265)

```python
if self.router_type == "switch":
    logits = self.router(x)
    topk_idx, probs, gate = switch_topk(logits, self.top_k)
elif self.router_type == "diff":
    logits = self.router(x)
    topk_idx, probs, gate = diff_routing(logits, self.top_k)
    # Extract only top-k weights to match switch format
    gate = torch.gather(gate, dim=-1, index=topk_idx)
```

For diff routing, there's an extra `torch.gather` call because `diff_routing` returns weights for ALL experts, while `switch_topk` returns weights only for top-k.

## Potential Reasons for Similar Results

### 1. Weight differences may not matter much
- Both methods select the same top-k experts
- Weight ratios might be similar enough that model performance is unaffected
- The subsequent MLP operations might be robust to weight variations

### 2. Router learns similar logits
- If the router learns to produce logits where the top-2 are much larger than others
- Both methods will give similar results (high confidence on same experts)

### 3. Implementation bug elsewhere?
- Check if `router_type` is being set/used correctly
- Verify the model is actually using the specified router
- Check if there's any code path that bypasses the routing

## Recommendations for Testing

1. **Verify router_type is being used:**
   - Print `model.transformer.h[0].mlp.router_type` during training
   - Confirm it matches the config

2. **Log gate weights during training:**
   - Add logging to compare switch vs diff gate weight distributions
   - Check if weights are actually different

3. **Test gradient magnitudes:**
   - Compare gradient norms for router weights in both methods
   - Different gradients → different learning dynamics

4. **Ablation study:**
   - Train with deliberately different hyperparameters
   - See if diff router advantages emerge with specific settings

5. **Check for identical results:**
   - If loss curves are EXACTLY identical, there might be a code path issue
   - If just similar, it might be that the differences don't matter much

## Conclusion

**The implementation looks mostly correct**, but both methods use discrete index selection (torch.topk), making them more similar than expected. The main difference is in weight computation, not expert selection.

The similar results might be:
1. **Expected:** Weight differences don't significantly affect training
2. **Unexpected:** There's a subtle bug causing them to compute identical weights
3. **Configuration issue:** The diff router isn't being used as expected

Recommend running tests to verify gate weights are actually different during training.
