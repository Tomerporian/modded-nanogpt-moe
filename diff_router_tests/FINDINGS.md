# Differential vs Switch Routing: Test Results and Analysis

## Summary
After comprehensive testing, **the implementations are CORRECT and DO behave differently**. However, there are important nuances about why training results might appear similar.

## Test Results

### 1. Gate Weight Differences ✓
- **Test 1 (known logits):** 42% relative difference in gate weights
- **Test 2 (random batch):** Mean absolute difference of 0.162 (16%)
- **Conclusion:** Gate weights are SIGNIFICANTLY different between methods

### 2. Expert Selection
- **Index agreement:** 100% in all tests
- Both methods select the same top-k experts
- Only the weighting of selected experts differs

### 3. Gradient Magnitude Differences ✓✓✓ (MAJOR FINDING)
- **Switch gradient norm:** 0.128
- **Diff gradient norm:** 3.502 (27x larger!)
- **Relative difference:** ~275% in gradient patterns
- **Conclusion:** Gradients are VASTLY different!

### 4. Gradient Flow Patterns ✓ (KEY DIFFERENCE)

**Switch Routing:**
- Gradients flow ONLY to the k selected experts
- Example: Selected [4, 6] → Only experts 4 and 6 get gradients
- Non-zero gradient count: 2 (exactly top-k)

**Differential Routing:**
- Gradients flow to MORE than k experts!
- Example: Selected [4, 6] → Experts 4, 5, and 6 get gradients
- Non-zero gradient count: 3 (more than top-k)
- Expert 5 wasn't selected but still received gradient of 0.5194!

**This is the key insight:** The relu-based threshold allows gradients to flow to experts near the threshold, even if they weren't selected in the forward pass.

## Why Training Results Might Be Similar

Despite the implementations being correct and different, training results can still be similar for several reasons:

### 1. Same Expert Selection
- Both methods always select the same top-k experts (100% agreement)
- The discrete routing decision is identical
- Only the combination weights differ

### 2. Weight Differences May Average Out
- Over many tokens and layers, the weight differences might average out
- The model learns to work with either weighting scheme
- Final loss might be similar even with different intermediate weights

### 3. Router Convergence
- If the router learns to produce high-confidence predictions (one expert much larger than others)
- Both methods will produce very similar weights
- Example: If logits are [0.1, 0.1, 10.0, 9.0, 0.1, 0.1, 0.1, 0.1]
  - Both methods will heavily weight expert 2, slightly weight expert 3
  - The exact weight ratios won't matter much

### 4. Model Architecture May Be Insensitive
- The subsequent MLP layers might be robust to weight variations
- If expert outputs are similar, weighting differences don't matter
- The model may learn to compensate

## Implementation Details Verified

### Threshold Calculation ✓ CORRECT
```python
kth_smallest_idx = num_experts - k  # For top-2 of 8: gives 6
threshold = kthvalue(exp_z, k=6)     # Gets 6th smallest = 3rd largest
relu(exp_z - threshold)              # Keeps values > 3rd largest = top 2
```
- Gets exactly k non-zero values
- Threshold is set correctly

### Forward Pass ✓ CORRECT
- Switch: Renormalizes top-k softmax probabilities
- Diff: Thresholds exp(logits), normalizes, then gathers top-k
- Both produce shape-consistent outputs
- Gate weights sum to 1 in both cases

### Backward Pass ✓ CORRECT BUT DIFFERENT
- Switch: Discrete topk → gradients only to selected experts
- Diff: Relu threshold → gradients to selected + near-threshold experts
- Diff has 27x larger gradient norms!
- Gradient patterns are fundamentally different

## Potential Issues to Check

Despite implementations being correct, here are things to verify in your training:

### 1. Verify Router Type is Actually Used
```python
# Add this to your training script
print(f"Model router type: {model.transformer.h[0].mlp.router_type}")
print(f"Expected: diff or switch")
```

### 2. Check Router Confidence During Training
```python
# Log router entropy - should be different for switch vs diff
# Already logged as: 'val/router_entropy' and 'train/router_entropy'
```

Low entropy (< 0.1) means high confidence → methods will be more similar

### 3. Verify Different Checkpoints Are Being Used
- Make sure you're not accidentally comparing the same checkpoint
- Check that output directories are different
- Verify wandb logs show different run IDs

### 4. Check Aux Loss Coefficient
```python
# From your config:
aux_coeff_train = 0.0  # If this is 0, aux loss has no effect
aux_coeff_val = 0.0
```

If aux_coeff = 0, both methods train without load balancing pressure. Try:
- aux_coeff_train = 0.01 for diff routing
- aux_coeff_train = 0.01 for switch routing
- Compare results with aux loss active

### 5. Check If Logits Are Too Confident
Add logging to see the logit distribution:
```python
# In MoE forward, after computing logits:
logit_max = logits.max(dim=-1).values.mean()
logit_std = logits.std(dim=-1).mean()
# Log these values
```

If logit_std is very large (> 5.0), the router is very confident and both methods will behave similarly.

## Recommendations

### To Verify Implementations Are Being Used:

1. **Add Explicit Logging:**
```python
# In MoE.__init__
if ddp_rank == 0:
    logging.info(f"Initializing MoE with router_type={router_type}")

# In MoE.forward (occasionally)
if step % 100 == 0 and ddp_rank == 0:
    logging.info(f"Router type: {self.router_type}, Gate mean: {gate.mean():.4f}")
```

2. **Log Gate Weight Statistics:**
```python
# Add to your training loop
gate_std = gate_flat.std()
gate_max = gate_flat.max()
gate_min = gate_flat.min()
# Log to wandb
```

3. **Compare Learning Curves More Carefully:**
   - Plot training loss, val loss, router entropy side-by-side
   - Check if curves diverge at any point
   - Look for subtle differences in convergence speed

### To Amplify Differences:

If you want to see larger differences between methods, try:

1. **Increase Routing Uncertainty:**
   - Use smaller learning rates for the router
   - Add noise to router inputs
   - Use temperature scaling on logits

2. **Enable Aux Loss:**
   - aux_coeff_train = 0.01 to 0.1
   - This will affect switch and diff differently

3. **Use Larger k or More Experts:**
   - With k=4 or k=6, weight differences become more important
   - With 16 experts, selection differences may emerge

4. **Monitor Per-Layer Statistics:**
   - Check if differences emerge in specific layers
   - Early layers vs late layers might show different patterns

## Conclusion

**The implementations are CORRECT.** The differential router:
1. ✓ Computes different gate weights (16-42% difference)
2. ✓ Has different gradient magnitudes (27x larger)
3. ✓ Flows gradients to more experts (k+1 vs k)
4. ✓ Uses relu-based threshold correctly

**However**, both methods:
- Select the same top-k experts (100% agreement)
- Produce normalized gates that sum to 1
- May converge to similar solutions

If your training results are extremely similar:
1. This might be expected for your particular task/dataset
2. Check that you're actually using different router types
3. Try amplifying differences with the recommendations above
4. The weight differences might not matter much for final performance

The implementation is correct - the similarity in results is likely a property of your specific training setup, not a bug.
