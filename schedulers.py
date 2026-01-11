# learning rate decay scheduler (linear warmup and warmdown)
def get_constant_lr(args, it, lr=None, min_lr=None):
    assert it <= args.num_iterations
    # 1) linear warmup for warmup_iters steps
    if it < args.warmup_iters:
        return (it+1) / args.warmup_iters
    # 2) constant lr for a while
    elif it <= args.num_iterations - args.warmdown_iters:
        return 1.0
    # 3) linear warmdown
    else:
        decay_ratio = (args.num_iterations - it) / args.warmdown_iters
        return decay_ratio
    
def get_inverse_decay_lr(args, it, lr=1, min_lr=0.1):
    assert it <= args.num_iterations
    
    # 1) linear warmup for warmup_iters steps
    if it < args.warmup_iters:
        return (it+1) / args.warmup_iters
    # 2) constant lr for a while
    elif it <= args.num_iterations - args.warmdown_iters:
        return 1.0
    # 3) decay warmdown
    else:
        decay_start = args.num_iterations - args.warmdown_iters
        T = args.warmdown_iters
        t = it - decay_start
        if T <= 0:
            return lr

        frac = t / T
        inv = frac * (1.0 / min_lr) + (1.0 - frac) * (1.0 / lr)
        
        return 1 / inv
    
SCHEDULER_TYPE = {
    'linear': get_constant_lr,
    'inverse': get_inverse_decay_lr
}