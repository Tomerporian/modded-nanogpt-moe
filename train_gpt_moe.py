import os
import sys
with open(sys.argv[0]) as f:
    code = f.read() # read the code of this file ASAP, for logging
import uuid
import glob
import time
import math
import gc
import logging
import warnings
import random
import re
import shutil

import numpy as np
import torch

# Suppress specific PyTorch Inductor warnings
warnings.filterwarnings("ignore", message="Online softmax is disabled on the fly since Inductor decides to split the reduction")
warnings.filterwarnings("ignore", category=UserWarning, module="torch._inductor.lowering")
import torch.distributed as dist
import torch._inductor.config as config
import torch._dynamo as dynamo
from torch.nn.parallel import DistributedDataParallel as DDP
import wandb
from wandb_logging import init_wandb, wandb_train_log, wandb_val_log
from params import parse_args
from optimizers import get_optimizers
from schedulers import SCHEDULER_TYPE
from torch.optim.swa_utils import AveragedModel
from logger import setup_default_logging
from data.dataloaders import create_dataloader
from gpt_moe_model import (
    GPT,
    GPTConfig,
    ROUTER_VALUE_KEYS,
    init_layer_router_value_tensors,
    init_total_router_value_tensors,
    GLUVariant
)

DTYPES = {
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}

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
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn

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
    hidden_dim_scale_factor=args.hidden_dim_scale_factor,
    num_experts=args.num_experts,
    top_k=args.top_k,
    router_type=args.router_type,
    router_depth=args.router_depth,
    router_layer_type=args.router_layer_type,
    router_activation=args.router_activation,
    topk_activation=args.topk_activation,
    global_load_balance=args.global_load_balance,
    aux_use_routed_prob=args.aux_use_routed_prob,
    maxvio_load_balance=args.maxvio_load_balance,
    loss_free_mode=args.loss_free_mode,
    loss_free_strength=args.loss_free_strength,
    loss_free_update_rate=args.loss_free_update_rate,
    router_logit_jitter=args.router_logit_jitter,
    use_router_temperature=args.use_router_temperature,
    diff_topk_reg_max_coeff=args.diff_topk_regularizer_max_coeff,
    diff_topk_reg_fp32=args.diff_topk_regularizer_fp32,
    theta_load_balance_coeff=args.theta_load_balance_coeff,
    theta_lb_detach_theta=args.theta_lb_detach_theta,
    theta_lb_detach_logits=args.theta_lb_detach_logits,
    topk_ste_width=args.topk_ste_width,
    load_balance_ste_width=args.load_balance_ste_width,
    qk_clip_tau=args.qk_clip_tau,
    qk_clip_block_size=args.qk_clip_block_size,
    log_attn_logits=args.log_attn_logits,
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
ctx = torch.amp.autocast(device_type='cuda', dtype=DTYPES[args.ops_dtype])

# CUDNN attention is ~4ms faster than Flash, but doesn't get selected by default in PyTorch 2.5.1
# from torch.backends.cuda import enable_cudnn_sdp, enable_flash_sdp, enable_math_sdp, enable_mem_efficient_sdp
# enable_cudnn_sdp(True)
# enable_flash_sdp(True)
# enable_mem_efficient_sdp(False)
# enable_math_sdp(False)

# init the optimizer(s)
optimizers, router_optimizer, router_temperature_optimizer = get_optimizers(raw_model, args)

# Initialize SWA averaged model if requested
swa_model = None
# If SWA is enabled, set default swa_start to the former decay phase start,
# and disable warmdown to keep a constant LR after warmup
if args.use_swa:
    if getattr(args, 'swa_start', 0) <= 0 and getattr(args, 'warmdown_iters', 0) > 0:
        args.swa_start = max(0, args.num_iterations - args.warmdown_iters)
        logging.info(f"SWA enabled; defaulting swa_start to {args.swa_start} (start of previous warmdown phase).")
    if getattr(args, 'warmdown_iters', 0) > 0:
        logging.info(f"SWA enabled; overriding warmdown_iters={args.warmdown_iters} to 0 for constant LR.")
        args.warmdown_iters = 0

    # Choose averaging function (uniform vs EMA)
    avg_fn = None
    if getattr(args, 'swa_avg_type', 'uniform') == 'ema':
        decay = float(getattr(args, 'swa_ema_decay', 0.99))
        def _ema_avg_fn(avg_param, param, n):
            # avg_param and param share dtype/device; EMA ignores n
            return avg_param.mul(decay).add(param, alpha=1.0 - decay)
        avg_fn = _ema_avg_fn
        logging.info(f"Using EMA averaging for SWA with decay={decay}")
    swa_model = AveragedModel(raw_model, avg_fn=avg_fn)
    swa_model.to(device)
    # Compile SWA model to reduce eval overhead when requested
    if getattr(args, 'swa_eval_during_val', False) or getattr(args, 'swa_eval_final', False):
        swa_model = torch.compile(swa_model)

# init the scheduler(s)
scheduler_type = SCHEDULER_TYPE[args.lr_scheduler]
schedulers = [torch.optim.lr_scheduler.LambdaLR(opt, lambda it: scheduler_type(args, it, min_lr=args.min_lr)) for opt in optimizers]

def get_diff_topk_reg_coeff(step):
    max_coeff = args.diff_topk_regularizer_max_coeff
    if max_coeff <= 0.0:
        return 0.0
    schedule = args.diff_topk_regularizer_schedule
    if schedule == 'cosine':
        progress = step / args.num_iterations - 1
        scale = 0.5 * (1.0 - math.cos(math.pi * progress))
    elif schedule == 'constant':
        scale = 1.0
    else:
        raise ValueError(f"Unsupported diff-topk regularizer schedule: {schedule}")
    return max_coeff * scale

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
        # If present, restore SWA model state
        if args.use_swa and 'swa_model' in checkpoint and swa_model is not None:
            try:
                swa_model.load_state_dict(checkpoint['swa_model'])
                logging.info("Loaded SWA state from checkpoint.")
            except Exception as e:
                logging.warning(f"Failed to load SWA state from checkpoint: {e}")
        start_step = checkpoint.get('step', 0) + 1
        start_step = min(start_step, args.num_iterations)
        resume_training_time_ms = checkpoint.get('training_time_ms', 0.0)
        args.resume = resolved_resume_path

        if os.path.dirname(resolved_resume_path) != args.output:
            new_resume_path = os.path.join(args.output, os.path.basename(resolved_resume_path))
            if master_process:
                shutil.copyfile(resolved_resume_path, new_resume_path)
            resolved_resume_path = new_resume_path
                
            logging.info(f"Resumed from checkpoint {resolved_resume_path} at step {start_step}.")
    elif args.resume and args.resume != 'auto' and master_process:
        raise ValueError(f"Resume requested but checkpoint {args.resume} not found")

last_checkpoint_path = (
    resolved_resume_path
    if args.save_only_latest and resolved_resume_path and os.path.isfile(resolved_resume_path)
    else None
)

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
        optimizers[2],
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
    diff_topk_reg_coeff = get_diff_topk_reg_coeff(step)
    if args.transition_start_iter == -1 or step < args.transition_start_iter:
        diff_weight = 1
    elif step > args.transition_end_iter:
        diff_weight = 0
    else:
        transition_duration = args.transition_end_iter - args.transition_start_iter
        diff_weight = (args.transition_end_iter - step) / transition_duration

    # once in a while evaluate the validation dataset
    if (last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0)):
        # stop the clock
        torch.cuda.synchronize()
        training_time_ms += 1000 * (time.time() - t0)
        # run validation batches
        model.eval()
        if args.use_swa and args.swa_eval_during_val and swa_model is not None:
            swa_model.eval()
        val_loss = 0.0
        val_ce_loss = 0.0
        val_aux_loss = 0.0
        val_theta_lb_loss = torch.tensor(0.0, device=device)
        val_diff_topk_reg = torch.tensor(0.0, device=device)
        val_router_entropy = torch.tensor(0.0, device=device)
        val_expert_balance = torch.zeros(num_experts, device=device)
        # per-layer
        n_layers = raw_model.config.n_layer
        val_layer_router_entropy = torch.zeros(n_layers, device=device)
        val_layer_expert_balance = torch.zeros(n_layers, num_experts, device=device)
        val_layer_router_values = init_layer_router_value_tensors(n_layers, device=device)
        val_total_router_values = init_total_router_value_tensors(device=device)
        # Optional SWA accumulators (only if evaluating during val)
        do_swa_val = args.use_swa and args.swa_eval_during_val and swa_model is not None and step >= args.swa_start
        if do_swa_val:
            swa_val_loss = torch.tensor(0.0, device=device)
            swa_val_ce_loss = torch.tensor(0.0, device=device)
            swa_val_aux_loss = torch.tensor(0.0, device=device)
            swa_val_theta_lb_loss = torch.tensor(0.0, device=device)
            swa_val_diff_topk_reg = torch.tensor(0.0, device=device)

        for _ in range(val_steps):
            x_val, y_val = val_loader.next_batch()
            with torch.no_grad():
                with ctx:
                    (
                        _,
                        loss,
                        ce_loss,
                        total_aux,
                        total_diff_topk_reg,
                        total_theta_lb,
                        router_entropy,
                        expert_balance,
                        layer_router_entropy,
                        layer_expert_balance,
                        layer_router_values,
                        total_router_values,
                    ) = model(
                        x_val,
                        y_val,
                        return_logits=False,
                        aux_coeff=args.aux_coeff_val,
                        diff_topk_reg_coeff=diff_topk_reg_coeff,
                        diff_weight=diff_weight
                    )
                    val_loss += loss.detach()
                    val_ce_loss += ce_loss.detach()
                    val_aux_loss += total_aux.detach()
                    val_theta_lb_loss = val_theta_lb_loss + total_theta_lb.detach()
                    val_router_entropy = val_router_entropy + router_entropy.detach()
                    val_expert_balance = val_expert_balance + expert_balance.detach()
                    val_layer_router_entropy = val_layer_router_entropy + layer_router_entropy.detach()
                    val_layer_expert_balance = val_layer_expert_balance + layer_expert_balance.detach()
                    val_diff_topk_reg = val_diff_topk_reg + total_diff_topk_reg.detach()
                    for key in ROUTER_VALUE_KEYS:
                        val_layer_router_values[key] = val_layer_router_values[key] + layer_router_values[key].detach()
                        val_total_router_values[key] = val_total_router_values[key] + total_router_values[key].detach()
                    del loss, ce_loss
            # Evaluate SWA on the same batch if requested and averaging has begun
            if do_swa_val and getattr(swa_model, 'n_averaged', 0) > 0:
                with torch.no_grad():
                    with ctx:
                        (
                            _,
                            loss,
                            ce_loss,
                            total_aux,
                            total_diff_topk_reg,
                            total_theta_lb,
                            *_
                        ) = swa_model(
                            x_val,
                            y_val,
                            return_logits=False,
                            aux_coeff=args.aux_coeff_val,
                            diff_topk_reg_coeff=diff_topk_reg_coeff,
                            diff_weight=diff_weight
                        )
                        swa_val_loss += loss.detach()
                        swa_val_ce_loss += ce_loss.detach()
                        swa_val_aux_loss += total_aux.detach()
                        swa_val_theta_lb_loss += total_theta_lb.detach()
                        swa_val_diff_topk_reg += total_diff_topk_reg.detach()
        dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)
        dist.all_reduce(val_ce_loss, op=dist.ReduceOp.AVG)
        dist.all_reduce(val_aux_loss, op=dist.ReduceOp.AVG)
        dist.all_reduce(val_theta_lb_loss, op=dist.ReduceOp.AVG)
        dist.all_reduce(val_diff_topk_reg, op=dist.ReduceOp.AVG)
        if do_swa_val and getattr(swa_model, 'n_averaged', 0) > 0:
            dist.all_reduce(swa_val_loss, op=dist.ReduceOp.AVG)
            dist.all_reduce(swa_val_ce_loss, op=dist.ReduceOp.AVG)
            dist.all_reduce(swa_val_aux_loss, op=dist.ReduceOp.AVG)
            dist.all_reduce(swa_val_theta_lb_loss, op=dist.ReduceOp.AVG)
            dist.all_reduce(swa_val_diff_topk_reg, op=dist.ReduceOp.AVG)
        val_loss /= val_steps
        val_ce_loss /= val_steps
        val_aux_loss /= val_steps
        val_theta_lb_loss /= val_steps
        val_diff_topk_reg /= val_steps
        if do_swa_val and getattr(swa_model, 'n_averaged', 0) > 0:
            swa_val_loss /= val_steps
            swa_val_ce_loss /= val_steps
            swa_val_aux_loss /= val_steps
            swa_val_theta_lb_loss /= val_steps
            swa_val_diff_topk_reg /= val_steps
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
            if do_swa_val and getattr(swa_model, 'n_averaged', 0) > 0:
                wandb.log({
                    'swa/val/loss': float(swa_val_loss.item()),
                    'swa/val/ce_loss': float(swa_val_ce_loss.item()),
                    'swa/val/aux_loss': float(swa_val_aux_loss.item()),
                    'swa/val/theta_lb_loss': float(swa_val_theta_lb_loss.item()),
                    'swa/val/diff_topk_reg': float(swa_val_diff_topk_reg.item()),
                }, step=step)
        # compute router grad norms (CE and AUX separately) occasionally at validation interval
        # Use a single micro-batch to avoid heavy cost
        model.train()
        # grab a fresh batch (train or val doesn't matter for grads inspection)
        x_probe, y_probe = val_loader.next_batch()
        # 1) CE-only
        model.zero_grad(set_to_none=True)
        gc.collect()
        with ctx:
            _, loss_ce, ce_loss_probe, total_aux_probe, _, _, _, _, _, _, _, _ = model(
                x_probe, y_probe, return_logits=False, aux_coeff=0.0, diff_topk_reg_coeff=0.0, diff_weight=diff_weight
            )
        loss_ce.backward()
        ce_router_layer_grad_norms = []
        for li in range(raw_model.config.n_layer):
            if raw_model.transformer.h[li].mlp.router_type != 'hash':
                l = raw_model.transformer.h[li].mlp.router[-1]
                if isinstance(l, GLUVariant):
                    gnorm = l.w_out.weight.grad.detach().float().norm(2) if l.w_out.weight.grad is not None else torch.tensor(0.0, device=device)
                else:
                    gnorm = l.weight.grad.detach().float().norm(2) if l.weight.grad is not None else torch.tensor(0.0, device=device)

                ce_router_layer_grad_norms.append(gnorm)
            else:
                ce_router_layer_grad_norms.append(torch.tensor(0.0, device=device))
        ce_router_layer_grad_norms = torch.stack(ce_router_layer_grad_norms)
        dist.all_reduce(ce_router_layer_grad_norms, op=dist.ReduceOp.AVG)
        # 2) AUX-only
        model.zero_grad(set_to_none=True)
        gc.collect()
        with ctx:
            _, _, _, total_aux_probe, _, _, _, _, _, _, _, _ = model(
                x_probe, y_probe, return_logits=False, aux_coeff=0.0, diff_topk_reg_coeff=0.0, diff_weight=diff_weight
            )
        # Backprop aux explicitly
        total_aux_probe.backward()
        aux_router_layer_grad_norms = []
        for li in range(raw_model.config.n_layer):
            if raw_model.transformer.h[li].mlp.router_type != 'hash':
                l = raw_model.transformer.h[li].mlp.router[-1]
                if isinstance(l, GLUVariant):
                    gnorm = l.w_out.weight.grad.detach().float().norm(2) if l.w_out.weight.grad is not None else torch.tensor(0.0, device=device)
                else:
                    gnorm = l.weight.grad.detach().float().norm(2) if l.weight.grad is not None else torch.tensor(0.0, device=device)
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
                    _, _, _, _, _, _, _, _, _, _, _, _, current_assignments = model(
                        tracking_x,
                        return_logits=False,
                        aux_coeff=0.0,
                        diff_topk_reg_coeff=0.0,
                        return_expert_assignments=True,
                        diff_weight=diff_weight
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
                ROUTER_VALUE_KEYS,
                diff_weight,
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
        if args.use_swa and swa_model is not None:
            try:
                log['swa_model'] = swa_model.state_dict()
            except Exception as e:
                logging.warning(f"Failed to serialize SWA state: {e}")
        checkpoint_path = os.path.join(args.output, f'state_step{step:06d}.pt')
        torch.save(log, checkpoint_path)
        if args.save_only_latest and (last_step or args.save_every > 0):
            previous_checkpoint = last_checkpoint_path
            last_checkpoint_path = checkpoint_path
            if previous_checkpoint and previous_checkpoint != checkpoint_path:
                try:
                    os.remove(previous_checkpoint)
                except OSError as err:
                    logging.warning(f"Failed to remove previous checkpoint {previous_checkpoint}: {err}")
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
                    model(x, y, return_logits=False, aux_coeff=0.0, diff_topk_reg_coeff=0.0, router_context=collect_context, diff_weight=diff_weight)
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
    diff_topk_reg_sum = torch.tensor(0.0, device=device)
    theta_lb_loss_sum = torch.tensor(0.0, device=device)
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
                total_diff_topk_reg,
                total_theta_lb,
                router_entropy,
                expert_balance,
                layer_router_entropy,
                layer_expert_balance,
                layer_router_values,
                total_router_values,
            ) = model(
                x_batch,
                y_batch,
                return_logits=False,
                aux_coeff=args.aux_coeff_train,
                diff_topk_reg_coeff=diff_topk_reg_coeff,
                router_context=router_context_use,
                diff_weight=diff_weight
            )
            train_loss = loss.detach()
            diff_topk_reg_sum = diff_topk_reg_sum + total_diff_topk_reg.detach()
            theta_lb_loss_sum = theta_lb_loss_sum + total_theta_lb.detach()
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

    # step the optimizers and schedulers
    for opt, sched in zip(optimizers, schedulers):
        opt.step()
        sched.step()
    # Update SWA parameters if enabled and within schedule
    if args.use_swa and swa_model is not None:
        if step >= args.swa_start and (args.swa_cadence <= 1 or ((step - args.swa_start) % args.swa_cadence) == 0):
            swa_model.update_parameters(raw_model)
    if args.qk_clip_tau > 0.0:
        raw_model.apply_qk_clip()

    # null the gradients
    model.zero_grad(set_to_none=True)
    raw_model.finalize_loss_free_updates()
    
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
    diff_topk_reg_avg = diff_topk_reg_sum / train_accumulation_steps
    theta_lb_loss_avg = theta_lb_loss_sum / train_accumulation_steps
    dist.all_reduce(router_entropy_avg, op=dist.ReduceOp.AVG)
    dist.all_reduce(expert_balance_avg, op=dist.ReduceOp.AVG)
    dist.all_reduce(layer_router_entropy_avg, op=dist.ReduceOp.AVG)
    dist.all_reduce(layer_expert_balance_avg, op=dist.ReduceOp.AVG)
    dist.all_reduce(diff_topk_reg_avg, op=dist.ReduceOp.AVG)
    dist.all_reduce(theta_lb_loss_avg, op=dist.ReduceOp.AVG)
    for key in ROUTER_VALUE_KEYS:
        dist.all_reduce(layer_router_values_avg[key], op=dist.ReduceOp.AVG)
        dist.all_reduce(total_router_values_avg[key], op=dist.ReduceOp.AVG)
    attn_logit_max = None
    if args.log_attn_logits:
        attn_logit_max = raw_model.collect_attn_logit_max()
    # --------------- TRAINING SECTION END -------------------

    if master_process:
        approx_time = training_time_ms + 1000 * (time.time() - t0)
        logging.info(f"step:{step+1}/{args.num_iterations} train_loss:{train_loss.item():.4f} train_time:{approx_time:.0f}ms step_avg:{approx_time/timed_steps:.2f}ms")
        with open(logfile, "a") as f:
            f.write(f"step:{step+1}/{args.num_iterations} train_loss:{train_loss.item():.4f} train_time:{approx_time:.0f}ms step_avg:{approx_time/timed_steps:.2f}ms\n")
        # wandb logging
        wandb_train_log(
            step + 1,
            train_loss,
            diff_topk_reg_avg,
            theta_lb_loss_avg,
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
            ROUTER_VALUE_KEYS,
            diff_topk_reg_coeff,
            attn_logit_max,
        )
    if args.log_attn_logits:
        raw_model.reset_attn_logit_max()

# Optional final SWA evaluation (no LR decay, averaged weights)
if args.use_swa and args.swa_eval_final and swa_model is not None:
    try:
        torch.cuda.synchronize()
        swa_model.eval()
        val_loss = torch.tensor(0.0, device=device)
        val_ce_loss = torch.tensor(0.0, device=device)
        val_aux_loss = torch.tensor(0.0, device=device)
        val_theta_lb_loss = torch.tensor(0.0, device=device)
        val_diff_topk_reg = torch.tensor(0.0, device=device)
        for _ in range(val_steps):
            x_val, y_val = val_loader.next_batch()
            with torch.no_grad():
                with ctx:
                    (
                        _,
                        loss,
                        ce_loss,
                        total_aux,
                        total_diff_topk_reg,
                        total_theta_lb,
                        *_
                    ) = swa_model(
                        x_val,
                        y_val,
                        return_logits=False,
                        aux_coeff=args.aux_coeff_val,
                        diff_topk_reg_coeff=get_diff_topk_reg_coeff(args.num_iterations),
                        diff_weight=1.0,
                    )
                    val_loss += loss.detach()
                    val_ce_loss += ce_loss.detach()
                    val_aux_loss += total_aux.detach()
                    val_theta_lb_loss += total_theta_lb.detach()
                    val_diff_topk_reg += total_diff_topk_reg.detach()
        dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)
        dist.all_reduce(val_ce_loss, op=dist.ReduceOp.AVG)
        dist.all_reduce(val_aux_loss, op=dist.ReduceOp.AVG)
        dist.all_reduce(val_theta_lb_loss, op=dist.ReduceOp.AVG)
        dist.all_reduce(val_diff_topk_reg, op=dist.ReduceOp.AVG)
        val_loss /= val_steps
        val_ce_loss /= val_steps
        val_aux_loss /= val_steps
        val_theta_lb_loss /= val_steps
        val_diff_topk_reg /= val_steps
        if master_process:
            wandb.log({
                'swa/val/loss': float(val_loss.item()),
                'swa/val/ce_loss': float(val_ce_loss.item()),
                'swa/val/aux_loss': float(val_aux_loss.item()),
                'swa/val/theta_lb_loss': float(val_theta_lb_loss.item()),
                'swa/val/diff_topk_reg': float(val_diff_topk_reg.item()),
            }, step=args.num_iterations)
            logging.info(
                f"SWA final eval — loss:{val_loss.item():.4f} ce:{val_ce_loss.item():.4f} aux:{val_aux_loss.item():.4f}"
            )
    except Exception as e:
        if master_process:
            logging.warning(f"SWA final evaluation failed: {e}")

if master_process:
    logging.info(f"peak memory consumption: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB")
    try:
        wandb.finish()
    except Exception:
        pass
    with open(os.path.join(args.output, 'done'), 'w') as f:
        f.write('')

# -------------------------------------------------------------------------
# clean up nice
dist.destroy_process_group()
