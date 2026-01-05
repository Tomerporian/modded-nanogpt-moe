import argparse
import yaml


def _str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.strip().lower()
    if value in ('True', 'true', '1', 'yes', 'y', 'on'):
        return True
    if value in ('False', 'false', '0', 'no', 'n', 'off'):
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got '{value}'")


def _build_training_parser():
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
    group.add_argument('--hidden-dim-scale-factor', default=4.0, type=float,
                       help='scale factor applied to n_embd to compute the MLP hidden dimension')
    group.add_argument('--num-experts', default=8, type=int,
                       help='number of MoE experts')
    group.add_argument('--top-k', default=2, type=int,
                       help='top-k experts to use')
    group.add_argument('--router-type', default='diff', type=str, choices=['switch', 'diff', 'diff_no_softmax', 'hash', 'scaled_diff_no_softmax'],
                       help='router type for MoE')
    group.add_argument('--router-depth', default=1, type=int,
                       help='number of layers in the router MLP for non-hash routing (hidden dim == input dim)')
    group.add_argument('--router-activation', default='gelu', type=str, choices=['gelu', 'relu', 'relu_squared'],
                       help='activation to use between router MLP layers (if depth > 1)')
    group.add_argument('--global-load-balance', action='store_true', default=False,
                       help='enable global batch load balancing for auxiliary router loss')
    group.add_argument('--aux-use-routed-prob', action='store_true', default=False,
                       help='compute aux load balancing loss with the probabilities actually used to route tokens')
    group.add_argument('--loss-free-mode', default='none', type=str, choices=['none', 'deepseek', 'stopgrad'],
                       help='loss-free router biasing strategy (deepseek for switch only, stopgrad supports switch/diff)')
    group.add_argument('--loss-free-strength', default=1.0, type=float,
                       help='scale factor applied to the loss-free routing bias')
    group.add_argument('--loss-free-update-rate', default=0.001, type=float,
                       help='per-expert bias update rate for sign-based loss-free routing')
    group.add_argument('--router-logit-jitter', default=0.0, type=float,
                       help='uniform multiplicative noise width applied to router logits during training')
    group.add_argument('--use-router-temperature', action='store_true', default=False,
                       help='enable a learnable router temperature applied to logits before routing')

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
                       help='iterations of linear warmup/warmdown for triangular or trapezoidal schedule')
    group.add_argument('--weight-decay', default=0.0, type=float,
                       help='weight decay')
    group.add_argument('--adamw-betas', nargs=2, type=float, default=(0.9, 0.95), metavar=('BETA1', 'BETA2'),
                       help='beta1 and beta2 for AdamW optimizers')
    group.add_argument('--adamw-fused', type=_str2bool, default=True, metavar='BOOL',
                       help='enable fused AdamW kernels (set to false to disable fused kernels)')
    group.add_argument('--use_adamw_opt3', action='store_true', default=False,
                       help='use AdamW instead of Muon for transformer blocks')
    group.add_argument('--use_adamw_router', action='store_true', default=False,
                       help='optimize router parameters with AdamW instead of Muon (requires learned routers)')
    group.add_argument('--only-router-muon', action='store_true', default=False,
                       help='use Muon only for router parameters and AdamW for the rest of the transformer blocks')
    group.add_argument('--muon-svd-backend', default='newtonschulz5', type=str, choices=['newtonschulz5', 'svd'],
                       help='method used for Muon orthogonalization backend')
    group.add_argument('--muon-nesterov', type=_str2bool, default=True, metavar='BOOL',
                       help='use Nesterov momentum in the Muon optimizer')
    group.add_argument('--muon-backend-steps', type=int, default=5,
                       help='iteration steps for the Muon backend orthogonalization')

    # Learning rate parameters
    group = parser.add_argument_group('Learning rate parameters')
    group.add_argument('--lr-embed', default=0.3, type=float,
                       help='learning rate for embedding layer')
    group.add_argument('--lr-head', default=0.002, type=float,
                       help='learning rate for head layer')
    group.add_argument('--lr-muon', default=0.02, type=float,
                       help='learning rate for muon optimizer (transformer blocks)')
    group.add_argument('--lr-theta', default=0.1, type=float,
                       help='learning rate for theta load balance parameters')
    group.add_argument('--momentum', default=0.95, type=float,
                       help='momentum for muon optimizer')

    # Evaluation and logging parameters
    group = parser.add_argument_group('Evaluation and logging parameters')
    group.add_argument('--val-loss-every', default=125, type=int,
                       help='every how many steps to evaluate val loss? 0 for only at the end')
    group.add_argument('--val-tokens', default=10485760, type=int,
                       help='number of tokens of validation data')
    group.add_argument('--save-every', default=0, type=int,
                       help='every how many steps to save the checkpoint? 0 for only at the end')
    group.add_argument('--save-only-latest', action='store_true', default=False,
                       help='if set, remove the previous checkpoint after saving a new one')
    group.add_argument('--n-tracked-seq', default=100, type=int,
                       help='number of sequences to track for expert assignment changes')
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
    group.add_argument('--diff-topk-regularizer-max-coeff', default=0.0, type=float,
                       help='maximum coefficient for the diff-topk normalization regularizer (0 disables it)')
    group.add_argument('--diff-topk-regularizer-schedule', default='constant', type=str,
                       choices=['constant', 'cosine'],
                       help="warm-up schedule for diff-topk regularizer coefficient")
    group.add_argument('--diff-topk-regularizer-fp32', action='store_true', default=False,
                       help='compute the diff-topk regularizer in fp32 precision')
    group.add_argument('--theta-load-balance-coeff', default=0.0, type=float,
                       help='coefficient for the theta-based load balancing loss during training')
    group.add_argument('--theta-lb-detach-theta', type=_str2bool, default=True,
                       help='if true, detach theta before adding it to router logits')
    group.add_argument('--theta-lb-detach-logits', type=_str2bool, default=True,
                       help='if true, detach logits before theta_lb_loss')

    # Misc parameters
    group = parser.add_argument_group('Run config')
    group.add_argument('--device_0', action='store_true', default=False,
                       help='always use device=0')
    group.add_argument('--seed', type=int, default=42,
                       help='random seed (default: 42)')
    group.add_argument('--resume', default='auto', type=str,
                       help="checkpoint path to resume from, or 'auto'")
    group.add_argument('--ops_dtype', default='bfloat16', type=str,
                       help="dtype for autocast")

    return parser


def parse_args(argv=None):
    config_parser = argparse.ArgumentParser(description='Training Config', add_help=False)
    config_parser.add_argument('-c', '--config', default='', type=str, metavar='FILE',
                               help='YAML config file specifying default arguments')
    args_config, remaining = config_parser.parse_known_args(argv)

    parser = _build_training_parser()
    if args_config.config:
        with open(args_config.config, 'r') as f:
            cfg = yaml.safe_load(f) or {}
            if not isinstance(cfg, dict):
                raise ValueError(f"Config file {args_config.config} must contain a mapping of defaults.")
            parser.set_defaults(**cfg)

    args = parser.parse_args(remaining)

    if args.only_router_muon and args.use_adamw_router:
        parser.error("--only-router-muon cannot be combined with --use_adamw_router")
    if args.only_router_muon and args.use_adamw_opt3:
        parser.error("--only-router-muon cannot be combined with --use_adamw_opt3")

    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
    return args, args_text


__all__ = ["parse_args"]
