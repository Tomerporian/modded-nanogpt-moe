from pathlib import Path
import tempfile

import numpy as np
import torch

import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.dataloaders import DistributedDataLoader


def _write_fineweb_shard(path: Path, tokens: np.ndarray) -> None:
    header = np.zeros(256, dtype=np.int32)
    header[0] = 20240520
    header[1] = 1
    header[2] = int(tokens.size)
    with open(path, "wb") as f:
        f.write(header.tobytes())
        f.write(tokens.astype(np.uint16).tobytes())


def _cpu_batch(batch):
    x, y = batch
    return x.cpu(), y.cpu()


def _batch_lists_equal(lhs, rhs):
    if len(lhs) != len(rhs):
        return False
    return all(torch.equal(x1, x2) and torch.equal(y1, y2) for (x1, y1), (x2, y2) in zip(lhs, rhs))


def main():
    original_cuda = torch.Tensor.cuda
    torch.Tensor.cuda = lambda self, *args, **kwargs: self
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            shard_a = tmp_path / "fineweb10B_train_000.bin"
            shard_b = tmp_path / "fineweb10B_train_001.bin"
            _write_fineweb_shard(shard_a, np.arange(0, 256, dtype=np.uint16))
            _write_fineweb_shard(shard_b, np.arange(10_000, 10_256, dtype=np.uint16))

            pattern = str(tmp_path / "fineweb10B_train_*.bin")
            kwargs = dict(
                filename_pattern=pattern,
                B=4,
                T=8,
                process_rank=1,
                num_processes=2,
                seed=123,
                split='train',
            )

            loader = DistributedDataLoader(**kwargs)
            initial_batches = [_cpu_batch(loader.next_batch()) for _ in range(5)]
            saved_state = loader.state_dict()
            resumed_target_batches = [_cpu_batch(loader.next_batch()) for _ in range(3)]

            resumed_loader = DistributedDataLoader(**kwargs)
            replayed_initial_batches = [_cpu_batch(resumed_loader.next_batch()) for _ in range(5)]
            resumed_loader.load_state_dict(saved_state)
            replayed_resumed_batches = [_cpu_batch(resumed_loader.next_batch()) for _ in range(3)]

            assert saved_state['num_batches_drawn'] == 5
            assert _batch_lists_equal(initial_batches, replayed_initial_batches), "fresh loaders with the same seed should match"
            assert _batch_lists_equal(resumed_target_batches, replayed_resumed_batches), "restored loader state should resume from the exact next batch"

            alt_split_loader = DistributedDataLoader(**{**kwargs, 'split': 'val'})
            alt_split_batch = _cpu_batch(alt_split_loader.next_batch())
            assert not (
                torch.equal(alt_split_batch[0], initial_batches[0][0]) and
                torch.equal(alt_split_batch[1], initial_batches[0][1])
            ), "train/val loaders should not share the same sampler stream"

            print("DataLoader resume-state checks passed.")
    finally:
        torch.Tensor.cuda = original_cuda


if __name__ == "__main__":
    main()
