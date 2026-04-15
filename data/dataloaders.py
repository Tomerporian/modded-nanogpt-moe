import glob
import logging
import os

import numpy as np
import torch

from megatron_indexed_dataset import MegatronDataLoader


_SPLIT_SEED_OFFSETS = {
    'train': 0,
    'val': 1_000_003,
    'test': 2_000_006,
}


def _loader_seed(base_seed, process_rank, split):
    return int(base_seed) + _SPLIT_SEED_OFFSETS.get(split, 0) + int(process_rank)


def _peek_data_shard(filename):
    """Only read the header of a .bin shard."""
    with open(filename, "rb") as f:
        header = np.frombuffer(f.read(256 * 4), dtype=np.int32)
    if header[0] != 20240520:
        logging.info("ERROR: magic number mismatch in the data .bin file!")
        logging.info("---> HINT: Are you passing in a correct file with --input_bin?")
        logging.info("---> HINT: Dataset encoding changed recently, re-run data prepro or refer again to README")
        logging.info("---> HINT: For example re-run: `python dev/data/tinyshakespeare.py`, then re-try")
        exit(1)
    assert header[1] == 1, "unsupported version"
    ntok = header[2]
    return ntok


def _load_data_shard(filename):
    """Load the full shard into memory – kept for parity with older flows."""
    with open(filename, "rb") as f:
        header = np.frombuffer(f.read(256 * 4), dtype=np.int32)
        assert header[0] == 20240520, "magic number mismatch in the data .bin file"
        assert header[1] == 1, "unsupported version"
        ntok = header[2]
        tokens = np.frombuffer(f.read(), dtype=np.uint16)
    assert len(tokens) == ntok, "number of tokens read does not match header?"
    return tokens


class DistributedDataLoader:
    def __init__(self, filename_pattern, B, T, process_rank, num_processes, seed=0, split='train'):
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.B = B
        self.T = T
        self.seed = int(seed)
        self.split = split

        self.files = sorted(glob.glob(filename_pattern))
        assert len(self.files) > 0, f"did not find any files that match the pattern {filename_pattern}"

        if 'fineweb10B' in filename_pattern:
            self.file_format = 'fineweb'
            self.header_size = 256 * 4
            self.dtype = np.uint16
        elif 'tokenized_owt' in filename_pattern:
            self.file_format = 'openwebtext'
            self.header_size = 0
            self.dtype = np.uint16
        elif 'tokenized_c4' in filename_pattern:
            self.file_format = 'c4'
            self.header_size = 0
            self.dtype = np.uint8
        else:
            raise ValueError(f"Unknown dataset format for pattern: {filename_pattern}")

        self.shard_lengths = []
        ntok_total = 0
        for fname in self.files:
            if self.file_format == 'fineweb':
                shard_ntok = _peek_data_shard(fname)
            else:
                file_size = os.path.getsize(fname)
                token_size = np.dtype(self.dtype).itemsize
                shard_ntok = file_size // token_size

            self.shard_lengths.append(shard_ntok)
            ntok_total += int(shard_ntok)

        self.ntok_total = ntok_total

        self.cumulative_lengths = []
        cumsum = 0
        for length in self.shard_lengths:
            effective_length = max(0, int(length) - self.T)
            cumsum += effective_length
            self.cumulative_lengths.append(cumsum)

        self.ntok_total = cumsum
        self.random_high = self.ntok_total - self.T
        assert self.random_high > 0, "dataset does not have enough tokens for the requested sequence length"

        self._base_seed = _loader_seed(self.seed, self.process_rank, self.split)
        self._generator = torch.Generator()
        self._num_batches_drawn = 0
        self._reset_sampler()

    def _reset_sampler(self):
        self._generator.manual_seed(self._base_seed)
        self._num_batches_drawn = 0

    def _advance_sampler(self, num_batches):
        remaining = int(num_batches)
        if remaining <= 0:
            return

        max_chunk_batches = max(1, 65536 // self.B)
        while remaining > 0:
            take = min(remaining, max_chunk_batches)
            torch.randint(0, self.random_high, (take, self.B), generator=self._generator)
            self._num_batches_drawn += take
            remaining -= take

    def state_dict(self):
        return {
            'num_batches_drawn': self._num_batches_drawn,
        }

    def load_state_dict(self, state):
        self._reset_sampler()
        self._advance_sampler(state.get('num_batches_drawn', 0))

    def next_batch(self):
        B = self.B
        T = self.T

        random_positions = torch.randint(0, self.random_high, (B,), generator=self._generator)
        self._num_batches_drawn += 1

        shard_info = []
        for pos in random_positions:
            pos = pos.item()
            shard_idx = 0
            for i, cum_len in enumerate(self.cumulative_lengths):
                if pos < cum_len:
                    shard_idx = i
                    break

            if shard_idx == 0:
                pos_in_shard = pos
            else:
                pos_in_shard = pos - self.cumulative_lengths[shard_idx - 1]

            shard_info.append((shard_idx, pos_in_shard))

        x_list = []
        y_list = []
        for shard_idx, pos_in_shard in shard_info:
            tokens = np.memmap(self.files[shard_idx], dtype=self.dtype, mode='r', offset=self.header_size)
            seq = tokens[pos_in_shard:pos_in_shard + T + 1]
            x_list.append(torch.from_numpy(seq[:T].astype(np.int64)))
            y_list.append(torch.from_numpy(seq[1:T+1].astype(np.int64)))

        x = torch.stack(x_list)
        y = torch.stack(y_list)

        return x.cuda(), y.cuda()


def is_megatron_dataset(path_pattern):
    """
    Detect if the path refers to a Megatron indexed dataset.
    """
    if '*' in path_pattern or '?' in path_pattern:
        return False

    if path_pattern.endswith('.bin'):
        idx_path = path_pattern[:-4] + '.idx'
    else:
        idx_path = path_pattern + '.idx'

    return os.path.exists(idx_path)


def create_dataloader(path_pattern, B, T, ddp_rank, ddp_world_size, split='train', seed=0):
    """
    Create appropriate dataloader based on dataset format.
    """
    if is_megatron_dataset(path_pattern):
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
            split=split,
            seed=seed,
        )

    if ddp_rank == 0:
        logging.info(f"Using DistributedDataLoader for multi-file dataset: {path_pattern}")

    return DistributedDataLoader(
        filename_pattern=path_pattern,
        B=B,
        T=T,
        process_rank=ddp_rank,
        num_processes=ddp_world_size,
        seed=seed,
        split=split,
    )
