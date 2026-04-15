"""
Simplified Megatron IndexedDataset reader for modded-nanogpt-moe.
Reads .bin and .idx files created by Megatron-LM preprocessing.

Adapted from: megatron/core/datasets/indexed_dataset.py
"""

import struct
from enum import Enum
from typing import Type, Tuple, Optional
import numpy
import torch


_INDEX_HEADER = b"MMIDIDX\x00\x00"
_SPLIT_SEED_OFFSETS = {
    'train': 0,
    'val': 1_000_003,
    'test': 2_000_006,
}


def _loader_seed(base_seed: int, process_rank: int, split: str) -> int:
    return int(base_seed) + _SPLIT_SEED_OFFSETS.get(split, 0) + int(process_rank)


class DType(Enum):
    """The NumPy data type Enum for reading IndexedDataset indices"""

    uint8 = 1
    int8 = 2
    int16 = 3
    int32 = 4
    int64 = 5
    float64 = 6
    float32 = 7
    uint16 = 8

    @classmethod
    def dtype_from_code(cls, value: int) -> Type[numpy.number]:
        """Get the dtype from the code"""
        return getattr(numpy, cls(value).name)

    @staticmethod
    def size(key) -> int:
        """Get the size of the dtype/code in bytes"""
        if isinstance(key, int):
            return DType.dtype_from_code(key)().itemsize
        elif numpy.number in key.__mro__:
            return key().itemsize
        else:
            raise ValueError


class _IndexReader:
    """Reads the index (.idx) file"""

    def __init__(self, idx_path: str) -> None:
        print(f"Loading index from {idx_path}")

        with open(idx_path, "rb") as stream:
            header = stream.read(9)
            assert header == _INDEX_HEADER, f"bad header, cannot read: {idx_path}"

            version = struct.unpack("<Q", stream.read(8))[0]
            assert version == 1, f"bad version, cannot read: {idx_path}"

            code = struct.unpack("<B", stream.read(1))[0]
            self.dtype = DType.dtype_from_code(code)
            self.dtype_size = DType.size(self.dtype)

            self.sequence_count = struct.unpack("<Q", stream.read(8))[0]
            self.document_count = struct.unpack("<Q", stream.read(8))[0]

            offset = stream.tell()

        # Memory-map the index file for efficient access
        self.bin_buffer_mmap = numpy.memmap(idx_path, mode="r", order="C")
        self.bin_buffer = memoryview(self.bin_buffer_mmap)

        # Read sequence lengths
        self.sequence_lengths = numpy.frombuffer(
            self.bin_buffer, dtype=numpy.int32, count=self.sequence_count, offset=offset
        )

        # Read sequence pointers (byte offsets)
        self.sequence_pointers = numpy.frombuffer(
            self.bin_buffer,
            dtype=numpy.int64,
            count=self.sequence_count,
            offset=offset + self.sequence_lengths.nbytes,
        )

        # Read document indices
        self.document_indices = numpy.frombuffer(
            self.bin_buffer,
            dtype=numpy.int64,
            count=self.document_count,
            offset=offset + self.sequence_lengths.nbytes + self.sequence_pointers.nbytes,
        )

        print(f"  Sequences: {self.sequence_count:,}")
        print(f"  Documents: {self.document_count:,}")
        print(f"  Data type: {self.dtype}")

    def __del__(self) -> None:
        if hasattr(self, "bin_buffer_mmap"):
            self.bin_buffer_mmap._mmap.close()
            del self.bin_buffer_mmap

    def __len__(self) -> int:
        return self.sequence_count

    def __getitem__(self, idx: int) -> Tuple[numpy.int64, numpy.int32]:
        """Return the pointer and length at the index"""
        return self.sequence_pointers[idx], self.sequence_lengths[idx]


class _MMapBinReader:
    """Memory-mapped binary file reader"""

    def __init__(self, bin_path: str) -> None:
        self.bin_path = bin_path
        self.bin_buffer_mmap = numpy.memmap(bin_path, mode="r", order="C")
        self.bin_buffer = memoryview(self.bin_buffer_mmap)

    def __del__(self) -> None:
        if hasattr(self, "bin_buffer_mmap"):
            self.bin_buffer_mmap._mmap.close()
            del self.bin_buffer_mmap

    def read(self, dtype: Type[numpy.number], count: int, offset: int) -> numpy.ndarray:
        """Read bytes into a numpy array"""
        return numpy.frombuffer(self.bin_buffer, dtype=dtype, count=count, offset=offset)


class IndexedDataset(torch.utils.data.Dataset):
    """
    Low-level interface for reading Megatron indexed datasets.

    Args:
        path_prefix: The index (.idx) and data (.bin) prefix (without extension)

    Example:
        ds = IndexedDataset('/path/to/merged')  # reads merged.idx and merged.bin
        doc = ds[0]  # get first document as numpy array
    """

    def __init__(self, path_prefix: str) -> None:
        super().__init__()
        idx_path = path_prefix + ".idx"
        bin_path = path_prefix + ".bin"

        self.path_prefix = path_prefix
        self.bin_reader = _MMapBinReader(bin_path)
        self.index = _IndexReader(idx_path)

    def __del__(self) -> None:
        del self.bin_reader
        del self.index

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> numpy.ndarray:
        """Return the tokens for document at index"""
        sequence_pointer, sequence_length = self.index[idx]
        sequence = self.bin_reader.read(
            dtype=self.index.dtype, count=sequence_length, offset=sequence_pointer
        )
        return sequence

    def get(self, idx: int, offset: int = 0, length: Optional[int] = None) -> numpy.ndarray:
        """
        Retrieve a portion of a document.

        Args:
            idx: Document index
            offset: Token offset within the document
            length: Number of tokens to read (None = read to end)
        """
        sequence_pointer, sequence_length = self.index[idx]
        if length is None:
            length = int(sequence_length) - offset
        sequence = self.bin_reader.read(
            dtype=self.index.dtype,
            count=length,
            offset=sequence_pointer + offset * DType.size(self.index.dtype),
        )
        return sequence

    @property
    def sizes(self):
        """Return array of document sizes (number of tokens per document)"""
        return self.index.sequence_lengths

    @property
    def doc_idx(self):
        """Return array of document boundary indices"""
        return self.index.document_indices


class MegatronDataLoader:
    """
    DataLoader for Megatron indexed datasets compatible with modded-nanogpt-moe.
    Samples random sequences of length T+1 from the dataset.

    Args:
        dataset_path: Path prefix to .idx/.bin files (e.g., '/path/to/merged')
        B: Batch size
        T: Sequence length
        process_rank: DDP rank (for distributed training)
        num_processes: Number of DDP processes
        split: Which split to use ('train', 'val', or 'test')
        split_ratios: Tuple of (train_pct, val_pct, test_pct). Must sum to 100.
                     Default: (98.9, 1.0, 0.1) like MegatronLM
    """

    def __init__(
        self,
        dataset_path: str,
        B: int,
        T: int,
        process_rank: int = 0,
        num_processes: int = 1,
        split: str = 'train',
        seed: int = 0,
        split_ratios: tuple = (98.9, 1.0, 0.1),
    ):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.split = split
        self.seed = int(seed)

        # Validate split
        assert split in ['train', 'val', 'test'], f"Invalid split: {split}"

        # Validate and normalize split ratios
        assert len(split_ratios) == 3, "split_ratios must have 3 values (train, val, test)"
        assert abs(sum(split_ratios) - 100.0) < 0.01, f"split_ratios must sum to 100, got {sum(split_ratios)}"
        self.split_ratios = split_ratios

        # Load the indexed dataset
        self.dataset = IndexedDataset(dataset_path)
        total_docs = len(self.dataset)

        # Calculate document indices for each split
        train_pct, val_pct, test_pct = split_ratios
        train_docs = int(total_docs * train_pct / 100.0)
        val_docs = int(total_docs * val_pct / 100.0)
        # test gets the remainder to ensure we use all docs

        # Split documents by index
        if split == 'train':
            self.doc_start = 0
            self.doc_end = train_docs
        elif split == 'val':
            self.doc_start = train_docs
            self.doc_end = train_docs + val_docs
        else:  # test
            self.doc_start = train_docs + val_docs
            self.doc_end = total_docs

        # Get document sizes for this split
        split_sizes = self.dataset.sizes[self.doc_start:self.doc_end]

        # Calculate total tokens in this split
        self.total_tokens = int(split_sizes.sum())

        if process_rank == 0:
            print(f"Split '{split}': documents [{self.doc_start}:{self.doc_end}] = {self.doc_end - self.doc_start:,} docs, {self.total_tokens:,} tokens")

        # Build cumulative document offsets for efficient sampling (only for this split)
        self.doc_cumsum = numpy.concatenate([[0], numpy.cumsum(split_sizes)])
        self.random_high = self.total_tokens - self.T
        assert self.random_high > 0, "dataset does not have enough tokens for the requested sequence length"
        self._base_seed = _loader_seed(self.seed, self.process_rank, self.split)
        self._generator = torch.Generator()
        self._num_batches_drawn = 0
        self._reset_sampler()

    def _reset_sampler(self):
        self._generator.manual_seed(self._base_seed)
        self._num_batches_drawn = 0

    def _advance_sampler(self, num_batches: int):
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
        """Sample a batch of random sequences"""
        B = self.B
        T = self.T

        # Sample random starting positions (within this split's tokens)
        random_positions = torch.randint(0, self.random_high, (B,), generator=self._generator)
        self._num_batches_drawn += 1

        x_list = []
        y_list = []

        for pos in random_positions:
            pos = pos.item()

            # Find which document contains this position (within the split)
            doc_idx_in_split = numpy.searchsorted(self.doc_cumsum, pos, side='right') - 1
            pos_in_doc = pos - self.doc_cumsum[doc_idx_in_split]

            # Map to actual document index in the full dataset
            doc_idx = self.doc_start + doc_idx_in_split

            # Get T+1 tokens starting from this position
            # Handle case where sequence spans multiple documents
            doc_len = self.dataset.sizes[doc_idx]
            tokens_available = doc_len - pos_in_doc

            if tokens_available >= T + 1:
                # Entire sequence fits in one document
                seq = self.dataset.get(doc_idx, offset=pos_in_doc, length=T + 1)
            else:
                # Need to span to next document(s)
                seq_parts = [self.dataset.get(doc_idx, offset=pos_in_doc, length=tokens_available)]
                tokens_needed = T + 1 - tokens_available
                current_doc = doc_idx + 1

                # Only span within the split's documents
                while tokens_needed > 0 and current_doc < self.doc_end:
                    doc_len = self.dataset.sizes[current_doc]
                    take = min(tokens_needed, doc_len)
                    seq_parts.append(self.dataset.get(current_doc, offset=0, length=take))
                    tokens_needed -= take
                    current_doc += 1

                seq = numpy.concatenate(seq_parts)[:T + 1]

            # Convert to torch tensors
            x_list.append(torch.from_numpy(seq[:T].astype(numpy.int64)))
            y_list.append(torch.from_numpy(seq[1:T + 1].astype(numpy.int64)))

        x = torch.stack(x_list)
        y = torch.stack(y_list)

        # Move to CUDA if available
        if torch.cuda.is_available():
            x = x.cuda()
            y = y.cuda()

        return x, y


if __name__ == "__main__":
    # Test the loader
    import sys

    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        path = "/p/data1/mmlaion/text-data/tokenized/c4/merged"

    print(f"Testing IndexedDataset with: {path}")
    ds = IndexedDataset(path)

    print(f"\nDataset statistics:")
    print(f"  Number of documents: {len(ds):,}")
    print(f"  Total tokens: {ds.sizes.sum():,}")
    print(f"  Min doc length: {ds.sizes.min()}")
    print(f"  Max doc length: {ds.sizes.max()}")
    print(f"  Mean doc length: {ds.sizes.mean():.1f}")

    print(f"\nFirst document:")
    doc0 = ds[0]
    print(f"  Length: {len(doc0)}")
    print(f"  Token range: {doc0.min()} - {doc0.max()}")
    print(f"  First 50 tokens: {doc0[:50].tolist()}")

    print(f"\nTesting MegatronDataLoader:")
    loader = MegatronDataLoader(path, B=4, T=128)
    x, y = loader.next_batch()
    print(f"  Batch shape: x={x.shape}, y={y.shape}")
    print(f"  Token range in batch: {x.min()} - {x.max()}")
