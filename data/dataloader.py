import glob
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, IterableDataset

__all__ = ["EOD_TOKEN", "PAD_TOKEN", "DocumentDataset"]

EOD_TOKEN = 50256
PAD_TOKEN = 0


class DocumentDataset(IterableDataset):
    def __init__(self, filename_pattern: str, seq_length: int, batch_size: int, rank: int, world_size: int):
        self.files = sorted(glob.glob(filename_pattern))
        self.seq_length = seq_length
        self.batch_size = batch_size
        self.rank = rank
        self.world_size = world_size

        self.local_files = self.files[rank::world_size]

    def _load_data_shard(self, file: Path):
        header = torch.from_file(filename=str(file), shared=False, size=256, dtype=torch.int32)
        assert header[0] == 20240520, "Magic number mismatch."
        assert header[1] == 1, "Unsupported version."

        num_tokens = int(header[2])
        with file.open("rb", buffering=0) as f:
            tokens = torch.empty(num_tokens, dtype=torch.uint16)
            f.seek(256 * 4)
            nbytes = f.readinto(tokens.numpy())
            assert nbytes == 2 * num_tokens
        return tokens

    def _split_into_documents(self, tokens: torch.Tensor):
        eod_positions = (tokens == EOD_TOKEN).nonzero(as_tuple=True)[0]
        boundaries = torch.cat([torch.tensor([-1], device=tokens.device), eod_positions, torch.tensor([len(tokens) - 1], device=tokens.device)])
        docs = [tokens[boundaries[i] + 1 : boundaries[i + 1] + 1] for i in range(len(boundaries) - 1)]
        return docs

    def _pad_document(self, doc: torch.Tensor) -> torch.Tensor:
        if len(doc) >= self.seq_length:
            return doc[: self.seq_length]
        else:
            return torch.cat([doc, torch.full((self.seq_length - len(doc),), PAD_TOKEN, dtype=doc.dtype)])

    def __iter__(self):
        for file in self.local_files:
            tokens = self._load_data_shard(Path(file))
            docs = self._split_into_documents(tokens)

            batch = []
            for doc in docs:
                padded = self._pad_document(doc)
                batch.append(padded)

                if len(batch) == self.batch_size:
                    batch_tensor = torch.stack(batch)  # [B, L]
                    inputs = batch_tensor.to(torch.int32)  # [B, L]
                    targets = batch_tensor.to(torch.int64)  # [B, L]

                    yield inputs, targets
                    batch = []


def benchmark_dataloader(
    filename_pattern: str,
    seq_length: int = 2048,
    batch_size: int = 8,
    rank: int = 0,
    world_size: int = 1,
    num_batches: int = 100,
):
    dataset = DocumentDataset(
        filename_pattern=filename_pattern,
        seq_length=seq_length,
        batch_size=batch_size,
        rank=rank,
        world_size=world_size,
    )
    loader = DataLoader(dataset, batch_size=None, num_workers=0, pin_memory=True)

    times = []
    total_tokens = 0

    for i, (inputs, targets) in enumerate(loader):
        start = time.time()

        inputs.sum()
        targets.sum()

        torch.cuda.synchronize()
        end = time.time()

        times.append(end - start)
        total_tokens += inputs.numel()

        if i + 1 >= num_batches:
            break

    avg_time = sum(times) / num_batches
    tokens_per_second = total_tokens / sum(times)

    print(f"Average batch load+process time: {avg_time * 1000:.2f} ms")
    print(f"Total tokens processed: {total_tokens}")
    print(f"Effective throughput: {tokens_per_second / 1e6:.2f}M tokens/sec")


if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser()

    parser.add_argument("--pattern", type=str, default="data/fineweb10B/fineweb_train_*.bin", help="Glob pattern for bin files")
    parser.add_argument("--seq_length", type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_batches", type=int, default=100)

    args = parser.parse_args()

    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    benchmark_dataloader(
        filename_pattern=args.pattern,
        seq_length=args.seq_length,
        batch_size=args.batch_size,
        rank=rank,
        world_size=world_size,
        num_batches=args.num_batches,
    )
