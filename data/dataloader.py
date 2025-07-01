import glob
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

__all__ = ["EOD_TOKEN", "PAD_TOKEN", "MambaDataloader"]

EOD_TOKEN = 50256
PAD_TOKEN = 0


class MambaDataloader:
    def __init__(self, data_directory: str, rank: int, seed: Optional[int] = None):
        self.data_directory = data_directory
        self.rank = rank
        self.seed = seed

        self.data: Dict[int, List[torch.Tensor]] = {}
        self.cursors: Dict[int, int] = {}
        self.generator = None
        if self.seed is not None:
            self.generator = torch.Generator().manual_seed(self.seed)

        all_files = glob.glob(str(Path(self.data_directory) / "*.bin"))
        file_pattern = re.compile(r"bucket_(\d+)_(\d+)\.bin")
        for file_path_str in all_files:
            match = file_pattern.search(Path(file_path_str).name)
            if not match:
                continue

            sequence_length, file_rank = int(match.group(1)), int(match.group(2))

            if file_rank == self.rank:
                tokens = self._load_data_shard(Path(file_path_str))
                documents = self._split_into_documents(tokens)
                if documents:
                    self.data[sequence_length] = self._shuffle_documents(documents)
                    self.cursors[sequence_length] = 0

        if not self.data:
            raise RuntimeError(f"No data files found for rank {self.rank} in directory {self.data_directory}.")

    def _shuffle_documents(self, documents: List[torch.Tensor]) -> List[torch.Tensor]:
        if not documents:
            return []

        num_documents = len(documents)
        perm = torch.randperm(num_documents, generator=self.generator)
        return [documents[i] for i in perm]

    def _load_data_shard(self, file: Path) -> torch.Tensor:
        tokens = torch.from_file(filename=str(file), shared=False, dtype=torch.uint16)
        return tokens

    def _split_into_documents(self, tokens: torch.Tensor) -> List[torch.Tensor]:
        eod_positions = (tokens == EOD_TOKEN).nonzero(as_tuple=True)[0]
        if len(eod_positions) == 0:
            return [tokens]

        boundaries = torch.cat([torch.tensor([-1]), eod_positions])
        documents = [tokens[boundaries[i] + 1 : boundaries[i + 1]] for i in range(len(boundaries) - 1)]
        if eod_positions[-1] < len(tokens) - 1:
            documents.append(tokens[eod_positions[-1] + 1 :])
        return [doc for doc in documents if len(doc) > 0]

    def _pad_document(self, document: torch.Tensor, sequence_length: int) -> torch.Tensor:
        document_length = len(document)
        if document_length > sequence_length:
            return document[:sequence_length]
        elif document_length < sequence_length:
            padding = torch.full((sequence_length - document_length,), PAD_TOKEN, dtype=document.dtype)
            return torch.cat([document, padding])
        else:
            return document

    def next_batch(self, batch_size: int, sequence_length: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if sequence_length not in self.data:
            raise ValueError(f"No data for sequence_length={sequence_length}. Available: {list(self.data.keys())}")

        documents = self.data[sequence_length]
        num_documents = len(documents)
        start_idx = self.cursors[sequence_length]

        batch_documents = []
        if start_idx + batch_size > num_documents:
            remainder_documents = documents[start_idx:]
            shuffled_documents = self._shuffle_documents(documents)
            self.data[sequence_length] = shuffled_documents

            needed = batch_size - len(remainder_documents)
            batch_documents = remainder_documents + shuffled_documents[:needed]
            self.cursors[sequence_length] = needed
        else:
            end_idx = start_idx + batch_size
            batch_documents = documents[start_idx:end_idx]
            self.cursors[sequence_length] = end_idx

        padded_batch = [self._pad_document(document, sequence_length) for document in batch_documents]
        batch_tensor = torch.stack(padded_batch)

        inputs = batch_tensor.to(torch.int32)
        targets = batch_tensor.to(torch.int64)
        return inputs, targets


def benchmark_dataloader(
    data_directory: str, sequence_length: int = 2048, batch_size: int = 8, rank: int = 0, num_batches: int = 100, seed: int = 42
):
    device = f"cuda:{rank}"

    init_start = time.time()
    dataloader = MambaDataloader(
        data_directory=data_directory,
        rank=rank,
        seed=seed,
    )
    init_end = time.time()
    print(f"MambaDataLoader initialization time: {init_end - init_start:.2f} seconds.")

    times = []
    total_tokens = 0

    _ = dataloader.next_batch(batch_size=batch_size, sequence_length=sequence_length)

    for i in range(num_batches):
        torch.cuda.synchronize()
        start = time.time()

        inputs_cpu, targets_cpu = dataloader.next_batch(batch_size=batch_size, sequence_length=sequence_length)
        inputs = inputs_cpu.pin_memory().to(device, non_blocking=True)
        targets = targets_cpu.pin_memory().to(device, non_blocking=True)

        inputs.sum()
        targets.sum()

        torch.cuda.synchronize()
        end = time.time()

        times.append(end - start)
        total_tokens += inputs.numel()

    if len(times) > 1:
        times = times[1:]

    avg_time = sum(times) / len(times)
    tokens_per_second = total_tokens / sum(times)

    print("\n--- Benchmark Results ---")
    print(f"Average batch fetch + H2D transfer time: {avg_time * 1000:.2f} ms")
    print(f"Total tokens processed: {total_tokens}")
    print(f"Effective throughput: {tokens_per_second / 1e6:.2f}M tokens/sec")


if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser()

    parser.add_argument("--data_directory", type=str, default="data/fineweb10B/bucketed", help="Directory holding bucketed files.")
    parser.add_argument("--sequence_length", type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_batches", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    benchmark_dataloader(
        data_directory=args.data_directory,
        sequence_length=args.sequence_length,
        batch_size=args.batch_size,
        rank=rank,
        num_batches=args.num_batches,
        seed=args.seed,
    )
