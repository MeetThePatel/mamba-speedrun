import argparse
import glob
import json
import multiprocessing
from io import BufferedWriter
from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np
from rich.console import Console
from rich.table import Table
from tqdm import tqdm

THRESHOLDS = [128, 256, 512, 1024, 2048]
EOD_TOKEN = 50256


def _load_data_shard(file: Path) -> np.ndarray:
    """
    Load a memory-mapped token array from a binary data shard file.

    Parameters
    ----------
    file : Path
        Path to the binary file containing the data shard.

    Returns
    -------
    np.ndarray
        A read-only memory-mapped array of tokens stored as unsigned 16-bit integers.

    Notes
    -----
    - The header is 256 integers (1024 bytes) long, with:
        - `header[0]`: magic number (== 20240520)
        - `header[1]`: version number (== 1)
        - `header[2]`: number of tokens in the shard.
    - Token data starts at byte offset 1024.
    """
    header = np.fromfile(file, dtype=np.int32, count=256)
    assert header[0] == 20240520, "Magic number mismatch."
    assert header[1] == 1, "Unsupported version."
    num_tokens = header[2]

    tokens = np.memmap(file, dtype=np.uint16, mode="r", offset=256 * 4, shape=(num_tokens,))
    return tokens


def get_document_lengths_from_shard(filepath: Path) -> List[int]:
    """
    Extract the lengths of individual documents from a token shard file.

    Documents are assumed to be separated by EOD token (50256).
    The length of each document (in tokens) is computed as the number of tokens
    between consecutive EOD tokens.

    Parameters
    ----------
    filepath : Path
        Path to the binary file containing the data shard.

    Returns
    -------
    List[int]
        A list of integers representing the number of tokens in each document.
        Only non-zero-length documents are returned.
    """
    tokens = _load_data_shard(filepath)
    eod_indices = np.where(tokens == EOD_TOKEN)[0]
    boundaries = np.concatenate(([-1], eod_indices, [len(tokens) - 1]))
    doc_lengths = np.diff(boundaries)
    return doc_lengths[doc_lengths > 0].tolist()


def collect_document_lengths(file_pattern: str, nworkers: int) -> List[int]:
    """
    Collects the lengths of documents from multiple files (shards) in parallel.

    This function searches for files matching the provided glob pattern and processes
    each file in parallel using a multiprocessing pool. It retrieves the lengths of
    the documents contained within each file and aggregates them into a single list.

    Parameters
    ----------
    file_pattern : str
        A glob pattern specifying which files to process (e.g. 'data/shard_*.bin').
    nworkers : int
        Number of worker processe

    Returns
    -------
    List[int]
        A list containing the document lengths from all processed files.
    """
    files = sorted(glob.glob(file_pattern))
    all_doc_lengths = []
    with multiprocessing.Pool(processes=nworkers) as pool:
        pbar = tqdm(pool.imap_unordered(get_document_lengths_from_shard, files), total=len(files))
        for lengths_from_shard in pbar:
            if lengths_from_shard:
                all_doc_lengths.extend(lengths_from_shard)
            pbar.set_description(f"Total documents found: {len(all_doc_lengths):,}")
    return all_doc_lengths


def calculate_statistics(document_lengths: List[int]) -> Dict[str, Union[int, float]]:
    """
    Calculate statistics for a list of document lengths.

    Parameters
    ----------
    doc_lengths : List[int]
        A list containing the document lengths.

    Returns
    -------
    Dict[str, Union[int, float]]
        A dictionary containing summary statistics.
        - total_docs (int)
        - total_tokens (int)
        - mean_length (float)
        - median_length (float)
        - min_length (int)
        - max_length (int)
        - std_dev (float)
    """
    lengths_np = np.array(document_lengths)
    return {
        "total_docs": len(lengths_np),
        "total_tokens": np.sum(lengths_np),
        "mean_length": np.mean(lengths_np),
        "median_length": np.median(lengths_np),
        "min_length": np.min(lengths_np),
        "max_length": np.max(lengths_np),
        "std_dev": np.std(lengths_np),
    }


def print_statistics_table(stats: Dict[str, Union[int, float]], console: Console):
    """
    Print formatted table of document length statistics.

    Parameters
    ----------
    stats : Dict[str, Union[int, float]]
        A dictionary containing precomputed document length statistics.
        Expected keys are:
        - 'total_docs'
        - 'total_tokens'
        - 'mean_length'
        - 'median_length'
        - 'min_length'
        - 'max_length'
        - 'std_dev'

    console : rich.console.Console
        A console instance used to render the output.
    """
    console.rule("Document length statistics")
    stats_table = Table(show_header=False, expand=False)
    stats_table.add_column("Metric")
    stats_table.add_column("Value")
    stats_table.add_row("Total Documents Found", f"{stats['total_docs']:,}")
    stats_table.add_row("Total Tokens in Documents", f"{stats['total_tokens']:,}")
    stats_table.add_row("Mean Length", f"{stats['mean_length']:,.2f}")
    stats_table.add_row("Median Length", f"{stats['median_length']:,.2f}")
    stats_table.add_row("Min Length", f"{stats['min_length']:,.2f}")
    stats_table.add_row("Max Length", f"{stats['max_length']:,.2f}")
    stats_table.add_row("Standard Deviation", f"{stats['std_dev']:,.2f}")
    console.print(stats_table)


def print_buckets_table(document_lengths: List[int], console: Console):
    """
    Print a formatted table showing document length distribution across predefined buckets.

    The buckets are: [128, 256, 512, 1024, 2048].

    Parameters
    ----------
    document_lengths : List[int]
        A list containing the lengths of documents, where each value represents
        the number of tokens (e.g., words or subwords) in a document.

    console : rich.console.Console
        A console instance used to render the output.
    """
    lengths_np = np.array(document_lengths)
    total_docs = len(lengths_np)
    buckets = []
    buckets.append((f"< {THRESHOLDS[0]}", lengths_np < THRESHOLDS[0]))
    for i in range(len(THRESHOLDS) - 1):
        buckets.append((f"{THRESHOLDS[i]} - {THRESHOLDS[i + 1]}", (lengths_np >= THRESHOLDS[i]) & (lengths_np < THRESHOLDS[i + 1])))
    buckets.append((f">= {THRESHOLDS[-1]}", lengths_np >= THRESHOLDS[-1]))

    console.rule("Length percentiles")
    buckets_table = Table(title="Sequence length buckets.")
    buckets_table.add_column("Bucket", justify="left")
    buckets_table.add_column("Documents", justify="right")
    buckets_table.add_column("Percentage", justify="right")
    buckets_table.add_column("Total Tokens", justify="right")

    for bucket_name, mask in buckets:
        count = np.sum(mask)
        total_tokens_in_bucket = np.sum(lengths_np[mask])
        percentage = (count / total_docs * 100) if total_docs > 0 else 0
        buckets_table.add_row(bucket_name, f"{count:,}", f"{percentage:.2f}%", f"{total_tokens_in_bucket:,}")

    console.print(buckets_table)


def create_histogram(document_lengths: List[int], output_image_path: str):
    """
    Create and save a log-log scaled histogram showing the distribution of document lengths.

    Vertical reference lines are drawn at 128, 256, 512, 1024, and 2048.

    Parameters
    ----------
    document_lengths : List[int]
        A list containing the document lengths.

    output_image_path : str
        File path where the generated histogram image will be saved.
    """
    import matplotlib.pyplot as plt

    lengths_np = np.array(document_lengths)
    plt.figure(figsize=(12, 7))
    plt.hist(lengths_np, bins=np.logspace(1, np.log10(lengths_np.max()), 100), color="skyblue", edgecolor="black")
    for length in [128, 256, 512, 1024, 2048]:
        plt.axvline(x=length, color="red")
    plt.title("Document Length Distribution")
    plt.xlabel("Document Length (Tokens)")
    plt.ylabel("Number of Documents")
    plt.grid(True, which="both", ls="--", c="0.7")
    plt.xscale("log")
    plt.yscale("log")
    plt.tight_layout()
    plt.savefig(output_image_path)


def process_shard(filepath: Path) -> Dict[int, List[np.ndarray]]:
    """
    Process a shard file and group documents by length thresholds.

    Parameters
    ----------
    filepath : Path
        Path to the binary file containing the data shard.

    Returns
    -------
    Dict[int, List[np.ndarray]]
        Dictionary mapping threshold values to lists of document arrays.
    """
    header = np.fromfile(filepath, dtype=np.int32, count=256)
    assert header[0] == 20240520
    assert header[1] == 1
    num_tokens = header[2]
    tokens = np.memmap(filepath, dtype=np.uint16, mode="r", offset=256 * 4, shape=(num_tokens,))

    eod_indices = np.where(tokens == EOD_TOKEN)[0]
    boundaries = np.concatenate(([-1], eod_indices, [len(tokens) - 1]))
    grouped_docs = {b: [] for b in THRESHOLDS}

    for i in range(len(boundaries) - 1):
        start_idx = boundaries[i] + 1
        end_idx = boundaries[i + 1]

        if start_idx >= end_idx:
            continue

        doc = tokens[start_idx:end_idx]
        doc_length = len(doc)

        assigned = False
        for threshold in THRESHOLDS:
            if doc_length <= threshold:
                grouped_docs[threshold].append(doc)
                assigned = True
                break

        if not assigned:
            max_threshold = THRESHOLDS[-1]
            clipped_doc = doc[:max_threshold]
            grouped_docs[max_threshold].append(clipped_doc)

    return grouped_docs


def setup_output_directory(output_dir: str) -> Path:
    """
    Ensure the output directory exists, creating it if necessary.

    Parameters
    ----------
    output_dir : str
        The path to the directory that should be created or validated.

    Returns
    -------
    pathlib.Path
        A Path object representing the absolute path to the output directory.
    """
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    return output_dir_path


def init_output_files(output_dir: Path, world_size: int) -> Tuple[Dict[Tuple[int, int], BufferedWriter], List[str]]:
    """
    Initialize binary write files for each bucket in the given output directory.

    For each threshold in [128, 256, 512, 1024, 2048] and each split 0-7, a binary file named `bucket_<threshold>_<split>.bin`
    is created in the output directory.

    Parameters
    ----------
    output_dir : pathlib.Path
        Directory where the bucket files will be created.
    world_size : int
        World size for training run.

    Returns
    -------
    Tuple[Dict[Tuple[int, int], io.BufferedWriter], List[str]]
        A tuple containing a dictionary mapping each threshold (bucket key) + split_id pair to its corresponding opened
        binary file stream, and a list of filenames.
    """
    output_files: Dict[Tuple[int, int], BufferedWriter] = {}
    filenames: List[str] = []
    for bucket in THRESHOLDS:
        for split_id in range(world_size):
            filename = f"bucket_{bucket}_{split_id}.bin"
            filenames.append(filename)
            output_path = output_dir / filename
            output_files[(bucket, split_id)] = open(output_path, "wb")
    return output_files, filename


def process_and_write_shards(
    files: List[str], nworkers: int, output_files: Dict[int, BufferedWriter], world_size: int
) -> Tuple[Dict[int, int], Dict[int, int], Dict[str, int]]:
    """
    Process input shards in parallel and write grouped documents into respective bucket files.

    Each shard is processed in parallel using a multiprocessing pool. Documents are grouped
    by length-based buckets and written to their corresponding output files along with an EOD token.

    Parameters
    ----------
    files : List[str]
        A list of paths to input shard files to be processed.
    nworkers : int
        Number of worker processes.
    output_files : Dict[int, io.BufferedWriter]
        Dictionary mapping bucket thresholds to pre-opened binary output files.
    world_size : int
        World size for training run.

    Returns
    -------
    Tuple[Dict[int, int], Dict[int, int], Dict[str, int]]
        - First dict maps each bucket to the number of documents written.
        - Second dict maps each bucket to the total number of tokens written.
        - Third dict maps each filename to token count.
    """
    doc_counts = {b: 0 for b in THRESHOLDS}
    token_counts = {b: 0 for b in THRESHOLDS}
    split_counters = {b: 0 for b in THRESHOLDS}
    eod_token_np = np.array([EOD_TOKEN], dtype=np.uint16)
    file_token_counts = {}

    with multiprocessing.Pool(processes=nworkers) as pool:
        pbar = tqdm(pool.imap_unordered(process_shard, files), total=len(files), desc="Processing shards")
        for grouped_docs in pbar:
            if not grouped_docs:
                continue
            for bucket, docs_list in grouped_docs.items():
                if not docs_list:
                    continue
                doc_counts[bucket] += len(docs_list)
                for doc in docs_list:
                    # Round-robin across splits
                    split_id = split_counters[bucket] % world_size
                    split_counters[bucket] += 1
                    filename = f"bucket_{bucket}_{split_id}.bin"

                    output_files[(bucket, split_id)].write(doc.tobytes())
                    output_files[(bucket, split_id)].write(eod_token_np.tobytes())
                    token_counts[bucket] += len(doc)
                    file_token_counts[filename] = file_token_counts.get(filename, 0) + len(doc)

    return doc_counts, token_counts, file_token_counts


def write_metadata_file(output_dir: Path, file_mapping: Dict[str, int]):
    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(file_mapping, f, indent=2)


def print_report(document_counts: Dict[int, int], token_counts: Dict[int, int], world_size: int, console: Console):
    """
    Print a formatted report showing document and token counts per bucket.

    Parameters
    ----------
    document_counts : Dict[int, int]
        Dictionary mapping each bucket to the number of documents written.
    token_counts : Dict[int, int]
        Dictionary mapping each bucket to the number of tokens written.
    world_size : int
        World size for training run.
    console : rich.console.Console
        A Console instance used to render the output.
    """
    console.rule("Document grouping report")
    report_table = Table(title="Document grouping statistics")
    report_table.add_column("Bucket", justify="left")
    report_table.add_column("Num Documents", justify="right")
    report_table.add_column("% of Docs", justify="right")
    report_table.add_column("Num Tokens", justify="right")
    report_table.add_column("Files Created", justify="right")

    total_docs = sum(document_counts.values())
    total_tokens = sum(token_counts.values())

    for bucket in THRESHOLDS:
        count = document_counts[bucket]
        tokens = token_counts[bucket]
        percent_docs = (count / total_docs * 100) if total_docs > 0 else 0
        files_created = f"bucket_{bucket}_0.bin to bucket_{bucket}_{world_size - 1}.bin"
        report_table.add_row(f"{bucket}", f"{count:,}", f"{percent_docs:.2f}%", f"{tokens:,}", files_created)

    report_table.add_row("TOTAL", f"{total_docs:,}", "100.00%", f"{total_tokens:,}", f"{len(THRESHOLDS) * world_size} files", style="bold")

    console.print(report_table)
    console.print(f"Processing finished. Each bucket was split into {world_size} files.")


def main():
    parser = argparse.ArgumentParser(description="Document sharding utility")
    subparsers = parser.add_subparsers(dest="command", required=True)

    stats_parser = subparsers.add_parser("stats", help="Show document length statistics.")
    stats_parser.add_argument("--file_pattern", type=str, required=True, help="Glob pattern for shard files")
    stats_parser.add_argument("--nworkers", type=int, default=4)
    stats_parser.add_argument("--output_image", type=str, help="Optional path to save histogram PNG")

    bucket_parser = subparsers.add_parser("bucket", help="Group and bucket documents by length.")
    bucket_parser.add_argument("--file_pattern", type=str, required=True, help="Glob pattern for shard files")
    bucket_parser.add_argument("--world_size", type=int, required=True, help="World size for training run")
    bucket_parser.add_argument("--nworkers", type=int, default=4)
    bucket_parser.add_argument("--output_dir", type=str, required=True, help="Output directory for bucketed binary files")

    args = parser.parse_args()
    console = Console()

    if args.command == "stats":
        document_lengths = collect_document_lengths(args.file_pattern, args.nworkers)
        stats = calculate_statistics(document_lengths)
        print_statistics_table(stats, console)
        print_buckets_table(document_lengths, console)
        if args.output_image:
            create_histogram(document_lengths, args.output_image)
            console.print(f"Plot saved to {args.output_image}")

    elif args.command == "bucket":
        files = sorted(glob.glob(args.file_pattern))
        if not files:
            console.print(f"[red]Error: no files found for pattern '{args.file_pattern}'[/red]")
            return
        output_dir = setup_output_directory(args.output_dir)
        output_files, all_filenames = init_output_files(output_dir, args.world_size)
        try:
            doc_counts, token_counts, file_token_counts = process_and_write_shards(files, args.nworkers, output_files, args.world_size)
            write_metadata_file(output_dir, file_token_counts)
            print_report(doc_counts, token_counts, args.world_size, console)
        finally:
            for f in output_files.values():
                f.close()


if __name__ == "__main__":
    main()
