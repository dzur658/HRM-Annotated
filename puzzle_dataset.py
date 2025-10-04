import os
import json

import numpy as np
import pydantic

import torch
from torch.utils.data import IterableDataset, get_worker_info

from models.losses import IGNORE_LABEL_ID
from dataset.common import PuzzleDatasetMetadata

# Sample a batch of puzzles, ensuring a good variety is chosen for each batch
def _sample_batch(rng: np.random.Generator, group_order: np.ndarray, puzzle_indices: np.ndarray, group_indices: np.ndarray, start_index: int, global_batch_size: int):
    # Pack examples into a full batch
    batch = []
    # indices of puzzles in the batch
    batch_puzzle_indices = []
    # Tracks the number of examples in the batch added so far
    current_size = 0

    # Continue sampling until batch is full or we run out of groups
    while (start_index < group_order.size) and (current_size < global_batch_size):
        # Pick a group and a puzzle from that group
        group_id = group_order[start_index]
        # Pick a random puzzle from the group, and ensure a batch isn't filled with augmentations of the same puzzle
        puzzle_id = rng.integers(group_indices[group_id], group_indices[group_id + 1])
        # shift the starting index
        start_index += 1

        # Get range of the puzzle using lookup arrays
        puzzle_start = puzzle_indices[puzzle_id]
        puzzle_size = int(puzzle_indices[puzzle_id + 1] - puzzle_start)

        # Calculate the size of the examples to append
        append_size = min(puzzle_size, global_batch_size - current_size)

        # Put into batch
        batch_puzzle_indices.append(np.full(append_size, puzzle_id, dtype=np.int32))
        # A random choice of examples from the puzzle are appended
        batch.append(puzzle_start + np.random.choice(puzzle_size, append_size, replace=False))

        # Increase size tracker appropriately
        current_size += append_size

    # return the updated position, a np array with the final list of data indices, and a np array with the corresponding puzzle indices
    return start_index, np.concatenate(batch), np.concatenate(batch_puzzle_indices)


# Configuration for the puzzle dataset
class PuzzleDatasetConfig(pydantic.BaseModel):
    # seed for random operations
    seed: int
    # path to the dataset
    dataset_path: str
    # global batch size across all GPUs
    global_batch_size: int
    # Enable test set mode, which iterates through the dataset sequentially without randomization
    test_set_mode: bool

    # Number of epochs to train in an iteration
    epochs_per_iter: int  # Batch X epochs in an iteration to reduce overhead.

    # Rank of the current process (for distributed training)
    rank: int
    # Total number of replicas (processes) in distributed training
    num_replicas: int

# Class representing the puzzle dataset
class PuzzleDataset(IterableDataset):
    # initialize the dataset with configuration and split type (train/test)
    def __init__(self, config: PuzzleDatasetConfig, split: str = "train"):
        # inherit from IterableDataset
        super().__init__()
        self.config = config
        self.split = split
        self.metadata = self._load_metadata()
        
        # Checks
        assert self.config.global_batch_size % self.config.num_replicas == 0, f"Global batch size {self.config.global_batch_size} must be multiples of nodes {self.config.num_replicas}."
        # Calculate the local batch size for each replica
        self.local_batch_size = self.config.global_batch_size // self.config.num_replicas

        # State
        # Part of lazy loading, this will be filled once it is needed
        self._data = None
        # initialize the iteration counter which is used to make sure random shuffling is different for each pass through the data
        self._iters = 0

    # Read and return the dataset metadata from dataset.json
    def _load_metadata(self) -> PuzzleDatasetMetadata:
        with open(os.path.join(self.config.dataset_path, self.split, "dataset.json"), "r") as f:
            return PuzzleDatasetMetadata(**json.load(f))

    # The lazy loading function to avoid loading the whole dataset into memory at once
    def _lazy_load_dataset(self):
        # exit if already loaded
        if self._data is not None:
            return

        # Contains instructions for memory mapping each field
        field_mmap_modes = {
            # allow lazy memory loading via memory mapping
            "inputs": "r",
            "labels": "r",

            # Keep indices (lookup arrays) in memory
            "puzzle_identifiers": None,
            "puzzle_indices": None,
            "group_indices": None
        }

        # Load data
        self._data = {}
        for set_name in self.metadata.sets:
            # Load subset
            self._data[set_name] = {
                # load inputs and labels as memory-mapped arrays according to field_mmap_modes
                # load indices fully into memory as per field_mmap_modes
                field_name: np.load(os.path.join(self.config.dataset_path, self.split, f"{set_name}__{field_name}.npy"), mmap_mode=mmap_mode)
                for field_name, mmap_mode in field_mmap_modes.items()
            }

    # Collect and combine batches to prepare them for the GPU(s)
    def _collate_batch(self, batch):
        # Convert dtype to 32 bit integers
        batch = {k: v.astype(np.int32) for k, v in batch.items()}

        # Convert ignore label IDs, these are used to not penalize the model for certain parts of its predictions
        if self.metadata.ignore_label_id is not None:
            batch["labels"][batch["labels"] == self.metadata.ignore_label_id] = IGNORE_LABEL_ID

        # Pad
        # Check if the batch size is less than the local batch size, and pad if necessary
        if batch["puzzle_identifiers"].size < self.local_batch_size:
            # Calculate padding size
            pad_size = self.local_batch_size - batch["puzzle_identifiers"].size

            # Fill pad values with IGNORE_LABEL_ID for labels, and appropriate pad IDs for other fields
            pad_values = {
                "inputs": self.metadata.pad_id,
                "labels": IGNORE_LABEL_ID,

                "puzzle_identifiers": self.metadata.blank_identifier_id
            }
            # Construct the padded batch
            batch = {k: np.pad(v, ((0, pad_size), ) + ((0, 0), ) * (v.ndim - 1), constant_values=pad_values[k]) for k, v in batch.items()}

        # To tensor (convert np arrays to torch tensors)
        return {k: torch.from_numpy(v) for k, v in batch.items()}
    
    # Evaluate performance on every single example in the dataset sequentially
    def _iter_test(self):
        for set_name, dataset in self._data.items():  # type: ignore
            # Calculate examples for the current set
            total_examples = len(dataset["inputs"])

            # Load examples one by one
            start_index = 0
            while start_index < total_examples:
                # Compute indices
                end_index = min(total_examples, start_index + self.config.global_batch_size)
                
                # Figure out where to start and end for the current set
                local_start = start_index + self.config.rank * self.local_batch_size
                local_end   = min(start_index + (self.config.rank + 1) * self.local_batch_size, end_index)
                
                # Get batch of examples, and also puzzle IDs
                puzzle_indices = []
                # Binary search to find the puzzle index for each example, since side right finds the next puzzle, we need to subtract 1 for the current puzzle
                puzzle_index = np.searchsorted(dataset["puzzle_indices"], local_start, side="right") - 1
                # Iterate through examples to find their corresponding puzzle indices
                for i in range(local_start, local_end):
                    while puzzle_index + 1 < len(dataset["puzzle_indices"]) and i >= dataset["puzzle_indices"][puzzle_index + 1]:
                        puzzle_index += 1

                    puzzle_indices.append(puzzle_index)
                
                # assemble the final batch
                batch = self._collate_batch({
                    "inputs": dataset["inputs"][local_start: local_end],
                    "labels": dataset["labels"][local_start: local_end],
                    "puzzle_identifiers": dataset["puzzle_identifiers"][puzzle_indices]
                })
                # yield the name, batch, along with its size
                yield set_name, batch, end_index - start_index
                
                # Advance to next batch
                start_index += self.config.global_batch_size

    # Iterate through the dataset in training mode, which involves randomization and shuffling
    def _iter_train(self):
        for set_name, dataset in self._data.items():  # type: ignore
            # Increase epoch count
            self._iters += 1

            # Randomly shuffle groups
            rng = np.random.Generator(np.random.Philox(seed=self.config.seed + self._iters))

            # Shuffle the order of groups
            group_order = np.concatenate([rng.permutation(dataset["group_indices"].size - 1) for _i in range(self.config.epochs_per_iter)])
            start_index = 0

            # Iterate through the dataset and create batches
            while start_index < group_order.size:
                # Sample a batch of puzzles
                start_index, batch_indices, batch_puzzle_indices = _sample_batch(
                    rng,
                    group_order=group_order,
                    puzzle_indices=dataset["puzzle_indices"],
                    group_indices=dataset["group_indices"],
                    start_index=start_index,
                    global_batch_size=self.config.global_batch_size,
                )

                # Select current rank and collate
                global_effective_batch_size = batch_puzzle_indices.size  # Global effective batch size, excluding pads

                # Drop last batch (it would have to be padded, so it's just easier to drop it)
                if global_effective_batch_size < self.config.global_batch_size:
                    break

                # Get batch and puzzle indices for the current rank and collate
                batch_indices        = batch_indices       [self.config.rank * self.local_batch_size: (self.config.rank + 1) * self.local_batch_size]
                batch_puzzle_indices = batch_puzzle_indices[self.config.rank * self.local_batch_size: (self.config.rank + 1) * self.local_batch_size]
                batch = self._collate_batch({
                    "inputs": dataset["inputs"][batch_indices],
                    "labels": dataset["labels"][batch_indices],
                    "puzzle_identifiers": dataset["puzzle_identifiers"][batch_puzzle_indices]
                })

                # yield the name, batch, along with its size
                yield set_name, batch, global_effective_batch_size

    # Special python function used for iterables            
    def __iter__(self):
        # Gather worker info
        worker_info = get_worker_info()
        # Ensure only a single thread is being used
        assert worker_info is None or worker_info.num_workers == 1, "Multithreaded data loading is not currently supported."
        
        # Lazy load the dataset if not already loaded
        self._lazy_load_dataset()
        
        # Iterate using specified mode
        if self.config.test_set_mode:
            yield from self._iter_test()
        else:
            yield from self._iter_train()
