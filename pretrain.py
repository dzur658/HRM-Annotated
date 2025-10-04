from typing import Optional, Any, Sequence, List
from dataclasses import dataclass
import os
import math
import yaml
import shutil

import torch
import torch.distributed as dist
from torch import nn
from torch.utils.data import DataLoader

import tqdm
import wandb
import coolname
import hydra
import pydantic
from omegaconf import DictConfig
from adam_atan2 import AdamATan2

from puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig, PuzzleDatasetMetadata
from utils.functions import load_model_class, get_model_source_path
from models.sparse_embedding import CastedSparseEmbeddingSignSGD_Distributed

# Defines the configuration for the loss function with a name
class LossConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra='allow')
    
    name: str

# Specify the architecture and associated loss configuration
class ArchConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra='allow')

    name: str
    loss: LossConfig

# Defines parameters for pretraining to run
class PretrainConfig(pydantic.BaseModel):
    # Configure the Architecture
    arch: ArchConfig
    # Path to dataset
    data_path: str

    # Hyperparams
    global_batch_size: int
    epochs: int

    lr: float
    lr_min_ratio: float
    lr_warmup_steps: int

    weight_decay: float
    beta1: float
    beta2: float

    # Puzzle embedding hyperparams
    puzzle_emb_lr: float
    puzzle_emb_weight_decay: float

    # Names
    project_name: Optional[str] = None
    run_name: Optional[str] = None
    checkpoint_path: Optional[str] = None

    # Extras
    seed: int = 0
    checkpoint_every_eval: bool = False
    eval_interval: Optional[int] = None
    # Optionally save certain tensors for inspection and analysis during evaluation
    eval_save_outputs: List[str] = []

# Holds all the "moving parts" of the training loop
@dataclass
class TrainState:
    # The HRM model itself at the current state of training
    model: nn.Module
    # Optimizer objects (AdamATan2, CastedSparseEmbeddingSignSGD_Distributed, etc.)
    optimizers: Sequence[torch.optim.Optimizer]
    # Learning rates for each optimizer
    optimizer_lrs: Sequence[float]
    # Holds the hidden state of the high and low modules
    carry: Any

    # Track steps
    step: int
    # Total number of steps to run for
    total_steps: int

# Set up the dataloader for training or evaluation
def create_dataloader(config: PretrainConfig, split: str, rank: int, world_size: int, **kwargs):
    # Initiate loading of the puzzle dataset
    dataset = PuzzleDataset(PuzzleDatasetConfig(
        # set seed for reproducibility
        seed=config.seed,

        # dataset path
        dataset_path=config.data_path,

        # multi-gpu optimization to ensure each GPU gets it's own unique subset of data
        rank=rank,
        num_replicas=world_size,
        
        **kwargs
    ), split=split)

    # Wrap in pytorch dataloader
    dataloader = DataLoader(
        # The puzzle dataset
        dataset,
        # Set to none because the puzzle dataset already handles batching internally
        batch_size=None,

        # Set 1 CPU process
        num_workers=1,
        # Each CPU will focus on loading 8 batches ahead of time
        prefetch_factor=8,

        # Optimization to speed up memory transfer from RAM to GPU
        pin_memory=True,
        # Keep workers alive to prevent overhead of respawning them
        persistent_workers=True
    )
    # Return both the dataloader and the metadata about the dataset (for tracking vocab size, etc.)
    return dataloader, dataset.metadata

# Build the HRM model!
def create_model(config: PretrainConfig, train_metadata: PuzzleDatasetMetadata, world_size: int):
    # Create a dictionary for all the model configuration parameters
    model_cfg = dict(
        **config.arch.__pydantic_extra__,  # type: ignore

        # For multi-gpu training get the local batch size for each GPU (world size is the total number of GPU workers)
        batch_size=config.global_batch_size // world_size,

        # Pull vocab size, sequence length, and number of puzzle identifiers from the dataset metadata
        vocab_size=train_metadata.vocab_size,
        seq_len=train_metadata.seq_len,
        num_puzzle_identifiers=train_metadata.num_puzzle_identifiers,
        # Encoder only architecture (Bert-like)
        causal=False  # Non-autoregressive
    )


    # Instantiate model with loss head based on names in the configuration file
    model_cls = load_model_class(config.arch.name)
    loss_head_cls = load_model_class(config.arch.loss.name)

    # Create model directly on the GPU
    with torch.device("cuda"):
        # Instantiate the base model
        model: nn.Module = model_cls(model_cfg)
        # Wrap the model with the loss head
        model = loss_head_cls(model, **config.arch.loss.__pydantic_extra__)  # type: ignore
        # Enable torch compile to optimize the model further
        # Disable for debugging and clearer error messages
        if "DISABLE_COMPILE" not in os.environ:
            model = torch.compile(model, dynamic=False)  # type: ignore

        # Broadcast parameters from rank 0
        # Copy the same random weights to all GPUs
        if world_size > 1:
            with torch.no_grad():
                for param in list(model.parameters()) + list(model.buffers()):
                    dist.broadcast(param, src=0)

    # Optimizers and lr
    optimizers = [
        # Set up the sparse puzzle embedding optimizer to address sparse gradients across the puzzle embeddings
        CastedSparseEmbeddingSignSGD_Distributed(
            model.model.puzzle_emb.buffers(),  # type: ignore
            
            lr=0,  # Needs to be set by scheduler
            weight_decay=config.puzzle_emb_weight_decay,

            world_size=world_size
        ),
        # AdamATan2 optimizer for the rest of the model parameters (variant of the Adam optimizer)
        # Addresses scale invariance making the model less sensitive to the scale of the weights
        # and ultimately make the learning rate easier to tune
        AdamATan2(
            model.parameters(),

            lr=0,  # Needs to be set by scheduler
            weight_decay=config.weight_decay,
            betas=(config.beta1, config.beta2)
        )
    ]
    # Put optimizer learning rates in a list
    optimizer_lrs = [
        config.puzzle_emb_lr,
        config.lr
    ]
    
    # Return the model, optimizers, and their learning rates
    return model, optimizers, optimizer_lrs

# learning rate schedule with warmup and cosine decay to dynamically adjust the learning rate during training
def cosine_schedule_with_warmup_lr_lambda(
    current_step: int, *, base_lr: float, num_warmup_steps: int, num_training_steps: int, min_ratio: float = 0.0, num_cycles: float = 0.5
):
    # increase linearlly during warmup towards the base learning rate
    if current_step < num_warmup_steps:
        return base_lr * float(current_step) / float(max(1, num_warmup_steps))

    # Gradually decrease learning rate following a cosine decay schedule
    progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
    return base_lr * (min_ratio + max(0.0, (1 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))))

# Create the starting train state
def init_train_state(config: PretrainConfig, train_metadata: PuzzleDatasetMetadata, world_size: int):
    # Estimated total training steps
    total_steps = int(config.epochs * train_metadata.total_groups * train_metadata.mean_puzzle_examples / config.global_batch_size)

    # Model
    model, optimizers, optimizer_lrs = create_model(config, train_metadata, world_size=world_size)

    # Return the train state dataclass
    return TrainState(
        # before first step
        step=0,
        total_steps=total_steps,

        model=model,
        optimizers=optimizers,
        optimizer_lrs=optimizer_lrs,
        # Initial carry is None, will be initialized at first forward pass
        carry=None
    )

# Save checkpoints
def save_train_state(config: PretrainConfig, train_state: TrainState):
    # FIXME: Only saved model.
    # Look for check point path in cfg_pretrain.yaml
    if config.checkpoint_path is None:
        return

    # Check for existence of checkpoint directory, create if it doesn't exist
    os.makedirs(config.checkpoint_path, exist_ok=True)

    # Save model checkpoint
    torch.save(train_state.model.state_dict(), os.path.join(config.checkpoint_path, f"step_{train_state.step}"))

# Helper function for learning rate scheduler
def compute_lr(base_lr: float, config: PretrainConfig, train_state: TrainState):
    return cosine_schedule_with_warmup_lr_lambda(
        current_step=train_state.step,
        base_lr=base_lr,
        num_warmup_steps=round(config.lr_warmup_steps),
        num_training_steps=train_state.total_steps,
        min_ratio=config.lr_min_ratio
    )

# Train a single batch
def train_batch(config: PretrainConfig, train_state: TrainState, batch: Any, global_batch_size: int, rank: int, world_size: int):
    # update the step
    train_state.step += 1
    # Stop if we have reached the max number of training steps
    if train_state.step > train_state.total_steps:  # At most train_total_steps
        return

    # Move batch to device
    batch = {k: v.cuda() for k, v in batch.items()}

    # Init carry if it is None
    # Initialize the hidden state of the high and low modules if this is the first batch
    if train_state.carry is None:
        with torch.device("cuda"):
            train_state.carry = train_state.model.initial_carry(batch)  # type: ignore

    # Forward returns new hidden state for next step, loss, metrics, and discards other returned values
    train_state.carry, loss, metrics, _, _ = train_state.model(carry=train_state.carry, batch=batch, return_keys=[])

    # Backward
    ((1 / global_batch_size) * loss).backward()

    # Allreduce make sure to sum gradients across GPUs and synchronize them
    if world_size > 1:
        for param in train_state.model.parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad)
            
    # Apply optimizer
    lr_this_step = None    
    for optim, base_lr in zip(train_state.optimizers, train_state.optimizer_lrs):
        # Compute and set the learning rate for this step
        lr_this_step = compute_lr(base_lr, config, train_state)

        # Update the optimizers with the new learning rate
        for param_group in optim.param_groups:
            param_group['lr'] = lr_this_step
        
        # Step the optimizer and zero the gradients
        optim.step()
        optim.zero_grad()

    # Reduce metrics
    if len(metrics):
        assert not any(v.requires_grad for v in metrics.values())

        metric_keys = list(sorted(metrics.keys()))  # Sort keys to guarantee all processes use the same order.
        # Reduce and reconstruct
        metric_values = torch.stack([metrics[k] for k in metric_keys])
        if world_size > 1:
            dist.reduce(metric_values, dst=0)

        # Process and return metrics on the first GPU
        if rank == 0:
            metric_values = metric_values.cpu().numpy()
            reduced_metrics = {k: metric_values[i] for i, k in enumerate(metric_keys)}
            
            # Postprocess
            count = max(reduced_metrics["count"], 1)  # Avoid NaNs
            reduced_metrics = {f"train/{k}": v / (global_batch_size if k.endswith("loss") else count) for k, v in reduced_metrics.items()}

            reduced_metrics["train/lr"] = lr_this_step
            return reduced_metrics


def evaluate(config: PretrainConfig, train_state: TrainState, eval_loader: torch.utils.data.DataLoader, eval_metadata: PuzzleDatasetMetadata, rank: int, world_size: int):
    # Use inference for evaluation to disable gradient tracking
    with torch.inference_mode():
        # Set up id, prediction, and metrics containers
        set_ids = {k: idx for idx, k in enumerate(eval_metadata.sets)}
        
        all_preds = {}

        metric_keys = []
        metric_values = None
        metric_global_batch_size = [0 for _ in range(len(set_ids))]
        
        carry = None

        # Iterate through all the batches in the test dataset
        for set_name, batch, global_batch_size in eval_loader:
            # Move batch to device
            batch = {k: v.cuda() for k, v in batch.items()}
            with torch.device("cuda"):

                # Solve each puzzle independently, so reset carry for each batch
                carry = train_state.model.initial_carry(batch)  # type: ignore

            # Forward (ACT solves the puzzle in multiple steps until it reaches a solution)
            while True:
                # Run a single thinking segment
                carry, _, metrics, preds, all_finish = train_state.model(carry=carry, batch=batch, return_keys=config.eval_save_outputs)
                
                # Break if the Q Head returns true signaling to stop
                if all_finish:
                    break
            
            # Save any tensors for inspection
            for collection in (batch, preds):
                for k, v in collection.items():
                    if k in config.eval_save_outputs:
                        all_preds.setdefault(k, [])
                        all_preds[k].append(v.cpu())  # Move to CPU for saving GPU memory

            # Free memory            
            del carry, preds, batch, all_finish

            # Aggregate
            set_id = set_ids[set_name]
            
            # Check if first batch to set up metric storage
            if metric_values is None:
                # dictionary of metric keys
                metric_keys = list(sorted(metrics.keys()))  # Sort keys to guarantee all processes use the same order.

                # assemble metric values tensor
                metric_values = torch.zeros((len(set_ids), len(metrics.values())), dtype=torch.float32, device="cuda")

            # Compress and add metric values
            metric_values[set_id] += torch.stack([metrics[k] for k in metric_keys])
            # Running total of examples processed for each dataset
            metric_global_batch_size[set_id] += global_batch_size

        # Save predictions made during evaluation
        if len(all_preds) and config.checkpoint_path is not None:
            all_preds = {k: torch.cat(v, dim=0) for k, v in all_preds.items()}

            os.makedirs(config.checkpoint_path, exist_ok=True)
            torch.save(all_preds, os.path.join(config.checkpoint_path, f"step_{train_state.step}_all_preds.{rank}"))

        # Logging
        # Reduce to rank 0
        if metric_values is not None:
            # Get all metrics across GPUs
            if world_size > 1:
                dist.reduce(metric_values, dst=0)
            
            if rank == 0:
                # On the main GPU process, process and return metrics
                reduced_metrics = metric_values.cpu().numpy()
                reduced_metrics = {set_name: {metric_name: reduced_metrics[set_id, metric_id] for metric_id, metric_name in enumerate(metric_keys)}
                                   for set_id, set_name in enumerate(set_ids)}
                
                # Postprocess
                for set_name, metrics in reduced_metrics.items():
                    # Get total number of examples processed
                    count = metrics.pop("count")
                    # Divide each metric by the count to get the average for each metric
                    reduced_metrics[set_name] = {k: v / count for k, v in metrics.items()}
                
                # Return all metrics
                return reduced_metrics

# Save configuration and source code for the run
def save_code_and_config(config: PretrainConfig):
    if config.checkpoint_path is None or wandb.run is None:
        return

    # Check for existence of checkpoint directory, create if it doesn't exist
    os.makedirs(config.checkpoint_path, exist_ok=True)

    # Copy code
    code_list = [
        # Get the architecture and loss source code paths
        get_model_source_path(config.arch.name),
        get_model_source_path(config.arch.loss.name)
    ]

    # Get code files and copy
    for code_file in code_list:
        if code_file is not None:
            code_name = os.path.basename(code_file)

            shutil.copy(code_file, os.path.join(config.checkpoint_path, code_name))

    # Dump config as yaml
    config_file = os.path.join(config.checkpoint_path, "all_config.yaml")
    with open(config_file, "wt") as f:
        yaml.dump(config.model_dump(), f)

    # Log code
    wandb.run.log_code(config.checkpoint_path)

# Load the configuration and sync across all GPUs
def load_synced_config(hydra_config: DictConfig, rank: int, world_size: int) -> PretrainConfig:
    # Initialize list to hold the config object
    objects = [None]
    # load the configuration on the main GPU
    if rank == 0:
        # Get the config from the hydra config and parse into the PretrainConfig dataclass
        config = PretrainConfig(**hydra_config)  # type: ignore

        # Naming
        if config.project_name is None:
            config.project_name = f"{os.path.basename(config.data_path).capitalize()} ACT-torch"
        if config.run_name is None:
            config.run_name = f"{config.arch.name.split('@')[-1]} {coolname.generate_slug(2)}"
        if config.checkpoint_path is None:
            config.checkpoint_path = os.path.join("checkpoints", config.project_name, config.run_name)
        
        # Put the config object in the list to be broadcasted (if multi-GPU environment)
        objects = [config]

    # Broadcast the config object to all GPUs
    if world_size > 1:
        dist.broadcast_object_list(objects, src=0)

    return objects[0]  # type: ignore

# Orchestrate the training process across multiple GPUs
@hydra.main(config_path="config", config_name="cfg_pretrain", version_base=None)
def launch(hydra_config: DictConfig):
    # Default to single GPU
    RANK = 0
    WORLD_SIZE = 1

    # Initialize distributed training if in distributed environment (e.g. torchrun)
    if "LOCAL_RANK" in os.environ:
        # Initialize communication between GPUs
        dist.init_process_group(backend="nccl")

        # Set the default device and dtype
        RANK = dist.get_rank()
        # Get the total number of GPU workers
        WORLD_SIZE = dist.get_world_size()

        # Set the GPU device to the local rank
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        
    # Load sync'ed config
    config = load_synced_config(hydra_config, rank=RANK, world_size=WORLD_SIZE)

    # Seed RNGs to ensure consistency
    torch.random.manual_seed(config.seed + RANK)

    # Dataset
    # number of epochs to train before each evaluation
    train_epochs_per_iter = config.eval_interval if config.eval_interval is not None else config.epochs
    # total number of train/eval cycles
    total_iters = config.epochs // train_epochs_per_iter

    # Ensure eval interval is a divisor of total epochs
    assert config.epochs % train_epochs_per_iter == 0, "Eval interval must be a divisor of total epochs."

    # Create dataloaders for training and evaluation
    train_loader, train_metadata = create_dataloader(config, "train", test_set_mode=False, epochs_per_iter=train_epochs_per_iter, global_batch_size=config.global_batch_size, rank=RANK, world_size=WORLD_SIZE)
    eval_loader,  eval_metadata  = create_dataloader(config, "test", test_set_mode=True, epochs_per_iter=1, global_batch_size=config.global_batch_size, rank=RANK, world_size=WORLD_SIZE)

    # Initialize the training state
    train_state = init_train_state(config, train_metadata, world_size=WORLD_SIZE)

    # Progress bar and logger
    progress_bar = None
    if RANK == 0:
        # progress bar for training
        progress_bar = tqdm.tqdm(total=train_state.total_steps)

        # Initialize wandb logging
        wandb.init(project=config.project_name, name=config.run_name, config=config.model_dump(), settings=wandb.Settings(_disable_stats=True))  # type: ignore
        # Log model parameters
        wandb.log({"num_params": sum(x.numel() for x in train_state.model.parameters())}, step=0)
        # Save code and config
        save_code_and_config(config)

    # Training Loop for specified number of epochs
    for _iter_id in range(total_iters):
        print (f"[Rank {RANK}, World Size {WORLD_SIZE}]: Epoch {_iter_id * train_epochs_per_iter}")

        ############ Train Iter
        # Set model to training state
        train_state.model.train()
        # Iterate through all batches in the current epoch
        for set_name, batch, global_batch_size in train_loader:
            # Get metrics from training on the batch
            metrics = train_batch(config, train_state, batch, global_batch_size, rank=RANK, world_size=WORLD_SIZE)

            if RANK == 0 and metrics is not None:
                # Use main GPU to log metrics and update progress bar
                wandb.log(metrics, step=train_state.step)
                progress_bar.update(train_state.step - progress_bar.n)  # type: ignore

        ############ Evaluation
        # Set model to evaluation state
        train_state.model.eval()
        # Evaluate the model on the evaluation dataset for each epoch
        metrics = evaluate(config, train_state, eval_loader, eval_metadata, rank=RANK, world_size=WORLD_SIZE)

        if RANK == 0 and metrics is not None:
            # Use main GPU to log metrics on evaluation
            wandb.log(metrics, step=train_state.step)
            
        ############ Checkpointing
        if RANK == 0 and (config.checkpoint_every_eval or (_iter_id == total_iters - 1)):
            save_train_state(config, train_state)

    # finalize
    if dist.is_initialized():
        dist.destroy_process_group()
    wandb.finish()


if __name__ == "__main__":
    # Initiate the training process if this script is run directly
    launch()
