from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F
from torch import nn
from pydantic import BaseModel

from models.common import trunc_normal_init_
from models.layers import rms_norm, SwiGLU, Attention, RotaryEmbedding, CosSin, CastedEmbedding, CastedLinear
from models.sparse_embedding import CastedSparseEmbedding


@dataclass
class HierarchicalReasoningModel_ACTV1InnerCarry:
    # Holds the hidden state of the high level module
    z_H: torch.Tensor

    # Holds the hidden state of the low level module
    z_L: torch.Tensor


@dataclass
class HierarchicalReasoningModel_ACTV1Carry:
    # Holds the inner state for the modules
    inner_carry: HierarchicalReasoningModel_ACTV1InnerCarry
    
    # Tracks the reasoning segments 1-d tensor (batch_size,)
    steps: torch.Tensor

    # a 1-d bool tensor (batch_size,) indicating whether the stop reasoning signal is given
    halted: torch.Tensor
    
    # A dictionary holding the vectorized inputs for items currently being processed
    current_data: Dict[str, torch.Tensor]


class HierarchicalReasoningModel_ACTV1Config(BaseModel):
    # integer defining the size of the current batch
    batch_size: int
    # sequence length of the input
    seq_len: int
    # puzzle embedding dimension
    puzzle_emb_ndim: int = 0
    # total number of unique tasks in the dataset
    num_puzzle_identifiers: int
    # vocabulary size of the input tokens
    vocab_size: int

    # How many updates are allowed for each module in the forward pass
    # "For each H module cycle the L module is run L_cycles times" preserving the nested architecture
    H_cycles: int
    L_cycles: int

    # Set the number of transformer blocks for the high-level and low-level modules
    H_layers: int
    L_layers: int

    # Transformer config
    # define the length of the corresponding vector for the token in the sequence
    hidden_size: int
    # Multiplies the hidden size vector by this number to temporarily expand the vector for processing by non-linear functions (SwiGLUs) and then reduces it
    # back to the original hidden size
    expansion: float
    # number of parallel attention heads assigned to each chunk. The hidden size vector is split across these heads,
    # allowing the model to focus on different parts of the input simultaneously
    num_heads: int
    # Will be either RoPE or learned positional embeddings for tracking the order of tokens
    pos_encodings: str

    # Root Mean Square Epsilon normalization of tensors, this sets the value for that epsilon to be added
    # to protect against potentially dividing by zero (see RMS Epsilon equation)
    rms_norm_eps: float = 1e-5
    # Theta value for RoPE, which defines the level of frequency for each token embedding,
    # essentially pointing at how precisely the model can know the position of a given token.
    # A larger value will result in a lower frequency of the rotational signals, which will lead to the
    # model being less sensitive to the exact positions of tokens, and more focused on broader positional
    # relationships. Conversely, a smaller theta value will increase the frequency of the rotational signals,
    # making the model more sensitive to the exact positions of tokens within the sequence.
    rope_theta: float = 10000.0
    
    # Halting Q-learning config
    # Maximum number of reasoning segments the model is allowed to perform before
    # it's cut off
    halt_max_steps: int
    # During training, a random minimum number of steps is defined to
    # encourage the model to think longer/deeper about the given
    # task. This float defines the probability of that happening
    halt_exploration_prob: float

    # Sets the precision of the forward pass (standard brain float 16, or bfloat16)
    forward_dtype: str = "bfloat16"

# Creates a simple transformer block to be used by both the H and L modules.
# The block, as is standard for a transformer block, consists of 1 attention layer,
# and one Feed Forward Network Layer (this uses SwiGLU as the non-linear activation function)
class HierarchicalReasoningModel_ACTV1Block(nn.Module):
    def __init__(self, config: HierarchicalReasoningModel_ACTV1Config) -> None:
        super().__init__()

        # Initialize the self attention layer
        self.self_attn = Attention(
            # hidden size and the number of heads come from the config
            hidden_size=config.hidden_size,
            # floor divide hidden size, by the number of heads to get the dimension of each head
            head_dim=config.hidden_size // config.num_heads,
            # number of heads comes from the config
            num_heads=config.num_heads,
            # Set the KV heads to be the same as the number of heads
            num_key_value_heads=config.num_heads,
            # Attention is bidirectional and encoder based (bert-like)
            causal=False
        )
        # Initialize the SwiGLU
        self.mlp = SwiGLU(
            # hidden size and expansion come from the config
            hidden_size=config.hidden_size,
            expansion=config.expansion,
        )
        # Store the epsilon value for RMS Epsilon normalization
        # to be used in the forward pass
        self.norm_eps = config.rms_norm_eps

    # Block computation (Attention + FFN). Takes in the hidden states and the cos/sin positional encodings (if RoPE is used)
    def forward(self, cos_sin: CosSin, hidden_states: torch.Tensor) -> torch.Tensor:
        # Post Norm - Normalization happens after computation as per the HRM foundational paper
        # Self Attention
        # Pass the hidden states through the attention layer first, and then added back to the original hidden states via a 
        # skip connection. Skip connections allow for training of very deep networks by mitigating the vanishing gradient problem
        # and allowing gradients to flow better. The result is then normalized using RMS Epsilon normalization.
        hidden_states = rms_norm(hidden_states + self.self_attn(cos_sin=cos_sin, hidden_states=hidden_states), variance_epsilon=self.norm_eps)
        # Fully Connected
        # Attention encoded hidden layer now flows into the Multi Layer Perceptron (MLP) block, which consists of two linear layers projecting up in parallel.
        # The resulting output of the first linear is then passed through an activation function, and then element-wise multiplied with the output of the second linear layer.
        # Only the resulting tensor of the first linear upscaler is passed through the activation function, and the second linear layer acts as a gating mechanism (aka remains at the pre swished value)
        # The resulting tensor is then projected back down to the original hidden size.
        # Hidden state passes through the MLP, then is added back to the original
        # hidden states via another skip connection, and then normalized again using RMS Epsilon normalization.
        hidden_states = rms_norm(hidden_states + self.mlp(hidden_states), variance_epsilon=self.norm_eps)
        # Final fully processed tensor is returned
        return hidden_states

# Stacks blocks to create a full reasoning module (either H or L)
class HierarchicalReasoningModel_ACTV1ReasoningModule(nn.Module):
    # Initializes all the individual blocks into a list of layers
    def __init__(self, layers: List[HierarchicalReasoningModel_ACTV1Block]):
        super().__init__()

        # Store the layers as a ModuleList so that PyTorch can properly register them
        # and track their parameters
        self.layers = torch.nn.ModuleList(layers)

    # Main computational logic for the module
    # If this is an L module it takes in the hidden state of the L module
    # and the input injection from the H module (and vice versa for the H module using the final state of the L module for injection)
    def forward(self, hidden_states: torch.Tensor, input_injection: torch.Tensor, **kwargs) -> torch.Tensor:
        # Input injection (add)
        hidden_states = hidden_states + input_injection
        # Layers
        for layer in self.layers:
            # The hidden state tensor is then passed through all the blocks in the module
            # and the input for the next block is the output of the previous block
            hidden_states = layer(hidden_states=hidden_states, **kwargs)
        
        # Final processed hidden states are returned
        return hidden_states

# This class provides the blue print for the orchestration of the H and L modules
class HierarchicalReasoningModel_ACTV1_Inner(nn.Module):
    def __init__(self, config: HierarchicalReasoningModel_ACTV1Config) -> None:
        super().__init__()
        # store config values of the current model
        self.config = config
        # Set the precision of the forward pass (ex bfloat16) as defined in the config class
        self.forward_dtype = getattr(torch, self.config.forward_dtype)

        # I/O
        # Used for setting the initial weights ONCE and ensuring random weights remain in the goldilocks zone
        # to avoid exploding or vanishing gradients.
        # Calculate the square root of the hidden size to be used for scaling the embeddings
        self.embed_scale  = math.sqrt(self.config.hidden_size)
        # Calculate the standard deviation for initializing the weights of the token embeddings
        embed_init_std = 1.0 / self.embed_scale

        # Converts token ids into their respective vector representations
        self.embed_tokens = CastedEmbedding(self.config.vocab_size, self.config.hidden_size, init_std=embed_init_std, cast_to=self.forward_dtype)
        # final layer to project the hidden states back to the vocabulary size for LM output (specifically to get logits for each token in the vocab)
        self.lm_head      = CastedLinear(self.config.hidden_size, self.config.vocab_size, bias=False)
        # Q head for halting decision (2 numerical outputs representing: halt & continue signals)
        self.q_head       = CastedLinear(self.config.hidden_size, 2, bias=True)

        # Calculate how many chunks of the size of the hidden size are needed to hold the puzzle embedding
        # Uses the negative value of hidden state to result in the absolute value of floor division effectively
        # rounding up (ceiling division)
        self.puzzle_emb_len = -(self.config.puzzle_emb_ndim // -self.config.hidden_size)  # ceil div
        # Checks to make sure puzzle feature is enabled correctly
        if self.config.puzzle_emb_ndim > 0:
            # Zero init puzzle embeddings
            # Only focus on updating vectors of puzzles in the current batch
            # All puzzles in the table will initially have vectors set to 0s
            self.puzzle_emb = CastedSparseEmbedding(self.config.num_puzzle_identifiers, self.config.puzzle_emb_ndim,
                                                    batch_size=self.config.batch_size, init_std=0, cast_to=self.forward_dtype)

        # LM Blocks
        # Creates either RoPE or learned positional embeddings based on the config
        if self.config.pos_encodings == "rope":
            self.rotary_emb = RotaryEmbedding(dim=self.config.hidden_size // self.config.num_heads,
                                              max_position_embeddings=self.config.seq_len + self.puzzle_emb_len,
                                              base=self.config.rope_theta)
        elif self.config.pos_encodings == "learned":
            self.embed_pos = CastedEmbedding(self.config.seq_len + self.puzzle_emb_len, self.config.hidden_size, init_std=embed_init_std, cast_to=self.forward_dtype)
        else:
            raise NotImplementedError()

        # Reasoning Layers
        # Put all the H blocks together to create the H module
        self.H_level = HierarchicalReasoningModel_ACTV1ReasoningModule(layers=[HierarchicalReasoningModel_ACTV1Block(self.config) for _i in range(self.config.H_layers)])
        # Put all the L blocks together to create the L module
        self.L_level = HierarchicalReasoningModel_ACTV1ReasoningModule(layers=[HierarchicalReasoningModel_ACTV1Block(self.config) for _i in range(self.config.L_layers)])
        
        # Initial states
        # Randomly initialize the static hidden state parameters of both the H and L modules
        self.H_init = nn.Buffer(trunc_normal_init_(torch.empty(self.config.hidden_size, dtype=self.forward_dtype), std=1), persistent=True)
        self.L_init = nn.Buffer(trunc_normal_init_(torch.empty(self.config.hidden_size, dtype=self.forward_dtype), std=1), persistent=True)

        # Q head special init
        # Init Q to (almost) zero for faster learning during bootstrapping
        # Prevents the Q Head from stopping prematurely at the start of training
        with torch.no_grad():
            self.q_head.weight.zero_()
            self.q_head.bias.fill_(-5)  # type: ignore

    # Take raw input token ids and puzzle identifiers to produce the input embeddings
    # It will create a single rich tensor combining token embeddings, puzzle embeddings (if enabled), and positional encodings
    def _input_embeddings(self, input: torch.Tensor, puzzle_identifiers: torch.Tensor):
        # Token embedding
        embedding = self.embed_tokens(input.to(torch.int32))

        # Puzzle embeddings
        # Check for puzzle configuration
        if self.config.puzzle_emb_ndim > 0:
            # Fetch the learned vector for the particular puzzle identifier
            puzzle_embedding = self.puzzle_emb(puzzle_identifiers)
            
            # Make sure the puzzle embedding is of the right shape, and add padding (0s) if not
            pad_count = self.puzzle_emb_len * self.config.hidden_size - puzzle_embedding.shape[-1]
            if pad_count > 0:
                puzzle_embedding = F.pad(puzzle_embedding, (0, pad_count))

            # Concatenate the puzzle embedding to the start of the token embeddings
            embedding = torch.cat((puzzle_embedding.view(-1, self.puzzle_emb_len, self.config.hidden_size), embedding), dim=-2)

        # Position embeddings
        if self.config.pos_encodings == "learned":
            # scale by 1/sqrt(2) to maintain forward variance
            embedding = 0.707106781 * (embedding + self.embed_pos.embedding_weight.to(self.forward_dtype))

        # Scale the embedding by the square root of the hidden size (from Transformers foundational paper)
        return self.embed_scale * embedding
    
    # Create an empty carry state for both the H and L modules
    def empty_carry(self, batch_size: int):
        return HierarchicalReasoningModel_ACTV1InnerCarry(
            z_H=torch.empty(batch_size, self.config.seq_len + self.puzzle_emb_len, self.config.hidden_size, dtype=self.forward_dtype),
            z_L=torch.empty(batch_size, self.config.seq_len + self.puzzle_emb_len, self.config.hidden_size, dtype=self.forward_dtype),
        )
    
    # State management function to reset the carry state of both the H and L modules
    # It uses a reset flag to determine which sequences in the batch need to be reset (e.g. new tasks)
    # and resets those sequences to the initial hidden states defined in the constructor.
    def reset_carry(self, reset_flag: torch.Tensor, carry: HierarchicalReasoningModel_ACTV1InnerCarry):
        # if the reset flag is true, it fills in the hidden state of the initial value,
        # but if it is false it keeps the current hidden state from the previous step
        return HierarchicalReasoningModel_ACTV1InnerCarry(
            z_H=torch.where(reset_flag.view(-1, 1, 1), self.H_init, carry.z_H),
            z_L=torch.where(reset_flag.view(-1, 1, 1), self.L_init, carry.z_L),
        )
    # Performs one full reasoning segment
    def forward(self, carry: HierarchicalReasoningModel_ACTV1InnerCarry, batch: Dict[str, torch.Tensor]) -> Tuple[HierarchicalReasoningModel_ACTV1InnerCarry, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        # Create RoPE embeddings if used
        seq_info = dict(
            cos_sin=self.rotary_emb() if hasattr(self, "rotary_emb") else None,
        )

        # Input encoding
        # Combine puzzle and input token embeddings (and positional encodings if learned)
        input_embeddings = self._input_embeddings(batch["inputs"], batch["puzzle_identifiers"])

        # Forward iterations without gradients being tracked
        with torch.no_grad():
            # set the states to the carried states
            z_H, z_L = carry.z_H, carry.z_L

            for _H_step in range(self.config.H_cycles):
                for _L_step in range(self.config.L_cycles):
                    # Update the z_L state so long as it's not the last L step of the last H step
                    if not ((_H_step == self.config.H_cycles - 1) and (_L_step == self.config.L_cycles - 1)):
                        z_L = self.L_level(z_L, z_H + input_embeddings, **seq_info)
                
                # Update the z_H state so long as it's not the last H step
                if not (_H_step == self.config.H_cycles - 1):
                    z_H = self.H_level(z_H, z_L, **seq_info)

        # Sanity check to confirm there is no gradient history from
        # the recursion loop
        assert not z_H.requires_grad and not z_L.requires_grad

        # 1-step grad
        # Get the gradients of the final states of both modules
        z_L = self.L_level(z_L, z_H + input_embeddings, **seq_info)
        z_H = self.H_level(z_H, z_L, **seq_info)

        # LM Output
        # New carry no grad
        new_carry = HierarchicalReasoningModel_ACTV1InnerCarry(z_H=z_H.detach(), z_L=z_L.detach())
        # Pass the final hidden state of the high level module to the lm head to get the prediction logits
        output = self.lm_head(z_H)[:, self.puzzle_emb_len:]

        # Q head
        # Use the hidden state of the first token (which is always a puzzle token) to decide whether to halt or continue
        # This is because the first token always has the full context of the entire sequence after processing
        q_logits = self.q_head(z_H[:, 0]).to(torch.float32)
        
        # return the new carry state, the output logits, and the halt/continue Q logits
        return new_carry, output, (q_logits[..., 0], q_logits[..., 1])

# Manages the inner model and the Adaptive Computation Time (ACT) logic
# Will call the forward pass method repeatedly and check the Q head
# for stoppage signals
class HierarchicalReasoningModel_ACTV1(nn.Module):
    """ACT wrapper."""

    # Extract the config and the model
    def __init__(self, config_dict: dict):
        super().__init__()
        self.config = HierarchicalReasoningModel_ACTV1Config(**config_dict)
        self.inner = HierarchicalReasoningModel_ACTV1_Inner(self.config)

    # Property to return the puzzle embedding layer if enabled
    @property
    def puzzle_emb(self):
        return self.inner.puzzle_emb

    # Set the initial state for ACT at the beginning of a new batch
    def initial_carry(self, batch: Dict[str, torch.Tensor]):
        # Get the batch size from the input batch
        batch_size = batch["inputs"].shape[0]

        return HierarchicalReasoningModel_ACTV1Carry(
            # Create an empty state for the z_H and z_L states of the inner model
            inner_carry=self.inner.empty_carry(batch_size),  # Empty is expected, it will be reseted in first pass as all sequences are halted.

            # Initialize the step counter for all items in the batch to 0
            steps=torch.zeros((batch_size, ), dtype=torch.int32),
            # Initialize the halted flag for all items in the batch to True so that will treat
            # every item as "new" and reset its state in the first forward pass
            halted=torch.ones((batch_size, ), dtype=torch.bool),  # Default to halted
            
            # Create an empty dictionary with the right tensor shapes to hold the current data for each item in the batch
            current_data={k: torch.empty_like(v) for k, v in batch.items()}
        )

    # The grand forward pass of ACT    
    def forward(self, carry: HierarchicalReasoningModel_ACTV1Carry, batch: Dict[str, torch.Tensor]) -> Tuple[HierarchicalReasoningModel_ACTV1Carry, Dict[str, torch.Tensor]]:
        # Update data, carry (removing halted sequences)
        new_inner_carry = self.inner.reset_carry(carry.halted, carry.inner_carry)
        
        # Set the step counter to 0 for any sequences that were halted in the previous step
        new_steps = torch.where(carry.halted, 0, carry.steps)

        # Update current data to include new sequences that were halted in the previous step
        # Replace finished data with new data from the input batch, otherwise keep the old data
        new_current_data = {k: torch.where(carry.halted.view((-1, ) + (1, ) * (batch[k].ndim - 1)), batch[k], v) for k, v in carry.current_data.items()}

        # Forward inner model
        # Call the forward method on the inner model to perform one reasoning segment
        new_inner_carry, logits, (q_halt_logits, q_continue_logits) = self.inner(new_inner_carry, new_current_data)

        # Store results from the forward pass in a dictionary
        outputs = {
            "logits": logits,
            "q_halt_logits": q_halt_logits,
            "q_continue_logits": q_continue_logits
        }
        
        with torch.no_grad():
            # Step
            new_steps = new_steps + 1
            is_last_step = new_steps >= self.config.halt_max_steps
            
            halted = is_last_step

            # if training, and ACT is enabled
            # Check if maximum allowed steps have been reached
            if self.training and (self.config.halt_max_steps > 1):
                # Halt signal
                # NOTE: During evaluation, always use max steps, this is to guarantee the same halting steps inside a batch for batching purposes
                halted = halted | (q_halt_logits > q_continue_logits)

                # Exploration
                # During training the model must think for the minimum number of steps defined by the exploration probability
                min_halt_steps = (torch.rand_like(q_halt_logits) < self.config.halt_exploration_prob) * torch.randint_like(new_steps, low=2, high=self.config.halt_max_steps + 1)

                halted = halted & (new_steps >= min_halt_steps)

                # Compute target Q
                # NOTE: No replay buffer and target networks for computing target Q-value.
                # As batch_size is large, there're many parallel envs.
                # Similar concept as PQN https://arxiv.org/abs/2407.04811
                # Peak into the future and see what the next Q would be
                next_q_halt_logits, next_q_continue_logits = self.inner(new_inner_carry, new_current_data)[-1]
                
                # Added to the outputs dictionary for loss computation of the Q head
                outputs["target_q_continue"] = torch.sigmoid(torch.where(is_last_step, next_q_halt_logits, torch.maximum(next_q_halt_logits, next_q_continue_logits)))

        # Return the updated carry and outputs
        return HierarchicalReasoningModel_ACTV1Carry(new_inner_carry, new_steps, halted, new_current_data), outputs
