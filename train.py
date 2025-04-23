"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU, example:
$ python train.py --batch_size=32 --compile=False

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 train.py

To run with DDP on 4 gpus across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)
"""

import os
import time
import math
import pickle
from contextlib import nullcontext

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import torch.profiler
from torch.profiler import profile, record_function, ProfilerActivity
import tiktoken # Added for sampling if meta.pkl is not found
import torch.nn as nn
import torch.nn.functional as F

from model import GPTConfig, GPT
# --- Debug Imports ---
# from model import debug_stats, print_and_reset_debug_stats, increment_debug_step
# -------------------

# -----------------------------------------------------------------------------
# default config values designed to train a gpt2 (124M) on OpenWebText
# I/O
out_dir = 'out'
eval_interval = 2000
log_interval = 1
eval_iters = 200
eval_only = False # if True, script exits right after the first eval
always_save_checkpoint = True # if True, always save a checkpoint after each eval
init_from = 'scratch' # 'scratch' or 'resume' or 'gpt2*'
# wandb logging
wandb_log = False # disabled by default
wandb_project = 'owt'
wandb_run_name = 'gpt2' # 'run' + str(time.time())
# data
dataset = 'openwebtext'
gradient_accumulation_steps = 5 * 8 # used to simulate larger batch sizes
batch_size = 12 # if gradient_accumulation_steps > 1, this is the micro-batch size
block_size = 1024
# model
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
bias = False # do we use bias inside LayerNorm and Linear layers?
# adamw optimizer
learning_rate = 6e-4 # max learning rate
max_iters = 600000 # total number of training iterations
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = True # whether to decay the learning rate
warmup_iters = 2000 # how many steps to warm up for
lr_decay_iters = 600000 # should be ~= max_iters per Chinchilla
min_lr = 6e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
# DDP settings
backend = 'nccl' # 'nccl', 'gloo', etc.
# system
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
compile = True # use PyTorch 2.0 to compile the model to be faster
# Sampling parameters (can be adjusted or moved to config)
sample_max_new_tokens = 100
sample_temperature = 0.8
sample_top_k = 200
sample_start = "First Citizen:\nWe are accounted poor citizens, the patricians good.\nWhat authority surfeits on would relieve us: if they\nwould yield us but the superfluity, while it were\nwholesome, we might guess they relieved us humanely;\nbut they think we are too dear: the leanness that\nafflicts us, the object of our misery, is as an\ninventory to particularise their abundance;\nour sufferance is a gain to them Let us revenge this with\nour pikes, ere we become rakes: for the gods know I\n speak this in hunger for bread, not \n" # Start prompt for sampling    
# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read()) # overrides from command line or config file
config = {k: globals()[k] for k in config_keys} # will be useful for logging
# -----------------------------------------------------------------------------

# various inits, derived attributes, I/O setup
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank # each process gets a different seed
    # world_size number of processes will be training simultaneously, so we can scale
    # down the desired gradient accumulation iterations per process proportionally
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    # if not ddp, we are running on a single gpu, and one process
    master_process = True
    seed_offset = 0
    ddp_world_size = 1
tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")

if master_process:
    os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# poor man's data loader
data_dir = os.path.join('data', dataset)
# probability of sampling a short sequence
p_short_seq = 0.0

def get_batch(split):
    # We recreate np.memmap every batch to avoid a memory leak, as per
    # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
    if split == 'train':
        data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    else:
        data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')

    # Decide sequence length for this batch
    if split == 'train' and np.random.rand() < p_short_seq:
        # Sample a short sequence length
        t = torch.randint(1, block_size, (1,)).item() # T_actual between 1 and block_size-1
    else:
        # Use full block size
        t = block_size

    ix = torch.randint(len(data) - t, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+t]).astype(np.int64)) for i in ix]) # Shape (B, t)
    y = torch.stack([torch.from_numpy((data[i+1:i+1+t]).astype(np.int64)) for i in ix]) # Shape (B, t)

    if device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9

# attempt to derive vocab_size from the dataset
meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

# --- Debug Activation ---
# debug_stats['active'] = True # Set to True to enable debugging stats
# debug_stats['log_interval'] = 100 # Print stats every 100 steps (adjust as needed)
# ----------------------

# model init
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=None, dropout=dropout) # start with model_args from command line
if init_from == 'scratch':
    # init a new model from scratch
    print("Initializing a new model from scratch")
    # determine the vocab size we'll use for from-scratch training
    if meta_vocab_size is None:
        print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
    model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
elif init_from == 'resume':
    print(f"Resuming training from {out_dir}")
    # resume training from a checkpoint.
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    # force these config attributes to be equal otherwise we can't even resume training
    # the rest of the attributes (e.g. dropout) can stay as desired from command line
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = checkpoint_model_args[k]
    # create the model
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']
elif init_from.startswith('gpt2'):
    print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
    # initialize from OpenAI GPT-2 weights
    override_args = dict(dropout=dropout)
    model = GPT.from_pretrained(init_from, override_args)
    # read off the created config params, so we can store them into checkpoint correctly
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = getattr(model.config, k)
# crop down the model block size if desired, using model surgery
if block_size < model.config.block_size:
    model.crop_block_size(block_size)
    model_args['block_size'] = block_size # so that the checkpoint will have the right value
model.to(device)

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

# optimizer
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])
checkpoint = None # free up memory

# compile the model
if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model) # requires PyTorch 2.0

# wrap model into DDP container
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        # --- Add storage for validation diagnostics ---
        if split == 'val':
            logit_means = torch.zeros(eval_iters)
            logit_stds = torch.zeros(eval_iters)
            logit_mins = torch.zeros(eval_iters)
            logit_maxs = torch.zeros(eval_iters)
            entropies = torch.zeros(eval_iters)
            # Store top-k from the first batch only
            first_batch_top_k_info = None
            # Store attention entropy stats
            comp_attn_entropies = torch.zeros(eval_iters)
        # ---------------------------------------------
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                # Modify model call to potentially return diagnostics
                # For now, run separate pass to get internal state (adds overhead)
                logits, loss = model(X, Y)

                # --- Get Compressed Attention Entropy --- 
                if split == 'val' and master_process:
                    with torch.no_grad():
                        # Rerun first block attention to get internal state 
                        # NOTE: This is inefficient, ideally modify model forward to return this
                        raw_model_local = model.module if ddp else model
                        x_emb = raw_model_local.transformer.drop(raw_model_local.transformer.wte(X) + raw_model_local.transformer.wpe(torch.arange(X.size(1), device=X.device)))
                        x_ln1 = raw_model_local.transformer.h[0].ln_1(x_emb)
                        # Call the attention forward pass to get internal att_comp if possible
                        # This requires NativeSparseAttention to store att_comp as an attribute or return it
                        # For now, we estimate by recalculating part of it (assuming NativeSparseAttention logic)
                        # This is approximate and needs NativeSparseAttention modification for accuracy
                        try:
                           # --- Replicate att_comp calculation (approximate) --- 
                           attn_module = raw_model_local.transformer.h[0].attn
                           hs = attn_module.n_embd // attn_module.n_head
                           q = attn_module.q_proj(x_ln1).view(X.size(0), X.size(1), attn_module.n_head, hs).transpose(1, 2)
                           k_comp_raw = attn_module.c_attn_comp(x_ln1).split(attn_module.n_embd, dim=2)[0]
                           k_comp_raw = k_comp_raw.view(X.size(0), X.size(1), attn_module.n_head, hs).transpose(1, 2)
                           k_comp = attn_module.comp_mlp(k_comp_raw)

                           if k_comp.size(2) > 0: # If num_blocks > 0
                                scale = 1.0 / math.sqrt(k_comp.size(-1))
                                att_scores = (q @ k_comp.transpose(-2, -1)) * scale
                                
                                # Recreate causal mask (simplified view for diag.)
                                T_q = q.size(2)
                                T_k_comp = att_scores.size(-1)
                                query_indices = torch.arange(T_q, device=q.device).unsqueeze(1)
                                block_start_indices = torch.arange(T_k_comp, device=q.device).unsqueeze(0) * attn_module.stride_length
                                allowed_mask = block_start_indices <= query_indices
                                causal_mask_bool = ~allowed_mask.unsqueeze(0).unsqueeze(0)
                                
                                att_scores = att_scores.masked_fill(causal_mask_bool, float('-inf'))
                                att_comp_probs = F.softmax(att_scores, dim=-1)
                                att_comp_probs = torch.nan_to_num(att_comp_probs)
                                
                                # Calculate entropy
                                log_probs_comp = torch.log(att_comp_probs + 1e-9)
                                entropy_per_attn = -torch.sum(att_comp_probs * log_probs_comp, dim=-1) # Shape (B, nh, T_q)
                                comp_attn_entropies[k] = entropy_per_attn.mean().item()
                           else:
                                comp_attn_entropies[k] = 0.0 # Or NaN, if no blocks
                        except Exception as e_attn:
                           print(f"Warning: Failed to calculate att_comp entropy: {e_attn}")
                           comp_attn_entropies[k] = -1 # Indicate error
            # -----------------------------------------

            losses[k] = loss.item()

            # --- Collect validation diagnostics ---
            if split == 'val':
                # 1. Logit Stats
                logit_means[k] = logits.mean().item()
                logit_stds[k] = logits.std().item()
                logit_mins[k] = logits.min().item()
                logit_maxs[k] = logits.max().item()

                # Calculate probs and entropy
                # Add epsilon for numerical stability with log(0)
                probs = F.softmax(logits, dim=-1)
                log_probs = torch.log(probs + 1e-9)
                entropy_per_token = -torch.sum(probs * log_probs, dim=-1) # Shape: (B, T)
                entropies[k] = entropy_per_token.mean().item() # Average over batch and sequence

                # 2. Top-k Probabilities/Tokens (only for first batch, first example)
                if k == 0 and master_process and decode is not None:
                    try:
                        # Get top 5 probs and indices for the last token of the first sequence
                        top_probs, top_indices = torch.topk(probs[0, -1, :], k=5)
                        # Decode indices to tokens
                        top_tokens = [itos[i.item()] for i in top_indices]
                        first_batch_top_k_info = list(zip(top_tokens, top_probs.tolist()))
                    except Exception as e:
                        # Handle potential errors if itos is not loaded or decoding fails
                        print(f"Warning: Could not decode top-k tokens: {e}")
                        first_batch_top_k_info = "Decoding unavailable"
            # ------------------------------------

        out[split] = losses.mean()

        # --- Print validation diagnostics (only from master process) ---
        if split == 'val' and master_process:
            print("--- Validation Diagnostics ---")
            print(f"  Avg Logit Stats: Mean={logit_means.mean():.4f}, Std={logit_stds.mean():.4f}, Min={logit_mins.mean():.4f}, Max={logit_maxs.mean():.4f}")
            print(f"  Avg Prediction Entropy: {entropies.mean():.4f}")
            if first_batch_top_k_info:
                print(f"  Top-5 Preds (First Batch, Last Token): {first_batch_top_k_info}")
            print("----------------------------")
            # --- Print extra diagnostics ---
            if split == 'val' and master_process:
                print(f"  Avg Comp Attention Entropy: {comp_attn_entropies.mean():.4f}")
                print("----------------------------")
        # -----------------------------------------------------------

    model.train()
    return out

# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)

# logging
if wandb_log and master_process:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)

# training loop
if master_process:  # Only profile on the master process
    prof = torch.profiler.profile(
        activities=[
            ProfilerActivity.CPU,
            ProfilerActivity.CUDA,
        ],
        schedule=torch.profiler.schedule(
            wait=1,      # Skip first iteration
            warmup=1,    # Warmup for one iteration
            active=3,    # Profile for 3 iterations
            repeat=1     # Stop after one cycle
        ),
        on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/pytorch_profiler'),
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    )
    prof.start()
else:
    prof = nullcontext()

X, Y = get_batch('train') # fetch the very first batch
t0 = time.time()
local_iter_num = 0 # number of iterations in the lifetime of this process
raw_model = model.module if ddp else model # unwrap DDP container if needed
running_mfu = -1.0

# Load tokenizer for sampling
encode = None
decode = None
if master_process: # Only load tokenizer on master process
    meta_path = os.path.join(data_dir, 'meta.pkl')
    load_meta = os.path.exists(meta_path)
    if load_meta:
        print(f"Loading meta from {meta_path} for sampling...")
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)
        stoi, itos = meta['stoi'], meta['itos']
        encode = lambda s: [stoi[c] for c in s]
        decode = lambda l: ''.join([itos[i] for i in l])
    else:
        print("No meta.pkl found, assuming GPT-2 encodings for sampling...")
        enc = tiktoken.get_encoding("gpt2")
        encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
        decode = lambda l: enc.decode(l)

# Backward hook to log gradient norms
def log_gradient_norm_hook(module, grad_input, grad_output, layer_idx, module_name):
    # Log only for layer 0 and every 100 steps to reduce spam
    if layer_idx == 0 and iter_num % 100 == 0 and master_process:
        # For dropout, grad_input is relevant; for Linear, grad_output is relevant
        if isinstance(module, nn.Dropout):
            grad = grad_input[0]
        else:
            grad = grad_output[0]

        grad_norm = grad.norm().item() if grad is not None else 0.0
        print(f"  [L{layer_idx} Grad Norm] {module_name}: {grad_norm:.4f}")

# Get the raw model to access nested modules
raw_model = model.module if ddp else model

# Attach gradient logging hooks
hook_handles = [] # Store hook handles to remove them later if needed
for i, block in enumerate(raw_model.transformer.h):
    if hasattr(block, 'attn') and hasattr(block.attn, 'c_attn_local'): # Check necessary components exist
        # Factory function to capture layer index and name
        def make_grad_hook(layer_idx, module_name):
            # Note: Need to pass the actual hook function here
            def hook_wrapper(module, grad_input, grad_output):
                log_gradient_norm_hook(module, grad_input, grad_output, layer_idx, module_name)
            return hook_wrapper

        # Attach hooks
        # Q projection hook
        if hasattr(block.attn, 'q_proj'):
            handle = block.attn.q_proj.register_full_backward_hook(make_grad_hook(i, 'q_proj')) # Input Q to attention score
            hook_handles.append(handle)

        # Keep hook for Key Compression MLP output gradient
        if hasattr(block.attn, 'comp_mlp') and hasattr(block.attn.comp_mlp, 'fc2'):
            handle = block.attn.comp_mlp.fc2.register_full_backward_hook(make_grad_hook(i, 'comp_mlp_k.fc2')) # Input K to attention score (from MLP)
            hook_handles.append(handle)

        # Add hooks for the combination LayerNorms

# Optional: Clean up hooks at the end (e.g., in a finally block)
# for handle in hook_handles:
#    handle.remove()

try:
    while True:
        # determine and set the learning rate for this iteration
        lr = get_lr(iter_num) if decay_lr else learning_rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        with record_function("train_step"):
            # evaluate the loss on train/val sets and write checkpoints
            if iter_num % eval_interval == 0 and master_process:
                with record_function("evaluation"):
                    losses = estimate_loss()
                print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

                # ---- GENERATION START ----
                if iter_num > 0: # Avoid sampling at iteration 0
                    print("--- Generating Sample ---")
                    raw_model.eval() # Set model to eval mode for generation
                    start_ids = encode(sample_start)
                    x_sample = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])
                    with torch.no_grad():
                        with ctx: # Use the same autocast context
                           y_sample = raw_model.generate(x_sample, sample_max_new_tokens, temperature=sample_temperature, top_k=sample_top_k)
                    print(decode(y_sample[0].tolist()))
                    print("-------------------------")
                    raw_model.train() # Set model back to train mode
                # ---- GENERATION END ----

                if wandb_log:
                    wandb.log({
                        "iter": iter_num,
                        "train/loss": losses['train'],
                        "val/loss": losses['val'],
                        "lr": lr,
                        "mfu": running_mfu*100, # convert to percentage
                    })
                if losses['val'] < best_val_loss or always_save_checkpoint:
                    best_val_loss = losses['val']
                    if iter_num > 0:
                        with record_function("save_checkpoint"):
                            checkpoint = {
                                'model': raw_model.state_dict(),
                                'optimizer': optimizer.state_dict(),
                                'model_args': model_args,
                                'iter_num': iter_num,
                                'best_val_loss': best_val_loss,
                                'config': config,
                            }
                            print(f"saving checkpoint to {out_dir}")
                            torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))
            if iter_num == 0 and eval_only:
                break

            # forward backward update, with optional gradient accumulation to simulate larger batch size
            # and using the GradScaler if data type is float16
            for micro_step in range(gradient_accumulation_steps):
                with record_function("microstep"):
                    if ddp:
                        # in DDP training we only need to sync gradients at the last micro step.
                        # the official way to do this is with model.no_sync() context manager, but
                        # I really dislike that this bloats the code and forces us to repeat code
                        # looking at the source of that context manager, it just toggles this variable
                        model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
                    with ctx:
                        with record_function("forward"):
                            logits, loss = model(X, Y)
                        loss = loss / gradient_accumulation_steps # scale the loss to account for gradient accumulation
                    # immediately async prefetch next batch while model is doing the forward pass on the GPU
                    with record_function("get_batch"):
                        X, Y = get_batch('train')
                    # backward pass, with gradient scaling if training in fp16
                    with record_function("backward"):
                        scaler.scale(loss).backward()

            # clip the gradient
            if grad_clip != 0.0:
                with record_function("grad_clip"):
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            # step the optimizer and scaler if training in fp16
            with record_function("optimizer_step"):
                scaler.step(optimizer)
                scaler.update()
            # flush the gradients as soon as we can, no need for this memory anymore
            optimizer.zero_grad(set_to_none=True)

        # timing and logging
        t1 = time.time()
        dt = t1 - t0
        t0 = t1
        if iter_num % log_interval == 0 and master_process:
            # get loss as float. note: this is a CPU-GPU sync point
            # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
            lossf = loss.item() * gradient_accumulation_steps
            if local_iter_num >= 5: # let the training loop settle a bit
                mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
                running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
            print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")

        # --- Debug Step Increment ---
        # if master_process: # Only increment/print on master process
        #     increment_debug_step()
        # --------------------------

        iter_num += 1
        local_iter_num += 1

        if master_process:
            prof.step()  # Record the profiling step

        # termination conditions
        if iter_num > max_iters:
            break

finally:
    if master_process:
        prof.stop()  # Make sure we stop the profiler even if there's an error

if ddp:
    destroy_process_group()
