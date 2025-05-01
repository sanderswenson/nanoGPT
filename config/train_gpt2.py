# config for training GPT-2 (124M) down to very nice loss of ~2.85 on 1 node of 8X A100 40GB
# launch as the following (e.g. in a screen session) and wait ~5 days:
# $ torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py

# NSA parameters
# Should be context size/16 according to the paper
local_window_size: int = 64  # Size of the local attention window
# Should be context size/64 according to the paper
block_length: int = 16  # Length of each block for sliding window compression
# These should be equal for now
stride_length: int = 16  # Stride between consecutive blocks for sliding window compression

wandb_log = True
wandb_project = 'owt'
wandb_run_name='gpt2-124M'

# these make the total batch size be ~0.5M
# 12 batch size * 1024 block size * 5 gradaccum * 8 GPUs = 491,520
batch_size = 12
block_size = 1024
gradient_accumulation_steps = 5 * 4

# this makes total number of tokens be 300B
max_iters = 1200000
lr_decay_iters = 1200000

# eval stuff
eval_interval = 1000
eval_iters = 200
log_interval = 11

# weight decay
weight_decay = 1e-1
