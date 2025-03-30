# train a GPT model on Wikipedia dataset
# based on the shakespeare config but adapted for Wikipedia scale

out_dir = 'out-wikipedia'
eval_interval = 2000 # less frequent for larger dataset
eval_iters = 200
log_interval = 100 # less frequent logging for larger dataset

# we're less likely to overfit on this large dataset
always_save_checkpoint = True

wandb_log = False # override via command line if you like
wandb_project = 'wikipedia-gpt'
wandb_run_name = 'wiki-gpt'

dataset = 'wikipedia'
gradient_accumulation_steps = 4  # increased due to larger dataset
batch_size = 32  # reduced batch size to handle longer sequences
block_size = 512  # matches the Wikipedia dataset's pre-tokenized size

# keeping the same model architecture for comparison
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.2

# adjusted learning parameters for the larger dataset
learning_rate = 5e-4  # slightly lower for stability
max_iters = 50000  # increased for larger dataset
lr_decay_iters = 50000  # make equal to max_iters usually
min_lr = 5e-5  # learning_rate / 10 usually
beta2 = 0.99

warmup_iters = 1000  # increased for larger dataset

# on macbook also add
# device = 'cpu'  # run on cpu only
# compile = False # do not torch compile the model 