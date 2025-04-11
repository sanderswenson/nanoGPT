"""
Full definition of a GPT Language Model, all of it in this single file.
References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class AttentionPattern(nn.Module):
    """Base class for different attention patterns"""
    def __init__(self, config):
        super().__init__()
        self.block_size = config.block_size
        
    def get_mask(self, T):
        """Return the attention mask for this pattern"""
        raise NotImplementedError
        
    def is_causal(self):
        """Whether this attention pattern is causal"""
        return False
    


class CausalAttentionPattern(AttentionPattern):
    def __init__(self, config):
        super().__init__(config)
        self.register_buffer("mask", torch.tril(torch.ones(config.block_size, config.block_size))
                                   .view(1, 1, config.block_size, config.block_size))
    
    def get_mask(self, T):
        return self.mask[:,:,:T,:T]
    
    def is_causal(self):
        return True

class LocalAttentionPattern(AttentionPattern):
    def __init__(self, config):
        super().__init__(config)
        self.window_size = config.local_window_size
        # Start with all zeros (mask everything initially)
        local_mask = torch.zeros(config.block_size, config.block_size)
        # Set positions WITHIN the CAUSAL window to ONE (allow attention)
        for i in range(config.block_size):
            start = max(0, i - self.window_size)
            # End is i+1 to include the current token i
            end = i + 1
            # Ensure start and end are within bounds for assignment
            if start < end:
              local_mask[i, start:end] = 1
        # Register the mask (which has ONEs inside the causal window, ZEROs outside)
        self.register_buffer("mask", local_mask.view(1, 1, config.block_size, config.block_size))

    def get_mask(self, T):
        return self.mask[:,:,:T,:T]

    def is_causal(self):
        # This pattern is now causal
        return True

class AttentionGating(nn.Module):
    """Trainable gating mechanism for multiple attention patterns"""
    def __init__(self, config, num_patterns):
        super().__init__()
        self.n_embd = config.n_embd
        
        self.gate = nn.Sequential(
            nn.Linear(config.n_embd * num_patterns, config.n_embd),
            LayerNorm(config.n_embd, bias=config.bias),
            nn.ReLU(),
            nn.Linear(config.n_embd, num_patterns),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, attention_outputs):
        B = attention_outputs[0].size(0)
        features = torch.cat([out.mean(dim=1) for out in attention_outputs], dim=-1)
        return self.gate(features).unsqueeze(1)

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:
            # manual implementation of attention
            # (B, nh, T, hs) @ (B, nh, hs, T) -> (B, nh, T, T)
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            # mask future positions (B, nh, T, T) with -inf
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            # normalize attention weights (B, nh, T, T)
            att = F.softmax(att, dim=-1)
            # dropout on attention matrix
            att = self.attn_dropout(att)
            # (B, nh, T, T) @ (B, nh, T, hs) -> (B, nh, T, hs)
            y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

class MultiAttentionWithGating(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        
        # Store config
        self.config = config
        
        # QKV projections
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        
        # Parameters
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.block_length = config.block_length
        self.stride_length = config.stride_length
        
        # Initialize attention patterns
        self.attention_patterns = nn.ModuleList([
            # CausalAttentionPattern(config),
            LocalAttentionPattern(config)
        ])
        self.compression_mlp = CompressionMLP(config)
        # Initialize gating mechanism
        self.gating = AttentionGating(config, len(self.attention_patterns))
        
        # Dropouts
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        
        # Flash attention support
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
    
    def apply_attention(self, q, k, v, pattern, T, compressed_T=None, is_causal=False):
        """Apply attention with a specific pattern"""

        # DEBUG: Check if flash is active - REMOVED
        # print(f"Inside apply_attention: self.flash = {self.flash}")

        # Determine if we should force the non-flash path for this specific pattern - REMOVED
        # force_non_flash = isinstance(pattern, LocalAttentionPattern)

        # Original logic, relying on pattern.is_causal() and self.flash
        if self.flash:
            # print("Using Flash Attention path") # DEBUG REMOVED
            attn_mask = None # Default for causal or no mask
            pattern_mask = pattern.get_mask(T) if pattern is not None else None

            if is_causal:
                 # Let flash handle causality if is_causal=True
                 pass # attn_mask remains None
            elif pattern_mask is not None:
                 # If pattern exists and is_causal=False (e.g., a future non-causal pattern)
                 # Convert 0/1 mask to 0.0/-inf float mask.
                 attn_mask = pattern_mask.float().masked_fill(pattern_mask == 0, float('-inf'))
            # Else: No pattern and not is_causal, attn_mask remains None (full attention)

            return torch.nn.functional.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask, # Use the processed mask or None
                dropout_p=self.dropout if self.training else 0,
                is_causal=is_causal # Pass the flag indicating if the overall call should be causal
            )
        else:
            # Use non-flash path if flash not available
            # print("Using Non-Flash Attention path") # DEBUG REMOVED
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            if pattern is not None:
                mask = pattern.get_mask(T)
                # Non-flash path correctly handles 0/1 mask with masked_fill
                att = att.masked_fill(mask == 0, float('-inf'))
            elif is_causal:
                # Manual causal mask for non-flash (used if pattern is None but is_causal=True)
                actual_T = compressed_T or T
                causal_mask = torch.tril(torch.ones(T, actual_T, device=q.device)).view(1, 1, T, actual_T)
                att = att.masked_fill(causal_mask == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            return att @ v
    
    def forward(self, x):
        B, T, C = x.size()
        
        # Compute QKV
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # Compress k and v using sliding window approach
        k_compressed = self.compression_mlp(k)  # [B, nh, num_blocks, hs]
        v_compressed = self.compression_mlp(v)  # [B, nh, num_blocks, hs]
        
        # Calculate number of blocks after compression
        num_blocks = max(1, (T - self.block_length) // self.stride_length + 1)
        
        # Apply each attention pattern
        attention_outputs = []
        for pattern in self.attention_patterns:
            # DEBUG: Print mask for LocalAttentionPattern - REMOVED
            # if isinstance(pattern, LocalAttentionPattern):
            #     mask = pattern.get_mask(T)
            #     print(f"LocalAttentionPattern mask shape: {mask.shape}")
            #     print(f"LocalAttentionPattern mask sample (first 5x5):\n{mask[0,0,:5,:5]}")

            # Pass pattern.is_causal() to the apply_attention function
            out = self.apply_attention(q, k, v, pattern, T, is_causal=pattern.is_causal())
            out = out.transpose(1, 2).contiguous().view(B, T, C)
            attention_outputs.append(out)  # [B, T, C]

            # DEBUG: Print output of LocalAttentionPattern - REMOVED
            # if isinstance(pattern, LocalAttentionPattern):
            #     print(f"LocalAttentionPattern output shape: {out.shape}")
            #     print(f"LocalAttentionPattern output sample (first 5 tokens, first 5 features):\n{out[0,:5,:5]}")
        
        # Apply compressed attention
        # compressed_out = self.apply_attention(q, k_compressed, v_compressed, 
        #                                    None, 
        #                                    T, compressed_T=num_blocks, is_causal=True)
        # compressed_out = compressed_out.transpose(1, 2).contiguous().view(B, T, C)
        # attention_outputs.append(compressed_out)
        
        # DEBUG: Temporarily disable gating
        # gates = self.gating(attention_outputs)  # Now all outputs are [B, T, C]
        # combined = sum(g * out for g, out in zip(gates.chunk(len(attention_outputs), dim=-1), 
        #                                        attention_outputs))
        combined = attention_outputs[0] # Still using only the first pattern output (LocalAttentionPattern)
        
        # Final projection
        return self.resid_dropout(self.c_proj(combined))
    
class CompressionMLP(nn.Module):
    """MLP for compressing key and value sequences in the temporal dimension using a sliding window approach"""
    def __init__(self, config):
        super().__init__()
        self.block_length = config.block_length
        self.stride_length = config.stride_length
        self.n_head = config.n_head
        self.head_size = config.n_embd // config.n_head
        
        # Simple position embeddings for intra-block positions
        self.pos_emb = nn.Embedding(config.block_size, self.head_size)
        
        # Compression MLP to process each block
        self.c_fc = nn.Linear(self.head_size * self.block_length, 4 * self.head_size, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * self.head_size, self.head_size, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)
        self.ln = LayerNorm(self.head_size, bias=config.bias)

    def forward(self, x):
        # x shape: (B, nh, T, hs)
        B, nh, T, hs = x.size()
        
        # Determine the number of blocks after sliding window
        num_blocks = max(1, (T - self.block_length) // self.stride_length + 1)
        
        # Prepare output tensor for compressed representations
        compressed = []
        
        # Apply sliding window and compress each block
        for i in range(num_blocks):
            # Get the current block
            start_idx = i * self.stride_length
            end_idx = min(start_idx + self.block_length, T)
            
            # Handle the case where the final block is smaller than block_length
            actual_block_length = end_idx - start_idx
            
            if actual_block_length < self.block_length:
                # Pad the block to block_length
                padding_size = self.block_length - actual_block_length
                block = F.pad(x[:, :, start_idx:end_idx, :], (0, 0, 0, padding_size))
            else:
                block = x[:, :, start_idx:end_idx, :]
            
            # Add position embeddings to tokens within the block
            pos = torch.arange(0, self.block_length, device=x.device)
            pos_emb = self.pos_emb(pos).view(1, 1, self.block_length, hs)
            block = block + pos_emb  # Broadcasting handles dimensions
            
            # Reshape for MLP processing
            block = block.reshape(B, nh, self.block_length * hs)
            
            # Apply compression MLP
            block_compressed = self.c_fc(block)
            block_compressed = self.gelu(block_compressed)
            block_compressed = self.c_proj(block_compressed)
            block_compressed = self.dropout(block_compressed)
            block_compressed = self.ln(block_compressed)
            
            compressed.append(block_compressed)
        
        # Stack all compressed representations
        compressed = torch.stack(compressed, dim=2)  # [B, nh, num_blocks, hs]
        
        return compressed  # (B, nh, num_blocks, hs)

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = MultiAttentionWithGating(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    local_window_size: int = 128  # Size of the local attention window
    block_length: int = 64  # Length of each block for sliding window compression
    stride_length: int = 32  # Stride between consecutive blocks for sliding window compression

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
        self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)

        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss

    def crop_block_size(self, block_size):
        # model surgery to decrease the block size if necessary
        # e.g. we may load the GPT2 pretrained model checkpoint (block size 1024)
        # but want to use a smaller block size for some smaller, simpler model
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
        for block in self.transformer.h:
            if hasattr(block.attn, 'bias'):
                block.attn.bias = block.attn.bias[:,:,:block_size,:block_size]

    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        override_args = override_args or {} # default to empty dict
        # only dropout can be overridden see more notes below
        assert all(k == 'dropout' for k in override_args)
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        print("forcing vocab_size=50257, block_size=1024, bias=True")
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        config_args['bias'] = True # always True for GPT model checkpoints
        # we can override the dropout rate, if desired
        if 'dropout' in override_args:
            print(f"overriding dropout rate to {override_args['dropout']}")
            config_args['dropout'] = override_args['dropout']
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd//cfg.n_head, cfg.block_size
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0/dt) # per second
        flops_promised = 312e12 # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
