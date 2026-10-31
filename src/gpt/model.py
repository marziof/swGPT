import torch
from torch import nn
import torch.nn.functional as F
import math


class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x): 
        # x: (B, T, d)
        xf = x.float()
        rms = torch.sqrt(xf.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (xf / rms).to(x.dtype) * self.weight


class RoPE(nn.Module):
    def __init__(self, head_dim, max_len, theta=10000.0):
        super().__init__()
        assert head_dim % 2 == 0
        inv_freq = theta ** (-torch.arange(0, head_dim, 2).float() / head_dim)  # w_i standard formula [hd/2]
        pos = torch.arange(max_len).float() # [max_len] -> positions in batch sequences
        angles = torch.outer(pos, inv_freq) # (max_len, hd/2) -> p*w_i for each position/token emb dimension index - same across heads/sequences
        self.register_buffer("cos", angles.cos(), persistent=False)
        self.register_buffer("sin", angles.sin(), persistent=False)

    def forward(self, x, start_pos=0):            # x: [B, H, T, hd]
        T = x.shape[-2]
        assert start_pos + T <= self.cos.shape[0], "sequence longer than max_len"
        cos = self.cos[start_pos:start_pos + T]   # (T, hd/2), broadcasts over B, H 
        sin = self.sin[start_pos:start_pos + T]
        x1, x2 = x.chunk(2, dim=-1)               # each (B, H, T, hd/2)
        cos = cos[None, None, :, :] 
        sin = sin[None, None, :, :] 
        rotated_x1 = x1 * cos - x2 * sin
        rotated_x2 = x1 * sin + x2 * cos
        return torch.cat([rotated_x1, rotated_x2], dim=-1).to(x.dtype)
        

class Attention(nn.Module):
    def __init__(self, d_model, n_head, max_len):
        super().__init__()
        self.H = n_head
        assert d_model % n_head == 0, "d_model must be divisible by n_head"
        self.hd = d_model // n_head
        self.rope = RoPE(head_dim=self.hd, max_len = max_len)
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, start_pos=0, kv=None):
        # (B,T,d)
        B, T, d = x.shape
        q, k, v = self.qkv(x).split(d, dim=-1) # each (B,T,d)
        q = q.view(B, T, self.H, self.hd).transpose(1, 2)   # (B,H,T,hd)
        k = k.view(B, T, self.H, self.hd).transpose(1, 2)
        v = v.view(B, T, self.H, self.hd).transpose(1, 2)
        q = self.rope(q, start_pos)               # rotate q and k only
        k = self.rope(k, start_pos)   
        if kv is None:  # training / plain path, unchanged                                          
            assert start_pos == 0, "start_pos != 0 only makes sense with a cache"
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)  # (B,H,T,hd)
        else: 
            assert T == 1 or start_pos == 0, "multi-token input with a non-empty cache needs an explicit mask"
            k_buf, v_buf = kv                                   # each (B, H, max_len, hd)
            end = start_pos + T
            assert k_buf.shape[0] == B and end <= k_buf.shape[2], "cache batch size or length mismatch"
            k_buf[:, :, start_pos:end] = k.to(k_buf.dtype)      # store the ROTATED keys, plus the values
            v_buf[:, :, start_pos:end] = v.to(v_buf.dtype)
            k_all = k_buf[:, :, :end].to(q.dtype)               # every position seen so far, (B, H, end, hd)
            v_all = v_buf[:, :, :end].to(q.dtype)
            y = F.scaled_dot_product_attention(q, k_all, v_all, is_causal=(T > 1))
            
        y = y.transpose(1, 2).contiguous().view(B, T, d)    # (B,T,d)
        return self.W_o(y) # (B,T,d)


class MLP(nn.Module):
    def __init__(self, d_model, hidden_dim):
        super().__init__()
        self.up = nn.Linear(d_model, hidden_dim, bias=False)
        self.down = nn.Linear(hidden_dim, d_model, bias=False)

    def forward(self, x):
        return self.down(F.gelu(self.up(x))) # (B, T, d)
    

class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.mha = Attention(cfg.d_model, cfg.n_head, cfg.context_len)
        self.mlp = MLP(cfg.d_model, cfg.mlp_mult*cfg.d_model)
        self.norm1 = RMSNorm(cfg.d_model)
        self.norm2 = RMSNorm(cfg.d_model)

    def forward(self, x, start_pos=0, kv=None):
        h = x + self.mha(self.norm1(x), start_pos=start_pos, kv=kv)
        y = h + self.mlp(self.norm2(h))
        return y



class GPT(nn.Module):
    def __init__(self, cfg): #B, T, d, H, V
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model) # embed input form (B,T) -> (B, T, d)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)]) # transformer blocks to refine embeddings, keep (B, T, d)
        self.norm_f = RMSNorm(cfg.d_model)
        self.apply(self._init_weights)  # normal_(0, 0.02) for Linear and Embedding
        n_scaled = 0
        for name, p in self.named_parameters():
            if name.endswith("W_o.weight") or name.endswith("down.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))
                n_scaled += 1
        assert n_scaled == 2 * cfg.n_layer, f"expected {2 * cfg.n_layer} scaled weights, got {n_scaled}"

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def make_cache(self, B, dtype=None):
        ref = self.tok_emb.weight                      # defines the device and the default dtype
        dtype = dtype or ref.dtype
        cfg = self.cfg
        shape = (B, cfg.n_head, cfg.context_len, cfg.d_model // cfg.n_head)   # [B, H, max_len, hd]
        # cache is a list of (k_buf, v_buf) for each head, in each block layer
        return [(torch.zeros(shape, device=ref.device, dtype=dtype),
                torch.zeros(shape, device=ref.device, dtype=dtype))
                for _ in range(cfg.n_layer)]

    def forward(self, idx, targets=None, start_pos=0, cache=None):
        # idx: batch of token ids, (B, T)
        # embed X: (B, T, d_emb)
        x = self.tok_emb(idx)
        # pass through n transformer blocks -> 
        for i, block in enumerate(self.blocks):
            x = block(x, start_pos=start_pos, kv=None if cache is None else cache[i])
        x = self.norm_f(x) # still (B, T, d_emb)
        logits = x @ self.tok_emb.weight.T # project back to logits over tokens for each initial token (predictinos) (B, T, V)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1)) if targets is not None else None
        return logits, loss