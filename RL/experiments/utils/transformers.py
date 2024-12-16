"""
Full definition of a GPT Language Model, all of it in this single file.
References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import inspect
import math

import torch as th
import torch.nn as nn
from loguru import logger
from torch.nn import functional as F


class LayerNorm(nn.Module):
    """LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False"""

    def __init__(self, ndim: int, bias: bool) -> None:
        super().__init__()
        self.weight = nn.Parameter(th.ones(ndim))
        self.bias = nn.Parameter(th.zeros(ndim)) if bias else None

    def forward(self, input: th.Tensor):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class CausalSelfAttention(nn.Module):

    def __init__(
        self,
        n_embd: int = 128,
        n_head: int = 8,
        bias: bool = True,
        drop_p: float = 0.1,
    ) -> None:
        super().__init__()
        assert n_embd % n_head == 0

        self.adapter_param = None
    
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=bias)
        # output projection
        self.c_proj = nn.Linear(n_embd, n_embd, bias=bias)
        # regularization
        self.n_head = n_head
        self.n_embd = n_embd

        # dropout
        self.drop_p = drop_p
        self.attn_drop = nn.Dropout(drop_p)
        self.resid_drop = nn.Dropout(drop_p)

    def forward(self, x: th.Tensor, attn_mask: th.Tensor | None = None) -> th.Tensor:
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)

        tmp_x = self.c_attn(x)
        

        q, k, v = tmp_x.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

    
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=self.drop_p, is_causal=False)

        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side

        tmp_y = self.c_proj(y)
    

        y = self.resid_drop(tmp_y)
        return y


class MLP(nn.Module):

    def __init__(self, sizes: list, bias: bool) -> None:
        super().__init__()
        assert len(sizes) == 3
        self.c_fc = nn.Linear(sizes[0], sizes[1], bias=bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(sizes[1], sizes[2], bias=bias)

    def forward(self, x: th.Tensor, hidden_state=None):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class Block(nn.Module):

    def __init__(
        self,
        n_embd: int = 128,
        n_head: int = 8,
        bias: bool = True,
        drop_p: float = 0.1,
    ) -> None:
        super().__init__()
        self.ln_1 = LayerNorm(n_embd, bias=bias)
        self.attn = CausalSelfAttention(n_embd, n_head, bias, drop_p)
        self.ln_2 = LayerNorm(n_embd, bias=bias)
        self.mlp = MLP([n_embd, n_embd * 4, n_embd], bias)
        self.mlp_drop = nn.Dropout(drop_p)

    def forward(self, x: th.Tensor, attn_mask: th.Tensor | None = None) -> th.Tensor:
        # ln -> attn/mlp -> residual

        x = x + self.attn(self.ln_1(x), attn_mask)
        x = x + self.mlp(self.ln_2(x))

        return x


class Transformer(nn.Module):

    def __init__(
        self,
        n_embd: int = 128,
        n_head: int = 8,
        bias: bool = True,
        drop_p: float = 0.1,
        n_layer: int = 3
    ) -> None:
        super().__init__()

        self.device = th.device("cuda")
        self.tblocks = nn.ModuleList([Block(n_embd, n_head, bias, drop_p) for _ in range(n_layer)])
        self.lm_head = nn.Linear(n_embd, n_embd, bias=False)

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                th.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * n_layer))

        self.to(self.device)

    def forward(self, tokens: th.Tensor, mask: th.Tensor | None = None) -> th.Tensor:

        b, t, e = tokens.size()
        x = tokens
        for block in self.tblocks:
            x = block(x, mask)
        x = self.lm_head(x.view(b * t, e)).view(b, t, -1)

        return x

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            th.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                th.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            th.nn.init.normal_(module.weight, mean=0.0, std=0.02)