"""Elements of Mega, adapted from the original implementation by
Meta AI

Authors
* Artem Ploujnikov, 2024
"""

from torch import nn
from speechbrain.utils.data_utils import pad_right_to
import torch
import torch.nn.functional as F
import math


def laplace(x, mu=0.707107, sigma=0.282095):
    x = (x - mu).div(sigma * math.sqrt(2.0))
    return 0.5 * (1.0 + torch.erf(x))


class SimpleRelativePositionalBias(nn.Module):
    def __init__(self, max_positions):
        super().__init__()
        self.max_positions = max_positions
        self.rel_pos_bias = nn.Parameter(torch.Tensor(2 * max_positions - 1))
        self.reset_parameters()

    def reset_parameters(self):
        std = 0.02
        nn.init.normal_(self.rel_pos_bias, mean=0.0, std=std)

    def forward(self, seq_len):
        if seq_len > self.max_positions:
            raise ValueError('Sequence length {} going beyond max length {}'.format(seq_len, self.max_positions))

        # seq_len * 2 -1
        b = self.rel_pos_bias[(self.max_positions - seq_len):(self.max_positions + seq_len - 1)]
        # seq_len * 3 - 1
        t = F.pad(b, (0, seq_len))
        # (seq_len * 3 - 1) * seq_len
        t = torch.tile(t, (seq_len,))
        t = t[:-seq_len]
        # seq_len x (3 * seq_len - 2)
        t = t.view(seq_len, 3 * seq_len - 2)
        r = (2 * seq_len - 1) // 2
        start = r
        end = t.size(1) - r
        t = t[:, start:end]
        return t
    
class RotaryRelativePositionalBias(nn.Module):
    def __init__(self, embed_dim, max_positions):
        super().__init__()
        assert embed_dim % 2 == 0
        self.embed_dim = embed_dim
        self.max_positions = max_positions
        self.sine, self.cosine = RotaryRelativePositionalBias.get_sinusoid_embeddings(max_positions, embed_dim)
        self.alpha = nn.Parameter(torch.Tensor(1, embed_dim))
        self.beta = nn.Parameter(torch.Tensor(1, embed_dim))
        self.register_buffer("_float_tensor", torch.FloatTensor(1))
        self.reset_parameters()

    def reset_parameters(self):
        std = 0.02
        nn.init.normal_(self.alpha, mean=0.0, std=std)
        nn.init.normal_(self.beta, mean=0.0, std=std)

    @staticmethod
    def get_sinusoid_embeddings(max_positions: int, embedding_dim: int):
        half_dim = embedding_dim // 2
        emb = math.log(10000) / half_dim
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        emb = torch.arange(max_positions, dtype=torch.float).unsqueeze(1) * emb.unsqueeze(0)
        return torch.sin(emb), torch.cos(emb)

    def rotary(self, x):
        n, d = x.size()
        x1, x2 = torch.chunk(x, 2, dim=-1)
        if self.sine is None or n > self.sine.size(0):
            self.sine, self.cosine = RotaryRelativePositionalBias.get_sinusoid_embeddings(n, d)
            self.max_positions = n
        self.sine = self.sine.to(self._float_tensor)
        self.cosine = self.cosine.to(self._float_tensor)

        sin = self.sine[:n]
        cos = self.cosine[:n]
        return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=1)

    def forward(self, seq_len):
        a = self.rotary(self.alpha.expand(seq_len, self.embed_dim))
        b = self.rotary(self.beta.expand(seq_len, self.embed_dim))
        t = torch.einsum('mk,nk->mn', a, b)
        return t
    
class GatedCrossAttention(nn.Module):
    """Gated Structured State Attention.

    See "" for more details.
    """

    def __init__(
        self,
        embed_dim,
        zdim,
        ndim=2,
        dropout=0.0,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        activation=nn.SiLU,
        attention_activation='softmax',
        prenorm=True,
        norm_affine=True,
        rel_pos_bias='simple',
        max_positions=1024,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.zdim = zdim
        self.ndim = ndim
        self.activation = activation()
        self.attention_activation = attention_activation
        self.scaling = self.zdim ** -0.5 if attention_activation == 'softmax' else None

        self.dropout = nn.Dropout(dropout)
        self.hidden_dropout = nn.Dropout(hidden_dropout)
        # Attention dropout is standard dropout
        self.attention_dropout = nn.Dropout(attention_dropout)

        self.prenorm = prenorm
        self.norm = nn.LayerNorm(
            embed_dim,
            elementwise_affine=norm_affine
        )

        self.k_proj = nn.Linear(embed_dim, zdim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, 2 * embed_dim + zdim)
        self.h_proj = nn.Linear(embed_dim, embed_dim)

        self.max_positions = max_positions
        if rel_pos_bias == 'simple':
            self.rel_pos_bias = SimpleRelativePositionalBias(max_positions)
        elif rel_pos_bias == 'rotary':
            self.rel_pos_bias = RotaryRelativePositionalBias(zdim, max_positions)
        else:
            raise ValueError('unknown relative position bias: {}'.format(rel_pos_bias))

        self.reset_parameters()

        self.onnx_trace = False
        self.tpu = False


    def reset_parameters(self):
        std = 0.02
        nn.init.normal_(self.k_proj.weight, mean=0.0, std=std)
        nn.init.constant_(self.k_proj.bias, 0.0)

        nn.init.normal_(self.v_proj.weight, mean=0.0, std=std)
        nn.init.constant_(self.v_proj.bias, 0.0)

        nn.init.normal_(self.q_proj.weight, mean=0.0, std=std)
        nn.init.constant_(self.q_proj.bias, 0.0)

        nn.init.normal_(self.h_proj.weight, mean=0.0, std=std)
        nn.init.constant_(self.h_proj.bias, 0.0)

    def element_attention(self, q, k, key_padding_mask, pidx, before_attn_fn):
        bsz, clen, _ = k.size()
        slen = q.size(1) if pidx is None else pidx + 1
        if key_padding_mask is not None:
            # B x L1
            inverse_mask = 1.0 - key_padding_mask.type_as(q)
            # B x 1 x 1
            lengths = inverse_mask.sum(dim=-1).view(bsz, 1, 1)
        else:
            lengths = clen
            inverse_mask = None

        # L x L1
        bias = self.rel_pos_bias(max(slen, clen))[:, :clen]
        if pidx is not None:
            assert q.size(1) == 1
            # L1
            bias = bias[pidx]
        else:
            # L2 x L1
            bias = bias[:slen]

        # B x L2 x L1
        qk = torch.bmm(q, k.transpose(1, 2)) / lengths + bias

        if before_attn_fn:
            return qk

        if self.attention_activation == 'relu2':
            attn_weights = F.relu(qk).square().type_as(qk)
        elif self.attention_activation == 'laplace':
            attn_weights = laplace(qk).type_as(qk)
        else:
            raise ValueError('Unknown attention activation function: {}'.format(self.attention_activation))

        if inverse_mask is not None:
            attn_weights = attn_weights * inverse_mask.unsqueeze(1)

        return attn_weights

    def softmax_attention(self, q, k, key_padding_mask, pidx, before_attn_fn):
        bsz, clen, _ = k.size()
        slen = q.size(1) if pidx is None else pidx + 1

        # L x L1
        bias = self.rel_pos_bias(max(slen, clen))[:, :clen]
        if pidx is not None:
            assert q.size(1) == 1
            # L1
            bias = bias[pidx]
        else:
            # L2 x L1
            bias = bias[:slen]

        # scaled attention
        q = q * self.scaling
        # B x L2 x L1
        qk = torch.bmm(q, k.transpose(1, 2)) + bias

        if key_padding_mask is not None:
            qk = qk.masked_fill(key_padding_mask.unsqueeze(1).to(torch.bool), float('-inf'))

        if before_attn_fn:
            return qk

        attn_weights = F.softmax(qk, dim=-1).type_as(qk)
        return attn_weights

    def forward(
        self,
        query,
        key,
        value,
        key_padding_mask,
        before_attn_fn=False,
    ):
        """Input shape: Time x Batch x Channel

        Args:
            padding_mask (ByteTensor, optional): mask to exclude
                queries that are pads, of shape `(batch, tgt_len)`, where
                padding elements are indicated by 1s.
            key_padding_mask (ByteTensor, optional): mask to exclude
                keys that are pads, of shape `(batch, src_len)`, where
                padding elements are indicated by 1s.
            need_weights (bool, optional): return the attention weights,
                averaged over heads (default: False).
            before_attn_fn (bool, optional): return the raw attention
                weights and values before the attention softmax.
        """
        query = query.permute(1, 0, 2)
        key = key.permute(1, 0, 2)
        value = value.permute(1, 0, 2)
        seq_len, bsz, embed_dim = query.size()
        assert embed_dim == self.embed_dim

        pidx = None

        q = query
        if self.prenorm:
            q = self.norm(q)

        # L2 x B x (2*D+S)
        base = self.q_proj(q)
        u, r, q = torch.split(base, [self.embed_dim, self.embed_dim, self.zdim], dim=-1)

        # L2 x B x D
        u = torch.sigmoid(u)
        r = F.silu(r)

        if key is None:
            assert value is None
            k = v = None
        else:
            # L1 x B x S
            k = self.k_proj(key)
            v = self.activation(self.v_proj(key))

        # L2 x B x S -> B x L2 x S
        q = q.transpose(0, 1)
        if k is not None:
            k = k.transpose(0, 1)
        if v is not None:
            v = v.transpose(0, 1)

        ctx_len = k.size(1)
        # This is part of a workaround to get around fork/join parallelism
        # not supporting Optional types.
        if key_padding_mask is not None and key_padding_mask.dim() == 0:
            key_padding_mask = None

        if key_padding_mask is not None:
            assert key_padding_mask.size(0) == bsz
            assert key_padding_mask.size(1) == ctx_len

        if self.attention_activation == 'softmax':
            attn_weights = self.softmax_attention(q, k, key_padding_mask, pidx, before_attn_fn)
        else:
            attn_weights = self.element_attention(q, k, key_padding_mask, pidx, before_attn_fn)

        if before_attn_fn:
            return attn_weights, v

        v = self.hidden_dropout(v)
        kernel = self.attention_dropout(attn_weights)
        # B x L2 x D -> L2 x B x D
        h = torch.bmm(kernel, v).transpose(0, 1)
        # L2 x B x D
        h = self.activation(self.h_proj(h * r))
        h = self.dropout(h)
        out = torch.addcmul(query, u, h - query)

        if not self.prenorm:
            out = self.norm(out)
        out = out.permute(1, 0, 2)
        return out, attn_weights

class MultiHeadEMA(nn.Module):
    """Exponential Moving Average Layer.

    See "https://arxiv.org/abs/2209.10655" for more details.
    """

    def __init__(
        self,
        embed_dim,
        ndim=2,
        bidirectional=False,
        truncation=None,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.ndim = ndim
        self.bidirectional = bidirectional
        self.truncation = truncation
        self.scale = math.sqrt(1.0 / self.ndim)

        kernel_dim = 2 * embed_dim if self.bidirectional else embed_dim
        self.delta = nn.Parameter(torch.Tensor(kernel_dim, ndim, 1))
        self.alpha = nn.Parameter(torch.Tensor(kernel_dim, ndim, 1))
        self.beta = nn.Parameter(torch.Tensor(kernel_dim, ndim, 1))
        self.gamma = nn.Parameter(torch.Tensor(kernel_dim, ndim))
        self.omega = nn.Parameter(torch.Tensor(embed_dim))
        self._kernel = None
        self._coeffs = None

        self.reset_parameters()

        self.onnx_trace = False
        self.tpu = False

    def reset_parameters(self):
        with torch.no_grad():
            # delta & alpha
            nn.init.normal_(self.delta, mean=0.0, std=0.2)
            nn.init.normal_(self.alpha, mean=0.0, std=0.2)
            # beta [1, -1, 1, -1, ...] seems more stable.
            val = torch.ones(self.ndim, 1)
            if self.ndim > 1:
                idx = torch.tensor(list(range(1, self.ndim, 2)))
                val.index_fill_(0, idx, -1.0)
            self.beta.normal_(mean=0.0, std=0.02).add_(val)
            # gamma & omega
            nn.init.normal_(self.gamma, mean=0.0, std=1.0)
            nn.init.normal_(self.omega, mean=0.0, std=1.0)

    def _calc_coeffs(self):
        self._coeffs = None
        # D x N x 1
        p = torch.sigmoid(self.delta)
        alpha = torch.sigmoid(self.alpha)
        q = 1.0 - p * alpha
        return p, q

    def _compute_kernel(self, length: int):
        self._kernel = None
        # D x N x 1
        p, q = self._calc_coeffs()
        # D x N x L
        vander = torch.arange(length).to(p).view(1, 1, length) * torch.log(q)
        kernel = (p * self.beta) * torch.exp(vander)
        # D x L
        return torch.einsum('dnl,dn->dl', kernel, self.gamma * self.scale)

    def coeffs(self):
        if self.training:
            return self._calc_coeffs()
        else:
            if self._coeffs is None:
                self._coeffs = self._calc_coeffs()
            return self._coeffs

    def kernel(self, length: int):
        kernel_size = length if self.truncation is None else min(self.truncation, length)
        if self.training:
            return self._compute_kernel(kernel_size)
        else:
            if self._kernel is None or self._kernel.size(-1) < kernel_size:
                self._kernel = self._compute_kernel(kernel_size)
            return self._kernel[..., :kernel_size]

    def forward(
        self,
        x,
        padding_mask=None
    ):
        """Input shape: Time x Batch x Channel
        Args:
            padding_mask (ByteTensor, optional): mask to exclude
                keys that are pads, of shape `(batch, src_len)`, where
                padding elements are indicated by 1s.
        """

        seq_len, bsz, embed_dim = x.size()
        assert embed_dim == self.embed_dim

        # L x B x D
        residual = x * self.omega

        # L x B x D -> B x D x L
        x = x.permute(1, 2, 0)
        if padding_mask is not None:
            x = x * (1.0 - padding_mask.unsqueeze(1).type_as(x))

        # Incremental state support was removed
        # D x L
        k = self.kernel(seq_len)
        fft_len = seq_len
        s = 0
        kernel_size = k.size(1)
        if self.bidirectional:
            k1, k2 = torch.split(k, [self.embed_dim, self.embed_dim], dim=0)
            # D x 2*L-1
            k = F.pad(k1, (kernel_size - 1, 0)) + F.pad(k2.flip(-1), (0, kernel_size - 1))
            x = F.pad(x, (kernel_size - 1, 0))
            fft_len = fft_len + kernel_size - 1
            s = 2 * kernel_size - 2

        k_f = torch.fft.rfft(k.float(), n=2 * fft_len)
        x_f = torch.fft.rfft(x.float(), n=2 * fft_len)
        # B x D x L
        out = torch.fft.irfft(x_f * k_f, n=2 * fft_len)[..., s:s + seq_len]
        out = out.type_as(x)
        # B x D x L -> L x B x D
        out = F.silu(out.permute(2, 0, 1) + residual)

        return out


class MovingAverageGatedAttention(nn.Module):
    """Exponential Moving Average Gated Attention.

    See "" for more details.
    """

    def __init__(
        self,
        embed_dim,
        zdim,
        hdim,
        ndim,
        dropout=0.0,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        activation=nn.SiLU,
        attention_activation='softmax',
        bidirectional=False,
        chunk_size=-1,
        truncation=None,
        prenorm=True,
        norm_affine=True,
        rel_pos_bias='simple',
        max_positions=1024,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.hdim = hdim
        self.zdim = zdim
        self.ndim = ndim
        self.activation = activation()
        self.attention_activation = attention_activation
        self.scaling = self.zdim ** -0.5 if attention_activation == 'softmax' else None
        
        self.dropout = nn.Dropout(dropout)
        self.hidden_dropout = nn.Dropout(hidden_dropout)
        # Attention dropout is standard dropout
        self.attention_dropout = nn.Dropout(attention_dropout)

        self.chunk_size = chunk_size
        self.prenorm = prenorm
        self.norm = nn.LayerNorm(embed_dim, elementwise_affine=norm_affine)

        self.move = MultiHeadEMA(embed_dim, ndim=ndim, bidirectional=bidirectional, truncation=truncation)

        self.v_proj = nn.Linear(embed_dim, hdim)
        self.mx_proj = nn.Linear(embed_dim, zdim + hdim + 2 * embed_dim)
        self.h_proj = nn.Linear(hdim, embed_dim)

        self.gamma = nn.Parameter(torch.Tensor(2, zdim))
        self.beta = nn.Parameter(torch.Tensor(2, zdim))

        self.max_positions = max_positions
        max_positions = max_positions if chunk_size < 0 else chunk_size
        if rel_pos_bias == 'simple':
            self.rel_pos_bias = SimpleRelativePositionalBias(max_positions)
        elif rel_pos_bias == 'rotary':
            self.rel_pos_bias = RotaryRelativePositionalBias(zdim, max_positions)
        else:
            raise ValueError('unknown relative position bias: {}'.format(rel_pos_bias))

        self.reset_parameters() 

    def reset_parameters(self):
        std = 0.02
        nn.init.normal_(self.v_proj.weight, mean=0.0, std=std)
        nn.init.constant_(self.v_proj.bias, 0.0)

        nn.init.normal_(self.mx_proj.weight, mean=0.0, std=std)
        nn.init.constant_(self.mx_proj.bias, 0.0)

        nn.init.normal_(self.h_proj.weight, mean=0.0, std=std)
        nn.init.constant_(self.h_proj.bias, 0.0)

        nn.init.normal_(self.gamma, mean=0.0, std=std)
        nn.init.constant_(self.beta, 0.0)

    def element_attention(self, q, k, padding_mask, attn_mask, before_attn_fn):
        slen = k.size(2)
        if padding_mask is not None:
            # B x K x C
            inverse_mask = 1.0 - padding_mask.type_as(q)
            # B x K x 1
            lengths = inverse_mask.sum(dim=-1, keepdim=True)
            # B x K x 1 x 1
            lengths = lengths.clamp(min=1.0).unsqueeze(-1)
        else:
            lengths = slen
            inverse_mask = None

        if attn_mask is not None:
            # C x 1
            lengths = attn_mask.sum(dim=-1, keepdim=True)

        # C x C
        bias = self.rel_pos_bias(slen)
        if slen != q.size(2):
            assert q.size(2) == 1
            # 1 x C
            bias = bias[-1:]

        # B x K x C x C
        qk = torch.matmul(q, k.transpose(2, 3)) / lengths + bias

        if before_attn_fn:
            return qk

        if self.attention_activation == 'relu2':
            attn_weights = F.relu(qk).square().type_as(qk)
        elif self.attention_activation == 'laplace':
            attn_weights = laplace.type_as(qk)
        else:
            raise ValueError('Unknown attention activation function: {}'.format(self.attention_activation))

        if inverse_mask is not None:
            attn_weights = attn_weights * inverse_mask.unsqueeze(2)

        if attn_mask is not None:
            attn_weights = attn_weights * attn_mask

        return attn_weights

    def softmax_attention(self, q, k, padding_mask, attn_mask, before_attn_fn):
        slen = k.size(2)
        # C x C
        bias = self.rel_pos_bias(slen)
        if slen != q.size(2):
            assert q.size(2) == 1
            # 1 x C
            bias = bias[-1:]

        # scaled attention
        q = q * self.scaling
        # B x K x C x C
        qk = torch.matmul(q, k.transpose(2, 3)) + bias

        if attn_mask is not None:
            qk = qk + attn_mask

        if padding_mask is not None:
            padding_mask_all = padding_mask.all(dim=-1, keepdim=True)
            padding_mask = torch.logical_and(padding_mask, ~padding_mask_all)
            qk = qk.masked_fill(padding_mask.unsqueeze(2).to(torch.bool), float('-inf'))

        if before_attn_fn:
            return qk

        attn_weights = F.softmax(qk, dim=-1).type_as(qk)
        return attn_weights

    def forward(
        self,
        x,
        padding_mask=None,
        incremental_state=None,
        attn_mask=None,
        before_attn_fn=False,
    ):
        """Input shape: Time x Batch x Channel

        Args:
            padding_mask (ByteTensor, optional): mask to exclude
                keys that are pads, of shape `(batch, src_len)`, where
                padding elements are indicated by 1s.
            need_weights (bool, optional): return the attention weights,
                averaged over heads (default: False).
            attn_mask (ByteTensor, optional): typically used to
                implement causal attention, where the mask prevents the
                attention from looking forward in time (default: None).
            before_attn_fn (bool, optional): return the raw attention
                weights and values before the attention softmax.
        """
        x = x.permute(1, 0, 2)
        seq_len, bsz, embed_dim = x.size()
        assert embed_dim == self.embed_dim

        residual = x
        if self.prenorm:
            x = self.norm(x)

        # L x B x E
        v = self.activation(self.v_proj(x))

        # L x B x D
        mx = self.move(x, padding_mask)
        mx = self.dropout(mx)

        # L x B x D -> L x B x (2*D+S+E)
        base = self.mx_proj(mx)
        u, zr, hx = torch.split(base, [self.embed_dim, self.zdim + self.hdim, self.embed_dim], dim=-1)
        # L x B x D
        u = torch.sigmoid(u)
        # L x B x (E+S)
        z, r = torch.split(F.silu(zr), [self.zdim, self.hdim], dim=-1)
        # L x B x S -> L x B x 1 x S -> L x B x 2 x S
        z = z.unsqueeze(2) * self.gamma + self.beta
        # L x B x 2 x S -> L x B x S
        q, k = torch.unbind(z, dim=2)

        # L x B x D -> B x L x D
        q = q.transpose(0, 1)
        k = k.transpose(0, 1)
        v = v.transpose(0, 1)
        ctx_len = k.size(1)
        
        if self.chunk_size < 0:
            # B x L x S -> B x 1 x L x S
            q = q.unsqueeze(1)
            k = k.unsqueeze(1)
            v = v.unsqueeze(1)
            if padding_mask is not None:
                # B x L -> B x 1 x L
                padding_mask = padding_mask.unsqueeze(1)
        else:
            if seq_len < self.chunk_size:
                q = q.unsqueeze(1)
            else:
                # B x L x S -> B x K x C x S
                nc = seq_len // self.chunk_size
                q = q.reshape(bsz, nc, self.chunk_size, self.zdim)

            if ctx_len < self.chunk_size:
                k = k.unsqueeze(1)
                v = v.unsqueeze(1)
                if padding_mask is not None:
                    padding_mask = padding_mask.unsqueeze(1)
            else:
                # B x L x S -> B x K x C x S
                nc = ctx_len // self.chunk_size
                k = k.reshape(bsz, nc, self.chunk_size, self.zdim)
                v = v.reshape(bsz, nc, self.chunk_size, self.hdim)
                if padding_mask is not None:
                    # B x L -> B x K x C
                    padding_mask = padding_mask.view(bsz, nc, self.chunk_size)

        
        # This is part of a workaround to get around fork/join parallelism
        # not supporting Optional types.
        if padding_mask is not None and padding_mask.dim() == 0:
            padding_mask = None

        if self.attention_activation == 'softmax':
            attn_weights = self.softmax_attention(q, k, padding_mask, attn_mask, before_attn_fn)
        else:
            attn_weights = self.element_attention(q, k, padding_mask, attn_mask, before_attn_fn)

        if before_attn_fn:
            return attn_weights, v

        v = self.hidden_dropout(v)
        kernel = self.attention_dropout(attn_weights)
        # B x K x C x E -> B x L x E -> L x B x E
        h = torch.matmul(kernel, v).view(bsz, seq_len, self.hdim).transpose(0, 1)
        # L x B x E -> L x B x D
        h = self.activation(hx + self.h_proj(h * r))
        h = self.dropout(h)
        # L x B x D
        out = torch.addcmul(residual, u, h - residual)

        if not self.prenorm:
            out = self.norm(out)

        out = out.permute(1, 0, 2)

        return out, attn_weights


class NormalizedFeedForwardNetwork(nn.Module):
    def __init__(
        self,
        embed_dim,
        ffn_hidden_dim,
        dropout=0.0,
        hidden_dropout=0.0,
        activation='silu',
        prenorm=True,
        norm_affine=True,
    ):
        super().__init__()

        self.embedding_dim = embed_dim
        self.hidden_dim = ffn_hidden_dim
        self.act_fn = activation
        self.activation = nn.SiLU(activation)

        self.dropout = nn.Dropout(dropout)
        self.hidden_dropout = nn.Dropout(hidden_dropout)

        self.prenorm = prenorm
        self.norm = nn.LayerNorm(embed_dim, elementwise_affine=norm_affine)

        self.fc1 = nn.Linear(embed_dim, ffn_hidden_dim)
        self.fc2 = nn.Linear(ffn_hidden_dim, embed_dim)

        self.reset_parameters()

    def reset_parameters(self):
        std = 0.02
        nn.init.normal_(self.fc1.weight, mean=0.0, std=std)
        nn.init.constant_(self.fc1.bias, 0.0)

        nn.init.normal_(self.fc2.weight, mean=0.0, std=std)
        nn.init.constant_(self.fc2.bias, 0.0)

    def forward(self, x):
        residual = x

        if self.prenorm:
            x = self.norm(x)

        x = self.activation(self.fc1(x))
        x = self.hidden_dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x = x + residual

        if not self.prenorm:
            x = self.norm(x)

        return x


class MegaDecoderLayer(nn.Module):
    def __init__(
        self,
        embed_dim=512,
        decoder_z_dim=128,
        decoder_hidden_dim=128,
        decoder_n_dim=16,
        decoder_ffn_embed_dim=2048,
        dropout=0.2,
        attention_dropout=0.2,
        hidden_dropout=0.2,
        activation_dropout=0.2,
        decoder_chunk_size=-1,
        truncation_length=None,
        rel_pos_bias="simple",
        activation=nn.SiLU,
        attention_activation="softmax",
        bidirectional=False,
        normalize_before=True,
        max_source_positions=1000,
        max_target_positions=1000,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.decoder_chunk_size = decoder_chunk_size
        self.mega_layer = MovingAverageGatedAttention(
            embed_dim=embed_dim,
            zdim=decoder_z_dim,
            hdim=decoder_hidden_dim,
            ndim=decoder_n_dim,
            dropout=dropout,
            attention_dropout=attention_dropout,
            hidden_dropout=hidden_dropout,
            chunk_size=decoder_chunk_size,
            truncation=truncation_length,
            rel_pos_bias=rel_pos_bias,
            max_positions=max_target_positions,
            activation=activation,
            attention_activation=attention_activation,
            bidirectional=bidirectional,
            prenorm=normalize_before,
        )

        self.cross_attn = GatedCrossAttention(
            embed_dim=embed_dim,
            zdim=decoder_z_dim,
            ndim=decoder_n_dim,
            dropout=dropout,
            attention_dropout=attention_dropout,
            hidden_dropout=hidden_dropout,
            activation=activation,
            attention_activation=attention_activation,
            prenorm=normalize_before,
            rel_pos_bias=rel_pos_bias,
            max_positions=max(max_target_positions, max_source_positions),
        )
        self.nffn = NormalizedFeedForwardNetwork(
            embed_dim=embed_dim,
            ffn_hidden_dim=decoder_ffn_embed_dim,
            dropout=dropout,
            hidden_dropout=activation_dropout,
            activation=activation,
            prenorm=normalize_before,
        )

    def forward(
        self,
        tgt,
        memory=None,
        memory_key_padding_mask=None,
        tgt_mask=None,
        tgt_key_padding_mask=None,
    ):
        batch_size, max_len, feature_dim = tgt.shape
        if self.decoder_chunk_size > 0:
            pad_len = math.ceil(max_len / self.decoder_chunk_size) * self.decoder_chunk_size
            x, _ = pad_right_to(
                tgt,
                (batch_size, pad_len, feature_dim)
            )
            len_diff = pad_len - max_len
            if tgt_key_padding_mask is not None:
                tgt_key_padding_mask = torch.cat([
                    tgt_key_padding_mask,
                    (
                        torch.tensor(True, device=tgt_key_padding_mask.device)[None, :]
                        .expand_as(batch_size, len_diff)
                    )
                ])
            if tgt_mask is not None:
                tgt_mask = tgt_mask[:self.decoder_chunk_size, :self.decoder_chunk_size]
                if tgt_mask.shape[0] < self.decoder_chunk_size:
                    old_tgt_mask = tgt_mask
                    tgt_mask = torch.ones(
                        (self.decoder_chunk_size, self.decoder_chunk_size),
                        device=tgt_mask.device
                    ) * -torch.inf
                    tgt_mask[:old_tgt_mask.size(0), :old_tgt_mask.size(1)] = old_tgt_mask
        else:
            x = tgt

        x, mega_attn = self.mega_layer(
            x=x,
            padding_mask=tgt_key_padding_mask,
            attn_mask=tgt_mask
        )

        x, cross_attn = self.cross_attn(
            query=x,
            key=memory,
            value=memory,
            key_padding_mask=memory_key_padding_mask
        )

        x = self.nffn(x)
        if self.decoder_chunk_size > 0:
            x = x[:, :max_len, :]
            mega_attn = [
                attn[:, :max_len, :]
                for attn in mega_attn
            ]
            cross_attn = cross_attn[:, :max_len, :]

        return x, mega_attn, cross_attn


class MegaEncoderLayer(nn.Module):
    """
        Implements a Flash-Quad encoder layer.
    """

    def __init__(
        self,
        embedding_dim=512,
        hidden_dim=1024,
        ffn_hidden_dim=1024,
        z_dim=128,
        n_dim=16,
        dropout=0.0,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        chunk_size=-1,
        truncation=None,
        rel_pos_bias='simple',
        max_positions=1024,
        activation=nn.SiLU,
        attention_activation='softmax',
        prenorm=True,
    ):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.chunk_size = chunk_size
        self.mega_layer = MovingAverageGatedAttention(
            embed_dim=embedding_dim,
            zdim=z_dim,
            hdim=hidden_dim,
            ndim=n_dim,
            dropout=dropout,
            attention_dropout=attention_dropout,
            hidden_dropout=hidden_dropout,
            chunk_size=chunk_size,
            truncation=truncation,
            rel_pos_bias=rel_pos_bias,
            max_positions=max_positions,
            activation=activation,
            attention_activation=attention_activation,
            bidirectional=True,
            prenorm=prenorm,
        )

        self.nffn = NormalizedFeedForwardNetwork(
            embed_dim=embedding_dim,
            ffn_hidden_dim=ffn_hidden_dim,
            dropout=dropout,
            hidden_dropout=hidden_dropout,
            activation=activation,
            prenorm=prenorm,
        )

    def forward(
        self,
        src,
        src_mask=None,
        src_key_padding_mask=None,
    ):

        x = src
        seq_len = x.size(0)
        if self.chunk_size > 0:
            assert seq_len % self.chunk_size == 0, 'the input sequence length {} cannot be divided by chunk size {}'.format(seq_len, self.chunk_size)
        x, attn = self.mega_layer(x, attn_mask=src_mask, padding_mask=src_key_padding_mask)
        x = self.nffn(x)

        return x, attn


class MegaEncoder(nn.Module):
    """
    Mega encoder consisting of *encoder_layers* layers. Each layer is a :class:`MegaEncoderLayer`.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): encoding dictionary
        embed_tokens (torch.nn.Embedding): input embedding
    """

    def __init__(
        self,
        num_layers=6,
        d_model=256,
        hidden_dim=1024,
        ffn_hidden_dim=1024,
        z_dim=128,
        n_dim=16,
        dropout=0.0,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        chunk_size=-1,
        truncation=None,
        rel_pos_bias='simple',
        max_positions=1024,
        activation=nn.SiLU,
        encoder_chunk_size=-1,
        no_scale_embedding=False,
        max_source_positions=1000,
        normalize_before=True,
        attention_activation='softmax',
    ):
        super().__init__()
        self.register_buffer("version", torch.Tensor([3]))

        self.embedding_dropout = nn.Dropout(dropout)

        self.max_source_positions = max_source_positions
        self.chunk_size = encoder_chunk_size
        self.embed_scale = 1.0 if no_scale_embedding else math.sqrt(d_model)
        self.embed_norm = nn.LayerNorm(d_model)

        self.layers = nn.ModuleList([
            MegaEncoderLayer(
                embedding_dim=d_model,
                hidden_dim=hidden_dim,
                ffn_hidden_dim=ffn_hidden_dim,
                z_dim=z_dim,
                n_dim=n_dim,
                dropout=dropout,
                attention_dropout=attention_dropout,
                hidden_dropout=hidden_dropout,
                chunk_size=chunk_size,
                truncation=truncation,
                rel_pos_bias=rel_pos_bias,
                max_positions=max_positions,
                activation=activation,
                attention_activation=attention_activation,
                prenorm=normalize_before
            )
            for i in range(num_layers)
        ])
        self.num_layers = len(self.layers)

        self.final_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        src,
        src_mask=None,
        src_key_padding_mask=None,
        pos_embs=None,
    ):
        x = src
        if self.embed_norm is not None:
            x = self.embed_norm(x)

        x = self.embedding_dropout(x)
        attn = []
        # encoder layers
        for layer in self.layers:
            x, layer_attn = layer(
                x,
                src_mask=src_mask,
                src_key_padding_mask=src_key_padding_mask
            )
            attn.append(layer_attn)

        if self.final_norm is not None:
            x = self.final_norm(x)

        return x, attn