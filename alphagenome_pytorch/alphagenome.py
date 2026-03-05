from __future__ import annotations

import inspect
import os
from functools import partial
from collections import namedtuple, defaultdict

import torch
import torch.distributed as dist
from torch import tensor, is_tensor, arange, nn, cat, stack, Tensor
import torch.nn.functional as F
from torch.nn import Conv1d, Linear, Sequential, Module, ModuleList, LayerNorm, ModuleDict

from torch.amp import autocast

from einx import add, multiply, greater
from einops.layers.torch import Rearrange, Reduce
from einops import rearrange, repeat, reduce, einsum

from torch_einops_utils import pack_with_inverse
from PoPE_pytorch import PoPE, compute_attn_similarity

# Enable TF32 by default for faster matmuls; set ALPHAGENOME_ALLOW_TF32=0 to disable.
_ALLOW_TF32 = os.environ.get("ALPHAGENOME_ALLOW_TF32", "1") == "1"
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = _ALLOW_TF32
    torch.backends.cudnn.allow_tf32 = _ALLOW_TF32
# Allow native bf16 compute paths in attention (enabled by default for better
# performance on modern GPUs). Set ALPHAGENOME_TORCH_BF16=0 to disable.
_ALLOW_NATIVE_BF16 = os.environ.get("ALPHAGENOME_TORCH_BF16", "1") == "1"

try:
    import matplotlib.pyplot as plt
    _HAS_MATPLOTLIB = True
except ImportError:
    _HAS_MATPLOTLIB = False

try:
    import logomaker
    _HAS_LOGOMAKER = True
except ImportError:
    _HAS_LOGOMAKER = False

# ein notation

# b - batch
# h - heads
# n - sequence
# p - relative positions
# d - feature dimension
# t - tracks

# constants

LinearNoBias = partial(Linear, bias = False)

TransformerUnetOutput = namedtuple('TransformerUnetOutput', [
    'unet_output',
    'single_repr',
    'pairwise_repr'
])

Embeds = namedtuple('Embeds', [
    'embeds_1bp',
    'embeds_128bp',
    'embeds_pair',
])

publication_heads_config = {
    'human': {
        'organism' : 'human',
        'num_tracks_1bp' : 3165, # 1174 total RNA-seq + 961 polyA plus RNA-seq + 516 hCAGE + 30 LQhCAGE + 167 ATAC-seq + 305 DNase-seq + 12 PRO-cap
        'num_tracks_128bp' : 2733, # 1617 TF ChIP-seq + 1116 Histone ChIP-seq
        'num_tracks_contacts' : 28, # 24 in situ Hi-C + 3 Micro-C + 1 Dilution Hi-C
        'num_splicing_contexts' : 282, # from Sup. Tab. 2
        'hidden_dim_splice_juncs' : 512 # max number of splice junctions to consider
    },
    'mouse': {
        'organism' : 'mouse',
        'num_tracks_1bp' : 730, # 168 total RNA-seq + 365 polyA plus RNA-seq + 172 hCAGE + 16 LQhCAGE + 18 ATAC-seq + 67 DNase-seq
        'num_tracks_128bp' : 310, # 127 TF ChIP-seq + 183 Histone ChIP-seq
        'num_tracks_contacts' : 8, # 8 in situ Hi-C
        'num_splicing_contexts' : 75, # from Sup. Tab. 2
        'hidden_dim_splice_juncs' : 512 # max number of splice junctions to consider
    }
}

# Reference (JAX) head specs from the official checkpoint
jax_genome_track_heads = {
    'rna_seq': dict(num_tracks = 768, resolutions = (1, 128)),
    'cage': dict(num_tracks = 640, resolutions = (1, 128)),
    'dnase': dict(num_tracks = 384, resolutions = (1, 128)),
    'procap': dict(num_tracks = 128, resolutions = (1, 128)),
    'atac': dict(num_tracks = 256, resolutions = (1, 128)),
    'chip_tf': dict(num_tracks = 1664, resolutions = (128,)),
    'chip_histone': dict(num_tracks = 1152, resolutions = (128,)),
}

jax_contact_maps_tracks = 28
jax_splice_usage_tracks = 734
jax_splice_junction_hidden_dim = 768
jax_splice_junction_max_tissues = 367

# functions

def exists(v):
    return v is not None

def divisible_by(num, den):
    return (num % den) == 0

def last(arr):
    return arr[-1]

def is_odd(num):
    return not divisible_by(num, 2)

def is_even(num):
    return divisible_by(num, 2)

def l2norm(t, dim = -1):
    return F.normalize(t, dim = dim, p = 2)

def default(*args):
    for arg in args:
        if exists(arg):
            return arg
    return None

def softclamp(t, value = 5.):
    return (t / value).tanh() * value

def symmetrize(t):
    return add('b i j ..., b j i ... -> b i j ...', t, t) * 0.5

def append_dims(t, ndims):
    return t.reshape(*t.shape, *((1,) * ndims))

def log(t, eps = 1e-7):
    return t.clamp(min = eps).log()

def gelu_1702(x):
    """Official AlphaGenome GELU variant: sigmoid(1.702 * x) * x"""
    return torch.sigmoid(1.702 * x) * x

class GELU1702(Module):
    """Module wrapper for gelu_1702 activation."""
    def forward(self, x):
        return gelu_1702(x)

def geomspace(start, stop, num, device = None):
    """Return numbers spaced evenly on a log scale (geometric progression).

    Matches numpy.geomspace / jax.numpy.geomspace behavior.
    """
    if num == 1:
        return torch.tensor([start], device=device)
    # Use log-linear interpolation: start * (stop/start)^(i/(num-1))
    t = torch.linspace(0, 1, num, device=device)
    return start * (stop / start) ** t

def safe_div(num, den, eps = 1e-7):
    return num / den.clamp(min = eps)

# losses

class MultinomialLoss(Module):
    def __init__(
        self,
        multinomial_resolution,
        positional_loss_weight = 5.,
        eps = 1e-7
    ):
        super().__init__()
        self.split_res = Rearrange('... (n resolution) t -> ... n resolution t', resolution = multinomial_resolution)

        self.eps = eps
        self.log = partial(log, eps = eps)
        self.resolution = multinomial_resolution
        self.positional_loss_weight = positional_loss_weight

    def forward(self, pred, target):
        pred = self.split_res(pred)
        target = self.split_res(target)

        pred_sum = reduce(pred, '... n resolution t -> ... n 1 t', 'sum')
        target_sum = reduce(target, '... n resolution t -> ... n 1 t', 'sum')

        poisson_loss = (pred_sum - target_sum * self.log(pred_sum)).sum()
        multinomial_prob = pred / pred_sum.clamp(min = self.eps)

        positional_loss = (-target * self.log(multinomial_prob)).sum()

        return poisson_loss / self.resolution + positional_loss * self.positional_loss_weight

class SoftClip(Module):
    def __init__(
        self,
        scale = 2.,
        gamma = 10.,
        threshold = 10.
    ):
        super().__init__()
        self.scale = scale
        self.gamma = gamma
        self.threshold = threshold

    def inverse(self, clipped):
        threshold, scale, gamma = self.threshold, self.scale, self.gamma
        return torch.where(clipped > threshold, ((clipped + threshold) ** 2) / (gamma * scale ** 2), clipped)

    def forward(self, x):
        threshold, scale, gamma = self.threshold, self.scale, self.gamma
        return torch.where(x > threshold, scale * torch.sqrt(x * gamma) - threshold, x)

class MultinomialCrossEntropy(Module):
    def __init__(
        self,
        dim = -1,
        eps = 1e-7
    ):
        super().__init__()
        self.dim = dim
        self.eps = eps

    def forward(self, pred, target):
        pred_ratios = safe_div(pred, pred.sum(dim = self.dim, keepdim = True), eps = self.eps)
        target_ratios = safe_div(target, target.sum(dim = self.dim, keepdim = True), eps = self.eps)
        return -(target_ratios * log(pred_ratios, eps = self.eps)).sum()

class PoissonLoss(Module):
    def __init__(
        self,
        dim = -1,
        eps = 1e-7,
        softclip_scale = 2.,
        softclip_gamma = 10.,
        softclip_threshold = 10.
    ):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.softclip = SoftClip(softclip_scale, softclip_gamma, softclip_threshold)

    def forward(self, pred, target):
        sum_pred = pred.sum(dim = self.dim)
        sum_target = self.softclip(target.sum(dim = self.dim))
        return (sum_pred - sum_target * log(sum_pred, eps = self.eps)).sum()

class JunctionsLoss(Module):
    def __init__(
        self, 
        softclip_scale = 2.,
        softclip_gamma = 10.,
        softclip_threshold = 10.
    ):
        super().__init__()
        self.multinomial_cross_entropy_row = MultinomialCrossEntropy(dim = 0)
        self.multinomial_cross_entropy_col = MultinomialCrossEntropy(dim = 1)

        self.poisson_loss_row = PoissonLoss(dim = 0, softclip_scale = softclip_scale, softclip_gamma = softclip_gamma, softclip_threshold = softclip_threshold)
        self.poisson_loss_col = PoissonLoss(dim = 1, softclip_scale = softclip_scale, softclip_gamma = softclip_gamma, softclip_threshold = softclip_threshold)

    def forward(
        self, 
        pred: Tensor, 
        target: Tensor
    ):
        loss = 0.0

        for pred_c, target_c in zip(pred.unbind(dim = -1), target.unbind(dim = -1)):

            ratios = self.multinomial_cross_entropy_row(pred_c, target_c) + \
                     self.multinomial_cross_entropy_col(pred_c, target_c)

            counts = self.poisson_loss_row(pred_c, target_c) + \
                     self.poisson_loss_col(pred_c, target_c)

            loss = loss + ( 0.2 * ratios + 0.04 * counts )

        return loss

# batch rmsnorm

def is_distributed():
    return dist.is_initialized() and dist.get_world_size() > 1

def get_maybe_dist_var(
    t,
    distributed = False
):
    t = rearrange(t, '... d -> (...) d')

    calc_distributed_var = distributed and is_distributed()

    if not calc_distributed_var:
        return t.var(dim = 0, unbiased = False)

    device = t.device
    numel = tensor(t[..., 0].numel(), device = device, dtype = t.dtype)
    dist.all_reduce(numel)

    summed = t.sum(dim = 0)
    dist.all_reduce(summed)

    mean = summed / numel
    centered = (t - mean)

    centered_squared_sum = centered.square().sum(dim = 0)
    dist.all_reduce(centered_squared_sum)

    return centered_squared_sum / numel

class BatchRMSNorm(Module):
    def __init__(
        self,
        dim_feat,
        channel_first = False,
        distributed = True,
        momentum = 0.9,
        eps = 1e-5,
        update_running_var = True
    ):
        super().__init__()
        self.eps = eps
        self.momentum = 1. - momentum
        self.channel_first = channel_first
        self.distributed = distributed

        self.gamma = nn.Parameter(torch.ones(dim_feat))
        self.beta = nn.Parameter(torch.zeros(dim_feat))

        self.update_running_var = update_running_var
        self.register_buffer('running_var', torch.ones((dim_feat,)))

    def forward(
        self,
        x,
        update_running_var = None
    ):
        gamma, beta, running_var, channel_first = self.gamma, self.beta, self.running_var, self.channel_first

        update_running_var = default(update_running_var, self.update_running_var, self.training)

        if update_running_var:

            with torch.no_grad():
                to_reduce = rearrange(x, 'b d ... -> b ... d') if channel_first else x

                batch_var = get_maybe_dist_var(to_reduce, distributed = self.distributed)

                running_var.lerp_(batch_var.to(running_var.dtype), self.momentum)

        # get denominator

        std = (running_var + self.eps).sqrt()

        # handle channel first

        if channel_first:
            std, gamma, beta = tuple(append_dims(t, x.ndim - 2) for t in (std, gamma, beta))

        std = std.to(x.dtype)
        gamma = gamma.to(x.dtype)
        beta = beta.to(x.dtype)

        # norm

        batch_rmsnormed = x / std

        # scale and offset

        return batch_rmsnormed * gamma + beta

# for easier time freezing the running variance

def set_update_running_var(
    root: Module,
    update_running_var
):
    for mod in root.modules():
        if isinstance(mod, BatchRMSNorm):
            mod.update_running_var = update_running_var

# channel first rmsnorm

class ChannelFirstRMSNorm(Module):
    def __init__(
        self,
        dim
    ):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        gamma = append_dims(self.gamma, x.ndim - 2)
        beta = append_dims(self.beta, x.ndim - 2)
        var = x.float().square().mean(dim = 1, keepdim = True)
        inv = torch.rsqrt(var + 1e-5).to(x.dtype)
        gamma = gamma.to(x.dtype)
        beta = beta.to(x.dtype)
        return x * inv * gamma + beta

class RMSNormWithBias(Module):
    def __init__(self, dim, eps = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        var = x.float().square().mean(dim = -1, keepdim = True)
        inv = torch.rsqrt(var + self.eps).to(x.dtype)
        weight = self.weight.to(x.dtype)
        bias = self.bias.to(x.dtype)
        return x * inv * weight + bias

# function for determining which rmsnorm

def RMSNorm(dim, channel_first = False, batch = False):

    if not batch and not channel_first:
        return RMSNormWithBias(dim)
    elif not batch and channel_first:
        return ChannelFirstRMSNorm(dim)
    else:
        return BatchRMSNorm(dim, channel_first = channel_first)

# convolutional unet related

class WeightStandardConv(Conv1d):
    def __init__(
        self,
        dim,
        dim_out,
        width,
        *args,
        **kwargs
    ):
        super().__init__(dim, dim_out, width, *args, **kwargs)

class ConvBlock(Module):
    def __init__(
        self,
        dim,
        width = 5,
        dim_out = None,
        batch_rmsnorm = True
    ):
        super().__init__()
        assert is_odd(width)
        dim_out = default(dim_out, dim)

        conv_klass = Conv1d if width == 1 else WeightStandardConv

        self.net = nn.Sequential(
            RMSNorm(dim, channel_first = True, batch = batch_rmsnorm),
            GELU1702(),
            conv_klass(dim, dim_out, width, padding = width // 2)
        )

    def forward(self, x):
        return self.net(x)

class DownresBlock(Module):
    def __init__(
        self,
        dim,
        channels_to_add = 128 # this is new as well? instead of doubling channels, they add 128 at a time, and use padding or slicing for the residual
    ):
        super().__init__()

        dim_out = dim + channels_to_add
        self.pad = channels_to_add

        self.conv = ConvBlock(dim, width = 5, dim_out = dim_out)
        self.conv_out = ConvBlock(dim_out, width = 5)

        self.max_pool = Reduce('b d (n pool) -> b d n', 'max', pool = 2)

    def forward(self, x, return_pre_pool = False):

        residual = F.pad(x, (0, 0, 0, self.pad), value = 0.)

        out = self.conv(x) + residual

        out = self.conv_out(out) + out

        pooled = self.max_pool(out)
        if return_pre_pool:
            return pooled, out
        return pooled

class UpresBlock(Module):
    def __init__(
        self,
        dim,
        channels_to_remove = 128,
        residual_scale_init = 1.,
        has_skip = True
    ):
        super().__init__()

        dim_out = dim - channels_to_remove
        self.pad = channels_to_remove

        self.conv = ConvBlock(dim, width = 5, dim_out = dim_out)
        self.unet_conv = ConvBlock(dim_out, width = 1) if has_skip else None  # skip conv stays width=1

        self.conv_out = ConvBlock(dim_out, width = 5)

        self.residual_scale = nn.Parameter(torch.ones(1,) * residual_scale_init)

    def forward(
        self,
        x,
        skip = None
    ):
        length = x.shape[1]
        residual = x[:, :(length - self.pad)]

        out = self.conv(x) + residual

        out = repeat(out, 'b c n -> b c (n upsample)', upsample = 2) * self.residual_scale

        if exists(self.unet_conv):
            assert exists(skip)
            out = out + self.unet_conv(skip)

        return self.conv_out(out) + out

# position related

def relative_shift(t):
    # Match JAX _shift in reference attention.py
    # Input: (..., query_len, query_len + key_len) with query_len == key_len in this model
    *leading_dims, n_rows, n_diags = t.shape
    t = t.reshape(*leading_dims, n_diags, n_rows)
    t = t[..., 1:, :]
    t = t.reshape(*leading_dims, n_rows, n_diags - 1)
    return t[..., :n_rows]

# rotary, but with attenuation of short relative distance frequencies

class RotaryEmbedding(Module):
    def __init__(
        self,
        dim,
        max_positions = 8192
    ):
        super().__init__()
        num_freqs = dim // 2
        inv_freq = 1. / (arange(num_freqs).float() + geomspace(1, max_positions - num_freqs + 1, num_freqs))
        self.register_buffer('inv_freq', inv_freq)

    @autocast('cuda', enabled = False)
    def forward(
        self,
        positions
    ):
        device = self.inv_freq.device
        if is_tensor(positions):
            t = positions.to(device = device, dtype = self.inv_freq.dtype)
            freqs = einsum(t, self.inv_freq, '... , j -> ... j')
        else:
            t = arange(positions, device = device).type_as(self.inv_freq)
            freqs = einsum(t, self.inv_freq, 'i , j -> i j')
        return freqs.repeat_interleave(2, dim = -1)

def rotate_half(x):
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim = -1).flatten(-2)

@autocast('cuda', enabled = False)
def apply_rotary_pos_emb(pos, t):
    return t * pos.cos() + rotate_half(t) * pos.sin()

# 'central mask features' - relative positions for constituting pairwise rep

class RelativePosFeatures(Module):
    def __init__(
        self,
        pool_size = 16,
        num_features = 32,
        max_distance = 8192
    ):
        super().__init__()
        self.pool_size = pool_size
        self.num_features = num_features
        self.max_distance = max_distance
        self.register_buffer('dummy', tensor(0))

    @property
    def device(self):
        return self.dummy.device

    def forward(self, single):

        _, seq_len, _ = single.shape

        seq_len //= self.pool_size
        max_seq_len = self.max_distance // self.pool_size

        rel_pos = arange(-seq_len, seq_len, device = self.device)

        center_widths = (
            arange(self.num_features, device = self.device) +
            geomspace(1, max_seq_len - self.num_features + 1, self.num_features + 1, device = self.device)[:-1]
        )

        abs_rel_pos, rel_pos_sign = rel_pos.abs(), rel_pos.sign()
        embeds = greater('j, i -> i j', center_widths, abs_rel_pos).float()
        embeds = cat((embeds, multiply('i, i j', rel_pos_sign, embeds)), dim = -1)
        return embeds.to(single.dtype)

# Polar embeddings
# Gopalakrishnan et al. https://arxiv.org/abs/2509.10534


# prenorm and sandwich norm - they use sandwich norm for single rep, prenorm for pairwise rep

class NormWrapper(Module):
    def __init__(
        self,
        dim,
        block: Module,
        dropout = 0.,
        sandwich = False,
        use_batch_rmsnorm = True
    ):
        super().__init__()
        norm_klass = partial(RMSNorm, batch = use_batch_rmsnorm)

        self.block = block
        self.pre_rmsnorm = norm_klass(dim)

        self.post_block_dropout = nn.Dropout(dropout)
        self.post_rmsnorm = norm_klass(dim) if sandwich else nn.Identity()

    def forward(
        self,
        x,
        *args,
        **kwargs
    ):
        x = self.pre_rmsnorm(x)
        out = self.block(x, *args, **kwargs)
        out = self.post_block_dropout(out)
        return self.post_rmsnorm(out)

# attention

class Attention(Module):
    def __init__(
        self,
        dim,
        dim_head = 64,
        heads = 8,
        kv_heads = 1,
        dim_head_qk = 128,
        dim_head_v = 192,
        dim_pairwise = None,
        softclamp_value = 5., # they employ attention softclamping
        use_qk_rmsnorm = False,  # official uses LayerNorm for q/k/v norms
        attn_bias_use_batch_rmsnorm = True
    ):
        super().__init__()
        dim_pairwise = default(dim_pairwise, dim)

        self.scale = dim_head_qk ** -0.5

        qkv_proj_dim_out = (dim_head_qk * heads, dim_head_qk, dim_head_v)

        # splitting and merging of attention heads

        assert divisible_by(heads, kv_heads)
        groups = heads // kv_heads

        self.split_q_heads = Rearrange('b n (g h d) -> b g h n d', h = kv_heads, g = groups)
        self.split_kv_heads = Rearrange('b n (h d) -> b h n d', h = kv_heads)

        self.merge_heads = Rearrange('b g h n d -> b n (g h d)')

        # projections

        self.to_qkv = LinearNoBias(dim, sum(qkv_proj_dim_out))
        self.to_out = Linear(dim_head_v * heads, dim)

        # they add layernorms to queries, keys, and interestingly enough, values as well. first time i've seen this

        norm_klass = nn.RMSNorm if use_qk_rmsnorm else nn.LayerNorm
        self.q_norm = norm_klass(dim_head_qk)
        self.k_norm = norm_klass(dim_head_qk)
        self.v_norm = norm_klass(dim_head_v)

        # to attention bias

        self.to_attn_bias = Sequential(
            RMSNorm(dim_pairwise, batch = attn_bias_use_batch_rmsnorm),
            nn.GELU(),
            LinearNoBias(dim_pairwise, heads),
            Rearrange('b i j (g h) -> b g h i j', g = groups)
        )
        # variables

        self.qkv_dim_splits = qkv_proj_dim_out
        self.softclamp_value = softclamp_value

    def forward(
        self,
        x,
        pairwise = None, # Float['b i j dp']
        rotary_emb = None,
        polar_emb = None
    ):

        use_bf16 = _ALLOW_NATIVE_BF16 and x.is_cuda and getattr(torch.cuda, "is_bf16_supported", lambda: False)()

        if use_bf16:
            x_qkv = x.to(torch.bfloat16)
            qkv = F.linear(x_qkv, self.to_qkv.weight.to(torch.bfloat16))
        else:
            qkv = self.to_qkv(x)

        q, k, v = qkv.split(self.qkv_dim_splits, dim = -1)

        # they use multi-query attention, with only 1 key / value head - pretty unconventional, but maybe enough for genomic modeling

        q = self.split_q_heads(q)
        k, v = tuple(self.split_kv_heads(t) for t in (k, v))

        q = self.q_norm(q.float())
        k = self.k_norm(k.float())
        v = self.v_norm(v.float())

        if use_bf16:
            q = q.to(torch.bfloat16)
            k = k.to(torch.bfloat16)
            v = v.to(torch.bfloat16)

        if exists(rotary_emb):
            rotary_dtype = q.dtype if use_bf16 else rotary_emb.dtype
            rotary = rotary_emb.to(rotary_dtype) if rotary_emb.dtype != rotary_dtype else rotary_emb
            q, k = tuple(apply_rotary_pos_emb(rotary, t) for t in (q, k))

        # similarities

        if exists(polar_emb):
            groups = q.shape[1] // k.shape[1]

            q, inverse_pack = pack_with_inverse(q, 'b * n d')
            k = rearrange(k, 'b h n d -> b h n d')

            sim = compute_attn_similarity(q, k, polar_emb)
            sim = inverse_pack(sim, 'b * i j')
        else:
            if use_bf16:
                sim = einsum(q.float(), k.float(), 'b g h i d, b h j d -> b g h i j')
            else:
                sim = einsum(q, k, 'b g h i d, b h j d -> b g h i j')

        sim = sim * self.scale


        # add attention bias + softclamping

        if exists(pairwise):
            attn_bias = self.to_attn_bias(pairwise)

            assert divisible_by(sim.shape[-1], attn_bias.shape[-1])
            expand_factor = sim.shape[-1] // attn_bias.shape[-1]

            attn_bias = repeat(attn_bias, 'b g h i j -> b g h (i r1) (j r2)', r1 = expand_factor, r2 = expand_factor)

            sim = softclamp(sim + attn_bias, value = self.softclamp_value)

        # attention

        attn = sim.softmax(dim = -1)

        # aggregate

        if use_bf16:
            out = einsum(attn, v.float(), 'b g h i j, b h j d -> b g h i d')
            out = out.to(torch.bfloat16)
        else:
            out = einsum(attn, v, 'b g h i j, b h j d -> b g h i d')

        out = self.merge_heads(out)

        if use_bf16:
            weight = self.to_out.weight.to(out.dtype)
            bias = self.to_out.bias.to(out.dtype) if exists(self.to_out.bias) else None
            return F.linear(out, weight, bias)

        return self.to_out(out)

# single to pairwise

class SingleToPairwise(Module):
    def __init__(
        self,
        dim,
        pool_size = 16,
        dim_pairwise = 128,
        heads = 32,
        rel_pos_features_dim = None
    ):
        super().__init__()
        self.avg_pool = Reduce('b (n pool) d -> b n d', 'mean', pool = pool_size)
        self.norm = RMSNorm(dim)

        dim_inner = heads * dim_pairwise

        rel_pos_features_dim = default(rel_pos_features_dim, heads * 2)

        self.split_heads = Rearrange('... (h d) -> ... h d', h = heads)

        self.to_outer_sum = Sequential(
            nn.GELU(),
            LinearNoBias(dim, dim_pairwise * 2),
        )

        self.to_qk = LinearNoBias(dim, dim_inner * 2)
        self.qk_to_pairwise = Linear(heads, dim_pairwise)

        # relative position related

        self.to_rel_pos_encoding = Linear(rel_pos_features_dim, heads * dim_pairwise)
        self.qk_rel_pos_bias = nn.Parameter(torch.zeros(2, 1, 1, heads, dim_pairwise))

    def forward(
        self,
        single,
        rel_pos_feats = None
    ):

        single = self.avg_pool(single)
        single = self.norm(single)
        pool_seq_len = single.shape[1]

        q, k = self.to_qk(single).chunk(2, dim = -1)
        q, k = tuple(self.split_heads(t) for t in (q, k))

        sim = einsum(q, k, 'b i h d, b j h d -> b i j h')

        if exists(rel_pos_feats):
            rel_pos_encoding = self.to_rel_pos_encoding(rel_pos_feats)
            rel_pos_encoding = self.split_heads(rel_pos_encoding)

            q_rel_bias, k_rel_bias = self.qk_rel_pos_bias

            rel_q = relative_shift(einsum(q + q_rel_bias, rel_pos_encoding, 'b n h d, p h d -> b h n p'))
            rel_k = relative_shift(einsum(k + k_rel_bias, rel_pos_encoding, 'b n h d, p h d -> b h n p'))

            rel_sim = add('b h i j, b h j i -> b i j h', rel_q, rel_k) * 0.5

            sim = sim + rel_sim

        pairwise_from_sim = self.qk_to_pairwise(sim)

        outer_q, outer_k = self.to_outer_sum(single).chunk(2, dim = -1)

        outer_sum = add('b i d, b j d -> b i j d', outer_q, outer_k)

        return outer_sum + pairwise_from_sim

# pairwise attention is a single headed attention across rows, they said columns did not help

class PairwiseRowAttention(Module):
    def __init__(
        self,
        dim
    ):
        super().__init__()
        self.scale = dim ** -0.5

        self.to_qk = LinearNoBias(dim, dim * 2)
        self.to_v = Linear(dim, dim)

    def forward(
        self,
        x
    ):

        q, k = self.to_qk(x).chunk(2, dim = -1)
        v = self.to_v(x)
        # similarity

        sim = einsum(q, k, 'b n i d, b n j d -> b n i j')
        sim = sim * self.scale

        # attention

        attn = sim.softmax(dim = -1)

        # aggregate

        return einsum(attn, v, 'b n i j, b n j d -> b n i d')

# feedforward for both single and pairwise

def FeedForward(
    dim,
    *,
    dropout = 0.,
    expansion_factor = 2.,  # they only do expansion factor of 2, no glu
):
    dim_inner = int(dim * expansion_factor)

    return Sequential(
        Linear(dim, dim_inner),
        nn.ReLU(),
        nn.Dropout(dropout),
        Linear(dim_inner, dim)
    )

# transformer

class TransformerTower(Module):
    def __init__(
        self,
        dim,
        *,
        depth = 9,
        heads = 8,
        dim_head_qk = 128,
        dim_head_v = 192,
        dropout = 0.,
        ff_expansion_factor = 2.,
        polar_pos_emb = False,
        max_positions = 8192,
        dim_pairwise = 128,
        pairwise_every_num_single_blocks = 2,   # how often to do a pairwise block
        single_to_pairwise_heads = 32,          # they did 32
        pool_size = 16,
        attn_kwargs: dict = dict(),
        ff_kwargs: dict = dict()
    ):
        super().__init__()
        dim_pairwise = default(dim_pairwise, dim)

        layers = []

        self.pairwise_every = pairwise_every_num_single_blocks

        self.rel_pos_features = RelativePosFeatures(pool_size, num_features = single_to_pairwise_heads, max_distance = max_positions)

        self.polar_pos_emb = polar_pos_emb

        self.rotary_emb, self.polar_emb = None, None

        if polar_pos_emb:
            inv_freqs = 1. / (arange(dim_head_qk).float() + geomspace(1, max_positions - dim_head_qk + 1, dim_head_qk))

            self.polar_emb = PoPE(dim_head_qk, heads = 1, inv_freqs = inv_freqs)
        else:
            self.rotary_emb = RotaryEmbedding(dim_head_qk, max_positions = max_positions)

        for layer_index in range(depth):

            attn = Attention(dim = dim, dim_head_qk = dim_head_qk, dim_head_v = dim_head_v, heads = heads, dim_pairwise = dim_pairwise)

            ff = FeedForward(dim = dim, expansion_factor = ff_expansion_factor)

            attn = NormWrapper(dim = dim, block = attn, dropout = dropout, sandwich = True)
            ff = NormWrapper(dim = dim, block = ff, dropout = dropout, sandwich = True)

            # maybe pairwise

            single_to_pairwise, pairwise_attn, pairwise_ff = None, None, None

            if divisible_by(layer_index, self.pairwise_every):
                single_to_pairwise = SingleToPairwise(
                    dim = dim,
                    dim_pairwise = dim_pairwise,
                    heads = single_to_pairwise_heads,
                    pool_size = pool_size,
                    rel_pos_features_dim = single_to_pairwise_heads * 2
                )
                pairwise_attn = PairwiseRowAttention(dim_pairwise)
                pairwise_ff = FeedForward(dim = dim_pairwise, expansion_factor = ff_expansion_factor)

                pairwise_attn = NormWrapper(dim = dim_pairwise, block = pairwise_attn, dropout = dropout, use_batch_rmsnorm = False)
                pairwise_ff = NormWrapper(dim = dim_pairwise, block = pairwise_ff, dropout = dropout, use_batch_rmsnorm = False)

            # add to layers

            layers.append(ModuleList([
                attn,
                ff,
                single_to_pairwise,
                pairwise_attn,
                pairwise_ff
            ]))


        self.layers = ModuleList(layers)

        # accessible attributes
        self.dim_pairwise = dim_pairwise

    def forward(
        self,
        single
    ):

        seq_len = single.shape[1]

        pairwise = None

        rel_pos_feats = self.rel_pos_features(single)

        if self.polar_pos_emb:
            polar_emb = self.polar_emb(seq_len)
            pos_emb = dict(polar_emb = polar_emb)
        else:
            rotary_emb = self.rotary_emb(seq_len)
            pos_emb = dict(rotary_emb = rotary_emb)

        for (
            attn,
            ff,
            maybe_single_to_pair,
            maybe_pairwise_attn,
            maybe_pairwise_ff
        ) in self.layers:

            if exists(maybe_single_to_pair):
                pairwise = maybe_single_to_pair(single, rel_pos_feats) + default(pairwise, 0.)
                pairwise = maybe_pairwise_attn(pairwise) + pairwise
                pairwise = maybe_pairwise_ff(pairwise) + pairwise

            single = attn(single,  pairwise = pairwise, **pos_emb) + single
            single = ff(single) + single

        return single, pairwise

# transformer unet

class TransformerUnet(Module):
    def __init__(
        self,
        dims: tuple[int, ...] = (
            768,
            896,
            1024,
            1152,
            1280,
            1408,
            1536
        ),
        basepairs = 4,  # 4 channels: A, C, G, T (official encoding)
        dna_embed_width = 15,
        transformer_kwargs: dict = dict()
    ):
        super().__init__()

        assert is_odd(dna_embed_width)

        assert len(dims) >= 2
        first_dim, *_, last_dim = dims

        self.dna_embed = DNAEmbed(first_dim, dim_input = basepairs, width = dna_embed_width)

        dim_with_input = (basepairs, *dims)
        dim_pairs = list(zip(dim_with_input[:-1], dim_with_input[1:]))

        downs = []

        for layer_num, (dim_in, dim_out) in enumerate(dim_pairs, start = 1):
            is_first = layer_num == 1

            channel_diff = dim_out - dim_in

            assert channel_diff > 0

            if not is_first:
                down = DownresBlock(dim_in, channels_to_add = channel_diff)
                downs.append(down)

        # Build ups to mirror JAX SequenceDecoder:
        # first up keeps 1536 -> 1536, then reduces by 128 each step.
        dims_rev = list(reversed(dims))
        input_dims = [dims[-1]] + dims_rev[:-1]

        ups = []
        for input_dim, output_dim in zip(input_dims, dims_rev):
            channels_to_remove = input_dim - output_dim
            up = UpresBlock(
                input_dim,
                channels_to_remove = channels_to_remove,
                has_skip = True
            )
            ups.append(up)

        self.downs = ModuleList(downs)
        self.ups = ModuleList(ups)

        self.transformer = TransformerTower(
            dim = last_dim,
            **transformer_kwargs
        )

    def forward(
        self,
        seq,
        pre_attend_embed = None
    ):
        skips = []

        # embed with one hot and add skip

        dna_embed, skip = self.dna_embed(seq)
        skips.append(skip)

        # downs

        x = dna_embed

        for down in self.downs:
            x, skip = down(x, return_pre_pool = True)
            skips.append(skip)

        x = rearrange(x, 'b d n -> b n d')

        # embed organism

        if exists(pre_attend_embed):
            x = add('b n d, b d', x, pre_attend_embed)

        # attention
        
        single, pairwise = self.transformer(x) # 1D 128bp resolution, 2D contact pairs
        
        # ups with skips from down
        
        x = rearrange(single, 'b n d -> b d n')

        for up in self.ups:
            skip = skips.pop()
            x = up(x, skip = skip)
            
        out = rearrange(x, 'b d n -> b n d') # 1D 1bp resolution

        return TransformerUnetOutput(out, single, pairwise) # the final output, as well as single and pairwise repr from the transformer

# embedding

class DNAEmbed(Module):
    """A one-hot encoder for DNA sequences.

      A -> [1, 0, 0, 0]
      C -> [0, 1, 0, 0]
      G -> [0, 0, 1, 0]
      T -> [0, 0, 0, 1]
    
    all other characters are encoded as zeros [0, 0, 0, 0].

    and here with input:
    'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N' and other characters: -1 
    """
    def __init__(
        self,
        dim,
        dim_input = 4, # 4 channels: A, C, G, T (official encoding)
        width = 15
    ):
        super().__init__()
        assert is_odd(width)
        self.dim_input = dim_input
        self.conv = Conv1d(dim_input, dim, width, padding = width // 2)
        self.pointwise = ConvBlock(dim, width = 5)

        self.pool = Reduce('b d (n pool) -> b d n', 'max', pool = 2)

    def forward(
        self,
        seq # Int['b n']
    ):
        onehot = F.one_hot(seq.clamp(min=0), num_classes = self.dim_input).float()
        
        # set zeros to other characters
        onehot[seq < 0] = 0.0
        x = rearrange(onehot, 'b n d -> b d n')

        out = self.conv(x)
        out = out + self.pointwise(out)
        pooled = self.pool(out) # think they downsample for dna embed block

        return pooled, out

class OrganismEmbedding(Module):
    def __init__(
        self,
        dim,
        num_organisms
    ):
        super().__init__()
        self.embed = nn.Embedding(num_organisms, dim)

    def forward(
        self,
        organism_index
    ):
        return self.embed(organism_index)

class OutputEmbedding(Module):
    def __init__(
        self,
        input_dim,
        num_organisms = 2,
        skip_dim = None,
        use_batch_rmsnorm = True
    ):
        super().__init__()
        self.double_features = Linear(input_dim, 2 * input_dim)
        self.skip_proj = LinearNoBias(skip_dim, 2 * input_dim) if exists(skip_dim) else None
        self.norm = RMSNorm(2 * input_dim, batch = use_batch_rmsnorm)
        self.embed = nn.Embedding(num_organisms, 2 * input_dim)
        self.activation = GELU1702()

    def forward(
        self,
        x,
        organism_index,
        skip = None
    ):
        seq_len = x.shape[1]
        x = self.double_features(x)  # double the input features

        if exists(skip) and exists(self.skip_proj):
            skip = self.skip_proj(skip)
            assert divisible_by(seq_len, skip.shape[1])

            repeat_factor = seq_len // skip.shape[1]
            skip = repeat(skip, 'b n d -> b (n repeat) d', repeat = repeat_factor)

            x = x + skip

        emb = self.embed(organism_index)

        x = add('b n d, b d', self.norm(x), emb)

        return self.activation(x)

class OutputPairEmbedding(Module):
    def __init__(
        self,
        pair_dim = 128,
        num_organisms = 2
    ):
        super().__init__()
        self.norm = RMSNorm(pair_dim)
        self.embed = nn.Embedding(num_organisms, pair_dim)
        self.activation = GELU1702()

    def forward(
        self,
        x,  # Float['b n n d']
        organism_index
    ):
        x = symmetrize(x)
        x = self.norm(x)

        organism_embed = self.embed(organism_index)
        x = add('b i j d, b d', x, organism_embed)

        return self.activation(x)

# some reflection to make adding prediction heads easy

def get_function_arg_names(fn):
    signature = inspect.signature(fn)
    parameters = signature.parameters.values()
    return [p.name for p in parameters]

def is_disjoint(a: set, b: set):
    return not any(a.intersection(b))

# output heads

class ContactMapHead(Module):
    def __init__(
        self,
        input_dim,
        num_tracks,
        num_organisms
    ):
        super().__init__()
        self.norm = RMSNorm(input_dim)
        self.embed = nn.Embedding(num_organisms, input_dim)
        self.gelu = GELU1702()
        self.proj = Linear(input_dim, num_tracks)
        
    def forward(
        self,
        embeds_pair,
        organism_index
    ):

        x = symmetrize(embeds_pair)
        x = self.norm(x)

        organism_embed = self.embed(organism_index)
        x = add('b i j d, b d', x, organism_embed)

        x = self.gelu(x)
        return self.proj(x)

class SpliceSiteClassifier(Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = Linear(input_dim, 5)  # Donor+, Acceptor+, Donor-, Acceptor-, None

    def forward(
        self,
        embeds_1bp
    ):
        return self.linear(embeds_1bp)

class SpliceSiteUsage(Module):
    def __init__(self, input_dim, n_contexts):
        super().__init__()
        self.linear = Linear(input_dim, n_contexts)

    def forward(
        self,
        embeds_1bp
    ):
        return self.linear(embeds_1bp)  # Return logits, not sigmoid

class SpliceJunctionHead(Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        n_contexts,
        rope_max_position = 2 ** 20
    ):
        super().__init__()
        self.project = Linear(input_dim, hidden_dim)
        self.scale = nn.Parameter(torch.ones(n_contexts, hidden_dim))
        self.offset = nn.Parameter(torch.zeros(n_contexts, hidden_dim))
        self.rope = RotaryEmbedding(hidden_dim, max_positions = rope_max_position)
        self.n_contexts = n_contexts

    def tissue_scaled_rope(
        self,
        embeds_1bp, # Float['b n h']
        indices # Int['b p']
    ):
    
        batch, device = embeds_1bp.shape[0], embeds_1bp.device
        n_contexts = self.n_contexts
    
        # index and rescale
        
        batch_indices = rearrange(arange(batch, device = device), 'b -> b 1')

        x = embeds_1bp[batch_indices, indices]  # [B, P, H]

        x = multiply('b p h, t h -> b p t h', x, self.scale)
        x = add('b p t h, t h -> b p t h', x, self.offset)
    
        # get rotary embeddings [T, H]
        
        rotary_emb = self.rope(n_contexts)
    
        # apply
        
        x = apply_rotary_pos_emb(rotary_emb, x)

        return x

    def forward(
        self,
        embeds_1bp,
        splice_donor_idx,
        splice_acceptor_idx
    ):
        x_proj = self.project(embeds_1bp)
        donor_embed = self.tissue_scaled_rope(x_proj, splice_donor_idx)
        acceptor_embed = self.tissue_scaled_rope(x_proj, splice_acceptor_idx)
        scores = einsum(donor_embed, acceptor_embed, 'b d t h, b a t h -> b d a t')
        return F.softplus(scores)

class TracksScaledPrediction(Module):
    def __init__(
        self,
        dim,
        num_tracks
    ):
        super().__init__()
        self.to_pred = Linear(dim, num_tracks)
        self.scale = nn.Parameter(torch.zeros(num_tracks))

    def forward(
        self,
        x
    ):
        track_pred = self.to_pred(x)
        return F.softplus(track_pred) * F.softplus(self.scale)

class GenomeTracksHead(Module):
    """JAX-style genome tracks head with per-resolution submodules."""
    def __init__(
        self,
        dim_1bp,
        dim_128bp,
        num_tracks,
        resolutions = (1, 128)
    ):
        super().__init__()
        self.resolutions = ModuleDict()
        if 1 in resolutions:
            self.resolutions["resolution_1"] = TracksScaledPrediction(dim_1bp, num_tracks)
        if 128 in resolutions:
            self.resolutions["resolution_128"] = TracksScaledPrediction(dim_128bp, num_tracks)

    def forward(
        self,
        embeds_1bp = None,
        embeds_128bp = None
    ):
        out = {}
        if "resolution_1" in self.resolutions:
            assert embeds_1bp is not None
            out["scaled_predictions_1bp"] = self.resolutions["resolution_1"](embeds_1bp)
        if "resolution_128" in self.resolutions:
            assert embeds_128bp is not None
            out["scaled_predictions_128bp"] = self.resolutions["resolution_128"](embeds_128bp)
        return out

class ContactMapsHead(Module):
    """JAX-style contact maps head (linear projection only)."""
    def __init__(
        self,
        input_dim,
        num_tracks
    ):
        super().__init__()
        self.to_pred = Linear(input_dim, num_tracks)

    def forward(
        self,
        embeds_pair,
        organism_index = None
    ):
        x = symmetrize(embeds_pair)
        return self.to_pred(x)

class SpliceSitesJunctionHead(Module):
    """JAX-style splice junction head with per-site rotary embeddings."""
    def __init__(
        self,
        input_dim,
        hidden_dim,
        max_num_tissues,
        rope_max_position = 2 ** 20
    ):
        super().__init__()
        self.project = Linear(input_dim, hidden_dim)
        self.hidden_dim = hidden_dim
        self.max_num_tissues = max_num_tissues

        param_size = 2 * max_num_tissues * hidden_dim
        self.pos_donor_embeddings = nn.Parameter(torch.zeros(param_size))
        self.pos_acceptor_embeddings = nn.Parameter(torch.zeros(param_size))
        self.neg_donor_embeddings = nn.Parameter(torch.zeros(param_size))
        self.neg_acceptor_embeddings = nn.Parameter(torch.zeros(param_size))

        self.rope = RotaryEmbedding(hidden_dim, max_positions = rope_max_position)

    def _apply_rope(
        self,
        embeds_1bp,
        indices,
        embeddings_param
    ):
        batch, device = embeds_1bp.shape[0], embeds_1bp.device
        indices = indices.to(device = device, dtype = torch.long)
        safe_indices = indices.clamp_min(0)
        batch_indices = arange(batch, device = device)[:, None]

        x = embeds_1bp[batch_indices, safe_indices]  # [B, P, H]

        params = embeddings_param.view(2, self.max_num_tissues, self.hidden_dim)
        scale = params[0]
        offset = params[1]

        x = x[:, :, None, :] * scale[None, None, :, :] + offset[None, None, :, :]

        rotary_emb = self.rope(safe_indices)
        # Expand rotary_emb to match the tissue dimension: [B, P, H] -> [B, P, 1, H]
        rotary_emb = rotary_emb[:, :, None, :]
        x = apply_rotary_pos_emb(rotary_emb, x)
        return x

    def forward(
        self,
        embeds_1bp,
        splice_site_positions = None
    ):
        if splice_site_positions is None:
            return None
        # splice_site_positions: [B, 4, P]
        pos_donor_idx = splice_site_positions[:, 0, :]
        pos_accept_idx = splice_site_positions[:, 1, :]
        neg_donor_idx = splice_site_positions[:, 2, :]
        neg_accept_idx = splice_site_positions[:, 3, :]

        x_proj = self.project(embeds_1bp)

        pos_donor_logits = self._apply_rope(x_proj, pos_donor_idx, self.pos_donor_embeddings)
        pos_accept_logits = self._apply_rope(x_proj, pos_accept_idx, self.pos_acceptor_embeddings)
        neg_donor_logits = self._apply_rope(x_proj, neg_donor_idx, self.neg_donor_embeddings)
        neg_accept_logits = self._apply_rope(x_proj, neg_accept_idx, self.neg_acceptor_embeddings)

        pos_counts = F.softplus(einsum(pos_donor_logits, pos_accept_logits, 'b d t h, b a t h -> b d a t'))
        neg_counts = F.softplus(einsum(neg_donor_logits, neg_accept_logits, 'b d t h, b a t h -> b d a t'))

        pos_mask = (pos_donor_idx >= 0)[:, :, None] & (pos_accept_idx >= 0)[:, None, :]
        neg_mask = (neg_donor_idx >= 0)[:, :, None] & (neg_accept_idx >= 0)[:, None, :]

        pos_counts = pos_counts * pos_mask[..., None]
        neg_counts = neg_counts * neg_mask[..., None]

        pred_counts = cat((pos_counts, neg_counts), dim = -1)

        splice_junction_mask = cat((
            pos_mask[..., None].expand(*pos_mask.shape, self.max_num_tissues),
            neg_mask[..., None].expand(*neg_mask.shape, self.max_num_tissues)
        ), dim = -1)

        return dict(
            predictions = pred_counts,
            splice_site_positions = splice_site_positions,
            splice_junction_mask = splice_junction_mask
        )

class TargetScaler(Module):
    def __init__(
        self,
        track_means: list[float] | Tensor,
        apply_squashing = False, # for rna-seq they squash the signal. not intimate enough with these assays to understand why yet
        squashing_factor = 0.75,
        softclip_scale = 2.,
        softclip_gamma = 10.,
        softclip_threshold = 10.
    ):
        super().__init__()

        if not is_tensor(track_means):
            track_means = tensor(track_means).float()

        self.register_buffer('track_means', track_means)

        self.apply_squashing = apply_squashing
        self.squashing_factor = squashing_factor

        self.softclip = SoftClip(threshold = softclip_threshold, scale = softclip_scale, gamma = softclip_gamma)

    def inverse(self, normalized):
        x = self.softclip.inverse(normalized)

        if self.apply_squashing:
            x = x ** (1. / self.squashing_factor)

        return x * self.track_means

    def forward(self, tracks):
        tracks = tracks / self.track_means

        if self.apply_squashing:
            tracks = tracks ** self.squashing_factor

        return self.softclip(tracks)

# classes

class AlphaGenome(Module):
    def __init__(
        self,
        dims: tuple[int, ...] = (
            768,
            896,
            1024,
            1152,
            1280,
            1408,
            1536
        ),
        basepairs = 4,  # 4 channels: A, C, G, T (official encoding)
        dna_embed_width = 15,
        num_organisms = 2,
        transformer_kwargs: dict = dict(),
    ):
        super().__init__()

        self.transformer_unet = TransformerUnet(
            dims = dims,
            basepairs = basepairs,
            dna_embed_width = dna_embed_width,
            transformer_kwargs = transformer_kwargs
        )

        # organism embed and output embed

        first_dim, *_, last_dim = dims

        self.organism_embed = OrganismEmbedding(last_dim, num_organisms)

        self.outembed_128bp = OutputEmbedding(last_dim, num_organisms)
        self.outembed_1bp = OutputEmbedding(first_dim, num_organisms, skip_dim = 2*last_dim)
        self.outembed_pair = OutputPairEmbedding(self.transformer_unet.transformer.dim_pairwise, num_organisms)

        # heads

        self.num_organisms = num_organisms
        self.dim_1bp = last_dim
        self.dim_128bp = 2 * last_dim
        self.dim_contacts = self.transformer_unet.transformer.dim_pairwise

        self.heads = ModuleDict()
        self.head_forward_arg_names = defaultdict(dict)
        self.head_forward_arg_maps = defaultdict(dict) # contains a map of forward argument names to the embed name to be used

    @property
    def total_parameters(self):
        return sum([p.numel() for p in self.parameters()])

    def add_head(
        self,
        organism,
        head_name,
        head: Module,
        head_input_kwarg_names: str | tuple[str, ...] | None = None
    ):
        if isinstance(head_input_kwarg_names, str):
            head_input_kwarg_names = (head_input_kwarg_names,)

        if organism not in self.heads:
            self.heads[organism] = ModuleDict()

        self.heads[organism][head_name] = head

        # store the list of inputs to the head

        head_forward_arg_names = get_function_arg_names(head.forward)

        self.head_forward_arg_names[organism][head_name] = default(head_input_kwarg_names, head_forward_arg_names)

        # create a dict[str, str] for if the custom head contains inputs that are named explicitly through `head_input_kwarg_names` in the order presented in function

        head_forwared_arg_map = dict()

        if exists(head_input_kwarg_names):
            head_forwared_arg_map = dict(zip(head_input_kwarg_names, head_forward_arg_names))

        self.head_forward_arg_maps[organism][head_name] = head_forwared_arg_map

    def add_heads(
        self,
        organism,
        num_tracks_1bp,
        num_tracks_128bp,
        num_tracks_contacts,
        num_splicing_contexts,
        hidden_dim_splice_juncs = 512
    ):
        # splice head related

        organism_heads = (

            # RNA-seq, CAGE, ATAC, DNAse and PRO-Cap
            ('1bp_tracks', TracksScaledPrediction(self.dim_1bp, num_tracks_1bp), ('embeds_1bp',)),
            
            # TF ChIP-seq and Histone ChIP-seq
            ('128bp_tracks', TracksScaledPrediction(self.dim_128bp, num_tracks_128bp), ('embeds_128bp',)),

            # Contact Maps
            ('contact_head', ContactMapHead(self.dim_contacts, num_tracks_contacts, self.num_organisms)),

            # Splicing
            ('splice_sites_classification', SpliceSiteClassifier(self.dim_1bp)),
            ('splice_sites_usage', SpliceSiteUsage(self.dim_1bp, num_splicing_contexts)),
            ('splice_sites_junction', SpliceJunctionHead(self.dim_1bp, hidden_dim_splice_juncs, num_splicing_contexts))
        )

        for add_head_args in organism_heads:
            self.add_head(organism, *add_head_args)

    def add_reference_heads(
        self,
        organism,
        genome_track_heads = None,
        num_tracks_contacts = jax_contact_maps_tracks,
        num_splice_usage_tracks = jax_splice_usage_tracks,
        splice_junction_hidden_dim = jax_splice_junction_hidden_dim,
        splice_junction_max_tissues = jax_splice_junction_max_tissues
    ):
        """Add JAX-aligned heads (rna_seq, cage, dnase, procap, atac, chip_tf, chip_histone, contact_maps, splice_*)."""
        genome_track_heads = default(genome_track_heads, jax_genome_track_heads)

        for head_name, spec in genome_track_heads.items():
            head = GenomeTracksHead(
                dim_1bp = self.dim_1bp,
                dim_128bp = self.dim_128bp,
                num_tracks = spec["num_tracks"],
                resolutions = spec["resolutions"]
            )
            self.add_head(organism, head_name, head)

        self.add_head(
            organism,
            "contact_maps",
            ContactMapsHead(self.dim_contacts, num_tracks_contacts)
        )

        self.add_head(
            organism,
            "splice_sites_classification",
            SpliceSiteClassifier(self.dim_1bp)
        )

        self.add_head(
            organism,
            "splice_sites_usage",
            SpliceSiteUsage(self.dim_1bp, num_splice_usage_tracks)
        )

        self.add_head(
            organism,
            "splice_sites_junction",
            SpliceSitesJunctionHead(
                self.dim_1bp,
                splice_junction_hidden_dim,
                splice_junction_max_tissues
            )
        )

    def load_from_official_jax_model(
        self,
        model_version = "all_folds",
        use_gpu = True,
        strict = False
    ):
        try:
            from alphagenome_pytorch.convert.convert_checkpoint import load_jax_checkpoint, convert_checkpoint
        except ImportError as e:
            print(f"Error: Missing required dependency for conversion. Details: {e}")
            return

        # Load and convert JAX weights

        flat_params, flat_state = load_jax_checkpoint(model_version, use_gpu = use_gpu)
        state_dict = convert_checkpoint(flat_params, flat_state)

        # Load into PyTorch model
        self.load_state_dict(state_dict, strict = strict)
        return self

    def get_embeds(
        self,
        seq, # Int['b n']
        organism_index,
        return_unet_transformer_outputs = False
    ):

        organism_embed = self.organism_embed(organism_index)

        unet_transformer_out = self.transformer_unet(seq, pre_attend_embed = organism_embed)

        unet_out, single, pairwise = unet_transformer_out

        # embed organism to outputs

        embeds_128bp = self.outembed_128bp(single, organism_index)
        embeds_1bp = self.outembed_1bp(unet_out, organism_index, embeds_128bp)
        embeds_pair = self.outembed_pair(pairwise, organism_index)

        embeds = Embeds(embeds_1bp, embeds_128bp, embeds_pair)

        if not return_unet_transformer_outputs:
            return embeds

        return embeds, unet_transformer_out
    
    def forward(
        self,
        seq,
        organism_index: int | Tensor,
        return_embeds = False,
        **head_kwargs
    ):

        # handle int organism_index (0 for human, 1 for mouse)

        if isinstance(organism_index, int):
            batch = seq.shape[0]
            organism_index = torch.full((batch,), organism_index, device = seq.device)

        # process sequence
        
        embeds, unet_transformer_out = self.get_embeds(seq, organism_index, return_unet_transformer_outputs = True)

        # early return embeds, if specified

        if return_embeds or len(self.heads) == 0:
            return embeds

        # get output tracks

        embeds_1bp, embeds_128bp, embeds_pair = embeds

        head_inputs = dict(
            embeds_1bp = embeds_1bp,
            embeds_128bp = embeds_128bp,
            embeds_pair = embeds_pair,
            organism_index = organism_index,
            unet_output = unet_transformer_out.unet_output,
            single_repr = unet_transformer_out.single_repr,
            pairwise_repr = unet_transformer_out.pairwise_repr,
        )
        assert is_disjoint(set(head_inputs.keys()), set(head_kwargs.keys()))

        head_inputs.update(**head_kwargs)
        head_inputs.setdefault("splice_site_positions", None)

        out = dict()

        for organism, heads in self.heads.items():

            organism_head_args = self.head_forward_arg_names[organism]

            organism_head_arg_map = self.head_forward_arg_maps[organism]

            organism_out = dict()

            for head_name, head in heads.items():

                # get the inputs needed for the head, which is determined by the arg / kwarg name on the forward

                head_args = organism_head_args[head_name]
                head_arg_map = organism_head_arg_map[head_name]

                head_kwargs = {head_arg: head_inputs[head_arg] for head_arg in head_args}

                # remap the kwargs

                head_kwargs = {(head_arg_map.get(k, k)): v for k, v in head_kwargs.items()}

                # forward gathered inputs through the specific head

                head_out = head(**head_kwargs)

                organism_out[head_name] = head_out

            out[organism] = organism_out

        return out

    def inference(
        self,
        seq: Tensor,
        organism_index: int | Tensor,
        *,
        requested_heads: set[str] | None = None,
        track_masks: dict[str, Tensor] | None = None,
        embeds: Embeds | None = None,
        return_embeds: bool = False,
        **head_kwargs,
    ) -> dict[str, dict] | tuple[dict[str, dict], Embeds]:
        """Memory-efficient inference with selective head execution.

        This method allows running only a subset of heads and applying track
        masks for memory-efficient inference. Unlike forward(), it does not
        return raw embeddings by default and is optimized for production use.

        Args:
            seq: Input DNA sequence tensor [B, S] (tokenized integers 0-3).
            organism_index: Organism index (0=human, 1=mouse) or batch tensor.
            requested_heads: Set of head names to run. None = all heads.
            track_masks: Optional per-head boolean masks for track filtering.
                Keys are head names, values are boolean tensors indicating
                which tracks to keep.
            embeds: Precomputed embeddings for reuse (from get_embeds()).
            return_embeds: If True, also return embeddings as second element.
            **head_kwargs: Additional kwargs passed to head forward methods
                (e.g., splice_site_positions).

        Returns:
            If return_embeds=False:
                Dict[organism][head_name] -> predictions (optionally masked).
            If return_embeds=True:
                Tuple of (predictions dict, Embeds namedtuple).

        Example:
            >>> # Run only RNA-seq and CAGE heads
            >>> outputs = model.inference(
            ...     seq, organism_index=0,
            ...     requested_heads={'rna_seq', 'cage'},
            ... )
            >>> # With track filtering
            >>> track_masks = {'rna_seq': torch.tensor([True, False, True, ...])}
            >>> outputs = model.inference(seq, 0, track_masks=track_masks)
            >>> # Embedding reuse for variant scoring
            >>> embeds = model.get_embeds(ref_seq, 0)
            >>> ref_out = model.inference(ref_seq, 0, embeds=embeds)
            >>> alt_out = model.inference(alt_seq, 0)
        """
        # Handle int organism_index (0 for human, 1 for mouse)
        if isinstance(organism_index, int):
            batch = seq.shape[0]
            organism_index = torch.full((batch,), organism_index, device=seq.device)

        # Compute or reuse embeddings
        if embeds is None:
            embeds = self.get_embeds(seq, organism_index)

        # Early return if no heads
        if len(self.heads) == 0:
            return (embeds if return_embeds else {})

        embeds_1bp, embeds_128bp, embeds_pair = embeds

        head_inputs = dict(
            embeds_1bp=embeds_1bp,
            embeds_128bp=embeds_128bp,
            embeds_pair=embeds_pair,
            organism_index=organism_index,
        )
        assert is_disjoint(set(head_inputs.keys()), set(head_kwargs.keys()))

        head_inputs.update(**head_kwargs)
        head_inputs.setdefault("splice_site_positions", None)

        out = dict()

        for organism, heads in self.heads.items():
            organism_head_args = self.head_forward_arg_names[organism]
            organism_head_arg_map = self.head_forward_arg_maps[organism]

            organism_out = dict()

            for head_name, head in heads.items():
                # Skip if not in requested_heads
                if requested_heads is not None and head_name not in requested_heads:
                    continue

                # Get the inputs needed for the head
                head_args = organism_head_args[head_name]
                head_arg_map = organism_head_arg_map[head_name]

                kwargs = {head_arg: head_inputs[head_arg] for head_arg in head_args}

                # Remap the kwargs
                kwargs = {(head_arg_map.get(k, k)): v for k, v in kwargs.items()}

                # Forward gathered inputs through the specific head
                head_out = head(**kwargs)

                # Apply track mask (early slicing for memory efficiency)
                if track_masks and head_name in track_masks:
                    head_out = self._apply_track_mask(head_out, track_masks[head_name])

                organism_out[head_name] = head_out

            if organism_out:
                out[organism] = organism_out

        return (out, embeds) if return_embeds else out

    def compute_input_gradients(
        self,
        seq: Tensor,
        organism_index: int | Tensor,
        *,
        head_name: str,
        organism: str | None = None,
        output_index: int | None = None,
        output_position: int | None = None,
        gradients_x_input: bool = False,
        center_per_position: bool = False,
        return_predictions: bool = False,
    ) -> Tensor:
        """Compute gradients of a head output w.r.t. the one-hot input sequence.

        Useful for saliency maps and attribution analysis.

        Args:
            seq: Input sequence as tokenized Int[B, S] or one-hot Float[B, S, 4].
            organism_index: Organism index (0=human, 1=mouse) or batch tensor.
            head_name: Name of the head to differentiate through.
            organism: Organism key for the head (e.g. 'human'). If None, uses
                the first organism that has the requested head.
            output_index: Track/class index to use as the scalar output. If None,
                uses the sum over all classes.
            output_position: Sequence position index in the output to differentiate
                through (e.g. the index of a specific splice site). If None,
                sums over all output positions. Use this to get gradients that
                reflect what the model attends to when predicting at a specific
                site rather than the entire sequence at once.
            gradients_x_input: If True, returns gradient × input
                (saliency × one-hot sequence).
            center_per_position: If True, subtract the per-position mean across
                nucleotide channels (A/C/G/T) from the attribution tensor.
            return_predictions: If True, also returns the raw head predictions.

        Returns:
            gradients Float[B, S, 4], or tuple (gradients, predictions) when
            return_predictions=True.
        """
        # Ensure deterministic behavior by setting model to eval mode
        was_training = self.training
        self.eval()
        
        # Explicitly disable running_var updates for reproducibility
        set_update_running_var(self, False)
        
        try:
            if isinstance(organism_index, int):
                batch = seq.shape[0]
                organism_index = torch.full((batch,), organism_index, device=seq.device)

            # Build float one-hot if integer tokens were passed
            if not seq.is_floating_point():
                onehot = F.one_hot(seq.clamp(min=0), num_classes=4).float()
                onehot[seq < 0] = 0.0
            else:
                onehot = seq.float()

            onehot = onehot.detach().requires_grad_(True)

            # --- replicate TransformerUnet.forward, bypassing F.one_hot in DNAEmbed ---
            dna_emb = self.transformer_unet.dna_embed
            x = rearrange(onehot, 'b n d -> b d n')
            out = dna_emb.conv(x)
            out = out + dna_emb.pointwise(out)
            pooled = dna_emb.pool(out)

            skips = [out]  # first skip (pre-pool from dna_embed)
            x = pooled
            for down in self.transformer_unet.downs:
                x, skip = down(x, return_pre_pool=True)
                skips.append(skip)

            x = rearrange(x, 'b d n -> b n d')
            organism_embed = self.organism_embed(organism_index)
            x = add('b n d, b d', x, organism_embed)

            single, pairwise = self.transformer_unet.transformer(x)

            x = rearrange(single, 'b n d -> b d n')
            for up in self.transformer_unet.ups:
                skip = skips.pop()
                x = up(x, skip=skip)
            unet_out = rearrange(x, 'b d n -> b n d')

            embeds_128bp = self.outembed_128bp(single, organism_index)
            embeds_1bp = self.outembed_1bp(unet_out, organism_index, embeds_128bp)
            embeds_pair = self.outembed_pair(pairwise, organism_index)

            head_inputs = dict(
                embeds_1bp=embeds_1bp,
                embeds_128bp=embeds_128bp,
                embeds_pair=embeds_pair,
                organism_index=organism_index,
                splice_site_positions=None,
            )

            # Run the requested head
            predictions = None
            for org_key, heads in self.heads.items():
                if organism is not None and org_key != organism:
                    continue
                if head_name in heads:
                    head = heads[head_name]
                    head_args = self.head_forward_arg_names[org_key][head_name]
                    head_arg_map = self.head_forward_arg_maps[org_key][head_name]
                    kwargs = {ha: head_inputs[ha] for ha in head_args if ha in head_inputs}
                    kwargs = {(head_arg_map.get(k, k)): v for k, v in kwargs.items()}
                    predictions = head(**kwargs)
                    break

            if predictions is None:
                available = [f"{org}/{h}" for org, hs in self.heads.items() for h in hs]
                raise ValueError(
                    f"Head '{head_name}' not found. Available heads: {available}"
                )

            # Reduce predictions to a scalar for backward
            if output_position is not None:
                # Select specific output position (sequence dim) before flattening
                if isinstance(predictions, dict):
                    predictions = {k: v[:, output_position] for k, v in predictions.items()
                                   if isinstance(v, Tensor)}
                else:
                    predictions = predictions[:, output_position]

            if isinstance(predictions, dict):
                tensors = [v for v in predictions.values() if isinstance(v, Tensor)]
                output = torch.cat([t.reshape(t.size(0), -1) for t in tensors], dim=-1)
            else:
                output = predictions

            if output_index is not None:
                output = output[..., output_index]

            output.sum().backward()

            gradients = onehot.grad.detach()

            if gradients_x_input:
                gradients = gradients * onehot.detach()

            if center_per_position:
                gradients = gradients - gradients.mean(dim = -1, keepdim = True)

            if return_predictions:
                return gradients, predictions
            return gradients
        finally:
            # Restore original training state
            if was_training:
                self.train()

    def compute_integrated_gradients(
        self,
        seq: Tensor,
        organism_index: int | Tensor,
        *,
        head_name: str,
        organism: str | None = None,
        output_index: int | None = None,
        output_position: int | None = None,
        baseline: Tensor | str | None = 'uniform',
        steps: int = 10,
        center_per_position: bool = False,
        return_predictions: bool = False,
    ) -> Tensor:
        """Compute integrated gradients of a head output w.r.t. the one-hot input sequence.

        Integrated gradients computes attributions by integrating gradients along a
        straight path from a baseline to the actual input. This provides better
        axiom compliance (completeness, sensitivity, etc.) compared to vanilla gradients.

        Args:
            seq: Input sequence as tokenized Int[B, S] or one-hot Float[B, S, 4].
            organism_index: Organism index (0=human, 1=mouse) or batch tensor.
            head_name: Name of the head to differentiate through.
            organism: Organism key for the head (e.g. 'human'). If None, uses
                the first organism that has the requested head.
            output_index: Track/class index to use as the scalar output. If None,
                uses the sum over all classes.
            output_position: Sequence position index in the output to differentiate
                through (e.g. the index of a specific splice site). If None,
                sums over all output positions.
            baseline: Baseline input for integration. Can be:
                - 'zeros': All-zero one-hot (N characters)
                - 'uniform': Uniform distribution [0.25, 0.25, 0.25, 0.25]
                - Tensor: Custom baseline with shape matching input
                Default is 'uniform'.
            steps: Number of interpolation steps for the integral approximation.
                More steps = more accurate but slower. Default 50.
            center_per_position: If True, subtract the per-position mean across
                nucleotide channels (A/C/G/T) from the attribution tensor.
            return_predictions: If True, also returns predictions at the actual input.

        Returns:
            integrated_gradients Float[B, S, 4], or tuple (gradients, predictions)
            when return_predictions=True.

        References:
            Sundararajan et al. (2017) "Axiomatic Attribution for Deep Networks"
            https://arxiv.org/abs/1703.01365
        """
        # Ensure deterministic behavior
        was_training = self.training
        self.eval()
        set_update_running_var(self, False)
        
        try:
            if isinstance(organism_index, int):
                batch = seq.shape[0]
                organism_index = torch.full((batch,), organism_index, device=seq.device)

            # Build float one-hot if integer tokens were passed
            if not seq.is_floating_point():
                onehot = F.one_hot(seq.clamp(min=0), num_classes=4).float()
                onehot[seq < 0] = 0.0
            else:
                onehot = seq.float()

            # Create baseline
            if baseline == 'zeros':
                baseline_tensor = torch.zeros_like(onehot)
            elif baseline == 'uniform':
                baseline_tensor = torch.full_like(onehot, 0.25)
            elif isinstance(baseline, Tensor):
                baseline_tensor = baseline.to(onehot.device, dtype=onehot.dtype)
                if baseline_tensor.shape != onehot.shape:
                    raise ValueError(
                        f"Baseline shape {baseline_tensor.shape} doesn't match input shape {onehot.shape}"
                    )
            else:
                raise ValueError(
                    f"baseline must be 'zeros', 'uniform', or a Tensor. Got {baseline}"
                )

            # Generate interpolated inputs: baseline + alpha * (input - baseline)
            # alphas shape: [steps]
            alphas = torch.linspace(0, 1, steps + 1, device=onehot.device, dtype=onehot.dtype)
            
            # Compute difference once
            input_diff = onehot - baseline_tensor  # [B, S, 4]
            
            # Accumulate gradients across all interpolation points
            integrated_grads = torch.zeros_like(onehot)
            predictions_at_input = None
            
            for alpha in alphas:
                # Interpolated input: baseline + alpha * (input - baseline)
                interpolated = baseline_tensor + alpha * input_diff
                interpolated = interpolated.detach().requires_grad_(True)
                
                # Forward pass through the model
                dna_emb = self.transformer_unet.dna_embed
                x = rearrange(interpolated, 'b n d -> b d n')
                out = dna_emb.conv(x)
                out = out + dna_emb.pointwise(out)
                pooled = dna_emb.pool(out)

                skips = [out]
                x = pooled
                for down in self.transformer_unet.downs:
                    x, skip = down(x, return_pre_pool=True)
                    skips.append(skip)

                x = rearrange(x, 'b d n -> b n d')
                organism_embed = self.organism_embed(organism_index)
                x = add('b n d, b d', x, organism_embed)

                single, pairwise = self.transformer_unet.transformer(x)

                x = rearrange(single, 'b n d -> b d n')
                for up in self.transformer_unet.ups:
                    skip = skips.pop()
                    x = up(x, skip=skip)
                unet_out = rearrange(x, 'b d n -> b n d')

                embeds_128bp = self.outembed_128bp(single, organism_index)
                embeds_1bp = self.outembed_1bp(unet_out, organism_index, embeds_128bp)
                embeds_pair = self.outembed_pair(pairwise, organism_index)

                head_inputs = dict(
                    embeds_1bp=embeds_1bp,
                    embeds_128bp=embeds_128bp,
                    embeds_pair=embeds_pair,
                    organism_index=organism_index,
                    splice_site_positions=None,
                )

                # Run the requested head
                preds = None
                for org_key, heads in self.heads.items():
                    if organism is not None and org_key != organism:
                        continue
                    if head_name in heads:
                        head = heads[head_name]
                        head_args = self.head_forward_arg_names[org_key][head_name]
                        head_arg_map = self.head_forward_arg_maps[org_key][head_name]
                        kwargs = {ha: head_inputs[ha] for ha in head_args if ha in head_inputs}
                        kwargs = {(head_arg_map.get(k, k)): v for k, v in kwargs.items()}
                        preds = head(**kwargs)
                        break

                if preds is None:
                    available = [f"{org}/{h}" for org, hs in self.heads.items() for h in hs]
                    raise ValueError(
                        f"Head '{head_name}' not found. Available heads: {available}"
                    )

                # Store predictions at actual input (alpha=1) for return
                if alpha == 1.0 and return_predictions:
                    predictions_at_input = preds

                # Reduce to scalar
                if output_position is not None:
                    if isinstance(preds, dict):
                        preds = {k: v[:, output_position] for k, v in preds.items()
                                 if isinstance(v, Tensor)}
                    else:
                        preds = preds[:, output_position]

                if isinstance(preds, dict):
                    tensors = [v for v in preds.values() if isinstance(v, Tensor)]
                    output = torch.cat([t.reshape(t.size(0), -1) for t in tensors], dim=-1)
                else:
                    output = preds

                if output_index is not None:
                    output = output[..., output_index]

                # Compute gradients for this interpolation point
                output.sum().backward()
                
                if interpolated.grad is not None:
                    # Accumulate gradients (trapezoidal rule: average of endpoints)
                    # Weight endpoints by 0.5, middle points by 1.0
                    weight = 0.5 if (alpha == 0.0 or alpha == 1.0) else 1.0
                    integrated_grads += interpolated.grad.detach() * weight
                
                # Zero gradients for next iteration
                if interpolated.grad is not None:
                    interpolated.grad.zero_()

            # Complete the trapezoidal integration: multiply by step size
            integrated_grads = integrated_grads / steps
            
            # Multiply by (input - baseline) to get final integrated gradients
            integrated_grads = integrated_grads * input_diff

            if center_per_position:
                integrated_grads = integrated_grads - integrated_grads.mean(dim = -1, keepdim = True)

            if return_predictions:
                return integrated_grads, predictions_at_input
            return integrated_grads
        
        finally:
            if was_training:
                self.train()

    def plot_sequence_logo(
        self,
        sequence: Tensor,
        gradients: Tensor,
        *,
        batch_idx: int = 0,
        positions: list[int] | Tensor | None = None,
        window: int = 50,
        splice_labels: Tensor | None = None,
        sharey: bool = False,
        figsize: tuple[int, int] = (20, 3),
        save_path: str | None = None,
        dpi: int = 150,
        threshold: float = 0.0,
        logo_type: str = 'information',
        mask_to_sequence: bool = True,
        use_absolute: bool = False,
        title: str | None = None,
        ylim: tuple[float | None, float | None] | None = None,
    ) -> None:
        """Plot a sequence logo visualization of attribution scores.

        Args:
            sequence: One-hot encoded input sequence Float[B, S, 4].
            gradients: Gradient/attribution scores Float[B, S, 4].
            batch_idx: Index of batch element to plot.
            positions: Sequence positions to plot around (e.g. splice site
                indices). If given, creates one subplot per position cropped
                to [pos - window, pos + window]. If None, plots the full
                sequence. Can be derived from splice_labels with e.g.
                ``(splice_labels[i] != 4).nonzero(as_tuple=True)[0]``.
            window: Half-window size in bp around each position (default 50).
            splice_labels: Integer label tensor Int[B, S] with values
                0=donor+, 1=acceptor+, 2=donor-, 3=acceptor-, 4=no site.
                When provided, all splice sites visible in each window are
                annotated with colored vertical lines and a legend.
            sharey: If True, all subplots share the same y-axis scale.
                Useful for comparing attribution magnitudes across sites.
            figsize: Figure size per subplot (width, height) in inches.
            save_path: Path to save figure. If None, displays it.
            dpi: Resolution for saved figure.
            threshold: Minimum attribution score to display.
            logo_type: 'information', 'probability', or 'weight'. Only used if 
                mask_to_sequence=False. 'information' scales heights by information 
                content (bits), 'probability' scales by probability, and 'weight' 
                uses raw attribution scores.
            mask_to_sequence: Multiply attributions by one-hot to show only
                the present nucleotide at each position.
            use_absolute: Use absolute attribution values for logo heights.
            title: Figure suptitle. If None, a default title is generated.
            ylim: Custom y-axis limits as (y_min, y_max). Either value can be None
                to auto-scale that dimension. For example: (None, 1.0) to only set max,
                (-0.5, None) to only set min. If fully None, automatically scales based
                on data with 10% padding.
        """
        if not _HAS_MATPLOTLIB:
            raise ImportError(
                "matplotlib is required for plotting. Install with: pip install matplotlib"
            )
        if not _HAS_LOGOMAKER:
            raise ImportError(
                "logomaker is required for sequence logo plotting. Install with: pip install logomaker"
            )

        import numpy as np
        import pandas as pd
        from matplotlib.lines import Line2D

        _CLASS_LABELS = {0: 'donor +', 1: 'acceptor +', 2: 'donor -', 3: 'acceptor -', 4: 'no splice site'}
        _CLASS_COLORS = {0: '#ff7f00', 1: '#33a02c', 2: '#fdbf6f', 3: '#b2df8a', 4: '#1f78b4'}

        seq_np = sequence[batch_idx].detach().cpu().float().numpy()    # (S, 4)
        grad_np = gradients[batch_idx].detach().cpu().float().numpy()  # (S, 4)

        if mask_to_sequence:
            grad_np = grad_np * seq_np

        values_np = np.abs(grad_np) if use_absolute else grad_np

        labels_np = None
        if splice_labels is not None:
            labels_np = splice_labels[batch_idx].cpu().numpy()  # (S,)

        def _make_logo_df(v):
            df = pd.DataFrame(v, columns=['A', 'C', 'G', 'T'])
            if threshold > 0:
                # Threshold by magnitude, preserving sign
                df = df.where(np.abs(df) >= threshold, 0)
            if not mask_to_sequence:
                # Only normalize if all values are non-negative
                # (probability/information normalization doesn't work with negative gradients)
                has_negative = (df < 0).any().any()
                if not has_negative:
                    if logo_type == 'probability':
                        df = df.div(df.sum(axis=1) + 1e-10, axis=0)
                    elif logo_type == 'information':
                        prob = np.asarray(df.div(df.sum(axis=1) + 1e-10, axis=0))
                        entropy = np.sum(prob * np.log2(prob + 1e-10), axis=1, keepdims=True)
                        df = pd.DataFrame(prob * (2 + entropy), columns=df.columns, index=df.index)
                # If has negative values, use raw attribution scores (no normalization)
            return df

        def _apply_ylim(ax, logo_df, custom_ylim=None):
            """Apply y-axis limits, respecting None values for partial specification."""
            if custom_ylim is None or (custom_ylim[0] is None and custom_ylim[1] is None):
                # Full auto-scale
                y_min = logo_df.min().min()
                y_max = logo_df.max().max()
                y_range = y_max - y_min
                if y_range > 0:
                    ax.set_ylim(y_min - 0.1 * y_range, y_max + 0.1 * y_range)
            else:
                # Partial specification: use custom values where provided, auto-scale where None
                y_min_custom, y_max_custom = custom_ylim
                y_min_data = logo_df.min().min()
                y_max_data = logo_df.max().max()
                y_range = y_max_data - y_min_data
                
                # Use custom values where provided, otherwise auto-scale
                y_min = y_min_custom if y_min_custom is not None else (y_min_data - 0.1 * y_range if y_range > 0 else y_min_data)
                y_max = y_max_custom if y_max_custom is not None else (y_max_data + 0.1 * y_range if y_range > 0 else y_max_data)
                
                ax.set_ylim(y_min, y_max)

        def _ylabel():
            if mask_to_sequence:
                return 'Attribution'
            elif logo_type == 'information':
                return 'Information Content (bits)'
            elif logo_type == 'probability':
                return 'Probability'
            return 'Attribution Score'

        def _annotate_sites(ax, start, end):
            """Draw vertical lines for all splice sites in [start, end)."""
            if labels_np is None:
                return
            seen_classes = set()
            for p in range(start, end):
                cls = int(labels_np[p])
                if cls == 4:
                    continue
                ax.axvline(p, color=_CLASS_COLORS[cls], linewidth=1.2,
                           linestyle='--', alpha=0.85)
                seen_classes.add(cls)
            if seen_classes:
                handles = [
                    Line2D([0], [0], color=_CLASS_COLORS[c], linewidth=1.5,
                           linestyle='--', label=_CLASS_LABELS[c])
                    for c in sorted(seen_classes)
                ]
                ax.legend(handles=handles, fontsize=8, loc='upper right',
                          framealpha=0.7)

        seq_len = values_np.shape[0]

        if positions is None:
            # --- full-sequence plot ---
            fig, ax = plt.subplots(figsize=figsize)
            logo_df = _make_logo_df(values_np)
            logomaker.Logo(logo_df, ax=ax,
                           color_scheme='classic', show_spines=True, baseline_width=0.5)
            # Set y-axis limits: use custom if provided, otherwise auto-scale
            _apply_ylim(ax, logo_df, ylim)
            ax.set_xlabel('Position in Sequence', fontsize=12)
            ax.set_ylabel(_ylabel(), fontsize=12)
            ax.set_title(title or 'Sequence Logo: Attributions', fontsize=14)
            ax.grid(True, alpha=0.3, axis='y', linestyle='--')
            _annotate_sites(ax, 0, seq_len)
        else:
            # --- one subplot per splice site ---
            if isinstance(positions, Tensor):
                positions = positions.cpu().tolist()
            positions = [int(p) for p in positions]
            n = len(positions)
            
            # Calculate global y_min and y_max if sharey=True
            global_y_min, global_y_max = None, None
            if sharey:
                all_dfs = [_make_logo_df(values_np[max(0, p - window):min(seq_len, p + window + 1)]) 
                           for p in positions]
                if all_dfs:
                    global_y_min = min(df.min().min() for df in all_dfs)
                    global_y_max = max(df.max().max() for df in all_dfs)
            
            fig, axes = plt.subplots(n, 1, figsize=(figsize[0], figsize[1] * n),
                                     squeeze=False, sharey=sharey)
            for idx, pos in enumerate(positions):
                start = max(0, pos - window)
                end = min(seq_len, pos + window + 1)
                logo_df = _make_logo_df(values_np[start:end])
                logo_df.index = range(start, start + len(logo_df))
                ax = axes[idx, 0]
                logomaker.Logo(logo_df, ax=ax, color_scheme='classic',
                               show_spines=True, baseline_width=0.5)
                # Set y-limits: use custom if provided, otherwise use global/local min/max
                if ylim is not None:
                    _apply_ylim(ax, logo_df, ylim)
                elif sharey and global_y_min is not None and global_y_max is not None:
                    y_range = global_y_max - global_y_min
                    if y_range > 0:
                        ax.set_ylim(global_y_min - 0.1 * y_range, global_y_max + 0.1 * y_range)
                else:
                    _apply_ylim(ax, logo_df, None)
                if labels_np is not None:
                    _annotate_sites(ax, start, end)
                else:
                    ax.axvline(pos, color='red', linewidth=1.0, linestyle='--', alpha=0.7)
                ax.set_xlabel('Sequence position', fontsize=10)
                ax.set_ylabel(_ylabel(), fontsize=10)
                ax.set_title(f'Position {pos}', fontsize=11)
                ax.grid(True, alpha=0.3, axis='y', linestyle='--')

            if title:
                fig.suptitle(title, fontsize=14, y=1.01)

        plt.tight_layout()

        if save_path is not None:
            plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
            print(f"Sequence logo saved to: {save_path}")
        else:
            plt.show()

        plt.close()

    def _apply_track_mask(self, head_output, mask: Tensor):
        """Slice output to only include masked tracks.

        Args:
            head_output: Output from a prediction head (Tensor or dict).
            mask: Boolean tensor indicating which tracks to keep.

        Returns:
            Filtered output with only the masked tracks.
        """
        if isinstance(head_output, Tensor):
            # Standard tensor output: [..., num_tracks]
            return head_output[..., mask]
        elif isinstance(head_output, dict):
            # Dict output (e.g., GenomeTracksHead returns dict with resolutions)
            result = {}
            for k, v in head_output.items():
                if isinstance(v, Tensor) and v.dim() >= 2:
                    result[k] = v[..., mask]
                else:
                    result[k] = v
            return result
        return head_output

    def compile(self, **compile_kwargs):
        """Compile the model with torch.compile for optimized inference.

        Args:
            **compile_kwargs: Arguments passed to torch.compile (mode, backend, etc.)

        Returns:
            Compiled model

        Example:
            model = AlphaGenome().compile(mode="reduce-overhead")
        """
        return torch.compile(self, **compile_kwargs)

    @torch.no_grad()
    def score_variant(
        self,
        seq_ref: Tensor,
        seq_alt: Tensor,
        organism_index: int | Tensor,
        scorer,
        settings,
        variant,
        interval,
        track_metadata,
        organism: str | None = None,
        requested_heads: set[str] | None = None,
        track_masks: dict[str, Tensor] | None = None,
        **head_kwargs,
    ):
        """Score a genetic variant using reference and alternate sequences.

        This is a convenience method for variant effect prediction that:
        1. Runs inference on both reference and alternate sequences
        2. Extracts masks and metadata using the scorer
        3. Computes variant scores
        4. Returns finalized results as AnnData

        Args:
            seq_ref: Reference sequence tensor [B, S] or [S] (tokenized), or
                one-hot [B, S, 4] / [S, 4]
            seq_alt: Alternate sequence tensor [B, S] or [S] (tokenized), or
                one-hot [B, S, 4] / [S, 4]
            organism_index: Organism index (0=human, 1=mouse)
            scorer: VariantScorer instance (e.g., GeneVariantScorer, CenterMaskVariantScorer)
            settings: Scorer-specific settings dataclass
            variant: alphagenome.data.genome.Variant object
            interval: alphagenome.data.genome.Interval object
            track_metadata: Output metadata from the model
            organism: Optional organism key for selecting outputs when multiple
                organisms are present in the model output.
            requested_heads: Set of head names to run. None = all heads.
            track_masks: Optional per-head boolean masks for track filtering.

        Returns:
            anndata.AnnData: Variant scores with gene/track annotations

        Example:
            >>> from alphagenome_pytorch.scoring import GeneVariantScorer
            >>> scorer = GeneVariantScorer(gene_mask_extractor)
            >>> result = model.score_variant(
            ...     ref_seq, alt_seq, organism_index=0,
            ...     scorer=scorer, settings=settings,
            ...     variant=variant, interval=interval,
            ...     track_metadata=metadata,
            ...     requested_heads={'rna_seq'},  # Only run RNA-seq head
            ... )
        """
        def _looks_one_hot(seq: Tensor) -> bool:
            if seq.dim() < 2 or seq.shape[-1] != 4:
                return False
            if seq.dtype.is_floating_point or seq.dtype == torch.bool:
                return True
            if seq.dtype in (
                torch.uint8,
                torch.int8,
                torch.int16,
                torch.int32,
                torch.int64,
            ):
                seq_min = seq.min().item()
                seq_max = seq.max().item()
                return seq_min >= 0 and seq_max <= 1
            return False

        def _normalize_seq(seq: Tensor) -> Tensor:
            if not isinstance(seq, Tensor):
                raise TypeError('Sequence inputs must be torch.Tensors.')
            if _looks_one_hot(seq):
                seq = seq.argmax(dim=-1)
            if seq.dim() == 1:
                seq = seq.unsqueeze(0)
            if seq.dim() != 2:
                raise ValueError(
                    f'Expected tokenized sequences with shape [B, S]. Got {tuple(seq.shape)}.'
                )
            return seq

        seq_ref = _normalize_seq(seq_ref)
        seq_alt = _normalize_seq(seq_alt)

        # Get predictions for both sequences using inference()
        ref_out = self.inference(
            seq_ref, organism_index,
            requested_heads=requested_heads,
            track_masks=track_masks,
            **head_kwargs,
        )
        alt_out = self.inference(
            seq_alt, organism_index,
            requested_heads=requested_heads,
            track_masks=track_masks,
            **head_kwargs,
        )

        # Select organism outputs if model returns multiple organisms
        if isinstance(ref_out, dict) and ref_out:
            sample_value = next(iter(ref_out.values()))
            if isinstance(sample_value, dict):
                if organism is None:
                    if len(ref_out) == 1:
                        organism = next(iter(ref_out.keys()))
                    else:
                        raise ValueError(
                            'Multiple organisms found in model output. '
                            'Pass organism="..." to select the correct outputs.'
                        )
                if organism not in ref_out:
                    raise KeyError(
                        f'Organism {organism!r} not found in model output. '
                        f'Available organisms: {list(ref_out.keys())}'
                    )
                ref_out = ref_out[organism]
                alt_out = alt_out[organism]

        # Normalize outputs to expected enum keys when possible
        requested_output = getattr(settings, 'requested_output', None)
        if requested_output is not None:
            from enum import Enum

            output_enum = type(requested_output) if isinstance(requested_output, Enum) else None

            head_to_output_name = {
                'atac': 'ATAC',
                'cage': 'CAGE',
                'dnase': 'DNASE',
                'rna_seq': 'RNA_SEQ',
                'procap': 'PROCAP',
                'chip_histone': 'CHIP_HISTONE',
                'chip_tf': 'CHIP_TF',
                'contact_maps': 'CONTACT_MAPS',
                'splice_sites_classification': 'SPLICE_SITES',
                'splice_sites_usage': 'SPLICE_SITE_USAGE',
                'splice_sites_junction': 'SPLICE_JUNCTIONS',
            }

            output_name_to_resolution = {
                'ATAC': 1,
                'CAGE': 1,
                'DNASE': 1,
                'RNA_SEQ': 1,
                'PROCAP': 1,
                'CHIP_HISTONE': 128,
                'CHIP_TF': 128,
                'SPLICE_SITES': 1,
                'SPLICE_SITE_USAGE': 1,
                'SPLICE_JUNCTIONS': 1,
                'CONTACT_MAPS': 2048,
            }

            def _select_prediction(head_output, output_name: str):
                if isinstance(head_output, dict):
                    resolution = output_name_to_resolution.get(output_name)
                    if resolution == 1 and 'scaled_predictions_1bp' in head_output:
                        return head_output['scaled_predictions_1bp']
                    if resolution == 128 and 'scaled_predictions_128bp' in head_output:
                        return head_output['scaled_predictions_128bp']
                    if resolution == 2048 and 'scaled_predictions_2048bp' in head_output:
                        return head_output['scaled_predictions_2048bp']
                return head_output

            def _normalize_outputs(outputs):
                if output_enum is None or not isinstance(outputs, dict):
                    return outputs
                mapped = {}
                for head_name, head_output in outputs.items():
                    output_name = head_to_output_name.get(head_name)
                    if output_name is None:
                        continue
                    if output_name not in output_name_to_resolution:
                        continue
                    try:
                        output_key = output_enum[output_name]
                    except Exception:
                        continue
                    mapped[output_key] = _select_prediction(head_output, output_name)
                return mapped if mapped else outputs

            ref_out = _normalize_outputs(ref_out)
            alt_out = _normalize_outputs(alt_out)

            if isinstance(ref_out, dict) and requested_output not in ref_out:
                raise KeyError(
                    f'Requested output {requested_output!r} not found in model '
                    'predictions. Ensure reference heads are added (e.g. '
                    'add_reference_heads) or provide outputs keyed by the '
                    'requested output enum.'
                )

        def _resolve_mapping_key(mapping, key):
            if key in mapping:
                return key
            if hasattr(key, 'name'):
                name = key.name
                if name in mapping:
                    return name
                if name.lower() in mapping:
                    return name.lower()
                for candidate in mapping:
                    if hasattr(candidate, 'name') and candidate.name == name:
                        return candidate
            if isinstance(key, str):
                if key in mapping:
                    return key
                if key.lower() in mapping:
                    return key.lower()
                if key.upper() in mapping:
                    return key.upper()
                for candidate in mapping:
                    if hasattr(candidate, 'name') and candidate.name == key.upper():
                        return candidate
            return None

        def _apply_track_mask(prediction, mask):
            if prediction is None or mask is None:
                return prediction
            if isinstance(prediction, dict) and 'predictions' in prediction:
                pred = dict(prediction)
                preds = pred.get('predictions')
                if isinstance(preds, Tensor):
                    tiled_mask = torch.from_numpy(mask).to(
                        device=preds.device, dtype=torch.bool
                    )
                    tiled_mask = tiled_mask.repeat(2)
                    pred['predictions'] = preds[..., tiled_mask]
                elif preds is not None:
                    pred['predictions'] = preds[..., np.tile(mask, 2)]
                return pred
            if isinstance(prediction, Tensor):
                mask_tensor = torch.from_numpy(mask).to(
                    device=prediction.device, dtype=torch.bool
                )
                return prediction[..., mask_tensor]
            return prediction[..., mask]

        def _reverse_complement_prediction(prediction, reindex):
            if prediction is None or reindex is None:
                return prediction
            if isinstance(prediction, dict) and 'predictions' in prediction:
                # Splice junction predictions are left as-is (strand handled in model).
                return prediction
            if isinstance(prediction, Tensor):
                if prediction.dim() == 3:
                    pred_rc = prediction.flip(1)
                elif prediction.dim() == 4:
                    pred_rc = prediction.flip(1).flip(2)
                else:
                    pred_rc = prediction
                reindex_tensor = torch.as_tensor(
                    reindex, device=pred_rc.device, dtype=torch.long
                )
                return pred_rc.index_select(-1, reindex_tensor)
            if prediction.ndim == 3:
                pred_rc = prediction[:, ::-1, :]
            elif prediction.ndim == 4:
                pred_rc = prediction[:, ::-1, ::-1, :]
            else:
                pred_rc = prediction
            return pred_rc[..., reindex]

        # Apply track masks if provided via metadata (padding mask)
        metadata_for_masks = track_metadata
        if (
            requested_output is not None
            and hasattr(metadata_for_masks, 'padding')
            and isinstance(ref_out, dict)
        ):
            import numpy as np

            padding = metadata_for_masks.padding
            padding_key = _resolve_mapping_key(padding, requested_output)
            if padding_key is not None:
                mask = ~np.asarray(padding[padding_key])
                output_key = _resolve_mapping_key(ref_out, requested_output) or requested_output
                if output_key in ref_out:
                    ref_out[output_key] = _apply_track_mask(ref_out[output_key], mask)
                if output_key in alt_out:
                    alt_out[output_key] = _apply_track_mask(alt_out[output_key], mask)

                # Filter metadata to match masked predictions
                if hasattr(track_metadata, 'get'):
                    meta = track_metadata.get(padding_key)
                    if meta is not None:
                        try:
                            meta = meta[mask]
                        except Exception:
                            pass
                        track_metadata = {output_key: meta}

        # Reverse complement predictions on negative strand when reindexing available
        if (
            requested_output is not None
            and getattr(interval, 'negative_strand', False)
            and hasattr(metadata_for_masks, 'strand_reindexing')
            and isinstance(ref_out, dict)
        ):
            reindexing = metadata_for_masks.strand_reindexing
            reindex_key = _resolve_mapping_key(reindexing, requested_output)
            if reindex_key is not None:
                reindex = reindexing[reindex_key]
                output_key = _resolve_mapping_key(ref_out, requested_output) or requested_output
                if output_key in ref_out:
                    ref_out[output_key] = _reverse_complement_prediction(
                        ref_out[output_key], reindex
                    )
                if output_key in alt_out:
                    alt_out[output_key] = _reverse_complement_prediction(
                        alt_out[output_key], reindex
                    )

        # Get masks and metadata
        masks, mask_metadata = scorer.get_masks_and_metadata(
            interval, variant, settings=settings, track_metadata=track_metadata
        )

        # Score the variant
        scores = scorer.score_variant(
            ref_out, alt_out,
            masks=masks,
            settings=settings,
            variant=variant,
            interval=interval,
        )

        # Convert scores to numpy for finalization
        import numpy as np
        scores_np = {
            k: v.cpu().numpy() if isinstance(v, Tensor) else v
            for k, v in scores.items()
        }

        # Finalize and return AnnData
        return scorer.finalize_variant(
            scores_np,
            track_metadata=track_metadata,
            mask_metadata=mask_metadata,
            settings=settings,
        )


