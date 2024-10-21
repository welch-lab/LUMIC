import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import einops
from einops import rearrange, reduce, repeat, pack, unpack
from einops.layers.torch import Rearrange
from torch import einsum
from einops_exts import rearrange_many
import math
def prob_mask_like(shape, prob, device):
    if prob == 1:
        return torch.ones(shape, device = device, dtype = torch.bool)
    elif prob == 0:
        return torch.zeros(shape, device = device, dtype = torch.bool)
    else:
        return torch.zeros(shape, device = device).float().uniform_(0, 1) < prob
        
def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def exists(x):
    return x is not None
class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        
    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x

def Upsample(dim, dim_out = None):
    return nn.Sequential(nn.Upsample(scale_factor = 2, mode = 'nearest'),
        nn.Conv1d(dim, default(dim_out, dim), 3, padding = 1), )

def Downsample(dim, dim_out = None):
    return nn.Conv1d(dim, default(dim_out, dim), 4, 2, 1)

class WeightStandardizedConv1d(nn.Conv1d):
    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        
        weight = self.weight
  
        mean  = reduce(weight, 'o ... -> o 1 1 1', 'mean')
        
        var = reduce(weight, 'o ... -> o 1 1 1', partial(torch.var, unbiased = False))
       
        normalized_weight = (weight - mean) * (var + eps).rsqrt()
        
        return F.conv1d(x, normalized_weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1))

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        var = torch.var(x, dim = 1, unbiased = False, keepdim = True)
        mean = torch.mean(x, dim = 1, keepdim = True)
        return (x - mean) * (var + eps).rsqrt() * self.g
    
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)
        
    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    
    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim -1)
        emb = torch.exp(torch.arange(half_dim, device = device) * -emb)
        emb = time[:,None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim = -1)
        return emb

class Block(nn.Module):
    def __init__(
        self,
        dim,
        dim_out,
        groups = 8
    ):
        super().__init__()
    
        self.proj = nn.Conv1d(dim, dim_out, 3, padding = 1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift = None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x
class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context=None, mask=None):
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale
        
        del q, k
    
        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        sim = sim.softmax(dim=-1)

        out = einsum('b i j, b j d -> b i d', sim, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return self.to_out(out)
class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, cond_dim = None, time_emb_dim = None, groups= 8, **attn_kwargs):
        super().__init__()
        if exists(time_emb_dim) or exists(cond_dim):
            self.mlp = nn.Sequential(nn.SiLU(), nn.Linear(int(time_emb_dim or 0) + int(cond_dim or 0) + int(cond_dim or 0), dim_out * 2))
        else:
            self.mlp = None
            
        self.block1 = Block(dim, dim_out, groups = groups)
        self.block2 = Block(dim_out, dim_out, groups = groups)
        self.res_conv = nn.Conv1d(dim, dim_out, 1) if dim != dim_out else nn.Identity()
        
    def forward(self, x, time_emb = None, cond = None):
        scale_shift = None
        if exists(self.mlp) and (exists(time_emb) or exists(cond)):
            cond_1 = cond[:, 0,:].squeeze()
            cond_2 = cond[:, 1, :].squeeze()
            cond_emb = tuple(filter(exists, (time_emb, cond_1, cond_2)))
            cond_emb = torch.cat(cond_emb, dim = -1)
            cond_emb = self.mlp(cond_emb)
            cond_emb = rearrange(cond_emb, 'b c -> b c 1')
            scale_shift = cond_emb.chunk(2, dim = 1)
        h = self.block1(x, scale_shift = scale_shift)
        h = self.block2(h)
        return h + self.res_conv(x)


#attention from Imagen
class Attention(nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.to_qkv = nn.Conv1d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv1d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, n = x.shape
        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) n -> b h c n', h = self.heads), qkv)

        q = q * self.scale

        sim = einsum('b h d i, b h d j -> b h i j', q, k)
        attn = sim.softmax(dim = -1)
        out = einsum('b h i j, b h d j -> b h i d', attn, v)

        out = rearrange(out, 'b h n d -> b (h d) n')
        return self.to_out(out)
class LinearAttention(nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv1d(dim, hidden_dim * 3, 1, bias = False)

        self.to_out = nn.Sequential(
            nn.Conv1d(hidden_dim, dim, 1),
            LayerNorm(dim)
        )

    def forward(self, x):
        b, c, n = x.shape
        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) n -> b h c n', h = self.heads), qkv)

        q = q.softmax(dim = -2)
        k = k.softmax(dim = -1)

        q = q * self.scale

        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c n -> b (h c) n', h = self.heads)
        return self.to_out(out)
class Unet1D(nn.Module):
    def __init__(self, *, dim, gene_emb_dim = 384, gene_emb = None, channels_out = None, cond_drop_prob = 0.5, init_dim = None, out_dim = None, dim_mults = (1, 2, 4, 8), channels = 3, cond_on_emb = True, resnet_block_groups = 8, learned_variance = False, attn_heads = 8, attn_dim_head = 64, attn_pool_num_latents = 32):
        super().__init__()
        self.cond_drop_prob = cond_drop_prob
        self.channels = channels
        input_channels = channels
        init_dim = default(init_dim, dim)
        self.init_conv = nn.Conv1d(input_channels, init_dim, 7, padding = 3)
        dims = [init_dim, *map(lambda m:dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        block_klass = partial(ResnetBlock, groups = resnet_block_groups)
        
        
        num_time_tokens = gene_emb_dim
        cond_dim = default(gene_emb_dim, dim)
        time_dim = cond_dim
        self.null_gene_emb = nn.Parameter(torch.randn(384))
        self.null_dmso_emb = nn.Parameter(torch.randn(384))
        self.cond_on_emb = cond_on_emb
        self.channels_out = default(channels_out, channels)
        sinu_pos_emb = SinusoidalPosEmb(dim)
       
        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )
        
        
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)
        layer_use_linear_cross_attn = False
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            
            self.downs.append(nn.ModuleList([
            block_klass(dim_in, dim_in, time_emb_dim = time_dim, cond_dim = cond_dim),
            block_klass(dim_in, dim_in, time_emb_dim = time_dim, cond_dim = cond_dim),
            Residual(PreNorm(dim_in, LinearAttention(dim_in))),
            Downsample(dim_in, dim_out) if not is_last else nn.Conv1d(dim_in, dim_out, 3, padding = 1)
            ]))
        
        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim = time_dim, cond_dim = cond_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim = time_dim, cond_dim = cond_dim)
        
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) -1)
            
            self.ups.append(nn.ModuleList([
            block_klass(dim_out+dim_in, dim_out, time_emb_dim = time_dim, cond_dim = cond_dim),
            block_klass(dim_out + dim_in, dim_out, time_emb_dim = time_dim, cond_dim = cond_dim),
            Residual(PreNorm(dim_out, LinearAttention(dim_out))),
            Upsample(dim_out, dim_in) if not is_last else nn.Conv1d(dim_out, dim_in, 3, padding = 1)]))
           
        default_out_dim = channels
        
        self.out_dim = default(out_dim, default_out_dim)
        
        self.final_res_block = block_klass(dim*2, dim, time_emb_dim = time_dim, cond_dim = cond_dim)
        self.final_conv = nn.Conv1d(dim, self.out_dim, 1)
    
    def forward_with_cond_scale(self, *args, cond_scale = 1, **kwargs):
        logits = self.forward(*args, cond_drop_prob = 0, **kwargs)
        
        if cond_scale == 1:
            return logits
        
        null_logits = self.forward(*args, cond_drop_prob = 1, **kwargs)
        return null_logits + (logits - null_logits) * cond_scale
    
    def forward(self, x, time, gene_emb = None, dmso_emb = None, cond_drop_prob = None):
        batch_size, device = x.shape[0], x.device
        
        x = self.init_conv(x)
        
        cond_drop_prob = default(cond_drop_prob, self.cond_drop_prob)
        t = self.time_mlp(time)
        if exists(gene_emb) and self.cond_on_emb:
            gene_keep_mask = prob_mask_like((batch_size,), 1-cond_drop_prob, device = device)
            null_gene_emb= repeat(self.null_gene_emb, 'd -> b d', b = batch_size)
            gene_emb = torch.nn.functional.pad(gene_emb, (0, 256), mode='constant', value=0)
            gene_emb = torch.where(rearrange(gene_keep_mask, 'b -> b 1'), gene_emb, null_gene_emb)
        if exists(dmso_emb) and self.cond_on_emb:
            dmso_keep_mask = prob_mask_like((batch_size,), 1-cond_drop_prob, device = device)
            null_dmso_emb= repeat(self.null_dmso_emb, 'd -> b d', b = batch_size)
            dmso_emb = torch.where(rearrange(dmso_keep_mask, 'b -> b 1'), dmso_emb, null_dmso_emb)
 
        gene_emb = gene_emb[:, None, :]
        dmso_emb = dmso_emb[:, None, :]

        c = torch.cat((gene_emb, dmso_emb), dim = -2)

        hiddens = []
 
        r = x.clone()
        
        for block1, block2, res, down in self.downs:
            x = block1(x, t, c)
            hiddens.append(x)
            
            x = block2(x, t, c)
            x = res(x)
            hiddens.append(x)
            x = down(x)
        
        x = self.mid_block1(x, t, c)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t, c)

        for block1, block2, res, up in self.ups:
            x = torch.cat((x, hiddens.pop()), dim = 1)
            x = block1(x, t, c)
            
            x = torch.cat((x, hiddens.pop()), dim = 1)
            x = block2(x, t, c)
            
            x = res(x)
            x = up(x)
        
        x = torch.cat((x, r), dim = 1)
        x = self.final_res_block(x, t, c)
        return self.final_conv(x)