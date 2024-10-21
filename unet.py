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

def l2norm(t):
    return F.normalize(t, dim = -1)
    
class Identity(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x
def cast_tuple(val, length = None):
    if isinstance(val, list):
        val = tuple(val)
    output = val if isinstance(val, tuple) else ((val,) * default(length, 1))
    if exists(length):
        assert len(output) == length
    return output
def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d
def exists(x):
    return x is not None
def prob_mask_like(shape, prob, device):
    if prob == 1:
        return torch.ones(shape, device = device, dtype = torch.bool)
    elif prob == 0:
        return torch.zeros(shape, device = device, dtype = torch.bool)
    else:
        return torch.zeros(shape, device = device).float().uniform_(0, 1) < prob

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        
    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

def Upsample(dim, dim_out = None):
    dim_out = default(dim_out, dim)

    return nn.Sequential(
        nn.Upsample(scale_factor = 2, mode = 'nearest'),
        nn.Conv2d(dim, dim_out, 3, padding = 1)
    )

def Downsample(dim, dim_out = None):
    dim_out = default(dim_out, dim)
    return nn.Sequential(Rearrange('b c (h s1) (w s2) -> b (c s1 s2) h w', s1 = 2, s2 = 2), nn.Conv2d(dim*4, dim_out, 1))

class LayerNorm(nn.Module):
    def __init__(self, feats, stable = False, dim = -1):
        super().__init__()
        self.stable = stable
        self.dim = dim

        self.g = nn.Parameter(torch.ones(feats, *((1,) * (-dim - 1))))

    def forward(self, x):
        dtype, dim = x.dtype, self.dim

        if self.stable:
            x = x / x.amax(dim = dim, keepdim = True).detach()

        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        var = torch.var(x, dim = dim, unbiased = False, keepdim = True)
        mean = torch.mean(x, dim = dim, keepdim = True)
        
        return (x - mean) * (var + eps).rsqrt().type(dtype) * self.g.type(dtype)
ChanLayerNorm = partial(LayerNorm, dim = -3)

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    
    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim -1)
        emb = torch.exp(torch.arange(half_dim, device = time.device) * -emb)
        emb = rearrange(time, 'i -> i 1') * rearrange(emb, 'j -> 1 j')
        emb = torch.cat((emb.sin(), emb.cos()), dim = -1)
        return emb
class LearnedSinusoidalPosEmb(nn.Module):
    """ following @crowsonkb 's lead with learned sinusoidal pos emb """
    """ https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8 """

    def __init__(self, dim):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim))

    def forward(self, x):
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim = -1)
        fouriered = torch.cat((x, fouriered), dim = -1)
        return fouriered
class Block(nn.Module):
    def __init__(
        self,
        dim,
        dim_out,
        groups = 8,
        norm = True
    ):
        super().__init__()
        
        self.groupnorm = nn.GroupNorm(groups, dim) if norm else Identity()
        self.activation = nn.SiLU()
        self.project = nn.Conv2d(dim, dim_out, 3, padding = 1)

    def forward(self, x, scale_shift = None):
        x = self.groupnorm(x)
        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift
      
        x = self.activation(x)
        return self.project(x)
class LinearAttentionTransformerBlock(nn.Module):
    def __init__(
        self,
        dim,
        *,
        depth = 1,
        heads = 8,
        dim_head = 32,
        ff_mult = 2,
        context_dim = None,
        **kwargs
    ):
        super().__init__()
        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                LinearAttention(dim = dim, heads = heads, dim_head = dim_head, context_dim = context_dim),
                ChanFeedForward(dim = dim, mult = ff_mult)
            ]))

    def forward(self, x, context = None):
        for attn, ff in self.layers:
            x = attn(x, context = context) + x
            x = ff(x) + x
        return x
class CrossAttention(nn.Module):
    def __init__(self, dim, *, context_dim = None, dim_head = 64, heads = 8, norm_context = False, scale = 8):
        super().__init__()
        self.scale = scale
        
        self.heads = heads
        inner_dim = dim_head * heads
        
        context_dim = default(context_dim, dim)
        
        self.norm = LayerNorm(dim)
        self.norm_context = LayerNorm(context_dim) if norm_context else Identity()
        
        self.null_kv = nn.Parameter(torch.randn(2, dim_head))
        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias = False)
        
        self.q_scale = nn.Parameter(torch.ones(dim_head))
        self.k_scale = nn.Parameter(torch.ones(dim_head))
        
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim, bias = False), LayerNorm(dim))
        
    def forward(self, x, context, mask = None):
        b, n, device = *x.shape[:2], x.device
        #x = x.cpu()
        x = self.norm(x)
     
        context = self.norm_context(context)
        q, k, v = (self.to_q(x), *self.to_kv(context).chunk(2, dim = -1))
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), (q, k, v))
 
        nk, nv = map(lambda t: repeat(t, 'd -> b h 1 d', h = self.heads,  b = b), self.null_kv.unbind(dim = -2))
        k = torch.cat((nk, k), dim = -2)
        v = torch.cat((nv, v), dim = -2)
        
        q, k = map(l2norm, (q, k))
        q = q * self.q_scale
        k = k * self.k_scale

        sim = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        
        max_neg_value = -torch.finfo(sim.dtype).max
        
        if exists(mask):
            mask = F.pad(mask, (1, 0), value = True)
            mask = rearrange(mask, 'b j -> b 1 1 j')
            sim = sim.masked_fill(~mask, max_neg_value)
        
        attn = sim.softmax(dim = -1, dtype = torch.float32)
        attn = attn.to(sim.dtype)
        
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = out.to(device)
        return self.to_out(out)
        
class LinearCrossAttention(CrossAttention):
    def forward(self, x, context, mask = None):
        b, n, device = *x.shape[:2], x.device
        
        x = self.norm(x)
        context = self.norm_context(context)
        
        q, k, v = (self.to_q(x), *self.to_kv(context).chunk(2, dim = -1))
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h = self.heads), (q, k, v))
        nk, nv = map(lambda t: repeat(t, 'd -> (b h) 1 d', h = self.heads,  b = b), self.null_kv.unbind(dim = -2))
        k = torch.cat((nk, k), dim = 0)
        v = torch.cat((nv, v), dim = 0)
        
        max_neg_value = -torch.finfo(x.dtype).max
        if exists(mask):
            mask = F.pad(mask, (1, 0), value = True)
            mask = rearrange(mask, 'b n -> b n 1')
            k = k.masked_fill(~mask, max_neg_value)
            v = v.masked_fill(~mask, 0.)
            
        q = q.softmax(dim = -1)
        k = k.softmax(dim = -2)
        
        q = q * self.scale
        context = einsum('b n d, b n e -> b d e', k, v)
        out = einsum('b n d, b d e -> b n e', q, context)
        out = rearrange(out, '(b h) n d -> b n (h d)', h = self.heads)
        return self.to_out(out)
        
def FeedForward(dim, mult = 2):
    hidden_dim = int(dim * mult)
    return nn.Sequential(LayerNorm(dim), nn.Linear(dim, hidden_dim, bias = False), nn.GELU(), LayerNorm(hidden_dim), nn.Linear(hidden_dim, dim, bias = False))
    
def ChanFeedForward(dim, mult = 2):  # in paper, it seems for self attention layers they did feedforwards with twice channel width
    hidden_dim = int(dim * mult)
    return nn.Sequential(
        ChanLayerNorm(dim),
        nn.Conv2d(dim, hidden_dim, 1, bias = False),
        nn.GELU(),
        ChanLayerNorm(hidden_dim),
        nn.Conv2d(hidden_dim, dim, 1, bias = False)
    )

class TransformerBlock(nn.Module):
    def __init__(self, dim, *, depth = 1, heads = 8, dim_head = 32, ff_mult = 2, context_dim = None):
        super().__init__()
        self.layers = nn.ModuleList([])
 
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim = dim, heads = heads, dim_head = dim_head, context_dim = context_dim),
                FeedForward(dim = dim, mult = ff_mult)
            ]))

    def forward(self, x, context = None):
        x = rearrange(x, 'b c h w -> b h w c')
        x, ps = pack([x], 'b * c')
        for attn, ff in self.layers:
            x = attn(x, context = context) + x
            x = ff(x) + x

        x, = unpack(x, ps, 'b * c')
        x = rearrange(x, 'b h w c -> b c h w')
        return x
class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, cond_dim = None, time_cond_dim = None, groups = 8, linear_attn = False, **attn_kwargs):
        super().__init__()
        self.time_mlp = None
        if exists(time_cond_dim):
            self.time_mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_cond_dim, dim_out * 2))
        self.cross_attn = None
        if exists(cond_dim):
            attn_klass = CrossAttention if not linear_attn else LinearCrossAttention
            
            self.cross_attn = attn_klass(dim = dim_out, context_dim = cond_dim, **attn_kwargs)
        self.block1 = Block(dim, dim_out, groups = groups)
        self.block2 = Block(dim_out, dim_out, groups = groups)
        
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else Identity()
    def forward(self, x, time_emb = None, cond = None):
        scale_shift = None
        
        if exists(self.time_mlp) and exists(time_emb):
            time_emb = self.time_mlp(time_emb)
            
            time_emb = rearrange(time_emb, 'b c -> b c 1 1')
            scale_shift = time_emb.chunk(2, dim = 1)
 
        h = self.block1(x)
      
        if exists(self.cross_attn):
            assert exists(cond)
       
            h = rearrange(h, 'b c h w -> b h w c')
            
            h, ps = pack([h], 'b * c')
           
            h = self.cross_attn(h, context = cond) + h
            
            h, = unpack(h, ps, 'b * c')
            h = rearrange(h, 'b h w c -> b c h w')
        
        h = self.block2(h, scale_shift = scale_shift)

        return h + self.res_conv(x)
        
class Attention(nn.Module):
    def __init__(self, dim, *, dim_head = 64, heads = 8, context_dim = None, scale = 8):
        super().__init__()
        self.scale = scale
        self.heads = heads
        hidden_dim = dim_head * heads
        self.norm = LayerNorm(dim)
        self.null_kv = nn.Parameter(torch.randn(2, dim_head))
        self.to_q = nn.Linear(dim, hidden_dim, bias = False)
        self.to_kv = nn.Linear(dim, dim_head * 2, bias = False)
    
        self.q_scale = nn.Parameter(torch.ones(dim_head))
        self.k_scale = nn.Parameter(torch.ones(dim_head))

        self.to_context = nn.Sequential(nn.LayerNorm(context_dim), nn.Linear(context_dim, dim_head * 2)) if exists(context_dim) else None

        self.to_out = nn.Sequential(
            nn.Linear(hidden_dim, dim, bias = False),
            LayerNorm(dim)
        )
        
    def forward(self, x, context = None, mask = None, attn_bias = None):
        b, n, device = *x.shape[:2], x.device
        x = self.norm(x)
        q, k, v = (self.to_q(x), *self.to_kv(x).chunk(2, dim = -1))
        q = rearrange(q, 'b n (h d) -> b h n d', h = self.heads)
      
        nk, nv = map(lambda t: repeat(t, 'd -> b 1 d', b = b), self.null_kv.unbind(dim = -2))
        k = torch.cat((nk, k), dim = -2)
        v = torch.cat((nv, v), dim = -2)

        if exists(context):
            assert exists(self.to_context)
            ck, cv = self.to_context(context).chunk(2, dim = -1)
            k = torch.cat((ck, k), dim = -2)
            v = torch.cat((cv, v), dim = -2)
        
        q, k = map(l2norm, (q, k))
        q = q * self.q_scale
        k = k * self.k_scale
       
        sim = einsum('b h i d, b j d -> b h i j', q, k) 
        sim *= self.scale
        
        if exists(attn_bias):
            sim = sim + attn_bias
        max_neg_value = -torch.finfo(sim.dtype).max
        
        if exists(mask):
            mask = F.pad(mask, (1, 0), value = True)
            mask = rearrange(mask, 'b j -> b 1 1 j')
            sim = sim.masked_fill(~mask, max_neg_value)
            
        attn = sim.softmax(dim = -1, dtype = torch.float32)
        attn = attn.to(sim.dtype)
        out = einsum('b h i j, b j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)
        
class PixelShuffleUpsample(nn.Module):
    """
    code shared by @MalumaDev at DALLE2-pytorch for addressing checkboard artifacts
    https://arxiv.org/ftp/arxiv/papers/1707/1707.02937.pdf
    """
    def __init__(self, dim, dim_out = None):
        super().__init__()
        dim_out = default(dim_out, dim)
        conv = nn.Conv2d(dim, dim_out * 4, 1)

        self.net = nn.Sequential(
            conv,
            nn.SiLU(),
            nn.PixelShuffle(2)
        )

        self.init_conv_(conv)

    def init_conv_(self, conv):
        o, i, h, w = conv.weight.shape
        conv_weight = torch.empty(o // 4, i, h, w)
        nn.init.kaiming_uniform_(conv_weight)
        conv_weight = repeat(conv_weight, 'o ... -> (o 4) ...')

        conv.weight.data.copy_(conv_weight)
        nn.init.zeros_(conv.bias.data)

    def forward(self, x):
        return self.net(x)

class LinearAttention(nn.Module):
    def __init__(
        self,
        dim,
        dim_head = 32,
        heads = 8,
        dropout = 0.05,
        context_dim = None,
        **kwargs
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        inner_dim = dim_head * heads
        self.norm = ChanLayerNorm(dim)

        self.nonlin = nn.SiLU()

        self.to_q = nn.Sequential(
            nn.Dropout(dropout),
            nn.Conv2d(dim, inner_dim, 1, bias = False),
            nn.Conv2d(inner_dim, inner_dim, 3, bias = False, padding = 1, groups = inner_dim)
        )

        self.to_k = nn.Sequential(
            nn.Dropout(dropout),
            nn.Conv2d(dim, inner_dim, 1, bias = False),
            nn.Conv2d(inner_dim, inner_dim, 3, bias = False, padding = 1, groups = inner_dim)
        )

        self.to_v = nn.Sequential(
            nn.Dropout(dropout),
            nn.Conv2d(dim, inner_dim, 1, bias = False),
            nn.Conv2d(inner_dim, inner_dim, 3, bias = False, padding = 1, groups = inner_dim)
        )

        self.to_context = nn.Sequential(nn.LayerNorm(context_dim), nn.Linear(context_dim, inner_dim * 2, bias = False)) if exists(context_dim) else None

        self.to_out = nn.Sequential(
            nn.Conv2d(inner_dim, dim, 1, bias = False),
            ChanLayerNorm(dim)
        )

    def forward(self, fmap, context = None):
        h, x, y = self.heads, *fmap.shape[-2:]

        fmap = self.norm(fmap)
        q, k, v = map(lambda fn: fn(fmap), (self.to_q, self.to_k, self.to_v))
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> (b h) (x y) c', h = h), (q, k, v))

        if exists(context):
            assert exists(self.to_context)
            ck, cv = self.to_context(context).chunk(2, dim = -1)
            ck, cv = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h = h), (ck, cv))
            k = torch.cat((k, ck), dim = -2)
            v = torch.cat((v, cv), dim = -2)

        q = q.softmax(dim = -1)
        k = k.softmax(dim = -2)

        q = q * self.scale

        context = einsum('b n d, b n e -> b d e', k, v)
        out = einsum('b n d, b d e -> b n e', q, context)
        out = rearrange(out, '(b h) (x y) d -> b (h d) x y', h = h, x = x, y = y)

        out = self.nonlin(out)
        return self.to_out(out)
class Unet(nn.Module):
    def __init__(self, *, dim, emb_dim = 384, drug_cond = None, img_cond = None, channels_out = None, cond_drop_prob = 0.5, init_dim = None, 
    out_dim = None, dim_mults = (1, 2, 4, 8), channels = 3, cond_on_emb = True, learned_variance = False, attn_heads = 8, attn_dim_head = 64,
    attn_pool_num_latents = 32, ff_mult = 2, layer_attns_depth = 1, num_resnet_blocks = 1, resnet_groups = 8, layer_attns = True, layer_cross_attns = True,
    use_linear_attn = False, use_linear_cross_attn = False, learned_sinu_pos_emb_dim = 16):
        super().__init__()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.to_gene_non_attn_cond = None
        self.channels = channels
        self.channels_out = default(channels_out, channels)
        
        num_time_tokens = emb_dim
        
        input_channels = channels
        self.cond_on_emb = cond_on_emb
        
        self.cond_drop_prob = cond_drop_prob
        init_dim = default(init_dim, dim)
       
        self.init_conv = nn.Conv2d(input_channels, init_dim, 7, padding = 3)
        
        dims = [init_dim, *map(lambda m:dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        
        cond_dim = default(emb_dim, dim)
        time_dim = emb_dim
       
        sinu_pos_emb = LearnedSinusoidalPosEmb(learned_sinu_pos_emb_dim)
        self.to_time_hiddens = nn.Sequential(sinu_pos_emb, nn.Linear(learned_sinu_pos_emb_dim + 1, time_dim), nn.SiLU())
        self.to_time_cond = nn.Sequential(nn.Linear(time_dim,emb_dim))
        self.to_time_tokens = nn.Sequential(nn.Linear(time_dim, time_dim))
        self.norm_cond = nn.LayerNorm(cond_dim)
    
        self.null_cond_emb = nn.Parameter(torch.randn(cond_dim))
        attn_kwargs = dict(heads = attn_heads, dim_head = attn_dim_head)
        layers = len(in_out)
        
        resnet_klass = partial(ResnetBlock, **attn_kwargs)
        num_resnet_blocks = cast_tuple(num_resnet_blocks, layers)
        resnet_groups = cast_tuple(resnet_groups, layers)
        
        layer_attns = cast_tuple(layer_attns, layers)
        layer_attns_depth = cast_tuple(layer_attns_depth, layers)
        layer_cross_attns = cast_tuple(layer_cross_attns, layers)
        
        use_linear_attn = cast_tuple(use_linear_attn, layers)
        use_linear_cross_attn = cast_tuple(use_linear_cross_attn, layers)
        
        self.init_resnet_block = resnet_klass(init_dim, init_dim, time_cond_dim = time_dim, groups = resnet_groups[0]) 
        layer_params = [num_resnet_blocks, resnet_groups, layer_attns, layer_attns_depth, layer_cross_attns, use_linear_attn, use_linear_cross_attn]
        
        reversed_layer_params = list(map(reversed, layer_params))
        
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        skip_connect_dims = []
        num_resolutions = len(in_out)
    
        for ind, ((dim_in, dim_out), layer_num_resnet_blocks, groups, layer_attn, layer_attn_depth, layer_cross_attn, layer_use_linear_attn, layer_use_linear_cross_attn) in enumerate(zip(in_out, *layer_params)):
            is_last = ind >= (num_resolutions - 1)
            layer_cond_dim = cond_dim if layer_cross_attn or layer_use_linear_cross_attn else None
            
            current_dim = dim_out
            skip_connect_dims.append(current_dim)
            self.downs.append(nn.ModuleList([
            Downsample(dim_in, dim_out),
            resnet_klass(current_dim, current_dim, cond_dim = layer_cond_dim, linear_attn = layer_use_linear_cross_attn, time_cond_dim = time_dim, groups = groups), 
            nn.ModuleList([ResnetBlock(current_dim, current_dim, time_cond_dim = time_dim, groups = groups) for _ in range(1)]), 
            TransformerBlock(dim = current_dim, depth = layer_attn_depth, ff_mult = ff_mult, context_dim = cond_dim, **attn_kwargs)
            ]))
        
        mid_dim = dims[-1]

        self.mid_block1 = ResnetBlock(mid_dim, mid_dim, time_cond_dim = time_dim, cond_dim = cond_dim, groups = resnet_groups[-1])
        self.mid_attn = TransformerBlock(mid_dim, depth = layer_attn_depth, **attn_kwargs)
        self.mid_block2 = ResnetBlock(mid_dim, mid_dim, time_cond_dim = time_dim, cond_dim = cond_dim, groups = resnet_groups[-1])
        upsample_fmap_dims = []
        for ind, ((dim_in, dim_out), layer_num_resnet_blocks, groups, layer_attn, layer_attn_depth, layer_cross_attn, layer_use_linear_attn, layer_use_linear_cross_attn) in enumerate(zip(reversed(in_out), *reversed_layer_params)):
            is_last = ind == (len(in_out) -1)
            layer_cond_dim = cond_dim if layer_cross_attn or layer_use_linear_cross_attn else None
            skip_connect_dim = skip_connect_dims.pop()
            upsample_fmap_dims.append(dim_out)
            
            self.ups.append(nn.ModuleList([
            resnet_klass(dim_out + skip_connect_dim, dim_out, cond_dim = layer_cond_dim, linear_attn = layer_use_linear_cross_attn, time_cond_dim = time_dim, groups = groups),
            nn.ModuleList([ResnetBlock(dim_out + skip_connect_dim, dim_out, time_emb_dim = time_dim, groups = groups) for _ in range(1)]), 
            TransformerBlock(dim = dim_out, depth = layer_attn_depth, ff_mult = ff_mult, context_dim = cond_dim, **attn_kwargs),
            PixelShuffleUpsample(dim_out, dim_in) if not is_last or True else Identity()]))
            
        default_out_dim = channels 
        
        self.out_dim = default(out_dim, default_out_dim)
        self.final_res_block = ResnetBlock(dim * 2, dim, time_cond_dim = time_dim, groups = resnet_groups[0])
        self.final_conv = nn.Conv2d(dim, default_out_dim, 3, padding = 1)
    
    def forward_with_cond_scale(self, *args, cond_scale = 1, **kwargs):
        logits = self.forward(*args, **kwargs)
        
        if cond_scale == 1:
            return logits
        
        null_logits = self.forward(*args, cond_drop_prob = 1, **kwargs)
        return null_logits + (logits - null_logits) * cond_scale
    
    def forward(self, x, time, cond_emb = None, cond_drop_prob = 0.2):
        batch_size, device = x.shape[0], x.device
        
        x = self.init_conv(x)
        
        time_hiddens = self.to_time_hiddens(time)
        time_tokens = self.to_time_tokens(time_hiddens)
        t = self.to_time_cond(time_hiddens)
 
        if exists(cond_emb) and self.cond_on_emb:
            cond_keep_mask = prob_mask_like((batch_size, ), 1 - cond_drop_prob, device = device)
            null_cond_emb = repeat(self.null_cond_emb, 'd -> b d', b = batch_size)
            cond_emb = torch.where(rearrange(cond_keep_mask, 'b -> b 1'), cond_emb, null_cond_emb)
        
        cond_emb = cond_emb[:, None, :]
        time_tokens = time_tokens[:, None, :]
        
        c = time_tokens if not exists(cond_emb) else torch.cat((time_tokens, cond_emb), dim = -2)
        c = c.float()
        c = self.norm_cond(c)
        
        if exists(self.init_resnet_block):
            x = self.init_resnet_block(x, t)
            
        hiddens = []

        r = x.clone()
        
        for down, block1, block2, res in self.downs:
            x = down(x)
            
            x = block1(x, t, c)
            
            for block in block2:
                x = block(x, t)
               
                hiddens.append(x)
      
            x = res(x, c)
            
            hiddens.append(x)
        
        x = self.mid_block1(x, t, c)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t, c)
        add_skip_connection = lambda x: torch.cat((x, hiddens.pop()), dim = 1)
        up_hiddens = []
        
        for block1, block2, res, up in self.ups:
            x = add_skip_connection(x)
            x = block1(x, t, c)
            for block in block2:
                x = add_skip_connection(x)
                
                x = block(x, t)
            
            x = res(x,  c)
           
            up_hiddens.append(x.contiguous())
            x = up(x)
        
        x = torch.cat((x, r), dim = 1)
 
        x = self.final_res_block(x, t)
        
        return self.final_conv(x)

class UnetSuperRes(nn.Module):
    def __init__(self, *, dim, emb_dim = 384, channels_out = None, cond_drop_prob = 0.5, init_dim = None,
    out_dim = None, dim_mults = (1, 2, 4, 8), channels = 3, cond_on_emb = True, learned_variance = False, attn_heads = 8, attn_dim_head = 64,
    attn_pool_num_latents = 32, ff_mult = 2, layer_attns_depth = 1, num_resnet_blocks = 1, resnet_groups = 2, layer_attns = True, layer_cross_attns = True,
    use_linear_attn = False, use_linear_cross_attn = False, learned_sinu_pos_emb_dim = 16, lowres_cond = True):
        super().__init__()
    
        self.to_gene_non_attn_cond = None
        self.channels = channels
        self.channels_out = default(channels_out, channels)
        
        num_time_tokens = emb_dim
        
        input_channels = channels
        self.cond_on_emb = cond_on_emb
        
        self.cond_drop_prob = cond_drop_prob
        init_dim = default(init_dim, dim)
       
        self.init_conv = nn.Conv2d(input_channels * 2, init_dim, 7, padding = 3)
        
        dims = [init_dim, *map(lambda m:dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        
        cond_dim = default(emb_dim, dim)
        time_dim = emb_dim
        
        sinu_pos_emb = LearnedSinusoidalPosEmb(learned_sinu_pos_emb_dim)
        self.to_time_hiddens = nn.Sequential(sinu_pos_emb, nn.Linear(learned_sinu_pos_emb_dim + 1, time_dim), nn.SiLU())
        self.to_time_cond = nn.Sequential(nn.Linear(time_dim,emb_dim))
        self.to_time_tokens = nn.Sequential(nn.Linear(time_dim, time_dim))
        self.norm_cond = nn.LayerNorm(cond_dim)
    
        self.null_cond_emb = nn.Parameter(torch.randn(cond_dim))
        
        
        attn_kwargs = dict(heads = attn_heads, dim_head = attn_dim_head)
        layers = len(in_out)
        
        resnet_klass = partial(ResnetBlock, **attn_kwargs)
        num_resnet_blocks = cast_tuple(num_resnet_blocks, layers)

        resnet_groups = cast_tuple(resnet_groups, layers)
        
        
        self.lowres_cond = lowres_cond
        self.to_lowres_time_hiddens = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, time_dim),
            nn.SiLU()
        )

        self.to_lowres_time_cond = nn.Sequential(
            nn.Linear(time_dim, time_dim)
        )

        self.to_lowres_time_tokens = nn.Sequential(
            nn.Linear(time_dim, time_dim))
        self.to_emb_non_attn_cond = None
        
        if cond_on_emb:
            self.to_emb_non_attn_cond = nn.Sequential(nn.LayerNorm(cond_dim), nn.Linear(cond_dim, time_dim), nn.SiLU(), nn.Linear(time_dim, time_dim))
        
        layer_attns = (False, False, False, True)
        layer_attns_depth = cast_tuple(layer_attns_depth, layers)
        layer_cross_attns = (False, False, False, True)
        use_linear_attn = (False, False, False, False)
        use_linear_cross_attn = (False, False, False, False)
        
        self.init_resnet_block = resnet_klass(init_dim, init_dim, time_cond_dim = time_dim, groups = resnet_groups[0])
        layer_params = [num_resnet_blocks, resnet_groups, layer_attns, layer_attns_depth, layer_cross_attns, use_linear_attn, use_linear_cross_attn]
        
        reversed_layer_params = list(map(reversed, layer_params))
        
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        skip_connect_dims = []
        num_resolutions = len(in_out)
        
        for ind, ((dim_in, dim_out), layer_num_resnet_blocks, groups, layer_attn, layer_attn_depth, layer_cross_attn, layer_use_linear_attn, layer_use_linear_cross_attn) in enumerate(zip(in_out, *layer_params)):
            is_last = ind == (num_resolutions - 1)
            layer_cond_dim = cond_dim if layer_cross_attn or layer_use_linear_cross_attn else None
            
            current_dim = dim_out
            skip_connect_dims.append(current_dim)
            self.downs.append(nn.ModuleList([
            Downsample(dim_in, dim_out),
            resnet_klass(current_dim, current_dim, cond_dim = layer_cond_dim, linear_attn = layer_use_linear_cross_attn, time_cond_dim = time_dim, groups = groups),
            
            nn.ModuleList(
                    [
                        ResnetBlock(current_dim,
                                    current_dim,
                                    time_cond_dim=time_dim,
                                    groups=groups
                                    )
                        for _ in range(layer_num_resnet_blocks)
                    ]
                ),
            LinearAttentionTransformerBlock(dim = current_dim, depth = layer_attn_depth, ff_mult = ff_mult, context_dim = cond_dim, **attn_kwargs)
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = ResnetBlock(mid_dim, mid_dim, time_cond_dim = time_dim, cond_dim = cond_dim, groups = resnet_groups[-1])
        self.mid_attn = TransformerBlock(mid_dim, depth = layer_attn_depth, **attn_kwargs)
        self.mid_block2 = ResnetBlock(mid_dim, mid_dim, time_cond_dim = time_dim, cond_dim = cond_dim, groups = resnet_groups[-1])
        upsample_fmap_dims = []
        
        for ind, ((dim_in, dim_out), layer_num_resnet_blocks, groups, layer_attn, layer_attn_depth, layer_cross_attn, layer_use_linear_attn, layer_use_linear_cross_attn) in enumerate(zip(reversed(in_out), *reversed_layer_params)):
            is_last = ind == (num_resolutions - 1)
            layer_cond_dim = cond_dim if layer_cross_attn or layer_use_linear_cross_attn else None
            skip_connect_dim = skip_connect_dims.pop()
            upsample_fmap_dims.append(dim_out)
            
            self.ups.append(nn.ModuleList([
            resnet_klass(dim_out + skip_connect_dim, dim_out, cond_dim = layer_cond_dim, linear_attn = layer_use_linear_cross_attn, time_cond_dim = time_dim, groups = groups),
            nn.ModuleList(
                    [
                        ResnetBlock(dim_out + skip_connect_dim,
                                    dim_out,
                                    time_cond_dim=time_dim,
                                    groups=groups)
                        for _ in range(layer_num_resnet_blocks)
                    ]),
            LinearAttentionTransformerBlock(dim = dim_out, depth = layer_attn_depth, ff_mult = ff_mult, context_dim = cond_dim, **attn_kwargs),
            PixelShuffleUpsample(dim_out, dim_in) if not is_last or True else Identity()]))
        default_out_dim = channels
        
        self.out_dim = default(out_dim, default_out_dim)
        self.final_res_block = ResnetBlock(dim * 2, dim, time_cond_dim = time_dim, groups = resnet_groups[0])
     
        self.final_conv = nn.Conv2d(dim + 3, default_out_dim, 3, padding = 1)
   
    def forward_with_cond_scale(self, *args, cond_scale = 1, **kwargs):
        logits = self.forward(*args, **kwargs)
        
        if cond_scale == 1:
            return logits
        
        null_logits = self.forward(*args, cond_drop_prob = 1, **kwargs)
        return null_logits + (logits - null_logits) * cond_scale
        
    def forward(self, x, time, *, lowres_cond_img = None, lowres_noise_times = None, cond_emb = None, cond_mask = None, cond_drop_prob = 0):
        batch_size, device = x.shape[0], x.device
        if exists(lowres_cond_img):
            x = torch.cat((x, lowres_cond_img), dim = 1)
 
        x = self.init_conv(x)
        time_hiddens = self.to_time_hiddens(time)
        time_tokens = self.to_time_tokens(time_hiddens)
        t = self.to_time_cond(time_hiddens)
        
        if self.lowres_cond:
            lowres_time_hiddens = self.to_lowres_time_hiddens(lowres_noise_times)
            lowres_time_tokens = self.to_lowres_time_tokens(lowres_time_hiddens)
            lowres_t = self.to_lowres_time_cond(lowres_time_hiddens)
            
            t = t + lowres_t
            time_tokens = time_tokens[:, None, :]
            lowres_time_tokens = lowres_time_tokens[:, None, :]
            time_tokens = torch.cat((time_tokens, lowres_time_tokens), dim = -2)
            
        if exists(cond_emb) and self.cond_on_emb:
            cond_keep_mask = prob_mask_like((batch_size, ), 1 - cond_drop_prob, device = device)
            null_cond_emb = repeat(self.null_cond_emb, 'd -> b d', b = batch_size)
            cond_emb = torch.where(rearrange(cond_keep_mask, 'b -> b 1'), cond_emb, null_cond_emb)
    
        cond_emb = cond_emb[:, None, :]

        c = time_tokens if not exists(cond_emb) else torch.cat((time_tokens, cond_emb), dim = -2)
        c = c.float()
        c = self.norm_cond(c)
        
        if exists(self.init_resnet_block):
            x = self.init_resnet_block(x, t)
            
        hiddens = []

        r = x.clone()
        for down, block1, block2, res in self.downs:
            x = down(x)
            x = block1(x, t, c)
            for resnet_block in block2:
                x = resnet_block(x, t)
                hiddens.append(x)
            x = res(x, c)
            hiddens.append(x)

        x = self.mid_block1(x, t, c)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t, c)
        
        add_skip_connection = lambda x: torch.cat((x, hiddens.pop()), dim = 1)
        up_hiddens = []
        
        for block1, block2, res, up in self.ups:
            x = add_skip_connection(x)
            x = block1(x, t, c)
            
            for resnet_block in block2:
                x = add_skip_connection(x)
                x = resnet_block(x, t)
            x = res(x, c)
           
            up_hiddens.append(x.contiguous())
            x = up(x)
        
        x = torch.cat((x, r), dim = 1)
        x = self.final_res_block(x, t, c)
        
        if exists(lowres_cond_img):
            x = torch.cat((x, lowres_cond_img), dim = 1)

        return self.final_conv(x)