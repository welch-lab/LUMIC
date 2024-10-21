from unet_1d import *
    
def extract(a, t, x_shape):
    b, *_ = t.shape
    t = t.to("cuda")
    a = a.to("cuda")
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def cosine_beta_schedule(timesteps, s = 0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5)**2
    alphas_cumprod = alphas_cumprod/alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:]/alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)
    
class GaussianDiffusion1D(nn.Module):
    def __init__(self, model, *, seq_length, batch_size = 64, timesteps = 1000, sampling_timesteps = None, loss_type = 'l2', pred_objective = 'pred_noise', p2_loss_weight_gamma = 0, p2_loss_weight_k = 1):
        super().__init__()
        self.device = "cuda"
        self.model = model
        self.seq_length = seq_length
        self.objective = pred_objective
        
        self.timesteps = 1000
        betas = cosine_beta_schedule(timesteps = timesteps)
        
        alphas = 1 - betas
        alphas_cumprod = torch.cumprod(alphas, axis = 0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value = 1.0)
        
        self.loss_type = loss_type
        
        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))
        
        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        
        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))
        
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        register_buffer('posterior_variance', posterior_variance)
        
        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min =1e-20)))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))
        
        snr = alphas_cumprod / (1 - alphas_cumprod)
        maybe_clipped_snr = snr.clone()
        
        if pred_objective == 'pred_noise':
            register_buffer('loss_weight', maybe_clipped_snr / snr)
        elif pred_objective == 'pred_x0':
            register_buffer('loss_weight', maybe_clipped_snr)
        elif pred_objective == 'pred_v':
            register_buffer('loss_weight', maybe_clipped_snr / (snr + 1))

    def predict_start_from_noise(self, x_t, t, noise):
        return (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise)
    def predict_noise_from_start(self, x_t, t, x0):
        return ((extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape))
    def predict_start_from_v(self, x_t, t, noise):
        return (extract(self.sqrt_alphas_cumprod, t, x_start.shape) * noise - extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * x_t)
    def predict_v(self, x_start, t, noise):
        return (extract(self.sqrt_alphas_cumprod, t, x_start.shape) * noise - extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise)

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (extract(self.posterior_mean_coef1, t, x_t.shape) * x_start + extract(self.posterior_mean_coef2, t, x_t.shape) * x_t)
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped
    def model_predictions(self, x, t, cond, dmso_emb, cond_scale = 3, clip_x_start = False, rederive_pred_noise = False):
        model_output = self.model.forward_with_cond_scale(x, t, cond, dmso_emb, cond_scale = cond_scale)
        maybe_clip = partial(torch.clamp, min = -1., max = 1.) if clip_x_start else identity
        if self.objective == "pred_noise":
            pred_noise = model_output
            x_start = self.predict_start_from_noise(x, t, pred_noise)
            x_start = maybe_clip(x_start)
            if clip_x_start and rederive_pred_noise:
                pred_noise = self.predict_noise_from_start(x, t, x_start)

        elif self.objective == 'pred_x0':
            x_start = model_output
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        elif self.objective == 'pred_v':
            v = model_output
            x_start = self.predict_start_from_v(x, t, v)
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        return ModelPrediction(pred_noise, x_start)
        
    def p_mean_variance(self, x, t, gene_emb, dmso_emb, cond_scale, clip_denoised = True):
        preds = self.model_predictions(x, t, gene_emb, dmso_emb, cond_scale)
        x_start = preds.pred_x_start
        if clip_denoised:
            x_start.clamp_(-1., 1.)
        mean, post_var, post_log_var = self.q_posterior(x_start = x_start, x_t = x, t = t)
        return mean, post_var, post_log_var, x_start

    @torch.no_grad()
    def p_sample(self, x, t, gene_emb, dmso_emb, cond_scale = 3, clip_denoised = True):
        batch, *_, device = *x.shape, x.device
        batched_times = torch.full((x.shape[0],), t, device = x.device, dtype = torch.long)
        noise = torch.randn_like(x) if t>0 else 0
        model_mean, _, model_log_var, x_start = self.p_mean_variance(x, t = batched_times, cond_scale = cond_scale, gene_emb = gene_emb, dmso_emb = dmso_emb, clip_denoised = clip_denoised)
        pred = model_mean + (0.5 * model_log_var).exp() * noise
        return pred, x_start

    @torch.no_grad()
    def p_sample_loop(self, gene_emb, dmso_emb, shape, cond_scale = 3):
        device = self.device
        batch = shape[0]
        img = torch.randn(shape, device = device)
        x_start = None
        
        for t in tqdm(reversed(range(0, self.timesteps)), desc = 'sampling loop time step', total = self.timesteps):
            img, x_start = self.p_sample(img, t,gene_emb = gene_emb, dmso_emb = dmso_emb, cond_scale= cond_scale)
        return img
        
    @torch.no_grad()
    def sample(self, gene_emb, dmso_emb, cond_scale = 3):
        batch_size = 32
        seq_length = self.seq_length
        channels = 1
        return self.ddim_sample(gene_emb, dmso_emb, (batch_size, channels, seq_length),cond_scale)
    @property
    def loss_fn(self):
        if self.loss_type == "l2":
            return F.mse_loss
        elif self.loss_type == "l1":
            return F.l1_loss
        else:
            raise ValueError
            
    @torch.no_grad()
    def ddim_sample(self, gene_emb, dmso_emb, shape, cond_scale = 3, clip_denoised = True):
        batch = shape[0]
        device = self.device
        total_timesteps = 1000
        sampling_timesteps = 250
        eta = 0
        times = torch.linspace(-1, total_timesteps - 1, steps = sampling_timesteps + 1)
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))
        
        img = torch.randn(shape, device = device)
        x_start = None
        
        for time, time_next in tqdm(time_pairs, desc = 'sampling loop time step'):
            time_cond = torch.full((batch,), time, device = device, dtype = torch.long)
            pred_noise, x_start, *_ = self.model_predictions(img, time_cond, gene_emb, dmso_emb, cond_scale = cond_scale, clip_x_start = clip_denoised, rederive_pred_noise = True)
            
            if time_next < 0:
                img = x_start
                continue
            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]
            
            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()
            
            noise = torch.randn_like(img)
            img = x_start * alpha_next.sqrt() + c * pred_noise + sigma * noise

        return img
            
    
    def p_losses(self, x_start, t, gene_emb, dmso_emb, noise = None):
        b, c, n = x_start.shape
        noise = default(noise, lambda: torch.randn_like(x_start))
        
        x = self.q_sample(x_start = x_start, t = t, noise = noise)
        model_out = self.model.forward(x, t, gene_emb = gene_emb, dmso_emb = dmso_emb)
        
        if self.objective == 'pred_noise':
            target = noise
        elif self.objective == 'pred_x_start':
            target = x_start
        elif self.objective == 'pred_v':
            target = self.predict_v(x_start, t, noise)
        else:
            raise ValueError()
        
        losses = self.loss_fn(model_out, target, reduction = 'none')
        losses = reduce(losses, 'b ... -> b', 'mean')
        
        losses = losses * extract(self.loss_weight, t, losses.shape)
        return losses.mean()
        
    def q_sample(self, x_start, t, noise = None):
        noise = default(noise, lambda:torch.randn_like(x_start))
        return (extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start + extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise)
        
    def forward(self, img, *args, **kwargs):
        b, c, n, device, seq_length = *img.shape, img.device, self.seq_length
        t = torch.randint(0, self.timesteps, (b,), device = device).long()
        return self.p_losses(img, t, *args, **kwargs)