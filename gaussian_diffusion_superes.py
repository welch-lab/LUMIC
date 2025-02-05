from unet import *
import torchvision
from tqdm import tqdm
from collections import namedtuple

ModelPrediction =  namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])
    
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
def linear_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype = torch.float32)
def eval_decorator(fn):
    """
    Decorator for sampling from Imagen. Temporarily sets the model in evaluation mode if it was training.
    """
    def inner(model, *args, **kwargs):
        was_training = model.training
        model.eval()
        out = fn(model, *args, **kwargs)
        model.train(was_training)
        return out

    return inner
def normalize_neg_one_to_one(img):
    return img * 2 - 1

def unnormalize_zero_to_one(t):
    return (t + 1) * 0.5
class GaussianDiffusionSuperRes(nn.Module):
    def __init__(self, *, timesteps: int, schedule = "cosine"):
        super().__init__()
        self.num_timesteps = timesteps
        if schedule == "cosine":
            betas = cosine_beta_schedule(timesteps = timesteps)
        elif schedule == "linear":
            betas = linear_beta_schedule(timesteps = timesteps)
        
        else:
            print("Not Implemented")
        
        alphas = 1 - betas
        alphas_cumprod = torch.cumprod(alphas, axis = 0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value = 1.0)
        
        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float16))
        
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
        
    def _get_times(self, batch_size, noise_level, *, device = torch.device):
        return torch.full((batch_size,), int(self.num_timesteps * noise_level), device = device, dtype = torch.long)
    def _sample_random_times(self, batch_size, *, device):
        return torch.randint(0, self.num_timesteps, (batch_size,), device = device, dtype = torch.long)
    def _get_sampling_timesteps(self, batch, *, device):
        time_transitions = []
        for i in reversed(range(self.num_timesteps)):
            time_transitions.append((torch.full((batch, ), i, device = device, dtype = torch.long)))
        return time_transitions
    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (extract(self.posterior_mean_coef1, t, x_t.shape) * x_start + extract(self.posterior_mean_coef2, t, x_t.shape) * x_t)
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped
    def q_sample(self, x_start, t, noise):
        noise = default(noise, lambda:torch.randn_like(x_start))
        return (extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start + extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise)

    def predict_start_from_noise(self, x_t, t, noise):
        return (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise)
    def predict_noise_from_start(self, x_t, t, x0):
        return ((extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape))
class SuperRes(nn.Module):
    def __init__(self, unet, loss_type = "l2", channels = 3, timesteps = 1000, lowres_sample_noise_level = 0.05, cond_drop_prob = 0.2, batch_size = 4):
        super().__init__()
        self.loss_type = loss_type
        
        self.noise_schedule = GaussianDiffusionSuperRes(timesteps = timesteps, schedule = "cosine")
        self.lowres_schedule = GaussianDiffusionSuperRes(timesteps = timesteps, schedule = "linear")
        self.lowres_sample_noise_level = lowres_sample_noise_level
        self.cond_drop_prob = cond_drop_prob
        self.normalize_img = normalize_neg_one_to_one
        self.unnormalize_img = unnormalize_zero_to_one
        self.model = unet
        self.channels = channels
        self.ema = None
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dynamic_thresholding_percentile = 0.9
        self.ddim_sampling_eta = 0
       
    @property
    def loss_fn(self):
        if self.loss_type == "l2":
            return F.mse_loss
        elif self.loss_type == "l1":
            return F.l1_loss
        else:
            raise ValueError
            
    def _p_mean_variance(self, x, t, *, noise_schedule, cond_emb = None, cond_mask = None, lowres_cond_img = None, lowres_noise_times = None, cond_scale = 5, model_output = None):
        pred = default(model_output, self.model.forward_with_cond_scale(x, t, cond_emb = cond_emb, cond_mask = cond_mask, cond_scale = cond_scale, lowres_cond_img = lowres_cond_img, lowres_noise_times = lowres_noise_times)) 
        x_start = noise_schedule.predict_start_from_noise(x, t= t, noise = pred)
        x_start.clamp_(-1, 1)
        
        return noise_schedule.q_posterior(x_start = x_start, x_t = x, t = t)
        
    @torch.no_grad()
    def _p_sample(self, x, t, *, noise_schedule, cond_emb = None, cond_mask = None, lowres_cond_img = None, lowres_noise_times = None, cond_scale = 5):
        b, *_, device = *x.shape, x.device
        
        model_mean, _, model_log_variance = self._p_mean_variance(x = x, t = t, noise_schedule = noise_schedule, cond_emb = cond_emb, cond_mask = cond_mask, cond_scale = cond_scale, lowres_cond_img = lowres_cond_img, lowres_noise_times = lowres_noise_times)
        noise = torch.randn_like(x)
        is_last_sampling_timestep = (t == 0)
        nonzero_mask = (1 - is_last_sampling_timestep.float()).reshape(b, *((1,) * (len(x.shape) - 1)))

        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def _p_sample_loop(self, shape, *, noise_schedule, cond_emb = None, cond_mask = None, lowres_cond_img = None, lowres_noise_times = None, cond_scale = 5):
        device = self.device
        lowres_cond_img = maybe(self.normalize_img)(lowres_cond_img)
        batch = shape[0]
        timesteps = noise_schedule._get_sampling_timesteps(batch, device = device)
        img = torch.randn(shape, device = device)
        
        for times in tqdm(timesteps, desc = "sampling loop time steps", total = len(timesteps)):
            img = self._p_sample(img, times, cond_emb = cond_emb, cond_mask = cond_mask, cond_scale = cond_scale, lowres_cond_img = lowres_cond_img, lowres_noise_times = lowres_noise_times, noise_schedule = noise_schedule)
        img.clamp_(-1, 1)
        unnormalize_img = self.unnormalize_img(img)
        return unnormalize_img
    @torch.no_grad()
    def ddim_sample(self, shape, cond_emb, cond_mask, lowres_cond_img, lowres_noise_times, noise_schedule, cond_scale = 5):
        batch, device, total_timesteps, alphas, eta = shape[0], self.device, noise_schedule.num_timesteps, noise_schedule.alphas_cumprod, self.ddim_sampling_eta
        timesteps = 250
        times = torch.linspace(0., total_timesteps, steps = timesteps + 2)[:-1].cuda()

        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))
        time_pairs = list(filter(lambda t: t[0] > t[1], time_pairs))

        img = torch.randn(shape, device = device).cuda()
        lowres_cond_img = maybe(self.normalize_img)(lowres_cond_img)
        for time, time_next in tqdm(time_pairs, desc = 'sampling loop time step'):
            is_last_timestep = time_next == 0
            for r in reversed(range(0, 1)):
                is_last_resample_step = r == 0

                alpha = alphas[time]
                alpha_next = alphas[time_next]
                t = torch.full((batch,), time, device = device, dtype = torch.long)
                unet_output = self.model.module.forward_with_cond_scale(img, t, cond_emb = cond_emb, cond_mask = cond_mask, cond_scale = cond_scale, lowres_cond_img = lowres_cond_img, lowres_noise_times = lowres_noise_times)
                x_start = noise_schedule.predict_start_from_noise(img, t= t, noise = unet_output)
                x_start.clamp_(-1, 1)

                pred_noise = noise_schedule.predict_noise_from_start(img, t = t, x0 = x_start)

                c1 = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
                c2 = ((1 - alpha_next) - torch.square(c1)).sqrt()
                noise = torch.randn_like(img) if not is_last_timestep else 0.

                img = x_start * alpha_next.sqrt() + \
                      c1 * noise + \
                      c2 * pred_noise

        img = self.unnormalize_img(img)
        return img

            
    @torch.no_grad()
    @eval_decorator
    def sample(self, img, cond_emb = None, cond_mask = None, cond_scale = 5):
        device = self.device
        batch_size = cond_emb.shape[0]
        image_size = 256
        lowres_sample_noise_level = self.lowres_sample_noise_level
        
        lowres_noise_times = self.lowres_schedule._get_times(batch_size, lowres_sample_noise_level, device = device)
        img = torchvision.transforms.Resize(64)(img)
        lowres_cond_img = torchvision.transforms.Resize(256)(img)
        lowres_cond_img = self.lowres_schedule.q_sample(x_start = lowres_cond_img, t = lowres_noise_times, noise = torch.randn_like(lowres_cond_img).cuda())
        shape = (batch_size, self.channels, image_size, image_size)
        
        img = self.ddim_sample(shape, cond_emb = cond_emb, cond_mask = cond_mask, lowres_cond_img = lowres_cond_img, lowres_noise_times = lowres_noise_times, noise_schedule = self.noise_schedule, cond_scale = cond_scale)
        outputs = img
        return outputs
        
    def _p_losses(self, x_start, times, *, noise_schedule, lowres_cond_img = None, lowres_aug_times = None, cond_emb = None, cond_mask = None, noise = None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        x_start = self.normalize_img(x_start)
        lowres_cond_img = self.normalize_img(lowres_cond_img)
        x_noisy = noise_schedule.q_sample(x_start = x_start, t = times, noise = noise)
        lowres_aug_times = default(lowres_aug_times, times)
        lowres_cond_img_noisy = self.lowres_schedule.q_sample(x_start = lowres_cond_img, t = lowres_aug_times, noise = torch.randn_like(lowres_cond_img))
        pred = self.model.forward(x_noisy, times, cond_emb = cond_emb, cond_mask = cond_mask, lowres_noise_times = lowres_aug_times, lowres_cond_img = lowres_cond_img_noisy, cond_drop_prob = self.cond_drop_prob)
        loss = self.loss_fn(pred, noise)
        return loss
        
    def forward(self, images, cond_emb = None, cond_mask = None):
        b, c, h, w, device = *images.shape, images.device
        target_image_size = 256
        prev_image_size = 64
        times = self.noise_schedule._sample_random_times(b, device = device)
        lowres_cond_img = torchvision.transforms.Resize(64)(images)
        lowres_cond_img = torchvision.transforms.Resize(256)(lowres_cond_img)
        lowres_aug_time = self.lowres_schedule._sample_random_times(1, device = device)
        lowres_aug_times = repeat(lowres_aug_time, '1 -> b', b = b)
        return self._p_losses(images, times, cond_emb = cond_emb, cond_mask = cond_mask, noise_schedule = self.noise_schedule, lowres_cond_img = lowres_cond_img, lowres_aug_times = lowres_aug_times)
