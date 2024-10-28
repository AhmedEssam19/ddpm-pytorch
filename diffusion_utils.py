import torch


class DiffusionUtils:
    def __init__(self, timesteps, device):
        self.timesteps = timesteps
        self.betas = torch.linspace(0.0001, 0.02, timesteps).to(device)
        alphas = 1 - self.betas
        self.sqrt_alphas = torch.sqrt(alphas).to(device)
        alpha_bars = torch.cumprod(alphas, dim=0).to(device)
        self.sqrt_alpha_bars = torch.sqrt(alpha_bars).to(device)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1 - alpha_bars).to(device)
        alpha_bars_prev = torch.cat([torch.tensor([1], device=device), alpha_bars[:-1]])
        self.sqrt_posterior_variance = torch.sqrt(self.betas * (1 - alpha_bars_prev) / (1 - alpha_bars)).to(device)

    def _extract(self, a, t, x_shape):
        a_t = a[t]
        batch_size = t.shape[0]
        return a_t.view(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

    @torch.no_grad()
    def p_sample(self, noise_predictor, x_t, t):
        beta = self._extract(self.betas, t, x_t.shape)
        sqrt_alpha = self._extract(self.sqrt_alphas, t, x_t.shape)
        sqrt_one_minus_alpha_bar = self._extract(self.sqrt_one_minus_alpha_bars, t, x_t.shape)
        model_mean = (x_t - beta * noise_predictor(x_t, t) / sqrt_one_minus_alpha_bar) / sqrt_alpha

        if t[0] == 0:
            return model_mean
        else:
            z = torch.randn_like(x_t, device=x_t.device)
            sqrt_posterior_variance = self._extract(self.sqrt_posterior_variance, t, x_t.shape)
            return model_mean + sqrt_posterior_variance * z
            
    @torch.no_grad()
    def p_sample_loop(self, noise_predictor, shape):
        device = next(noise_predictor.parameters()).device
        x_t = torch.randn(shape, device=device)
        interval = self.timesteps // 10
        imgs = []
        for i in reversed(range(self.timesteps)):
            x_t = self.p_sample(noise_predictor, x_t, torch.tensor([i] * shape[0], device=device))
            if i % interval == 0:
                imgs.append(x_t)
        return torch.cat(imgs, dim=-1)
    
    def q_sample(self, x_0, t, noise):
        sqrt_alpha_bar = self._extract(self.sqrt_alpha_bars, t, x_0.shape)
        sqrt_one_minus_alpha_bar = self._extract(self.sqrt_one_minus_alpha_bars, t, x_0.shape)
        return sqrt_alpha_bar * x_0 + sqrt_one_minus_alpha_bar * noise
