"""Request-local RF sigma lookup, preserving upstream arithmetic precision."""
import torch


class RFSchedule:
    def __init__(self, scheduler, steps, device, high_dtype=torch.float64, *, fused_pointwise=False):
        self.scheduler = scheduler
        self.fused_pointwise = fused_pointwise
        # pred_noise_to_pred_video receives integer model timesteps and searches
        # in FP64 (FP32 on platforms without double support).
        times = scheduler.timesteps.to(device=device, dtype=high_dtype)
        integers = torch.arange(1001, device=device, dtype=torch.long)
        nearest = (times[None] - integers[:, None]).abs().argmin(1)
        self.flow_sigmas = scheduler.sigmas.to(device=device, dtype=torch.float64)[nearest]
        native_times = scheduler.timesteps.to(device)
        native_sigmas = scheduler.sigmas.to(device)
        # add_noise receives FP32, unrounded warped steps. Don't reuse the
        # integer model lookup: near ties can select a different sigma.
        self.noise_sigmas = {}
        for value in steps.detach().cpu().tolist():
            step = float(value)
            query = torch.tensor(step, device=device, dtype=torch.float32)
            index = (native_times - query).abs().argmin()
            self.noise_sigmas[step] = native_sigmas[index].reshape(1, 1, 1, 1)

    def x0(self, prediction, noisy, timestep):
        if (self._can_fuse(prediction, noisy) and timestep.dtype == torch.long
                and timestep.is_contiguous() and timestep.numel() == prediction.shape[0]
                and timestep.device == prediction.device):
            from .schedule_kernels import x0
            return x0(prediction, noisy, timestep, self.flow_sigmas)
        sigma = self.flow_sigmas[timestep.long().flatten()].reshape(-1, 1, 1, 1)
        return (noisy.double() - sigma * prediction.double()).to(prediction.dtype)

    def add_noise_at(self, original, noise, step):
        sigma = self.noise_sigmas[float(step)]
        if self._can_fuse(original, noise) and sigma.dtype == torch.float32:
            from .schedule_kernels import add_noise
            return add_noise(original, noise, sigma)
        return ((1 - sigma) * original + sigma * noise).to(noise.dtype)

    def _can_fuse(self, first, second):
        return (self.fused_pointwise and first.is_cuda and first.ndim == second.ndim == 4
                and first.shape == second.shape and first.numel() > 0
                and first.dtype in (torch.bfloat16, torch.float32)
                and second.dtype in (torch.bfloat16, torch.float32) and first.device == second.device)


def predict_x0(stage, schedule, batch, server_args, **kw):
    """Same stage metadata/autocast/forward path with a cached flow sigma."""
    timestep = kw.pop('timestep_bf').to(device=kw['latent_model_input'].device, dtype=torch.long)
    noisy = kw.pop('noise_latents_btchw')
    kw.pop('scheduler')
    metadata = stage._build_causal_attn_metadata(batch, server_args,
        current_timestep=kw['current_timestep'],
        raw_latent_shape=kw.pop('attn_raw_latent_shape'), device=kw.pop('device'))
    prediction = stage._forward_causal_transformer(batch, timestep=timestep,
                                                  attn_metadata=metadata, **kw)
    prediction = prediction.permute(0, 2, 1, 3, 4)
    result = schedule.x0(prediction.flatten(0, 1), noisy.flatten(0, 1), timestep)
    return result.unflatten(0, prediction.shape[:2]), metadata
