"""Request-local conditioning reuse using the original sgldiff projection modules."""
import torch


class ConditioningCache:
    def __init__(self):
        self.reset()
        self.timesteps = None

    def configure(self, timesteps):
        # RF stage casts frame timesteps to long before calling the model.
        self.timesteps = torch.cat((timesteps.detach().long().flatten(),
                                   timesteps.new_zeros(1, dtype=torch.long))).unique()
        if self.timesteps.min().item() < 0 or self.timesteps.max().item() > 1000:
            raise ValueError('RF timesteps must be within [0, 1000]')
        self.reset()

    def reset(self):
        self.text = self.temb = self.proj = self.lookup = None

    def prepare(self, embedder, context, device):
        if self.timesteps is None:
            raise RuntimeError('Configure the RF timestep schedule before model inference')
        if self.text is None:
            # Called from inside the RF forward/autocast context, exactly as the
            # original condition_embedder.forward. No independent dtype policy.
            self.text = embedder.text_embedder(context)
        if self.temb is None:
            times = self.timesteps.to(device)
            self.temb = embedder.time_embedder(times, None)
            self.proj = embedder.time_modulation(self.temb)
            # DMD timesteps are integer indices in the 1000-step training schedule.
            # Sentinel index is deliberately out of range: an unknown timestep
            # cannot silently alias a cached valid embedding.
            self.lookup = torch.full((1001,), times.numel(), device=device, dtype=torch.long)
            self.lookup[times] = torch.arange(times.numel(), device=device)

    def __call__(self, embedder, timestep, context):
        self.prepare(embedder, context, timestep.device)
        index = self.lookup.index_select(0, timestep.long().flatten())
        return self.temb.index_select(0, index), self.proj.index_select(0, index), self.text
