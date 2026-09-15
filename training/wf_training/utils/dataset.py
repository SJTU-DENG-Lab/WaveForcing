from torch.utils.data import Dataset


class TextDataset(Dataset):
    def __init__(self, prompt_path, extended_prompt_path=None):
        with open(prompt_path, encoding="utf-8") as f:
            self.prompt_list = [line.rstrip() for line in f]

        if extended_prompt_path is not None:
            with open(extended_prompt_path, encoding="utf-8") as f:
                self.extended_prompt_list = [line.rstrip() for line in f]
            assert len(self.extended_prompt_list) == len(self.prompt_list)
        else:
            self.extended_prompt_list = None

    def __len__(self):
        return len(self.prompt_list)

    def __getitem__(self, idx):
        batch = {
            "prompts": self.prompt_list[idx],
            "idx": idx,
        }
        if self.extended_prompt_list is not None:
            batch["extended_prompts"] = self.extended_prompt_list[idx]
        return batch


class CyclingLoader:
    """Yield batches forever, calling ``sampler.set_epoch`` at each pass.

    ``DistributedSampler(shuffle=True)`` is seeded by ``seed + epoch``. Leaving
    epoch at 0 makes every pass identical, so Stage 2's generator/critic pair
    of draws permanently splits a shard whose length divides ``2 * accum``.
    """

    def __init__(self, dataloader, sampler=None):
        self.dataloader = dataloader
        self.sampler = sampler
        self.epoch = 0
        self._iterator = None
        if self.sampler is not None:
            self.sampler.set_epoch(self.epoch)

    def _start_epoch(self, epoch):
        self.epoch = int(epoch)
        if self.sampler is not None:
            self.sampler.set_epoch(self.epoch)
        self._iterator = iter(self.dataloader)

    def __iter__(self):
        return self

    def __next__(self):
        if self._iterator is None:
            self._start_epoch(self.epoch)
        try:
            return next(self._iterator)
        except StopIteration:
            self._start_epoch(self.epoch + 1)
            return next(self._iterator)

    def seek(self, batches_seen):
        epoch_len = len(self.dataloader)
        if epoch_len < 1:
            raise ValueError("dataloader must contain at least one batch")
        batches_seen = int(batches_seen)
        if batches_seen < 0:
            raise ValueError("batches_seen must be nonnegative")
        self._start_epoch(batches_seen // epoch_len)
        for _ in range(batches_seen % epoch_len):
            next(self._iterator)


def cycle(dl, sampler=None):
    return CyclingLoader(dl, sampler)
