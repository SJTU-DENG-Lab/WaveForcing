"""CPU checks that each sampler epoch reshuffles and G/C no longer lock halves."""

import unittest

import torch
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from wf_training.utils.dataset import CyclingLoader


class IndexDataset(Dataset):
    def __init__(self, size):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        return {"idx": index}


def index_of(batch):
    value = batch["idx"]
    return int(value.item() if torch.is_tensor(value) else value)


def cycling_loader(size, *, seed=0, replicas=1, rank=0):
    dataset = IndexDataset(size)
    sampler = DistributedSampler(
        dataset, num_replicas=replicas, rank=rank, shuffle=True,
        drop_last=True, seed=seed)
    loader = DataLoader(dataset, batch_size=1, sampler=sampler)
    return CyclingLoader(loader, sampler), sampler


def take(loader, count):
    return [index_of(next(loader)) for _ in range(count)]


def generator_critic_sets(sequence):
    return set(sequence[0::2]), set(sequence[1::2])


class CyclingLoaderTest(unittest.TestCase):
    def test_later_epochs_use_a_new_permutation(self):
        loader, _ = cycling_loader(8, seed=0)
        first = take(loader, 8)
        second = take(loader, 8)
        self.assertEqual(sorted(first), list(range(8)))
        self.assertEqual(sorted(second), list(range(8)))
        self.assertNotEqual(first, second)
        self.assertEqual(loader.epoch, 1)

    def test_s2_generator_and_critic_do_not_lock_disjoint_halves(self):
        frozen, frozen_sampler = cycling_loader(8, seed=0)
        frozen_sampler.set_epoch = lambda epoch: None
        locked = take(frozen, 32)
        locked_generator, locked_critic = generator_critic_sets(locked)
        self.assertEqual(locked_generator, generator_critic_sets(locked[:8])[0])
        self.assertEqual(locked_critic, generator_critic_sets(locked[:8])[1])
        self.assertTrue(locked_generator.isdisjoint(locked_critic))
        self.assertEqual(locked_generator | locked_critic, set(range(8)))

        loader, _ = cycling_loader(8, seed=0)
        generator, critic = generator_critic_sets(take(loader, 80))
        self.assertGreater(len(generator), 4)
        self.assertGreater(len(critic), 4)
        self.assertTrue(generator & critic)
        self.assertEqual(generator | critic, set(range(8)))

    def test_seek_matches_linear_consumption(self):
        reference, _ = cycling_loader(8, seed=7)
        sequence = take(reference, 12)
        for seen in (0, 1, 8, 9, 11):
            with self.subTest(seen=seen):
                loader, _ = cycling_loader(8, seed=7)
                loader.seek(seen)
                self.assertEqual(loader.epoch, seen // 8)
                self.assertEqual(index_of(next(loader)), sequence[seen])
