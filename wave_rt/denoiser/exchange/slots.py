"""One paged slot per layer; reuse only after the tick migration collective."""
from dataclasses import dataclass


@dataclass(frozen=True)
class PagedSlots:
    layers: int

    def __post_init__(self):
        if self.layers < 1:
            raise ValueError('Positive layer count required')

    @property
    def depth(self):
        # A producer may publish its entire forward before any consumer reads.
        # The next tick's compute follows migration, which follows every rank's
        # attention releases and publication drain. No same-tick slot reuse.
        return self.layers

    def address(self, tick, consumer, layer):
        # Consumer r first becomes active at tick r. Counting its actual
        # forwards avoids waiting for releases from nonexistent fill ticks.
        if tick < consumer or not 0 <= layer < self.layers:
            raise ValueError('Inactive consumer or invalid layer')
        sequence = (tick - consumer) * self.layers + layer + 1
        if sequence >= 2**31:
            raise ValueError('Paged credit sequence exceeds signed wrap-safe range')
        return (sequence - 1) % self.depth, sequence, max(0, sequence - self.depth)
