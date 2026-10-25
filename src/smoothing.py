"""One Euro filter: adaptive smoothing for noisy, real-time signals.

When the hand is still, the cutoff frequency is low (strong smoothing, no
jitter). When the hand moves fast, the cutoff rises (little lag).
Paper: Casiez, Roussel, Vogel - "1 Euro Filter" (CHI 2012).
"""

from __future__ import annotations

import math


def _alpha(cutoff: float, dt: float) -> float:
    tau = 1.0 / (2.0 * math.pi * cutoff)
    return 1.0 / (1.0 + tau / dt)


class OneEuroFilter:
    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.0, d_cutoff: float = 1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.reset()

    def reset(self, value: float | None = None, t: float | None = None) -> None:
        self._x = value
        self._dx = 0.0
        self._t = t

    def __call__(self, x: float, t: float) -> float:
        if self._x is None or self._t is None:
            self._x, self._t, self._dx = x, t, 0.0
            return x
        dt = t - self._t
        if dt <= 0:
            return self._x
        self._t = t
        dx = (x - self._x) / dt
        a_d = _alpha(self.d_cutoff, dt)
        self._dx = a_d * dx + (1 - a_d) * self._dx
        cutoff = self.min_cutoff + self.beta * abs(self._dx)
        a = _alpha(cutoff, dt)
        self._x = a * x + (1 - a) * self._x
        return self._x


class OneEuroFilter2D:
    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.0, d_cutoff: float = 1.0):
        self.fx = OneEuroFilter(min_cutoff, beta, d_cutoff)
        self.fy = OneEuroFilter(min_cutoff, beta, d_cutoff)

    def reset(self, point: tuple[float, float] | None = None, t: float | None = None) -> None:
        if point is None:
            self.fx.reset()
            self.fy.reset()
        else:
            self.fx.reset(point[0], t)
            self.fy.reset(point[1], t)

    def __call__(self, x: float, y: float, t: float) -> tuple[float, float]:
        return self.fx(x, t), self.fy(y, t)
