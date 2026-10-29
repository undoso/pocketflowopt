"""Active-learning outer loop: GP surrogate + expected improvement (WP4).

Implements the RBF-kernel GP (Eq. 16) and EI acquisition (Eq. 17) in numpy,
plus the closed design-make-test cycle (Algorithm 2). Generator, scorer and
assay backend are injected callables -- FEP+ orchestration and lab logistics
live behind those interfaces.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.stats import norm


class RBFKernelGP:
    """Zero-mean GP with RBF kernel and closed-form posterior (Eq. 16)."""

    def __init__(self, length_scale: float = 1.0, sigma_f: float = 1.0,
                 noise: float = 1e-4):
        self.ls, self.sigma_f, self.noise = length_scale, sigma_f, noise
        self._x: np.ndarray | None = None
        self._y: np.ndarray | None = None
        self._l: np.ndarray | None = None  # cholesky factor

    def _k(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        d2 = ((a[:, None, :] - b[None, :, :]) ** 2).sum(-1)
        return self.sigma_f**2 * np.exp(-0.5 * d2 / self.ls**2)

    def fit(self, x: np.ndarray, y: np.ndarray) -> "RBFKernelGP":
        self._x, self._y = np.asarray(x, float), np.asarray(y, float)
        k = self._k(self._x, self._x) + self.noise * np.eye(len(self._x))
        self._l = np.linalg.cholesky(k)
        return self

    def predict(self, x: np.ndarray):
        x = np.asarray(x, float)
        ks = self._k(self._x, x)
        alpha = np.linalg.solve(self._l.T, np.linalg.solve(self._l, self._y))
        mu = ks.T @ alpha
        v = np.linalg.solve(self._l, ks)
        var = self._k(x, x).diagonal() - (v**2).sum(0)
        return mu, np.maximum(var, 1e-12)


def expected_improvement(mu: np.ndarray, sigma: np.ndarray,
                         y_best: float) -> np.ndarray:
    """EI(q) = (mu - y+) Phi(z) + sigma phi(z),  z = (mu - y+)/sigma (Eq. 17)."""
    sigma = np.maximum(sigma, 1e-12)
    z = (mu - y_best) / sigma
    return (mu - y_best) * norm.cdf(z) + sigma * norm.pdf(z)


@dataclass
class ActiveLearningLoop:
    """Algorithm 2: generate -> filter -> score -> EI rank -> assay -> update."""

    generate: object          # callable(round_idx, n) -> list of candidates
    describe: object          # callable(candidate) -> np.ndarray descriptor q
    score: object             # callable(candidate) -> float fast score
    assay: object             # callable(candidate) -> float measured activity
    n_generate: int = 256
    n_assay: int = 8
    ei_tolerance: float = 0.05
    data: list = field(default_factory=list)   # (q, y) assay records

    def run(self, rounds: int = 4) -> list:
        stall = 0
        for r in range(rounds):
            cands = self.generate(r, self.n_generate)
            scored = sorted(cands, key=self.score, reverse=True)
            shortlist = scored[: max(4 * self.n_assay, 16)]
            if len(self.data) >= 4:
                q = np.stack([self.describe(c) for c in shortlist])
                gp = RBFKernelGP().fit(
                    np.stack([d[0] for d in self.data]),
                    np.array([d[1] for d in self.data]))
                mu, var = gp.predict(q)
                ei = expected_improvement(mu, np.sqrt(var),
                                          max(d[1] for d in self.data))
                order = np.argsort(-ei)
                picked = [shortlist[i] for i in order[: self.n_assay]]
                stall = stall + 1 if ei.max() < self.ei_tolerance else 0
            else:
                picked = shortlist[: self.n_assay]
            for c in picked:
                self.data.append((self.describe(c), float(self.assay(c))))
            if stall >= 2:
                break
        return sorted(self.data, key=lambda d: -d[1])
