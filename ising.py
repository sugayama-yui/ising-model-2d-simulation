"""2次元イジング模型のメトロポリスシミュレーションを提供するクラス実装。"""

import math
import random
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


@dataclass
class IsingModel:
    L: int
    spins: np.ndarray
    energy: float
    magnetization: int

    @classmethod
    def create_random(cls, L: int, seed: int | None = None) -> "IsingModel":
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        spins = np.random.choice([-1, 1], size=(L, L)).astype(np.int8)
        energy = cls._compute_total_energy(spins)
        magnetization = int(np.sum(spins))
        return cls(L=L, spins=spins, energy=energy, magnetization=magnetization)

    @staticmethod
    def _periodic_index(i: int, L: int) -> int:
        return i % L

    @staticmethod
    def _compute_total_energy(spins: np.ndarray) -> float:
        L = spins.shape[0]
        energy = 0
        for i in range(L):
            for j in range(L):
                energy -= int(spins[i, j]) * (
                    int(spins[IsingModel._periodic_index(i + 1, L), j])
                    + int(spins[i, IsingModel._periodic_index(j + 1, L)])
                )
        return float(energy)

    @staticmethod
    def _local_delta_energy(spins: np.ndarray, i: int, j: int) -> int:
        L = spins.shape[0]
        s = spins[i, j]
        neighbor_sum = (
            int(spins[IsingModel._periodic_index(i + 1, L), j])
            + int(spins[IsingModel._periodic_index(i - 1, L), j])
            + int(spins[i, IsingModel._periodic_index(j + 1, L)])
            + int(spins[i, IsingModel._periodic_index(j - 1, L)])
        )
        return 2 * int(s) * neighbor_sum

    def attempt_flip(self, i: int, j: int, beta: float, rng: random.Random) -> None:
        dE = self._local_delta_energy(self.spins, i, j)
        if dE <= 0 or rng.random() < math.exp(-beta * dE):
            self.spins[i, j] = -self.spins[i, j]
            self.energy += dE
            self.magnetization += -2 * int(self.spins[i, j])

    def monte_carlo_step(self, beta: float, rng: random.Random) -> None:
        for _ in range(self.L * self.L):
            i = rng.randrange(self.L)
            j = rng.randrange(self.L)
            self.attempt_flip(i, j, beta, rng)

    def susceptibility(self, beta: float, measurements: List[int]) -> float:
        m2 = float(np.mean(np.array(measurements) ** 2))
        m1 = float(np.mean(np.abs(np.array(measurements))))
        return beta * (m2 - m1 ** 2) / (self.L * self.L)

    def binder_cumulant(self, measurements: List[int]) -> float:
        m2 = float(np.mean(np.array(measurements) ** 2))
        m4 = float(np.mean(np.array(measurements) ** 4))
        return 1.0 - m4 / (3.0 * m2 * m2)

    def copy(self) -> "IsingModel":
        return IsingModel(
            L=self.L,
            spins=self.spins.copy(),
            energy=float(self.energy),
            magnetization=int(self.magnetization),
        )


def main() -> None:
    import numpy as np

    seed = 2026
    model = IsingModel.create_random(16, seed=seed)
    rng = random.Random(seed)
    beta = 1.0 / 2.269185
    print(f"initial energy={model.energy}, magnetization={model.magnetization}")
    model.monte_carlo_step(beta, rng)
    print(f"after one MCS energy={model.energy}, magnetization={model.magnetization}")


if __name__ == "__main__":
    main()
