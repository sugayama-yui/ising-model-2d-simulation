import math
import random
import unittest

import numpy as np

from ising_simulation import initialize_spins, run_simulation, total_energy, magnetization


def compute_exact_l2_expectation(temperature: float) -> tuple[float, float]:
    """L=2 の全状態を列挙し、エネルギー密度と磁化密度の期待値を計算する。"""
    beta = 1.0 / temperature
    Z = 0.0
    E_sum = 0.0
    M_sum = 0.0
    for config in range(16):
        spins = np.empty((2, 2), dtype=np.int8)
        for bit in range(4):
            spins[bit // 2, bit % 2] = 1 if (config >> bit) & 1 else -1
        state_energy = total_energy(type("S", (), {"L": 2, "spins": spins}))
        state_mag = abs(magnetization(type("S", (), {"L": 2, "spins": spins})))
        weight = math.exp(-beta * state_energy)
        Z += weight
        E_sum += state_energy * weight
        M_sum += state_mag * weight
    E_avg = E_sum / Z / 4.0
    M_avg = M_sum / Z / 4.0
    return E_avg, M_avg


class TestL2ExactSolution(unittest.TestCase):
    def test_l2_exact_vs_simulation(self):
        temperature = 2.269185
        exact_e, exact_m = compute_exact_l2_expectation(temperature)

        temperatures = [temperature]
        n_thermal = 2000
        n_measure = 10000
        measure_interval = 1
        seed = 12345

        temps, E_avg, M_avg, chi, binder = run_simulation(
            L=2,
            temperatures=temperatures,
            n_thermal=n_thermal,
            n_measure=n_measure,
            measure_interval=measure_interval,
            seed=seed,
        )

        self.assertAlmostEqual(E_avg[0], exact_e, delta=0.1)
        self.assertAlmostEqual(M_avg[0], exact_m, delta=0.1)


if __name__ == "__main__":
    unittest.main()
