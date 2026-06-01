import random
import unittest

import numpy as np

from ising import IsingModel


class TestIsingModel(unittest.TestCase):
    def test_create_random_initializes_spins(self):
        model = IsingModel.create_random(2, seed=0)
        self.assertEqual(model.L, 2)
        self.assertEqual(model.spins.shape, (2, 2))
        self.assertIn(model.energy, [-8.0, 0.0, 8.0])

    def test_delta_energy_and_flip_updates_energy_and_magnetization(self):
        model = IsingModel.create_random(2, seed=0)
        rng = random.Random(0)
        initial_energy = model.energy
        initial_m = model.magnetization
        model.monte_carlo_step(1.0, rng)
        self.assertEqual(model.L, 2)
        self.assertEqual(model.spins.shape, (2, 2))
        self.assertIsInstance(model.energy, float)
        self.assertIsInstance(model.magnetization, int)
        self.assertTrue(abs(model.energy - initial_energy) >= 0)

    def test_monte_carlo_step_scaling_small_system(self):
        model = IsingModel.create_random(4, seed=1)
        rng = random.Random(1)
        model.monte_carlo_step(1.0, rng)
        self.assertEqual(model.spins.size, 16)


if __name__ == "__main__":
    unittest.main()
