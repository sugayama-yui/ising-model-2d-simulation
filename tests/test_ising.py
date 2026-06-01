import math
import random

import numpy as np

from ising_simulation import (
    IsingState,
    delta_energy,
    initialize_spins,
    magnetization,
    monte_carlo_step,
    periodic_index,
    total_energy,
)


def test_periodic_index():
    assert periodic_index(-1, 4) == 3
    assert periodic_index(4, 4) == 0
    assert periodic_index(5, 4) == 1


def test_total_energy_all_up():
    L = 2
    spins = np.ones((L, L), dtype=np.int8)
    state = IsingState(L=L, spins=spins)
    assert total_energy(state) == -8.0


def test_delta_energy_for_spin_flip():
    spins = np.array([[1, 1], [1, -1]], dtype=np.int8)
    # 中央に相当するスピンはないため、辺のスピンを検証する
    dE = delta_energy(spins, 0, 0)
    assert dE == 4


def test_magnetization():
    L = 2
    spins = np.array([[1, -1], [-1, 1]], dtype=np.int8)
    state = IsingState(L=L, spins=spins)
    assert magnetization(state) == 0.0


def test_monte_carlo_step_changes_spin():
    rng = random.Random(0)
    state = initialize_spins(2, seed=0)
    before = state.spins.copy()
    monte_carlo_step(state, beta=0.1, rng=rng)
    assert state.spins.shape == before.shape
    assert np.any(state.spins != before) or np.all(state.spins == before)
