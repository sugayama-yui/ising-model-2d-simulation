"""2次元強磁性イジング模型の古典モンテカルロシミュレーション

このスクリプトは、2次元正方格子イジング模型でメトロポリス法を用いて
転移温度を調べるための実装です。

主要な測定量:
- エネルギー密度 E/N
- 磁化密度 |M|/N
- 磁化率 chi
- ビンダー累積量 U4

1 モンテカルロステップ (1 MCS) は、L^2 回のスピン更新試行を意味します。
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


@dataclass
class IsingState:
    L: int
    spins: np.ndarray


def initialize_spins(L: int, seed: int | None = None) -> IsingState:
    """すべてのスピンを +1 で初期化する (Cold start)。
    ランダムスタートよりも Tc 付近で平衡状態に達しやすい。
    """
    spins = np.ones((L, L), dtype=np.int8)
    return IsingState(L=L, spins=spins)


def periodic_index(i: int, size: int) -> int:
    return i % size


def delta_energy(spins: np.ndarray, i: int, j: int) -> int:
    """スピン (i, j) を反転したときのエネルギー差を計算する。"""
    L = spins.shape[0]
    s = spins[i, j]
    neighbor_sum = (
        spins[periodic_index(i + 1, L), j]
        + spins[periodic_index(i - 1, L), j]
        + spins[i, periodic_index(j + 1, L)]
        + spins[i, periodic_index(j - 1, L)]
    )
    return 2 * int(s) * int(neighbor_sum)


def total_energy(state: IsingState) -> float:
    """全格子のエネルギーを計算する。"""
    spins = state.spins
    L = state.L
    energy = 0
    for i in range(L):
        for j in range(L):
            energy -= int(spins[i, j]) * (
                int(spins[periodic_index(i + 1, L), j])
                + int(spins[i, periodic_index(j + 1, L)])
            )
    return float(energy)


def magnetization(state: IsingState) -> float:
    return float(np.sum(state.spins))


def monte_carlo_step(state: IsingState, beta: float, rng: random.Random) -> None:
    """1 MCS: L^2 回のスピン更新試行を行う。"""
    L = state.L
    for _ in range(L * L):
        i = rng.randrange(L)
        j = rng.randrange(L)
        dE = delta_energy(state.spins, i, j)
        if dE <= 0 or rng.random() < math.exp(-beta * dE):
            state.spins[i, j] = -state.spins[i, j]


def measure_observables(state: IsingState) -> Tuple[float, float, float, float]:
    """現在の状態からエネルギー、磁化、M^2、M^4 を計算する。"""
    E = total_energy(state)
    M = magnetization(state)
    return E, M, M * M, M * M * M * M


def run_simulation(
    L: int,
    temperatures: List[float],
    n_thermal: int,
    n_measure: int,
    measure_interval: int,
    seed: int | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """複数温度でシミュレーションを行い、平均値と Binder cumulant を求める。"""
    rng = random.Random(seed)
    temps = np.array(temperatures, dtype=float)
    E_avg = np.zeros_like(temps)
    M_avg = np.zeros_like(temps)
    chi = np.zeros_like(temps)
    binder = np.zeros_like(temps)

    for idx, T in enumerate(temps):
        beta = 1.0 / T
        state = initialize_spins(L, seed=rng.randint(0, 2**31 - 1))

        for _ in range(n_thermal):
            monte_carlo_step(state, beta, rng)

        E_samples = []
        M_samples = []
        M2_samples = []
        M4_samples = []
        for sample in range(n_measure):
            for _ in range(measure_interval):
                monte_carlo_step(state, beta, rng)
            E, M, M2, M4 = measure_observables(state)
            E_samples.append(E)
            M_samples.append(abs(M))
            M2_samples.append(M2)
            M4_samples.append(M4)

        E_avg[idx] = np.mean(E_samples) / (L * L)
        M_avg[idx] = np.mean(M_samples) / (L * L)
        chi[idx] = beta * (np.mean(M2_samples) - np.mean(M_samples) ** 2) / (L * L)
        binder[idx] = 1.0 - np.mean(M4_samples) / (3.0 * np.mean(M2_samples) ** 2)

    return temps, E_avg, M_avg, chi, binder


def estimate_tc_from_susceptibility(temperatures: np.ndarray, susceptibility: np.ndarray) -> float:
    """磁化率ピーク位置を転移温度の近似として返す。"""
    index = int(np.argmax(susceptibility))
    return float(temperatures[index])


def estimate_tc_from_binder_crossing(
    temperatures: np.ndarray,
    binders: List[np.ndarray],
) -> float:
    """複数サイズの Binder 累積量の交差点付近を簡単に推定する。"""
    # ここでは最も近い交差点を単純に取り、近似値を返す。
    if len(binders) < 2:
        raise ValueError("Binder crossing requires at least 2 sizes.")
    tcs = []
    base = binders[0]
    for other in binders[1:]:
        diff = base - other
        sign = np.sign(diff)
        # 交差点の近傍は符号の変化点
        cross_indices = np.where(np.diff(sign) != 0)[0]
        if len(cross_indices) > 0:
            idx = cross_indices[0]
            tcs.append((temperatures[idx] + temperatures[idx + 1]) / 2.0)
    if not tcs:
        return float(temperatures[np.argmin(np.abs(base - binders[1]))])
    return float(np.mean(tcs))


def scaling_test(L_values: List[int], seed: int | None = None) -> List[Tuple[int, float]]:
    """1 MCS の実行時間を計測し、O(L^2) であることを確認する。"""
    rng = random.Random(seed)
    timings = []
    for L in L_values:
        state = initialize_spins(L, seed=rng.randint(0, 2**31 - 1))
        beta = 1.0 / 2.269185
        start = time.perf_counter()
        monte_carlo_step(state, beta, rng)
        end = time.perf_counter()
        timings.append((L, end - start))
    return timings


def main() -> None:
    L_values = [16, 24, 32]
    temperatures = list(np.linspace(2.0, 2.6, 17))
    n_thermal = 200
    n_measure = 120
    measure_interval = 1
    seed = 42

    print("2次元イジング模型の古典モンテカルロ解析")
    print("L, Tc(susceptibility), Tc(binder)")
    binder_list = []
    for L in L_values:
        temps, E_avg, M_avg, chi, binder = run_simulation(
            L,
            temperatures,
            n_thermal,
            n_measure,
            measure_interval,
            seed=seed,
        )
        tc_sus = estimate_tc_from_susceptibility(temps, chi)
        binder_list.append(binder)
        print(f"L={L}: Tc_sus={tc_sus:.4f}")

    try:
        tc_binder = estimate_tc_from_binder_crossing(temps, binder_list)
        print(f"Binder crossing estimate Tc ≈ {tc_binder:.4f}")
    except ValueError:
        print("Binder crossing could not be estimated.")

    print("\nO(L^2) scaling test")
    for L, dt in scaling_test([8, 16, 24, 32], seed=seed):
        print(f"L={L}: time={dt:.6f} s")


if __name__ == "__main__":
    main()
