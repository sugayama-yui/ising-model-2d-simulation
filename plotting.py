import numpy as np
import matplotlib.pyplot as plt
import random
import math
from ising_simulation import initialize_spins, monte_carlo_step, measure_observables

def run_simulation_with_progress(L, temperatures, n_thermal, n_measure, measure_interval, seed):
    rng = random.Random(seed)
    temps = np.array(temperatures, dtype=float)
    E_avg, M_avg, chi, binder = [], [], [], []

    for T in temps:
        print(f"  Temperature T={T:.3f} ...", end="", flush=True)
        beta = 1.0 / T
        state = initialize_spins(L, seed=rng.randint(0, 2**31 - 1))
        # 緩和
        for _ in range(n_thermal):
            monte_carlo_step(state, beta, rng)
        
        # 測定
        Es, Ms, M2s, M4s = [], [], [], []
        for _ in range(n_measure):
            for __ in range(measure_interval):
                monte_carlo_step(state, beta, rng)
            E, M, M2, M4 = measure_observables(state)
            Es.append(E); Ms.append(abs(M)); M2s.append(M2); M4s.append(M4)
        
        E_avg.append(np.mean(Es)/(L*L))
        M_avg.append(np.mean(Ms)/(L*L))
        chi.append(beta*(np.mean(M2s)-np.mean(Ms)**2)/(L*L))
        binder.append(1.0 - np.mean(M4s)/(3.0*np.mean(M2s)**2))
        print(" Done")
    return temps, np.array(E_avg), np.array(M_avg), np.array(chi), np.array(binder)

def generate_plots():
    L_values = [8, 16, 24] 
    temperatures = np.linspace(2.1, 2.5, 11) 
    n_thermal = 3000
    n_measure = 2000
    measure_interval = 2
    seed = 42

    results = {}
    for L in L_values:
        print(f"--- Running simulation for L={L} ---")
        temps, E_avg, M_avg, chi, binder = run_simulation_with_progress(
            L, temperatures, n_thermal, n_measure, measure_interval, seed=seed
        )
        results[L] = {'E': E_avg, 'M': M_avg, 'chi': chi, 'binder': binder}

    # Plotting
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    # Energy
    for L in L_values:
        axs[0, 0].plot(temperatures, results[L]['E'], 'o-', label=f'L={L}')
    axs[0, 0].set_title('Energy Density vs Temperature')
    axs[0, 0].set_xlabel('Temperature T')
    axs[0, 0].set_ylabel('Energy <E>/N')
    axs[0, 0].legend()
    axs[0, 0].grid(True)

    # Magnetization
    for L in L_values:
        axs[0, 1].plot(temperatures, results[L]['M'], 'o-', label=f'L={L}')
    axs[0, 1].set_title('Magnetization vs Temperature')
    axs[0, 1].set_xlabel('Temperature T')
    axs[0, 1].set_ylabel('Magnetization <|M|>/N')
    axs[0, 1].legend()
    axs[0, 1].grid(True)

    # Susceptibility
    for L in L_values:
        axs[1, 0].plot(temperatures, results[L]['chi'], 'o-', label=f'L={L}')
    axs[1, 0].set_title('Susceptibility vs Temperature')
    axs[1, 0].set_xlabel('Temperature T')
    axs[1, 0].set_ylabel('Susceptibility chi')
    axs[1, 0].legend()
    axs[1, 0].grid(True)

    # Binder Cumulant
    for L in L_values:
        axs[1, 1].plot(temperatures, results[L]['binder'], 'o-', label=f'L={L}')
    axs[1, 1].axvline(x=2.269, color='k', linestyle='--', label='Exact Tc')
    axs[1, 1].set_title('Binder Cumulant vs Temperature')
    axs[1, 1].set_xlabel('Temperature T')
    axs[1, 1].set_ylabel('U4')
    axs[1, 1].legend()
    axs[1, 1].grid(True)

    plt.tight_layout()
    plt.savefig('ising_results.png')
    print(f"\nPlots saved as ising_results.png")

if __name__ == "__main__":
    generate_plots()
