import numpy as np
import matplotlib.pyplot as plt
from ising_simulation import run_simulation

def generate_plots():
    L_values = [8, 16, 32]
    temperatures = np.linspace(2.0, 2.6, 20)
    n_thermal = 2000
    n_measure = 1000
    measure_interval = 1
    seed = 42

    results = {}
    for L in L_values:
        print(f"Running simulation for L={L}...")
        temps, E_avg, M_avg, chi, binder = run_simulation(
            L, temperatures, n_thermal, n_measure, measure_interval, seed=seed
        )
        results[L] = {
            'E': E_avg,
            'M': M_avg,
            'chi': chi,
            'binder': binder
        }

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
    print("Plots saved as ising_results.png")

if __name__ == "__main__":
    generate_plots()
