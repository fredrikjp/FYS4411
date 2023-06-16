from VMC import VMC
import matplotlib.pyplot as plt
import numpy as np


# Task 1b
alpha = 0.1
N_particles = [1, 10, 100]
beta = 1
a = 0  # No interaction
N_cycles = 1
StepSize = 1
MaxVariations = 10
Dimension = [1, 2, 3]

for N in N_particles:
    for dim in Dimension:
        print(f"N_particles = {N}, dim = {dim}")
        VMC_obj = VMC(N, alpha, beta, a)
        (
            alpha_values,
            analytic_local_energies,
            Energies,
            Variances,
            time_stats,
        ) = VMC_obj.MC_Sampling(N_cycles, StepSize, MaxVariations, dim, KE=True)
        if dim == 1:
            plt.plot(
                alpha_values,
                analytic_local_energies,
                label=f"Exact EL N = {N}",
                linestyle="-.",
            )
        plt.plot(alpha_values, Energies["local_energy"], label=f"VMC EL {dim}D: time = {np.mean(time_stats['avarage_time_EL']):.2e}")
        plt.plot(alpha_values, Energies["kinetic_energy"], label=f"VMC KE {dim}D: time = {np.mean(time_stats['avarage_time_KE']):.2e}", linestyle="--")

        
    plt.title(f"Energy for N={N}")
    plt.xlabel("alpha")
    plt.ylabel("Energy")
    plt.legend()
    plt.savefig(f"fig/Energy_N{N}.png")
    plt.show()


