from VMC import VMC
import matplotlib.pyplot as plt
import numpy as np


alpha = 0.1
N_particles = [1, 10, 100]
beta = 1
a = 0  # No interaction
N_cycles = 1000
StepSize = 1
MaxVariations = 10
Dimension = [1, 2, 3]

""" Task 1b
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

        
    plt.title(f"Energy for N={N}, dt = {StepSize}")
    plt.xlabel("alpha")
    plt.ylabel("Energy")
    plt.legend()
    plt.savefig(f"fig/Energy_N{N}.png")
    plt.show()
#"""

""" Task 1c
StepSizes = [0.1, 0.5, 1, 2, 5]
N = 10

for StepSize in StepSizes:
    for dim in Dimension:
        print(f"N_particles = {N}, dim = {dim}")
        VMC_obj = VMC(N, alpha, beta, a)
        (
            alpha_values,
            analytic_local_energies,
            Energies,
            Variances,
            time_stats,
        ) = VMC_obj.MC_Sampling(N_cycles, StepSize, MaxVariations, dim, KE=True, Importance_sampling = True)
        if dim == 1:
            plt.plot(
                alpha_values,
                analytic_local_energies,
                label=f"Exact EL N = {N}",
                linestyle="-.",
            )
        plt.plot(alpha_values, Energies["local_energy"], label=f"VMC EL {dim}D: time = {np.mean(time_stats['avarage_time_EL']):.2e}")
        plt.plot(alpha_values, Energies["kinetic_energy"], label=f"VMC KE {dim}D: time = {np.mean(time_stats['avarage_time_KE']):.2e}", linestyle="--")

        
    plt.title(f"Energy for N={N}, dt = {StepSize}")
    plt.xlabel("alpha")
    plt.ylabel("Energy")
    plt.legend()
    plt.savefig(f"fig/IS_Energy_dt{StepSize}.png")
    plt.show()
#"""

#""" Task 1d

N = 10
StepSize = 0.5
alpha = 1

VMC_obj = VMC(N, alpha, beta, a)
(
    alpha_values,
    analytic_local_energies,
    Energies,
    Variances,
    time_stats,
) = VMC_obj.MC_Sampling(N_cycles, StepSize, MaxVariations, 3, KE=True,
                        Importance_sampling = True, lr = 0.00001)

variation = np.arange(0, MaxVariations)

plt.plot(variation, alpha_values)
plt.title(f"Alpha variations using GD for N = {N}, dt = {StepSize}, dim = 3, lr = 0.01")
plt.xlabel("Variation cycle")
plt.ylabel("alpha")
plt.savefig(f"fig/GD_alpha_variation.png")
plt.show()


#"""
