from VMC import VMC
import matplotlib.pyplot as plt

alpha = 0.1
N_particles = [1, 10, 100, 500]
beta = 1
a = 0  # No interaction
N_cycles = 1
StepSize = 1
MaxVariations = 10
Dimension = [1, 2, 3]

for dim in Dimension:
    for N in N_particles:
        print(f"N_particles = {N}, dim = {dim}")
        VMC_obj = VMC(N, alpha, beta, a)
        (
            alpha_values,
            analytic_local_energies,
            Energies,
            Variances,
            time_stats,
        ) = VMC_obj.MC_Sampling(N_cycles, StepSize, MaxVariations, dim, KE=True)
        plt.plot(alpha_values, Energies["local_energy"], label=f"VMC EL N = {N}")
        plt.plot(alpha_values, Energies["kinetic_energy"], label=f"VMC KE N = {N}")
        plt.plot(
            alpha_values,
            analytic_local_energies,
            label=f"Exact EL N = {N}",
            linestyle="--",
        )
    plt.title(f"Energy for {dim}D")
    plt.xlabel("alpha")
    plt.ylabel("Energy")
    plt.legend()
    plt.savefig(f"fig/Energy_{dim}D.png")
    plt.show()

"""
VMC_obj = VMC(N_particles, alpha, beta, a)
(
    alpha_values,
    analytic_local_energies,
    Energies,
    Variances,
    time_stats,
) = VMC_obj.MC_Sampling(N_cycles, StepSize, MaxVariations, Dimension, KE=True)
print("alpha_values: ", alpha_values)
print("analytic_local_energies: ", analytic_local_energies)
print("Energies: ", Energies)
print("Variances: ", Variances)
print("time_stats: ", time_stats)
"""

