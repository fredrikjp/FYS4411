import numpy as np
from random import random, seed
from autograd import grad
import time

"""
r = np.array([[0,0,1],[0,0,2]])
print(r.shape[1])
"""


class VMC:
    def __init__(self, N_particles, alpha, beta, a):
        self.alpha = alpha
        self.N_particles = N_particles
        self.beta = beta
        self.a = a
        self.m = 1
        self.h_bar = 1

        # External potential parameters
        a_ho = 1  # Characteristic dimension of the trap
        self.om_ho = 1  # Frequency of the harmonic oscillator in xy_plane
        self.om_z = 1  # Frequency of the harmonic oscillator in z-direction

    def wavefunction(self, r):
        g = 1
        for i in range(self.N_particles):
            if r.shape[1] == 3:
                r[i, 2] *= np.sqrt(self.beta)
            g *= np.exp(-self.alpha * np.linalg.norm(r[i, :]) ** 2)
        return g

    def V_ext(self, r):
        V = 0
        if len(r[0]) == 3:
            r[:, 2] *= self.om_z / self.om_ho
        for i in range(self.N_particles):
            V += (
                1
                / 2
                * self.m
                * self.om_ho**2
                * (np.linalg.norm(r[i, :]) ** 2)
            )
        return V

    def V_int(self, r):
        V = 0
        for i in range(self.N_particles):
            for j in range(i + 1, self.N_particles):
                r_ij = np.linalg.norm(r[i, :] - r[j, :])
                if r_ij < self.a:
                    V += np.inf
        return V

    def local_energy(self, r):
        alpha = self.alpha
        a = self.a
        # sum of laplacian_k acting on Psi_T and divided by Psi_T
        laplacian_sum = 0
        for k in range(self.N_particles):
            laplacian_term = 4 * alpha**2 * r[k, :] @ r[k, :] - 6 * alpha
            for l in range(self.N_particles):
                if l != k:
                    r_kl = np.linalg.norm(r[k, :] - r[l, :])
                    laplacian_term += (
                        -4
                        * alpha
                        * r[k, :]
                        @ (r[k, :] - r[l, :])
                        * a
                        / (r_kl**3 - a * r_kl**2)
                        + (a**2 - 2 * a * r_kl) / (r_kl**2 - a * r_kl) ** 2
                        + 2 * a / (r_kl**3 - a * r_kl**2)
                    )
                    for j in range(self.N_particles):
                        if j != k:
                            r_kj = np.linalg.norm(r[k, :] - r[j, :])
                            laplacian_term += (
                                a**2
                                * (r[k, :] - r[j, :])
                                @ (r[k, :] - r[l, :])
                                / (
                                    (r_kl**2 - a * r_kl)
                                    * (r_kj**2 - a * r_kj)
                                    * (r_kl * r_kj)
                                )
                            )
            laplacian_sum += laplacian_term
        # Local energy
        EL = (
            -self.h_bar**2 / (2 * self.m) * laplacian_sum
            + self.V_ext(r)
            + self.V_int(r)
        )
        return EL

    def MC_Sampling(self, N_cycles, StepSize, MaxVariations, Dimension, KE = False):
        alpha = self.alpha
        alpha_values = np.zeros(MaxVariations)
        local_energies = np.zeros(MaxVariations)
        Variances = np.zeros(MaxVariations)
        Kinetic_energies = np.zeros(MaxVariations)
        avg_time_EL = np.zeros(MaxVariations)
        avg_time_KE = np.zeros(MaxVariations)

        PositionOld = np.zeros((self.N_particles, Dimension), np.double)
        PositionNew = np.zeros((self.N_particles, Dimension), np.double)

        for ia in range(MaxVariations):
            alpha_values[ia] = alpha
            energy = 0.0
            energy2 = 0.0
            kinetic_energy = 0.0
            # Initial position
            for i in range(self.N_particles):
                for j in range(Dimension):
                    PositionOld[i, j] = StepSize * (random() - 0.5)
            wfold = self.wavefunction(PositionOld)

            time_EL = 0
            time_KE = 0
            for cycle in range(N_cycles):
                for i in range(self.N_particles):
                    for j in range(Dimension):
                        # New position
                        PositionNew[i, j] = PositionOld[i, j] + StepSize * (
                            random() - 0.5
                        )
                    wfnew = self.wavefunction(PositionNew)
                    # Metropolis test
                    if random() <= wfnew**2 / wfold**2:
                        for j in range(Dimension):
                            PositionOld[i, j] = PositionNew[i, j]
                        wfold = wfnew
                start_time_EL = time.time()
                EL = self.local_energy(PositionOld)
                end_time_EL = time.time()
                time_EL += end_time_EL - start_time_EL 

                energy += EL
                energy2 += EL**2
                
                if KE:
                    start_time_KE = time.time()
                    kinetic_energy += self.kinetic_energy(PositionOld)
                    end_time_KE = time.time()
                    time_KE += end_time_KE - start_time_KE


            # Calculate mean, variance and error
            energy /= N_cycles
            energy2 /= N_cycles
            variance = energy2 - energy**2
            error = np.sqrt(variance / N_cycles)
            local_energies[ia] = energy
            Variances[ia] = variance
            kinetic_energy /= N_cycles
            Kinetic_energies[ia] = kinetic_energy

            avg_time_EL[ia] = time_EL / N_cycles
            avg_time_KE[ia] = time_KE / N_cycles
            

            alpha += 0.01
            self.alpha = alpha

        Energies = {"local_energy": local_energies, "kinetic_energy": Kinetic_energies}
        time_stats = {"avarage_time_EL": avg_time_EL, "avarage_time_KE": avg_time_KE}
        return alpha_values, Energies, Variances, time_stats

    def kinetic_energy(self, r, h=0.001):
        # Calculate the laplace operator acting on the wavefunction numerically 
        # using finite difference
        laplacian = 0
        for i in range(self.N_particles):
            for dim in range(len(r[0])):
                r1 = r.copy() 
                r1[i, dim] -= h
                r2 = r.copy() 
                r2[i, dim] += h
                laplacian += (self.wavefunction(r1) - 2 * self.wavefunction(r) + self.wavefunction(r2)) / h**2
        # Estimation of total kinetic energy of the wavefunction's boson(s) at r 
        T = -1/2 * laplacian / self.wavefunction(r)
        return T


if __name__ == "__main__":
    alpha = 0.1
    N_particles = 1 
    beta = 1
    a = 0  # No interaction
    N_cycles = 10000
    StepSize = 1
    MaxVariations = 100
    Dimension = 3
    VMC = VMC(N_particles, alpha, beta, a)
    alpha_values, Energies, Variances, time_stats = VMC.MC_Sampling(
        N_cycles, StepSize, MaxVariations, Dimension, KE = True
    )
    print("alpha_values: ", alpha_values)
    print("Energies: ", Energies)
    print("Variances: ", Variances)
    print("time_stats: ", time_stats)
