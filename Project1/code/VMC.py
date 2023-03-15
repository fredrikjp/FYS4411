import numpy as np
from math import exp, sqrt
from random import random, seed

"""
r = np.array([[0,0,1],[0,0,2]])
print(r.shape[1])
"""


class VMC:
    def __init__(self, alpha, N_particles, beta, a):
        self.alpha = alpha
        self.N_particles = N_particles
        self.beta = beta
        self.a = a
        self.m = 1
        self.h_bar = 1

    def wavefunction(self, r):
        g = 1
        for i in range(self.N_particles):
            if r.shape[1] == 3:
                r[i, 2] *= np.sqrt(self.beta)
            g *= np.exp(-self.alpha * np.linalg.norm(r[i, :]) ** 2)
        return g

    def V_ext(self, r, om_z, om_ho):
        V = 0
        for i in range(self.N_particles):
            V += (
                1
                / 2
                * self.m
                * (
                    om_ho**2 * (r[i, 0] ** 2 + r[i, 1] ** 2)
                    + om_z**2 * r[i, 2] ** 2
                )
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
                        / (r_kl**3 / a - r_kl**2)
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

    def MC_Sampling(self, N_cycles, StepSize, MaxVariations, Dimension):
        alpha = self.alpha
        alpha_values = np.zeros(MaxVariations)
        Energies = np.zeros(MaxVariations)
        Variances = np.zeros(MaxVariations)

        PositionOld = np.zeros((self.N_particles, Dimension), np.double)
        PositionNew = np.zeros((self.N_particles, Dimension), np.double)

        for ia in range(1, MaxVariations):
            alpha_values[ia] = alpha
            energy = energy2 = 0.0
            DeltaE = 0.0
            # Initial position
            for i in range(self.N_particles):
                for j in range(Dimension):
                    PositionOld[i, j] = StepSize * (random() - 0.5)
            wfold = self.wavefunction(PositionOld, alpha)
            for cycle in range(N_cycles):
                for i in range(self.N_particles):
                    for j in range(Dimension):
                        # New position
                        PositionNew[i, j] = PositionOld[i, j] + StepSize * (
                            random() - 0.5
                        )
                    wfnew = self.wavefunction(PositionNew, alpha)
                    # Metropolis test
                    if random() <= wfnew**2 / wfold**2:
                        for j in range(Dimension):
                            PositionOld[i, j] = PositionNew[i, j]
                        wfold = wfnew
                EL = self.local_energy(PositionOld)
                energy += EL
                energy2 += EL**2
            # We calculate mean, variance and error ...
            energy /= NumberMCcycles
            energy2 /= NumberMCcycles
            variance = energy2 - energy**2
            error = sqrt(variance / NumberMCcycles)
            Energies[ia] = energy
            Variances[ia] = variance
            alpha += 0.025
        return alpha_values, Energies, Variances
