import numpy as np
from random import random, seed
import time
from numba import int32, float64
from numba.experimental import jitclass

spec = [
    ("alpha", float64),
    ("N_particles", int32),
    ("beta", float64),
    ("a", float64),
    ("m", float64),
    ("h_bar", float64),
    ("a_ho", float64),
    ("om_ho", float64),
    ("om_z", float64),
]

#@jitclass(spec)
class VMC:
    def __init__(self, N_particles, alpha, beta, a):
        self.alpha = alpha
        self.N_particles = N_particles
        self.beta = beta
        self.a = a
        self.m = 1
        self.h_bar = 1

        # External potential parameters
        self.a_ho = 1  # Characteristic dimension of the trap
        self.om_ho = 1  # Frequency of the harmonic oscillator in xy_plane
        self.om_z = 1  # Frequency of the harmonic oscillator in z-direction

    def wavefunction(self, r):
        g = 1.0
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
        """
        alpha = self.alpha
        a = self.a
        # sum of laplacian_k acting on Psi_T and divided by Psi_T
        laplacian_sum2 = 0
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
                                    * (r_kl * r_kj))
                            )
            laplacian_sum2 += laplacian_term
            print(laplacian_term)
        print(laplacian_sum2)
        """

        #######################################################################
        alpha = self.alpha
        a = self.a
        N = self.N_particles
        # Cache repeated calculations
        r_diff = r[:, np.newaxis, :] - r[np.newaxis, :, :]
        r_norm = np.linalg.norm(r_diff, axis=-1)
        r_norm[r_norm < 1e-12] = 1e-12  # avoid division by zero

        # Remove diagonal elements as these are not used in the
        # calculation of the laplacian term
        r_diff = np.delete(
            r_diff.reshape(N**2, 3), np.arange(N**2, step=N + 1), axis=0
        ).reshape(N, N - 1, 3)
        r_norm = np.delete(
            r_norm.reshape(N**2), np.arange(N**2, step=N + 1), axis=0
        ).reshape(N, N - 1)

        # Calculate laplacian sum
        # breakpoint()

        laplacian_sum = (
            4 * alpha**2 * np.einsum("ij,ij->i", r, r)
            - 6 * alpha
            - 4
            * alpha
            * a
            * np.sum(r
            @ np.sum(r_diff, axis=(1)).T, axis=1)
            / np.sum(r_norm**3 - a * r_norm**2, axis=-1)
            + np.sum((a**2 - 2 * a * r_norm) / (r_norm**2 - a * r_norm)**2
                     + 2 * a / (r_norm**3 - a * r_norm**2), axis=-1)
            #+ (a**2 * np.einsum("ijk,ik->i", r_diff, np.sum(r_diff, axis=1)))

        ) 
        """
        for i in range(N-1):
            laplacian_sum += (
                a**2
                * np.einsum(
                    "jk,k->j", r_diff[:, i, :], np.sum(r_diff[:, i, :], axis=0)
                )
                / np.sum(
                    (r_norm[i, :] ** 2 - a * r_norm[i, :])
                    * np.sum(r_norm[i, :] ** 2 - a * r_norm[i, :])
                    * r_norm[i, :]
                    * np.sum(r_norm[i, :])
                )
            )
        """
        laplacian_sum = np.sum(laplacian_sum)
        #######################################################################

        # Local energy
        EL = (
            -self.h_bar**2 / (2 * self.m) * laplacian_sum
            + self.V_ext(r)
            + self.V_int(r)
        )
        return EL

    def MC_Sampling(
        self, N_cycles, StepSize, MaxVariations, Dimension, KE=False
    ):
        alpha = self.alpha
        alpha_values = np.zeros(MaxVariations)
        analytic_local_energies = np.zeros(MaxVariations)
        local_energies = np.zeros(MaxVariations)
        Variances = np.zeros(MaxVariations)
        Kinetic_energies = np.zeros(MaxVariations)
        avg_time_EL = np.zeros(MaxVariations)
        avg_time_KE = np.zeros(MaxVariations)

        PositionOld = np.zeros((self.N_particles, Dimension), np.double)
        PositionNew = np.zeros((self.N_particles, Dimension), np.double)

        for ia in range(MaxVariations):
            print(f"{ia*100/MaxVariations}%")
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

            analytic_local_energies[ia] = self.analytic_solution(PositionOld)
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

            alpha += 0.1
            self.alpha = alpha

        Energies = {
            "local_energy": local_energies,
            "kinetic_energy": Kinetic_energies,
        }
        time_stats = {
            "avarage_time_EL": avg_time_EL,
            "avarage_time_KE": avg_time_KE,
        }
        return (
            alpha_values,
            analytic_local_energies,
            Energies,
            Variances,
            time_stats,
        )

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
                laplacian += (
                    self.wavefunction(r1)
                    - 2 * self.wavefunction(r)
                    + self.wavefunction(r2)
                ) / h**2
        # Estimation of total kinetic energy of the wavefunction's boson(s) at r
        T = -1 / 2 * laplacian / self.wavefunction(r)
        return T

    def analytic_solution(self, r):
        # Analytic solution for the spherical trap without interaction
        if len(r) == 1:
            EL = (
                np.sum(r * r)
                * (
                    1 / 2 * self.m * self.om_ho**2
                    - self.h_bar**2 / (2 * self.m) * self.a_ho**4
                )
                + 3 * self.h_bar**2 / (2 * self.m) * self.a_ho**2
            )
        elif len(r) > 1:
            EL = (
                -2
                * self.h_bar**2
                * 1 / 4
                * self.a_ho**4 / self.m
                * np.sum(r * r)
                + 3 * self.h_bar**2 / (2*self.m) * self.a_ho**2 * self.N_particles
                + np.sum(self.V_ext(r))
            )
        else:
            raise ValueError("Positions not initialized")
        return EL


if __name__ == "__main__":
    alpha = 0.1
    N_particles = 10
    beta = 1
    a = 0  # No interaction
    N_cycles = 100
    StepSize = 1
    MaxVariations = 10
    Dimension = 3
    VMC_obj = VMC(N_particles, alpha, beta, a)
    (
        alpha_values,
        analytic_local_energies,
        Energies,
        Variances,
        time_stats,
    ) = VMC_obj.MC_Sampling(
        N_cycles, StepSize, MaxVariations, Dimension, KE=True
    )
    print("alpha_values: ", alpha_values)
    print("analytic_local_energies: ", analytic_local_energies)
    print("Energies: ", Energies)
    print("Variances: ", Variances)
    print("time_stats: ", time_stats)





