"""
    File name: calculate_kappa.py
    Author: Matthias Blaschke
    Python Version: 3.9
"""

import numpy as np
from turbomoleOutputProcessing import turbomoleOutputProcessing as top
import matplotlib
#matplotlib.use('Agg') #for cluster usage!
import matplotlib.pyplot as plt


def calculate_kappa(tau_ph, E, T):
    """Calculates the thermal conductance according to "Tuning the thermal conductance of molecular junctions with interference effects" (https://doi.org/10.1103/PhysRevB.96.245419)

    Args:
     tau_ph (array): Phonon transmission
     E (array): Phonon transmission
     T (float): Temperature in Kelvin

    Returns:
     kappa (np.array): Thermal conductance in Hartree/sK
    """
    #Planck sontant in hartree s
    h = 1.51983003855935e-16
    #Boltzmann constant in hartree/K
    k_B = 3.167*10**(-6)

    prefactor = 1./(h*k_B*T**2)
    beta = 1/(k_B*T)
    exp_ = np.exp(E*beta)
    integrand = E ** 2 * thau_ph * exp_ / ((exp_ - 1) ** 2)

    integral = np.trapz(integrand, E)

    return integral * prefactor

if __name__ == '__main__':
    #path to data
    path = "C:/Users/Matthias Blaschke/OneDrive - Universität Augsburg/Promotion/phonon_transport/test/"
    #filename of datafile
    filename = "transp_benz_dia_au56_tips.dat"
    pOPE3_hh_small_tips = top.read_plot_data(path + filename)[0]
    #Energy must be in Hartree! -> Convert otherwise
    E = np.asarray(pOPE3_hh_small_tips[0,:], dtype=np.float64)
    thau_ph = np.asarray(pOPE3_hh_small_tips[1, :], dtype=np.float64)

    #Temperature range in Kelvin with given Resolution res
    res = 1000
    T = np.linspace(1, 400, res, dtype=np.float64)

    kappa = list()
    for i in range(0, len(T)):
        kappa.append(calculate_kappa(thau_ph, E, T[i]))
    kappa = np.asarray(kappa)

    har2pJ = 4.35974e-6
    kappa = kappa*har2pJ

    #save kappa data
    top.write_plot_data(path + "kappa.dat", (T, kappa), "T [K], kappa [pW/K]")

    #now plot everything
    plt.plot(T, kappa)
    plt.xlabel('Temperature ($K$)', fontsize=12)
    plt.ylabel(r'Thermal Conductance $\mathrm{pw/K}$', fontsize=12)
    plt.show()
