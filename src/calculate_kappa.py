"""
    File name: calculate_kappa.py
    Author: Matthias Blaschke
    Python Version: 3.9
"""
import codecs
import configparser
import sys

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
    integrand = E ** 2 * tau_ph * exp_ / ((exp_ - 1) ** 2)

    integral = np.trapz(integrand, E)

    return integral * prefactor

if __name__ == '__main__':
    config_path = sys.argv[1]
    cfg = configparser.ConfigParser()
    cfg.read_file(codecs.open(config_path, "r", "utf8"))

    try:
        data_path = str(cfg.get('Data Input', 'data_path'))
        transp_name = str(cfg.get('Data Input', 'transp_name'))


        # for thermal conducatance
        T_min = float(cfg.get('Calculation', 'T_min'))
        T_max = float(cfg.get('Calculation', 'T_max'))
        kappa_grid_points = int(cfg.get('Calculation', 'kappa_grid_points'))

    except configparser.NoOptionError:
        print("Missing option in config file. Check config file!")
        exit(-1)
    except ValueError:
        print("Wrong value in config file. Check config file!")
        exit(-1)

    transport = top.read_plot_data(data_path + "/" + transp_name)[0]
    #Energy must be in Hartree! -> Convert otherwise
    E = np.asarray(transport[0,:], dtype=np.float64)
    thau_ph = np.asarray(transport[1, :], dtype=np.float64)

    #Temperature range in Kelvin with given Resolution res
    res = kappa_grid_points
    T = np.linspace(T_min, T_max, res, dtype=np.float64)

    kappa = list()
    for i in range(0, len(T)):
        kappa.append(calculate_kappa(thau_ph, E, T[i]))
    kappa = np.asarray(kappa)

    har2pJ = 4.35974e-6
    kappa = kappa*har2pJ

    #save kappa data
    top.write_plot_data(data_path + "/kappa.dat", (T, kappa), "T [K], kappa [pW/K]")

    #now plot everything
    plt.tick_params(axis="x", labelsize=15)
    plt.tick_params(axis="y", labelsize=15)
    #plt.legend(fontsize=13)
    plt.rc('xtick', labelsize=15)
    plt.plot(T, kappa)
    plt.xlabel('Temperature ($K$)', fontsize=17)
    plt.ylabel(r'Thermal Conductance $\mathrm{pw/K}$', fontsize=17)
    plt.savefig(data_path + "/kappa.pdf", bbox_inches='tight')
    plt.show()
