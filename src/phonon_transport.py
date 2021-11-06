'''
    File name: phonon_transport.py
    Author: Matthias Blaschke
    Python Version: 3.9
'''
import codecs
import copy
import sys
import json

import numpy as np
from scipy.linalg import eig
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from turbomoleOutputProcessing import turbomoleOutputProcessing as top
import fnmatch
import scipy.signal
from multiprocessing import Pool
from functools import partial
import time
import configparser
import calculate_kappa as ck
from scipy import integrate


#h_bar in Js
h_bar = 1.0545718*10**(-34)
eV2hartree = 0.0367493
ang2bohr = 1.88973
har2J = 4.35974E-18
bohr2m = 5.29177E-11
u2kg = 1.66054E-27
har2pJ = 4.35974e-6

def calculate_g0(w, w_D):
	"""Calculates surface greens function according to Markussen, T. (2013). Phonon interference effects in molecular junctions. The Journal of chemical physics, 139(24), 244101 (https://doi.org/10.1063/1.4849178).

    Args:
    w (array_like): Frequency where g0 is calculated
	w_D (float): Debeye frequency

    Returns:
    g0	(array_like) Surface greens function g0
	"""

	def im_g(w):
		if(w<=w_D):
			Im_g = -np.pi*3.0*w/(2*w_D**3)
		else:
			Im_g = 0
		return Im_g
	Im_g = map(im_g,w)
	Im_g = np.asarray(list(Im_g))
	Re_g = -np.asarray(np.imag(scipy.signal.hilbert(Im_g)))
	g0=np.asarray((Re_g+1.j*Im_g),complex)
	return g0


def calculate_Sigma(w,g0,gamma, M_L, M_C):
	"""Calculates self energy according to Markussen, T. (2013). Phonon interference effects in molecular junctions. The Journal of chemical physics, 139(24), 244101  (https://doi.org/10.1063/1.4849178).

    Args:
    w (np.array): frequency
    g0 (np.array): g0
	gamma (float): gamma
	M_L (str): M_L atom type in reservoir
	M_C (str): M_C atom type coupled to reservoir

    Returns:
    sigma_nu (array_like) self energy term
	"""

	#convert to hartree/Bohr**2
	gamma_hb = gamma * (eV2hartree/ang2bohr**2)

	M_L = top.atom_weight(M_L, u2kg=False)
	M_C = top.atom_weight(M_C, u2kg=False)

	gamma_prime = gamma_hb/np.sqrt(M_C*M_L)

	g = g0/(1+gamma_prime*g0)
	sigma_nu = gamma_prime**2*g

	return sigma_nu

def calculate_P(i,para):
	"""Calculates Greens Function with given parameters at given frequency w.

	Args:
		i: (int): frequency index
		para: (tuple): frequency w (array), self energy sigma (complex), filename_hessian (str), filename_coord (str), left atom for transport calculation n_l (int), right atom for transport calculation n_r (int), coupling constant Gamma (complex), in_plane (boolean)

	Returns:
		P (array_like): phonon transmission
	"""

	w = para[0]
	sigma = para[1]
	filename_hessian = para[2]
	filename_coord = para[3]
	n_l = para[4]
	n_r = para[5]
	gamma = para[6]
	in_plane = para[7]
	D = para[8]
	D = copy.copy(D)




	n_atoms = int(D.shape[0]/3)

	"""
	remover = np.zeros((len(eigenvalues),len(eigenvalues)))
	for i in range(0,6):
		#print(i)
		remover += eigenvalues[i]*np.outer(np.transpose(eigenvectors[:,i]),eigenvectors[:,i])/np.sqrt(np.dot(eigenvectors[:,i],eigenvectors[:,i]))
	K=K-remover
	"""

	#set up self energies
	sigma_L = np.zeros((n_atoms*3,n_atoms*3), complex)
	sigma_R = np.zeros((n_atoms*3,n_atoms*3), complex)
	if(in_plane==True):
		lower = 2
	else:
		lower = 0

	for n_l_ in n_l:
		for u in range(lower,3):
			sigma_L[n_l_*3+u,n_l_*3+u] = sigma[i]
	for n_r_ in n_r:
		for u in range(lower,3):
			sigma_R[n_r_*3+u,n_r_*3+u] = sigma[i]
	sigma_i = sigma[i]

	#correct momentum conservation
	#convert to hartree/Bohr**2
	gamma_hb = gamma * eV2hartree/ang2bohr**2

	D_save = copy.deepcopy(D)
	for u in range(lower,3):
		for n_l_ in n_l:
			#remove mass weighting
			K_ = D[n_l_*3+u][n_l_*3+u]*top.atom_weight(M_C)
			#correct momentum
			K_ = K_ - gamma_hb
			#add mass weighting again
			D_ = K_/top.atom_weight(M_C)
			D[n_l_ * 3 + u][n_l_ * 3 + u] = D_

		for n_r_ in n_r:
			#remove mass weighting
			K_ = D[n_r_*3+u][n_r_*3+u]*top.atom_weight(M_C)
			#correct momentum
			K_ = K_ - gamma_hb
			#add mass weighting again
			D_ = K_/top.atom_weight(M_C)
			D[n_r_ * 3 + u][n_r_ * 3 + u] = D_
		#"""
	"""
	n_l = n_l[0]
	n_r = n_r[0]
	D[n_l * 3 + u][n_l * 3 + u] = (D[n_l * 3 + u][n_l * 3 + u] * top.atom_weight(M_C) - gamma_hb) / top.atom_weight(
		M_C)
	D[n_r * 3 + u][n_r * 3 + u] = (D[n_r * 3 + u][n_r * 3 + u] * top.atom_weight(M_C) - gamma_hb) / top.atom_weight(
		M_C)
	"""



	#extra broadening
	eta = np.full((n_atoms*3, n_atoms*3), 1.j*1E-8)

	#calculate greens function
	G = np.linalg.inv(w[i]**2*np.identity(3*n_atoms)-D-sigma_L-sigma_R)
	Gamma_L = -2*np.imag(sigma_L)
	Gamma_R = -2*np.imag(sigma_R)
	P = np.real(np.trace(np.dot(np.dot(Gamma_L,G),np.dot(Gamma_R,np.conj(np.transpose(G))) )))
	return P


def calculate_kappa(P,w, T):
	"""Calculates Thermal conductance up to given Temperature
	Args:
		P: (array_like): Phonon transmission
		w: (array_like): frequency in atomic units
		T: (float): Temperature in Kelvin

	Returns:
		kappa (float) Thermal conductance
	"""
	w_si=w*np.sqrt(9.375821464623672e+29)
	#Boltzmann constant in Si units
	k_B=1.38064852*10E-23
	factor = np.sqrt(9.375821464623672e+29)*h_bar/k_B
	#print("factor " +str(factor))
	prefactor = h_bar**2/(2*np.pi*k_B*T**2)*9.375821464623672e+29*np.sqrt(9.375821464623672e+29)*1E12
	#print(prefactor)
	kappa = list()
	exp = np.exp((w)/(T)*factor)
	#print(T)
	#print(exp)

	#print(exp)

	integrand= w**2*P*exp/((exp-1)**2)
	#print(integrand)
	#integral = np.cumsum(integrand)[-1]
	integral = np.trapz(integrand, w)

	return integral*prefactor



if __name__ == '__main__':
	config_path = sys.argv[1]
	cfg = configparser.ConfigParser()
	#cfg.read(config_path)
	cfg.read_file(codecs.open(config_path, "r", "utf8"))

	try:
		data_path= str(cfg.get('Data Input', 'data_path'))
		hessian_name=str(cfg.get('Data Input', 'hessian_name'))
		coord_name = str(cfg.get('Data Input', 'coord_name'))
		filename_hessian = data_path + "/" + hessian_name
		filename_coord = data_path + "/" + coord_name

		#atoms which are coupled to the leads -> self energy
		n_l = np.asarray(str(cfg.get('Calculation', 'n_l')).split(','),dtype=int)
		n_r = np.asarray(str(cfg.get('Calculation', 'n_r')).split(','),dtype=int)

		#atom type in resevoir M_L and molecule M_C
		M_L=str(cfg.get('Calculation', 'M_L'))
		M_C=str(cfg.get('Calculation', 'M_C'))
		#coupling force constant resevoir in eV/Ang**2
		gamma = float(cfg.get('Calculation', 'gamma'))

		#Debeye energy in meV
		E_D = float(cfg.get('Calculation', 'E_D'))
		#Number of grid points
		N = int(cfg.get('Calculation', 'N'))
		#only in plane motion (-> set x and y coupling to zero)
		in_plane = json.loads(str(cfg.get('Calculation', 'in_plane')).lower())

		#for thermal conducatance
		T_min = float(cfg.get('Calculation', 'T_min'))
		T_max = float(cfg.get('Calculation', 'T_max'))
		kappa_grid_points = int(cfg.get('Calculation', 'kappa_grid_points'))

		#check if g0 should be plotted
		plot_g0 = json.loads(str(cfg.get('Data Output', 'plot_g')).lower())

	except configparser.NoOptionError:
		print("Missing option in config file. Check config file!")
		exit(-1)
	except ValueError:
		print("Wrong value in config file. Check config file!")
		exit(-1)

	#convert to J
	E_D = E_D*1.60217656535E-22
	#convert to 1/s
	w_D = E_D/h_bar
	#convert to har/(bohr**2*u)
	w_D = w_D/np.sqrt(9.375821464623672e+29)



	w = np.linspace(0.0,w_D*1.1,N)
	i =np.linspace(0,N,N,False,dtype=int)
	g0 = calculate_g0(w,w_D)

	if(plot_g0==True):
		fig, ax1 = plt.subplots()
		ax1.plot(np.real(g0))
		ax1.plot(np.imag(g0), color="red", label="Im(g0)")
		ax1.plot(np.real(g0), color="green", label="Re(g0)")
		plt.grid()
		plt.savefig(data_path + "/g0.pdf", bbox_inches='tight')


	Sigma = calculate_Sigma(w,g0,gamma,M_L,M_C)
	# set up dynamical matrix K
	D = top.create_dynamical_matrix(filename_hessian, filename_coord, t2SI=False)

	Pv = list()
	p = Pool()
	params = w,Sigma,filename_hessian,filename_coord,n_l,n_r,gamma,in_plane,D
	start = time.time()
	result = map(partial(calculate_P, para=params), i)
	stop = time.time()
	start = time.time()
	print("calculated in " + str(stop-start))
	P_vals = list()
	for P in result:
		P_vals.append(P)
	P_vals = np.asarray(P_vals)
	stop = time.time()

	print("transformed in " + str(stop-start))

	T=np.linspace(T_min,T_max,kappa_grid_points)
	#w_int = np.linspace(0.000,w_D*100,N*100)
	kappa=list()
	#w to SI
	w_kappa = w*np.sqrt(9.375821464623672e+29)
	E = h_bar*w_kappa
	#joule to hartree
	E = E/har2J
	for i in range(0,len(T)):
		kappa.append(ck.calculate_kappa(P_vals[1:len(P_vals)], E[1:len(E)], T[i])*har2pJ)

	#save data
	top.write_plot_data(data_path + "/phonon_trans.dat", (w, P_vals), "w (weird units), P_vals")
	top.write_plot_data(data_path + "/kappa.dat", (T, kappa), "T (K), kappa (pW/K)")

	#now plot everything
	E = w*np.sqrt(9.375821464623672e+29)*h_bar/(1.60217656535E-22)
	fig,(ax1,ax2) = plt.subplots(2,1)
	fig.tight_layout()
	ax1.plot(E,P_vals)
	ax1.set_yscale('log')
	ax1.set_xlabel('Phonon Energy ($\mathrm{meV}$)',fontsize=12)
	ax1.set_ylabel(r'Transmission $\tau_{\mathrm{ph}}$',fontsize=12)
	ax1.axvline(w_D*np.sqrt(9.375821464623672e+29)*h_bar/(1.60217656535E-22),ls="--", color="black")
	ax1.set_ylim(10E-7,1)

	ax2.plot(T,kappa)
	ax2.set_xlabel('Temperature ($K$)',fontsize=12)
	ax2.set_ylabel(r'Thermal Conductance $\mathrm{pw/K}$',fontsize=12)
	plt.savefig(data_path + "/transport.pdf", bbox_inches='tight')
	plt.show()



