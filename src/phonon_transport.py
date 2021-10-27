'''
    File name: phonon_transport.py
    Author: Matthias Blaschke
    Python Version: 3.9
'''


import numpy as np
from scipy.linalg import eig
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from turbomoleOutputProcessing import turbomoleOutputProcessing as top 
import fnmatch
import scipy.signal
from multiprocessing import Pool
from functools import partial
import time
from scipy import integrate

h_bar = 1.0545718*10**(-34)

def calculate_g0(w, w_D):
	"""
    Calculates surface greens function according to Markussen, T. (2013). Phonon interference effects in molecular junctions. The Journal of chemical physics, 139(24), 244101.

    Args:
    	param1 (float): frequency
		param2 (float): Debeye frequency

    Returns:
    	(np.array) surface greens function
	"""  

	def im_g(w):
		if(w<=w_D):
			Im_g = -np.pi*3.0*w/(2*w_D)
		else:
			Im_g = 0
		return Im_g
	Im_g = map(im_g,w)
	Im_g = np.asarray(list(Im_g))	
	Re_g = -np.asarray(np.imag(scipy.signal.hilbert(Im_g)))
	g0=np.asarray((Re_g+1.j*Im_g),complex)
	return g0


def calculate_Sigma(w,g0,gamma, M_L, M_C):
	"""
    Calculates self energy according to Markussen, T. (2013). Phonon interference effects in molecular junctions. The Journal of chemical physics, 139(24), 244101.

    Args:
    	param1 (np.array): frequency
    	param2 (np.array): g0
		param2 (float): gamma
		param3 (String): M_L atom type in resevoir
		param4 (String): M_C atom type coupled to resevoir           
    Returns:
    	(population, fitness_values)
	"""   

	#convert to hartree/Bohr**2
	gamma = gamma * 0.010290855869847846

	M_L = top.atom_weight(M_L, u2kg=False)
	M_C = top.atom_weight(M_C, u2kg=False)

	gamma_prime = gamma/np.sqrt(M_C*M_L)

	g = g0/(1+gamma_prime*g0)
	sigma_nu = gamma_prime**2*g

	return sigma_nu

def calculate_P(i,para):
	"""
    Calculates Greens Function with given parameters at given frequency w for three coupled masses.

    Args:
    	param1 (float): frequency
		param2 (float): spring constant k
		param3 (float): mass m
		param4 (float): coupling constant Lambda
		param5 (float): coupling constant Gamma            
    Returns:
    	(population, fitness_values)
            
    """
	w = para[0]
	sigma = para[1]
	filename_hessian = para[2]
	filename_coord = para[3]
	n_l = para[4]
	n_r = para[5]
	gamma = para[6]
	in_plane = para[7]


	#set up dynamical matrix K
	K = top.create_dynamical_matrix(filename_hessian, filename_coord, t2SI=False)

	n_atoms = int(K.shape[0]/3)

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
	for u in range(lower,3):
		sigma_L[n_l*3+u][n_l*3+u] = sigma[i]
		sigma_R[n_r*3+u][n_r*3+u] = sigma[i]

	#correct momentum conservation
	gamma = gamma * 0.010290855869847846
	for u in range(lower,3):
		K[n_l*3+u][n_l*3+u]=(K[n_l*3+u][n_l*3+u]*top.atom_weight(M_C)-gamma)/top.atom_weight(M_C)
		K[n_r*3+u][n_r*3+u]=(K[n_r*3+u][n_r*3+u]*top.atom_weight(M_C)-gamma)/top.atom_weight(M_C)
	

	#calculate greens function
	G = np.linalg.inv(w[i]**2*np.identity(3*n_atoms)-K-sigma_L-sigma_R)
	Gamma_L = -2*np.imag(sigma_L)
	Gamma_R = -2*np.imag(sigma_R)
	P = np.real(np.trace(np.dot(np.dot(Gamma_L,G),np.dot(Gamma_R,np.conj(np.transpose(G))) )))
	return P


def calculate_kappa(P,w, T):
	w_si=w*np.sqrt(9.375821464623672e+29)
	k_B=1.38064852*10E-23
	factor = np.sqrt(9.375821464623672e+29)*h_bar/k_B
	#print("factor " +str(factor))
	prefactor = h_bar**2/(2*np.pi*k_B)*9.375821464623672e+29*np.sqrt(9.375821464623672e+29)*1E12
	#print(prefactor)
	kappa = list()
	exp = np.exp((w)/(T)*factor)	
	#print(T)
	#print(exp)
	
	#print(exp)
	
	integrand= (1/T**2)*w**2*P*exp/((exp-1)**2)
	print(integrand)
	#integral = np.cumsum(integrand)[-1]
	integral = np.trapz(integrand, w)

	return integral*prefactor



if __name__ == '__main__':
	filename="benzenediamine"
	filename_hessian = filename+"/hessian"
	filename_coord = filename+"/coord.xyz"
	n_l = 4
	n_r = 10
	#atom type in resevoir M_L and molecule M_C
	M_L="N"
	M_C="Au"
	#coupling force constant resevoir in eV/Ang**2
	gamma = -1.401986386	
	#Debeye energy in meV
	E_D=20
	#Number of grid points
	N = 2000
	#only in plane motion (-> set x and y coupling to zero)
	in_plane = False


	#convert to J
	E_D = E_D*1.60217656535E-22
	#convert to my units 
	w_D = E_D/h_bar
	w_D = w_D/np.sqrt(9.375821464623672e+29)


	
	w = np.linspace(0.000,w_D*1.1,N)
	i =np.linspace(0,N,N,False,dtype=int)    
	g0 = calculate_g0(w,w_D)
	Sigma = calculate_Sigma(w,g0,gamma,M_L,M_C)

	Pv = list()
	p = Pool()
	params = w,Sigma,filename_hessian,filename_coord,n_l,n_r,gamma,in_plane
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
	
	T=np.linspace(5,600,1000)
	#w_int = np.linspace(0.000,w_D*100,N*100)
	kappa=list()
	for i in range(0,len(T)):
		kappa.append(calculate_kappa(P_vals[1:len(P_vals)], w[1:len(w)], T[i]))

	#save data
	top.write_plot_data(filename + "/phonon_trans.dat", (w,P_vals), "w (weird units), P_vals")
	top.write_plot_data(filename + "/kappa.dat", (T,kappa),"T (K), kappa (pW/K)")

	#now plot everything
	E = w*np.sqrt(9.375821464623672e+29)*h_bar/(1.60217656535E-22)
	fig,(ax1,ax2) = plt.subplots(2,1)
	fig.tight_layout()
	ax1.plot(E,P_vals)
	ax1.set_yscale('log')
	ax1.set_xlabel('Phonon Energy ($\mathrm{meV}$)',fontsize=12)
	ax1.set_ylabel(r'Transmission $\tau_{\mathrm{ph}}$',fontsize=12)
	ax1.axvline(w_D*np.sqrt(9.375821464623672e+29)*h_bar/(1.60217656535E-22),ls="--", color="black")
	ax1.set_ylim(10E-12,1)

	ax2.plot(T,kappa)
	ax2.set_xlabel('Temperature ($K$)',fontsize=12)
	ax2.set_ylabel(r'Thermal Conductance $\mathrm{pw/K}$',fontsize=12)
	plt.savefig(filename + "/transport.pdf", bbox_inches='tight')
	plt.show()
	


