# Phonon Transport
Calculates the phononic transport according to [[1]](#1).
## Requirements
* Python 3.x
* [TurbomoleOutputProcessing](https://topdocu.github.io/)


## Calculation of phonon transmission
```` 
python3 phonon_transport.py config_file
```` 
### Preparation
* Build molecule in Avogadro. Save as .xyz file 
* Relaxation and hessian:
    * Geometry optimization e.g. using xtb (https://xtb-docs.readthedocs.io/en/latest/optimization.html)
    * Align molecule -> x axis through anchoring atoms (not hydrogen)
    * Calculate hessian e.g using xtb (https://xtb-docs.readthedocs.io/en/latest/hessian.html)

### Config file
```` 
[Data Input]
data_path= #path where data is located
hessian_name=hessian #filename of hessian
coord_name=coord.xyz #filename of coord file (.xyz and Turbomole format allowed)

[Calculation]
n_l=5,1,2 #indices of atoms (-> ordering of coord file) connected to left lead
n_r=1,2,3 #indices of atoms (-> ordering of coord file) connected to right lead
M_L=Au # atom type in left/right lead
M_C=Au # atom type in center connected to lead
gamma= #coupling constant from [Phonon interference effects in molecular junctions](https://doi.org/10.1063/1.4849178)
E_D= #Debey energy
N= #number of grid points for transmission calculation
in_plane=False #in_plane option from [Phonon interference effects in molecular junctions](https://doi.org/10.1063/1.4849178)
T_min= #lower bound for thermal conductance integral (avoid zero)
T_max= #upper bound for thermal conductance integral
kappa_grid_points= #number of grid point in thermal conductance integral

[Data Output]
plot_g=True #plot surface green function 

````

### Output
* data_path/phonon_trans.dat
* data_path/kappa.dat
* data_path/transport.pdf
* data_path/g0.pdf (optional, see plot_g)

## Calculation of thermal conductance
```` 
python3 calculate_kappa config_file
```` 
Calculates thermal conductance from phonon transmission. Energy must be in Hartrees!
### Config file
A reduced config file is sufficient for this
```` 
[Data Input]
data_path= #path where data is located
transp_name= #name of file containing phonon transmission

[Calculation]
T_min= #lower bound for thermal conductance integral (avoid zero)
T_max= #upper bound for thermal conductance integral
kappa_grid_points= #number of grid point in thermal conductance integral
````
### Output
* data_path/kappa.dat
* data_path/kappa.pdf



## References
<a id="1">[1]</a> 
Markussen, T. (2013).  
Phonon interference effects in molecular junctions. 
The Journal of chemical physics, 139(24), 244101.
[https://doi.org/10.1063/1.4849178]( https://doi.org/10.1063/1.4849178)

***
Matthias Blaschke [matthias.blaschke@physik.uni-augsburg.de](matthias.blaschke@pyhsik.uni-augsburg.de)
