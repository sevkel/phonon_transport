# Phonon Transport

##Requirements
* Python 3.x
* [TurbomoleOutputProcessing](https://topdocu.github.io/)

##Usage
python3 phonon_transport.py config_file

###Config file
```` 
[Data Input]
data_path=
hessian_name=hessian
coord_name=coord.xyz

[Calculation]
n_l=5,1,2
n_r=1,2,3
M_L=Au
M_C=Au
gamma=
E_D=
N=
in_plane=False 
T_min=
T_max=
kappa_grid_points=

````