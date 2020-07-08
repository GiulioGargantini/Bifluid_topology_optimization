# This script contains all the parameters of the model a
import os, shutil                       # tools to work with folders
import numpy as np
# Output folder
output_folder = 'output'
to_be_loaded_folder = "to_be_loaded"
output_to_be_loaded_folder = output_folder + "/" + to_be_loaded_folder

# Domain Shape
length_y = 1.0e-3       # heigth of the domain (length in the y direction) [m]
length_x_f = 1.0e-3     # length (in the x direction) of Omega_f [m]
length_x_d = 1.7e-3     # length (in the x direction) of Omega_d [m]
length_x_tot = 2*length_x_f + length_x_d    # total length of the domain [m]
area_d = length_x_d * length_y          # surface of Omega_d [m^2]
area_tot = length_x_tot * length_y      # surface of Omega_tot [m^2]
length_x_d_adim = length_x_d / length_y
length_x_f_adim = length_x_f / length_y
length_x_tot_adim = length_x_tot / length_y
length_y_adim = length_y / length_y
area_d_adim = area_d / length_y**2    # adimensionalized surface of Omega_d
area_tot_adim = area_tot / length_y**2       # adimensionalized surface of Omega_tot

# Physical constants
## air
c_f = 1006          # heat capacity of air [J/(kg K)]
k_f = 0.024         # thermal conductivity of air [W/(m K)]
mu_f = 1.94e-5      # dynamic fluid viscosity of air [Pa s]
rho_f = 1.204       # density of air [kg / m^3]

## Inverse permeabilities
Da = 1e-5       # Darcy's number [-]
alpha_f = 0.0   # inverse permeability of the fluid
alpha_s = mu_f/(Da * length_y**2)  # inverse permeability of the solid

## others
C_k = 1e-3  # ratio k_f/k_s, i.e. ratio btw thermal conductivities [-]
k_s = k_f / C_k     # thermal conductivity of the solid [W/(m K)]
k_s_base = 400.0      # thermal conductivity of the base plate [W/(m K)]
h_f = 50.0    # heat transfer coefficient of air [W / (m^2 K)]
h_s = 2.0e5   # heat transfer coefficient of metal [W / (m^2 K)]
C_h = h_f / h_s     # ratio r_f/r_s, i.e. ratio btw heat transfer coeffs [-]

p_in = 3.0     # Pressure at the inflow [Pa]
p_out = 0.0    # Pressure at the outflow [Pa]

# Model parameters
dQ_prod = 0.175     # heat flux in the base plate [W]
T_in = 20.0           # fluid temperature at the inflow [°C]
Dz_base = 0.2*1e-3  # thickness of the base plate (in the z direction) [m]
Dz_ch = 8e-3        # thickness of the channel (in the z direction) [m]
Vol_base = Dz_base * area_d     # volume of the base plate [m^3]

max_fluid_frac = 0.70   # maximal fluid fraction
min_fluid_frac = 0.20   # minimal fluid fraction

## Filtering
r_filter = 2e-5     # filtering radius [m]
heaviside_eta = 0.5     # projection treshold parameter

# Adimensionalization constants
Lbar = length_y     # adim of length [m]
pbar = p_in         # adim of pressure [Pa]
ubar = np.sqrt(pbar/rho_f)

Re =  Lbar * np.sqrt(rho_f * pbar) / mu_f    # Reynolds number [-]
Stokes_pressure_coeff = pbar / Lbar         # coefficient before \nabla p in Stokes

Transp_coeff_F2 = np.sqrt(pbar * rho_f) * c_f /Lbar    # Transport coefficient for heat eq [W/m^3]
Diff_coeff_F2_f = k_f/(Lbar ** 2)   # diffusion coefficient for T_f [W/m^3]
Heat_coupling_coeff_F2_f = h_f/Dz_ch    # coefficient for the thermal coupling for T_f [W/m^3]

Diff_coeff_F2_s = k_s_base/(Lbar ** 2)   # diffusion coefficient for T_s [W/m^3]
Heat_coupling_coeff_F2_s = h_f/Dz_base      # coefficient for the thermal coupling for T_s [W/m^3]
Heat_generation_F2 = dQ_prod / Vol_base     # known, constant heat generation term [W/m^3]

# Expectation
u_max = np.sqrt(pbar/rho_f) * Re /(8 * length_x_d_adim)
u_max_adim = u_max * np.sqrt(rho_f / pbar)

# Numerical parameters
N = 50     # Mesh refinement
Nx_tot = round(N * length_x_tot * 1e3)    # Mesh refinement in x direction
Nx_d = round(N * length_x_d * 1e3)    # Mesh refinement in x direction
Ny = round(N * length_y * 1e3)            # Mesh refinement in y direction

iterations = [1] #, 50, 50, 50, 50, 50, 50, 50, 50, 50]

## Convexity parameters
b_alpha =  [8.0] #[1.0, 1.0, 2.0] #, 4.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]       # conv. param. for the inverse permeability
b_k = [5.0]#[5.0, 5.0, 10.0]#, 5.0, 20.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0]          # conv. param. for the thermal conuctivity
b_h = [2.0]#[2.0, 4.0, 4.0]#, 2.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0]           # conv. param. for the heat transfer coefficient
heaviside_beta = [5.0] #, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 7.0, 7.0]

## Bounds for the control
lb = 0.0    # Lower bound for gamma [-]
ub = 1.0    # Upper bound for gamma [-]

# File names

controls = output_folder + "/control"
controls_d = output_folder + "/control_d"
velprestem = output_folder + "/velprestem"
velocity = output_folder + "/velocity"
pressure = output_folder + "/pressure"
temperature_f = output_folder + "/temperature_f"
temperature_s = output_folder + "/temperature_s"
temperature_s_d = output_folder + "/temperature_s_d"

gamma_viz_name = "Control_visualisation"
gamma_d_viz_name = "Control_d_visualisation"
vpt_viz_name = "Velocity_pressure_temperature_visualization"
vel_viz_name = "Velocity_visualization"
pre_viz_name = "Pressure_visualization"
temp_f_viz_name = "Temperature_f_visualization"
temp_s_viz_name = "Temperature_s_visualization"
temp_s_d_viz_name = "Temperature_s_d_visualization"
output_log_file_name = output_folder + "/output_log.txt"

load_mesh = False
load_path = "load"
load_name_mesh = load_path + "/prova_13042020/mesh.h5"
load_name_init = load_path + "/prova_13042020/init_2.h5"


# Clear path function
def clear_output():
    print(os.getcwd())
    current_folder = os.getcwd()
    real_output_folder = os.getcwd() + '/' + output_folder
    print(real_output_folder)
    if(os.path.isdir(real_output_folder)):
        shutil.rmtree(real_output_folder)
    os.mkdir(real_output_folder)
    os.mkdir(real_output_folder + "/" + to_be_loaded_folder)
