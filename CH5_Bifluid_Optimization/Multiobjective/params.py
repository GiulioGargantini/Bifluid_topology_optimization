# This script contains all the parameters for Tawk fig 3.25

import os, shutil         # tools to work with folders
import numpy as np

# Output folder
output_folder = "output"
to_be_loaded_folder = "to_be_loaded"
output_to_be_loaded_folder = output_folder + "/" + to_be_loaded_folder

# Domain Shape
length_y = 70.0e-3         # heigth of the domain (length in the y direction) [m]
length_x = 110.0e-3        # length (in the x direction) of Omega_f [m]
length_y_wall_hole = 3.0e-3    # distance btw the wall and the inlet
length_y_hole = 15.0e-3    # size of the inlets and outlets

# Physical constants
## fluids
mu_f = 1.0e-3      # dynamic fluid viscosity of fluid [Pa s]
rho_f = 1.0e3       # density of the fluid [kg/m^3]
C_p = 4200.0    # thermal capacity [J / (K kg)]

## Inverse permeabilities
alpha_f = 0.0   # inverse permeability of the fluid


## Thermal conductivities
k_f = 1.0   # thermal conductivity of the fluid [W / (K m)]
k_s = 10.0   # thermal conductivity of the solid [W / (K m)]

## Inlet temperatures
T_cold_in = 0.0     # temperature of the cold fluid at the inlet
T_hot_in = 200.0     # temperature of the hot fluid at the inlet

# Model parameters
phi_f1 = 0.23    # max porosity of fluid 1
phi_f2 = 0.23    # max porosity of fluid 2

Re = 15.0   # Reynolds number

u_avg_in = Re * mu_f / (length_y_hole * rho_f)   # average inlet velocity [m/s]
u_max_in = 3.0 * u_avg_in / 2.0     # maximum inlet velocity [m/s]
u_max_out = u_max_in      # maximum outlet velocity [m/s]

contiguity_penalization = 1.0e2         # weight of the contiguity penalization
gradient_penalization = 1.0e2   # weight for the penalization of the gradient term


r_filtering_eta2 = 0.03  # radius for the helmholtz filtering for the fluid contiguity penalization
r_filtering_eta1 = r_filtering_eta2    # radius for the helmholtz filtering at every outer cycle
start_with_filtering_eta1 = False

# scaling the costs
w_cost_weight = 0.50      # weight to balance the costs:  w = 1.0 -> J = J_pressure
                         #                               W = 0.0 -> J = J_thermal

max_pressure_cost = 18.3893762834507
min_pressure_cost = 6.481887395243737
max_thermal_cost = 20.433758740745244
min_thermal_cost = 12.641422223573892
scale_pressure_cost = max_pressure_cost - min_pressure_cost
scale_thermal_cost = max_thermal_cost - min_thermal_cost
cost_constant_pressure = min_pressure_cost / scale_pressure_cost
cost_constant_thermal = min_thermal_cost / scale_thermal_cost

# Adimensionalization constants
L_bar = length_y   # length unit for adimensionalization
U_bar = u_avg_in   # velocity unit for adimensionalization
T_bar = 1.0        # temperature for adimensionalization

length_x_adim = length_x / L_bar    # length on direction x adimensionalized
length_y_adim = length_y / L_bar    # length on direction y adimensionalized
length_y_wall_hole_adim = length_y_wall_hole / L_bar
length_y_hole_adim = length_y_hole / L_bar
area_adim = length_x_adim * length_y_adim   # adimensionalized surface

NS_to_adim = rho_f * U_bar**2 / L_bar   # coeff for the adimensionalization of NS
heat_to_adim = 1.0 / (L_bar * U_bar * rho_f * C_p)  # coeff for the adimensionalization of the heat eq.

u_max_in_adim = u_max_in / U_bar    # max inlet velocity adimensionalized
u_max_out_adim = u_max_out / U_bar    # max outlet velocity adimensionalized

const_adim_cost_pressure = U_bar**3 * rho_f * L_bar
const_adim_cost_thermal = L_bar * U_bar * rho_f * C_p * T_bar

# Numerical parameters
N = 40     # Mesh refinement
Nx = round(N * length_x / L_bar)     # Mesh refinement in x direction
Ny = round(N * length_y / L_bar)     # Mesh refinement in y direction

# Iterative parameters
iterations = [300, 150]
alpha_penalizations = [1.0e2, 1.0e4]  # inverse permeability of the solid
velocity_penalizations = [1.0e2, 1.0e1]
## Convexity parameters
b_alpha =  [10.0, 10.0]    # conv. param. for the inverse permeability
b_k = [1.0, 1.0]     # conv. param. for the thermal conductivity

# Bounds for the control
lb = 0.0    # Lower bound for eta1 and eta2 [-]
ub = 1.0    # Upper bound for eta1 and eta2 [-]

# File names

control_eta1 = output_folder + "/control_eta1"
control_eta2 = output_folder + "/control_eta2"

eta1_viz_name = "Control_eta1_visualisation"
eta2_viz_name = "Control_eta2_visualisation"
vel_viz_name = "Velocity_visualization"
vel_f1_viz_name = "Velocity_f1_visualization"
vel_f2_viz_name = "Velocity_f2_visualization"
pre_viz_name = "Pressure_visualization"
pre_f1_viz_name = "Pressure_f1_visualization"
pre_f2_viz_name = "Pressure_f2_visualization"
tem_viz_name = "Temperature_visualization"
Q1f_viz_name = "Q1_filtered_visualization"
Q2f_viz_name = "Q2_filtered_visualization"
eta2f_viz_name = "eta2_filtered_visualization"
output_log_file_name = output_folder + "/output_log.txt"

load_mesh = False
load_path = "load/prova_27052020_1"
load_name_mesh = load_path + "/mesh.h5"
load_name_init_eta1 = load_path + "/eta1_final.h5"
load_name_init_eta2 = load_path + "/eta2_final.h5"
eta1_name = "eta1"
eta2_name = "eta2"
