# This script contains all the parameters of the model
# This script contains all the parameters of the model

# Output folder
output_folder = 'output'

# Physical constants
mu = 0.001 * 6.6                      # dynamic viscosity [Pa s]
rho_fluid = 1000                # density [kg m^-3]
nu = mu / rho_fluid             # kinematic viscosity [m^2 s^-1]
l = 1.0                         # side of the domain
delta_l = 0.005                 # thickness of the domain

Cp = 5000                       # heat capacity [J kg^-1 K^-1]
ks_phy = 100                     # thermal conductivity of solid [W m^-1 K^-1]
kf_phy = 10                   # thermal conductivity of fluid [W m^-1 K^-1]
ks = ks_phy / kf_phy            # adimensional ks
kf = 1.0 # = kf_phy / kf_phy    # adimensional kf

Re = 3.0                        # Reynolds number
Pe = Re * Cp * mu / kf_phy      # Péclet number
ubar = Re * 15.0/2.0 * nu / l   # average inlet velocity


alphaf = 0.1                     # lower bound for \alpha (fluid)
alphas = 10**4                   # upper bound for \alpha (solid)

lb = 0.0                        # lower bound for the control eta
ub = 1.0                        # upper bound for the control eta

# Shape of the rectangular domain
V = 0.4                         # fraction of the volume occupied by the fluid
delta = 1.0                     # ratio length / height
T_in = 0.0                      # inlet temperature [°C]
T_updown = 10.0                 # top and bottom walls temperature [°C]

# Numerical parameters
N = 100                         # Mesh refinement
w = 1.0                        # Fraction of the two costs
NavierStokes = False            # Boolean, set to False if the Stokes approximation is enough

# Navier-Stokes numerical parameters
iter_max_NS = 5
tol_NS = 0.0001

qds = [0.0001, 0.001]           # refinement constant
qrs = [0.0001, 0.001]
iterations = [30, 20]

# File names
eta_viz_name = "Control_visualisation"
vpt_viz_name = "Velocity_pressure_temperature_visualization"
vel_viz_name = "Velocity_visualization"
pre_viz_name = "Pressure_visualization"
tmp_viz_name = "Temperature_visualization"
output_log_file_name = output_folder + "/output_log.txt"

controls_guess = output_folder + "/control_guess"
velprestem_guess = output_folder + "/velprestem_guess"
velocity_guess = output_folder + "/velocity_guess"
pressure_guess = output_folder + "/pressure_guess"
temperature_guess = output_folder + "/temperature_guess"

controls = output_folder + "/control"
velprestem = output_folder + "/velprestem"
velocity = output_folder + "/velocity"
pressure = output_folder + "/pressure"
temperature = output_folder + "/temperature"

def compute_diff(list1, list2):
    """ Computes the mathematical difference list1 - list2 """
    if len(list1) != len(list2):
        raise Exception("The two lists have different lengths")
    else:
        u = []
        for i in range(len(list1)):
            u.append(list1[i] - list2[i])
    return u

# Cost bounds
Jd_max = [60.905082380459056, 61.50057137630121, 73.47436935158525, 73.47436935158525]
Jd_min = [4.455305990349442, 5.703032100949015, 5.5067874599442455, 5.5067874599442455]
Jr_max = [-0.17349130694906034, -0.09978256099375189, -0.09931649693543847, -0.09931649693543847]
Jr_min = [-0.44958335493508167, -0.4554014567994763, -0.8705829779127432, -0.8705829779127432]
diff_Jd = compute_diff(Jd_max, Jd_min)
diff_Jr = compute_diff(Jd_max, Jd_min)

# Loadind Mesh

load_mesh = False
load_path = "load"
load_name_mesh = load_path + "/mesh.xml"
load_name_init = load_path + "/init.xml"



# A few compatibility checks
if (len(qrs) != len(iterations)) or (len(qrs) != len(qds)):
    raise Exception("qrs, qds and iterations have different lengths")





#
# # Output folder
# output_folder = 'output'
#
# # Physical constants
# mu = 0.001 * 6.6                      # dynamic viscosity [Pa s]
# rho_fluid = 1000                # density [kg m^-3]
# nu = mu / rho_fluid             # kinematic viscosity [m^2 s^-1]
# l = 1.0                         # side of the domain
# delta_l = 0.005                 # thickness of the domain
#
# Cp = 5000                       # heat capacity [J kg^-1 K^-1]
# ks_phy = 10                     # thermal conductivity of solid [W m^-1 K^-1]
# kf_phy = 1                   # thermal conductivity of fluid [W m^-1 K^-1]
# ks = ks_phy / kf_phy            # adimensional ks
# kf = 1.0 # = kf_phy / kf_phy    # adimensional kf
#
# Re = 3.0                        # Reynolds number
# Pe = Re * Cp * mu / kf_phy      # Péclet number
# ubar = Re * 15.0/2.0 * nu / l   # average inlet velocity
#
#
# alphaf = 0.1                     # lower bound for \alpha (fluid)
# alphas = 10**4                   # upper bound for \alpha (solid)
#
# lb = 0.0                        # lower bound for the control eta
# ub = 1.0                        # upper bound for the control eta
#
# # Shape of the rectangular domain
# V = 0.4                         # fraction of the volume occupied by the fluid
# delta = 1.0                     # ratio length / height
# T_in = 0.0                      # inlet temperature [°C]
# T_updown = 10.0                 # top and bottom walls temperature [°C]
#
# # Numerical parameters
# N = 10                         # Mesh refinement
# w = 0.0                        # Fraction of the two costs
# NavierStokes = False            # Boolean, set to False if the Stokes approximation is enough
#
# # Navier-Stokes numerical parameters
# iter_max_NS = 5
# tol_NS = 0.0001
#
# qds = [0.0001, 0.001, 0.01, 0.1]           # refinement constant
# qrs = [0.0001, 0.001, 0.01, 0.1]
# iterations = [20, 20, 20, 30]
#
# # File names
# eta_viz_name = "Control_visualisation"
# vpt_viz_name = "Velocity_pressure_temperature_visualization"
# vel_viz_name = "Velocity_visualization"
# pre_viz_name = "Pressure_visualization"
# tmp_viz_name = "Temperature_visualization"
# output_log_file_name = output_folder + "/output_log.txt"
#
# controls_guess = output_folder + "/control_guess"
# velprestem_guess = output_folder + "/velprestem_guess"
# velocity_guess = output_folder + "/velocity_guess"
# pressure_guess = output_folder + "/pressure_guess"
# temperature_guess = output_folder + "/temperature_guess"
#
# controls = output_folder + "/control"
# velprestem = output_folder + "/velprestem"
# velocity = output_folder + "/velocity"
# pressure = output_folder + "/pressure"
# temperature = output_folder + "/temperature"
#
# def compute_diff(list1, list2):
#     """ Computes the mathematical difference list1 - list2 """
#     if len(list1) != len(list2):
#         raise Exception("The two lists have different lengths")
#     else:
#         u = []
#         for i in range(len(list1)):
#             u.append(list1[i] - list2[i])
#     return u
#
# # Cost bounds
# Jd_max = [60.905082380459056, 61.50057137630121]
# Jd_min = [4.455305990349442, 5.703032100949015]
# Jr_max = [-0.17349130694906034, -0.09978256099375189]
# Jr_min = [-0.44958335493508167, -0.4554014567994763]
# diff_Jd = compute_diff(Jd_max, Jd_min)
# diff_Jr = compute_diff(Jd_max, Jd_min)
#
# # Loadind Mesh
#
# load_mesh = False
# load_path = "load"
# load_name_mesh = load_path + "/mesh.xml"
# load_name_init = load_path + "/init.xml"
#
#
#
# # A few compatibility checks
# if (len(qrs) != len(iterations)) or (len(qrs) != len(qds)):
#     raise Exception("qrs, qds and iterations have different lengths")
