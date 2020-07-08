# This script contains all the parameters of the model

# Output folder
output_folder = 'output'

# Physical constants
mu = 0.001                      # dynamic viscosity [Pa s]
rho_fluid = 1000                # density [kg m^-3]
nu = mu / rho_fluid             # kinematic viscosity [m^2 s^-1]
l = 1.0                         # side of the domain
delta_l = 0.005                 # thickness of the domain

Re = 32.0                        # Reynolds number
ubar = nu*Re/l            # Average unscaled inlet velocity

alphaf = 0.1                     # lower bound for \alpha (fluid)
alphas = 10**4                   # upper bound for \alpha (solid)

lb = 0.0                        # lower bound for the control eta
ub = 1.0                        # upper bound for the control eta

# Shape of the rectangular domain
V = 0.5                         # fraction of the volume occupied by the fluid
delta = 1.0                     # ratio length / height

# Numerical parameters
N = 100                         # Mesh refinement
NavierStokes = False            # Boolean, set to False if the Stokes approximation is enough

# Navier-Stokes numerical parameters
iter_max_NS = 5
tol_NS = 0.0001

qds = [0.001, 0.01, 0.1]           # refinement constant
iterations = [20, 30, 30]

# File names
eta_viz_name = "Control_visualisation"
vpt_viz_name = "Velocity_pressure_temperature_visualization"
vel_viz_name = "Velocity_visualization"
pre_viz_name = "Pressure_visualization"
output_log_file_name = output_folder + "/output_log.txt"

controls_guess = output_folder + "/control_guess"
velprestem_guess = output_folder + "/velprestem_guess"
velocity_guess = output_folder + "/velocity_guess"
pressure_guess = output_folder + "/pressure_guess"

controls = output_folder + "/control"
velprestem = output_folder + "/velprestem"
velocity = output_folder + "/velocity"
pressure = output_folder + "/pressure"

def compute_diff(list1, list2):
    """ Computes the mathematical difference list1 - list2 """
    if len(list1) != len(list2):
        raise Exception("The two lists have different lengths")
    else:
        u = []
        for i in range(len(list1)):
            u.append(list1[i] - list2[i])
    return u

# Loadind Mesh

load_mesh = False
load_path = "load"
load_name_mesh = load_path + "/mesh.xml"
load_name_init = load_path + "/init.xml"



# A few compatibility checks
if (len(qds) != len(iterations)):
    raise Exception("qrs, qds and iterations have different lengths")
