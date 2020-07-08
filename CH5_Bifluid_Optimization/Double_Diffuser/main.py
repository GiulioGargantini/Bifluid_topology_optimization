# In this script we try to implement a topology optimization with two fluids subject
# to Stokes eq. with the separation of the fluids as in Tawk.
# IOT achieve separation, we add a penalization term inspired Helmholtz filtering.
# In this script we try to model the temperature as well, aiming to replicate
# fig. 4.20 of Tawk
# In this script we use a single velocity field for both fluids, forcing the value
# of eta2 at the inlets and outlets.

## Import libraries
from dolfin import *                    # Library for the optimization
from dolfin_adjoint import *            # Library for the optimization
import pyipopt                          # Library for the optimization
import params as data                   # numerical parameters and physical constant values
import os, shutil                       # tools to work with folders
import numpy as np
import meshes as meshes


## Interpolation Functions
def I_alpha(eta1, eta2):     # I_alpha(0) = Alpha_penalization,   I_alpha(1) = 0
    """Interpolation of the inverse permeability"""
    eta_c = eta1 * (1.0 - 4.0 * eta2 * (1.0 - eta2))
    return Alpha_penalization * (1.0 - eta_c) / (1.0 + b_alpha * eta_c)

def I_k(eta1):
    """Interpolation of the thermal conductivity"""
    return Thermal_k_fluid * (eta1 * (C_k * (1.0 + b_k) - 1.0) + 1.0) / (C_k * (1.0 + b_k * eta1))

def Q_f1(eta1, eta2):    # Quantity of fluid 1
    return eta1 * (Constant(1.0) - eta2)

def Q_f2(eta1, eta2):    # Quantity of fluid 1
    return eta1 * eta2

## Setting output path and file names
def set_path_and_names(step):
    if step in range(0,len(data.b_alpha) - 1):
        str_suffix = "_" + str(step) + "_guess.pvd"
    elif step == (len(data.b_alpha) - 1):
        str_suffix = "_solution.pvd"
    elif step == -1:
        str_suffix = "_original.pvd"
    else:
        raise Exception("No further precision step defined")

    controls_eta1 = File(data.control_eta1 + str_suffix)
    controls_eta2 = File(data.control_eta2 + str_suffix)

    eta1_viz = Function(meshes.A, name = data.eta1_viz_name)
    eta2_viz = Function(meshes.A, name = data.eta2_viz_name)
    vel_viz = Function(meshes.U, name = data.vel_viz_name)
    pre_viz = Function(meshes.A, name = data.pre_viz_name)
    tem_viz = Function(meshes.A, name = data.tem_viz_name)
    eta2f_viz = Function(meshes.A, name = data.eta2f_viz_name)
    out_file = open(data.output_log_file_name, 'w+')

    return controls_eta1, controls_eta2, eta1_viz, eta2_viz, vel_viz, pre_viz, tem_viz, eta2f_viz, out_file

def save_for_output(eta1, eta2, str_iter):
    fFile1 = HDF5File(MPI.comm_world, data.output_to_be_loaded_folder + "/eta1_" + str_iter + ".h5", "w")
    fFile2 = HDF5File(MPI.comm_world, data.output_to_be_loaded_folder + "/eta2_" + str_iter + ".h5", "w")
    fFile1.write(eta1,"/" + data.eta1_name)
    fFile2.write(eta2,"/" + data.eta2_name)
    fFile1.close()
    fFile2.close()

def update_constants(iteration):
    b_alpha.assign(data.b_alpha[iteration])
    b_k.assign(data.b_k[iteration])
    Alpha_penalization.assign(data.alpha_penalizations[iteration] * data.U_bar / data.NS_to_adim)
    Cost_velocity.assign(data.velocity_penalizations[iteration])

##############################################################################################

## Funzione forward
def forward(eta1, eta2, annotation = True):
    """Solve the forward problem for a given fluid distribution gamma(x)."""

    ## Boundary conditions
    class VelocityBC(UserExpression):
    # BC for the velocity
        def eval(self, values, x):
            values[1] = 0.0
            values[0] = 0.0

            if ((x[0] <= DOLFIN_EPS) or (x[0] >= data.length_x_adim - DOLFIN_EPS)):
                # Cold inflow
                if (((x[1] >= data.length_y_wall_hole_in_adim) and (x[1] <= data.length_y_wall_hole_in_adim + data.length_y_hole_in_adim)) and near(x[0], 0.0)):
                    t = (x[1] - data.length_y_wall_hole_in_adim - data.length_y_hole_in_adim / 2) * 2 / data.length_y_hole_in_adim
                    values[0] =  data.u_max_in_adim * (1.0 - t**2)

                # Hot inflow
                elif (((x[1] <= data.length_y_adim - data.length_y_wall_hole_in_adim) and (x[1] >= data.length_y_adim - data.length_y_wall_hole_in_adim - data.length_y_hole_in_adim)) and near(x[0], 0.0)):
                    t = (data.length_y_adim - x[1] - data.length_y_wall_hole_in_adim - data.length_y_hole_in_adim / 2) * 2 / data.length_y_hole_in_adim
                    values[0] =  data.u_max_in_adim * (1.0 - t**2)

                # Cold outflow
                elif (((x[1] >= data.length_y_wall_hole_out_adim) and (x[1] <= data.length_y_wall_hole_out_adim + data.length_y_hole_out_adim)) and near(x[0], data.length_x_adim)):
                    t = (x[1] - data.length_y_wall_hole_out_adim - data.length_y_hole_out_adim / 2) * 2 / data.length_y_hole_out_adim
                    values[0] =  data.u_max_out_adim * (1.0 - t**2)

                # Hot outflow
                elif (((x[1] <= data.length_y_adim - data.length_y_wall_hole_out_adim) and (x[1] >= data.length_y_adim - data.length_y_wall_hole_out_adim - data.length_y_hole_out_adim)) and near(x[0], data.length_x_adim)):
                    t = (data.length_y_adim - x[1] - data.length_y_wall_hole_out_adim - data.length_y_hole_out_adim / 2) * 2 / data.length_y_hole_out_adim
                    values[0] =  data.u_max_out_adim * (1.0 - t**2)

        # tells that values are 2D vectors (they are 1D by default)
        def value_shape(self):
            return (2,)


    bc_vel = DirichletBC(meshes.UPTQ.sub(0), VelocityBC(degree=1), "on_boundary")
    bc_T_cold_in = DirichletBC(meshes.UPTQ.sub(2), Constant(data.T_cold_in), meshes.inflow_cold_bound)
    bc_T_hot_in = DirichletBC(meshes.UPTQ.sub(2), Constant(data.T_hot_in), meshes.inflow_hot_bound)
    bc_cold_eta2_in = DirichletBC(meshes.UPTQ.sub(3), Constant(0.0), meshes.inflow_cold_bound)
    bc_cold_eta2_out = DirichletBC(meshes.UPTQ.sub(3), Constant(0.0), meshes.outflow_cold_bound)
    bc_hot_eta2_in = DirichletBC(meshes.UPTQ.sub(3), Constant(1.0), meshes.inflow_hot_bound)
    bc_hot_eta2_out = DirichletBC(meshes.UPTQ.sub(3), Constant(1.0), meshes.outflow_hot_bound)

    bc_complete = [bc_vel, bc_T_cold_in, bc_T_hot_in, bc_cold_eta2_in, bc_cold_eta2_out, bc_hot_eta2_in,bc_hot_eta2_out]

    (v, q, S, R) = TestFunctions(meshes.UPTQ)
    UPTQ0 = Function(meshes.UPTQ)
    u, p, T, eta2f = split(UPTQ0)


    F_complete = (I_alpha(eta1, eta2) * inner(u,v) * dx
                    + inner(grad(u) * u, v) * dx
                    + Rem1 * inner(grad(u), grad(v)) * dx
                    - inner(div(v), p) * dx
                    - inner(div(u), q) * dx
                    + inner(u, grad(T)) * S * dx
                    + Heat_adim_coeff * I_k(eta1) * inner(grad(T), grad(S)) * dx
                    + Radius_filtering_squared_penalization * inner(grad(eta2f), grad(R)) * dx
                    + eta2f * R * dx
                    - eta2 * R * dx)

    solve(F_complete == 0, UPTQ0, bcs = bc_complete, annotate = annotation)

    return UPTQ0

## Cost function
def pressure_cost_unassembled(eta1_a, eta2_a, u_a, p_a, n_a):
    cost = - inner(u_a, n_a) * p_a * ds
    return cost

def thermal_cost_unassembled(eta1_a, eta2_a, u_a, p_a, T_a, n_a):
    cost_f1 =   inner(u_a, n_a) * T_a * ds(3) + inner(u_a, n_a) * T_a * ds(1)
    cost_f2 = - inner(u_a, n_a) * T_a * ds(4) - inner(u_a, n_a) * T_a * ds(2)
    return (cost_f1 + cost_f2)

def pressure_cost(eta1_a, eta2_a, u_a, p_a, n_a):
    return assemble(pressure_cost_unassembled(eta1_a, eta2_a, u_a, p_a, n_a))

def thermal_cost(eta1_a, eta2_a, u_a, p_a, T_a, n_a):
    return assemble(thermal_cost_unassembled(eta1_a, eta2_a, u_a, p_a, T_a, n_a))

def combined_cost(eta1_a, eta2_a, u_a, p_a, T_a, n_a):
    cost_pressure = (assemble(1 / Scale_pressure_cost * pressure_cost_unassembled(eta1_a, eta2_a, u_a, p_a, n_a)) - data.cost_constant_pressure) * data.w_cost_weight
    cost_thermal = (assemble(1 / Scale_thermal_cost * thermal_cost_unassembled(eta1_a, eta2_a, u_a, p_a, T_a, n_a)) - data.cost_constant_pressure) * (1.0 - data.w_cost_weight)
    return cost_pressure - cost_thermal
    #return assemble(-thermal_cost_unassembled(eta1_a, eta2_a, u_a, p_a, T_a, n_a))

def penalization_cost(eta1_a, eta2_a, eta2f_a):
    contiguity_penalization = (Cost_contiguity_penalization * Q_f1(eta1_a, eta2_a) * Q_f2(eta1_a, eta2f_a) * dx
                             + Cost_contiguity_penalization * Q_f2(eta1_a, eta2_a) * Q_f1(eta1_a, eta2f_a) * dx)
    return assemble(contiguity_penalization)

def gradient_cost(eta1_a, eta2_a):
    gradient_penalization = Cost_gradient_penalization * inner(grad(eta2_a), grad(eta2_a)) * eta1_a**2 * dx
    return assemble(gradient_penalization)

def velocity_penalization_cost(u_a, eta1_a):
    velocity_penalization = Cost_velocity * (1.0 - eta1_a) * inner(u_a, u_a) * dx
    return assemble(velocity_penalization)

def intermediate_velocity_penalization_cost(u_a, eta2_a):
    intermediate_velocity_penalization = Cost_intermediate_velocity * (1.0 - eta2_a) * eta2_a * inner(u_a, u_a) * dx
    return assemble(intermediate_velocity_penalization)

def define_cost_fct(eta1_a, eta2_a, u_a, p_a, T_a, eta2f_a, n_a):
    cost = combined_cost(eta1_a, eta2_a, u_a, p_a, T_a, n_a)
    contiguity_penalization = penalization_cost(eta1_a, eta2_a, eta2f_a)
    velocity_penalization = velocity_penalization_cost(u_a, eta1_a)
    intermediate_velocity_penalization = intermediate_velocity_penalization_cost(u_a, eta2_a)
    return cost + contiguity_penalization + velocity_penalization + intermediate_velocity_penalization# + gradient_penalization

def save_and_compute_cost(eta1_opt, eta2_opt, iteration, save = True):
    UPTQ_opt = forward(eta1_opt, eta2_opt, False)
    (u_opt, p_opt, T_opt, eta2f_opt) = UPTQ_opt.split(True)

    if save:
        eta1_opt.rename("eta1", "label")
        eta2_opt.rename("eta2", "label")
        u_opt.rename("velocity", "label")
        p_opt.rename("pressure", "label")
        T_opt.rename("temperature", "label")
        eta2f_opt.rename("eta2_filtered", "label")

        if iteration == -1:
            str_suffix = "_initial.xdmf"
        elif iteration < len(data.b_alpha) - 1:
            str_suffix = "_guess_" + str(iteration)+".xdmf"
        else:
            str_suffix = "_solution.xdmf"

        eta1_opt_xdmf = XDMFFile(MPI.comm_world, data.output_folder + "/eta1" + str_suffix)
        eta2_opt_xdmf = XDMFFile(MPI.comm_world, data.output_folder + "/eta2" + str_suffix)
        u_opt_xdmf = XDMFFile(MPI.comm_world, data.output_folder + "/velocity" + str_suffix)
        p_opt_xdmf = XDMFFile(MPI.comm_world, data.output_folder + "/pressure" + str_suffix)
        T_opt_xdmf = XDMFFile(MPI.comm_world, data.output_folder + "/temperature" + str_suffix)
        eta2f_opt_xdmf = XDMFFile(MPI.comm_world, data.output_folder + "/eta2_filtered" + str_suffix)

        eta1_opt_xdmf.write(eta1_opt)
        eta2_opt_xdmf.write(eta2_opt)
        u_opt_xdmf.write(u_opt)
        p_opt_xdmf.write(p_opt)
        T_opt_xdmf.write(T_opt)
        eta2f_opt_xdmf.write(eta2f_opt)

    #### Compute the cost
    costvec = {}
    costvec["tot"] = define_cost_fct(eta1_opt, eta2_opt, u_opt, p_opt, T_opt, eta2f_opt, n)
    costvec["val"] = combined_cost(eta1_opt, eta2_opt, u_opt, p_opt, T_opt, n)
    costvec["pre"] = pressure_cost(eta1_opt, eta2_opt, u_opt, p_opt, n)
    costvec["hea"] = thermal_cost(eta1_opt, eta2_opt, u_opt, p_opt, T_opt, n)
    costvec["penalization"] = penalization_cost(eta1_opt, eta2_opt, eta2f_opt)
    costvec["velocity"] = velocity_penalization_cost(u_opt, eta1_opt)
    costvec["intermediate_velocity"] = intermediate_velocity_penalization_cost(u_opt, eta2_opt)
    avg_temperatures = avg_temp(T_opt)
    return costvec, avg_temperatures

## Function solving the optimization problem for a given precision level, defined by "iteration"
def solve_optimization(eta1_opt, eta2_opt, iteration):
    """This function solves, for a given initial condition gamma_opt, one iteration of the optimization."""
    #### Assigning constant values and resetting tape
    if iteration >= 0:
        b_alpha.assign(data.b_alpha[iteration])
        b_k.assign(data.b_k[iteration])
        eta1.assign(eta1_opt)
        eta2.assign(eta2_opt)
        set_working_tape(Tape())
    else:
        raise Exception("iteration must be an integer >= 0")
    #### Setting path and names
    controls_eta1, controls_eta2, eta1_viz, eta2_viz, vel_viz, pre_viz, tem_viz, eta2f_viz, out_file = set_path_and_names(iteration)

    UPTQ = forward(eta1, eta2, True)
    (u, p, T, eta2f) = split(UPTQ)

    #### Call Back function: used to save each step
    def simple_eval_cb(j, eta1e2):
        eta1 = eta1e2[0]
        eta2 = eta1e2[1]
        eta1_viz.assign(eta1)
        eta2_viz.assign(eta2)
        controls_eta1 << eta1_viz
        controls_eta2 << eta2_viz
        print("******** J_tot = ", j)


    #### Defining the optimization problem
    J = define_cost_fct(eta1, eta2, u, p, T, eta2f, n)
    m = [Control(eta1), Control(eta2)]
    Jhat = ReducedFunctional(J, m, eval_cb_post = simple_eval_cb)

    ######## Constraints
    class VolumeConstraints_combined(InequalityConstraint):
        """A class that enforces the volume constraint g_i(eta1, eta2) >= 0."""
        # g_1(eta1, eta2) = (phi_f1 - eta1 * (1 - eta2)) * dx
        # g_2(eta1, eta2) = (phi_f2 - eta1 * eta2) * dx

        def __init__(self, phi_f1, phi_f2):
            self.phi_f1 = float(phi_f1)
            self.phi_f2 = float(phi_f2)
            self.smass = assemble(TestFunction(meshes.A) * Constant(1) * dx)
            self.tmpvec_eta1 = Function(meshes.A)
            self.tmpvec_eta2 = Function(meshes.A)
            self.vector_ones = np.ones(len(self.tmpvec_eta2.vector()))

        def compute_integral_f1(self, eta1, eta2):
            return assemble(eta1 * (Constant(1) - eta2) * dx) / data.area_adim

        def compute_integral_f2(self, eta1, eta2):
            return assemble(eta1 * eta2 * dx) / data.area_adim

        def function(self, m):
            from pyadjoint.reduced_functional_numpy import set_local
            len_m = int(len(m) / 2)
            set_local(self.tmpvec_eta1, m[0:len_m])
            set_local(self.tmpvec_eta2, m[len_m:])
            # Compute the integral of the control over the domain
            integral_f1 = self.compute_integral_f1(self.tmpvec_eta1, self.tmpvec_eta2)
            integral_f2 = self.compute_integral_f2(self.tmpvec_eta1, self.tmpvec_eta2)
            if MPI.rank(MPI.comm_world) == 0:
                print("***** Current control integral_f1: ", integral_f1)
                print("***** Current constraint g_f1: ", str(self.phi_f1 - integral_f1))
                print("***** Current control integral_f2: ", integral_f2)
                print("***** Current constraint g_f2: ", str(self.phi_f2 - integral_f2))

            return [(self.phi_f1 - integral_f1), (self.phi_f2 - integral_f2)]

        def jacobian(self, m):
            len_m = len(m)

            from pyadjoint.reduced_functional_numpy import set_local
            set_local(self.tmpvec_eta1, m[0:len_m])
            set_local(self.tmpvec_eta2, m[len_m:])

            jacobian_eta1_f1 = assemble(TestFunction(meshes.A) * (self.tmpvec_eta2 - Constant(1)) * dx)
            jacobian_eta2_f1 = assemble(TestFunction(meshes.A) * self.tmpvec_eta1 * dx)
            jacobian_f1 = np.concatenate([jacobian_eta1_f1.get_local(), jacobian_eta2_f1.get_local()])

            jacobian_eta1_f2 = assemble( - TestFunction(meshes.A) * self.tmpvec_eta2 * dx)
            jacobian_eta2_f2 = assemble( - TestFunction(meshes.A) * self.tmpvec_eta1 * dx)
            jacobian_f2 = np.concatenate([jacobian_eta1_f2.get_local(), jacobian_eta2_f2.get_local()])

            return [jacobian_f1, jacobian_f2]

        def output_workspace(self):
            return [0.0, 0.0]

        def length(self):
            """Return the number of components in the constraint vector (here, one)."""
            return 2

    ######## problem
    #volume_constraints = [VolumeConstraint_f1(data.phi_f1), VolumeConstraint_f2(data.phi_f2)]
    volume_constraints = VolumeConstraints_combined(data.phi_f1, data.phi_f2)
    problem = MinimizationProblem(Jhat, bounds = [(data.lb, data.ub), (data.lb, data.ub)], constraints = volume_constraints)
    parameters = {'maximum_iterations': data.iterations[iteration], "output_file": data.output_log_file_name}
    solver = IPOPTSolver(problem, parameters = parameters)

    #### Solve problem
    eta1e2_opt = solver.solve()
    return eta1e2_opt

def avg_temp(T_opt):
    avg_T_cold_in = assemble(T_opt * ds(1)) / size_channel_cold_in
    avg_T_hot_in = assemble(T_opt * ds(2)) / size_channel_hot_in
    avg_T_cold_out = assemble(T_opt * ds(3)) / size_channel_cold_out
    avg_T_hot_out = assemble(T_opt * ds(4)) / size_channel_hot_out

    return [avg_T_cold_in, avg_T_hot_in, avg_T_cold_out, avg_T_hot_out]

# Clear path function
def clear_output():
    if MPI.rank(MPI.comm_world) == 0:
        print(os.getcwd())
        current_folder = os.getcwd()
        real_output_folder = os.getcwd() + '/' + data.output_folder
        print(real_output_folder)
        if(os.path.isdir(real_output_folder)):
            shutil.rmtree(real_output_folder)
        os.mkdir(real_output_folder)
        os.mkdir(real_output_folder + "/" + data.to_be_loaded_folder)

##############################################################################################

## Clear path
clear_output()

## Save mesh
meshFile = HDF5File(MPI.comm_world, data.output_to_be_loaded_folder + "/mesh.h5", "w")
meshFile.write(meshes.mesh, "/mesh")
meshFile.close()

## Constants definition
b_alpha = Constant(data.b_alpha[0]) # Convexity param. for inverse permeability
b_k = Constant(data.b_k[0]) # Convexity param. for thermal conductivity
Alpha_penalization = Constant(data.alpha_penalizations[0] * data.U_bar / data.NS_to_adim)
Cost_velocity = Constant(data.velocity_penalizations[0])

mu_f = Constant(data.mu_f)          # Dynamic viscosity of the fluid
rho_f = Constant(data.rho_f)        # Density of the fluid
nu_f = Constant(data.mu_f / data.rho_f)       # Cynematic viscosity of the fluid

max_fluid1_frac = Constant(data.phi_f1)  # maximal fluid fraction
max_fluid2_frac = Constant(data.phi_f2)  # maximal fluid fraction



#dx = dx(subdomain_data=meshes.domains)
ds = ds(subdomain_data = meshes.sub_domains)
n = FacetNormal(meshes.mesh)

## Coefficients for the adimensionalized equations
Rem1 = Constant(1/data.Re)  # Re^-1
Heat_adim_coeff = Constant(data.heat_to_adim)   # adimensionalization of the heat equation

Thermal_k_fluid = Constant(data.k_f)    # Thermal conductivity of the fluid
C_k = Constant(data.k_f / data.k_s)     # ratio of the thermal conductivities

Cost_contiguity_penalization = Constant(data.contiguity_penalization)   # weight of the contiguity penalization
Cost_gradient_penalization = Constant(data.gradient_penalization)  # weight for the gradient penalization term in the cost function
Radius_filtering_squared_penalization = Constant(data.r_filtering_eta2**2) # radius for Helmholtz filtering in the contiguity penalization
Cost_weight = Constant(data.w_cost_weight)   # weight to balance pressure and heat costs
Complement_Cost_weight = Constant(1.0 - data.w_cost_weight)
Scale_pressure_cost = Constant(data.scale_pressure_cost)    # max(J_pressure) - min(J_pressure)
Scale_thermal_cost = Constant(data.scale_thermal_cost)    # max(J_thermal) - min(J_thermal)
Cost_constant_pressure = Constant(data.cost_constant_pressure)
Cost_constant_thermal = Constant(data.cost_constant_thermal)
Cost_intermediate_velocity = Constant(data.cost_intermediate_velocity)
Area_adim = Constant(data.area_adim)
Const_adim_cost_pressure = Constant(data.const_adim_cost_pressure)
Const_adim_cost_thermal = Constant(data.const_adim_cost_thermal)
size_channel_cold_in = assemble(Constant(1.0) * ds(1, domain = meshes.mesh))
size_channel_hot_in = assemble(Constant(1.0) * ds(2, domain = meshes.mesh))
size_channel_cold_out = assemble(Constant(1.0) * ds(3, domain = meshes.mesh))
size_channel_hot_out = assemble(Constant(1.0) * ds(4, domain = meshes.mesh))



## Initialization of a few vectors
J_tots = []
J_vals = []
J_pres = []
J_thes = []
J_penalizations = []
J_velocities = []
J_intermediate_vels = []
steps_vector = []
avg_T_cold_in = []
avg_T_hot_in = []
avg_T_cold_out = []
avg_T_hot_out = []

## Print the coeffs of the equations:
print("----------------------------------------------------")
print("Coeffs of the equations:")
print("Re = " + str(data.Re))
print("Heat_adim_coeff = ", str(data.heat_to_adim))
print("Scale_pressure_cost = ", str(data.scale_pressure_cost))
print("Scale_thermal_cost = ", str(data.scale_thermal_cost))
print("----------------------------------------------------")

## Computing the initial state
eta1_0 = meshes.eta1_initial
eta2_0 = meshes.eta2_initial
eta1_opt = eta1_0
eta2_opt = eta2_0
eta1 = eta1_0
eta2 = eta2_0



##########################
#  Solving the optimization problem and saving results (also for the initial conditions!)
for j in range(0, len(data.b_alpha) + 1):
    i = j - 1
    if i > 0:
        update_constants(i)
    if i >= 0:
        eta1e2_opt = solve_optimization(eta1_opt, eta2_opt, i)
        eta1_opt = eta1e2_opt[0]
        eta2_opt = eta1e2_opt[1]

    if i == -1:
        save_for_output(eta1_opt, eta2_opt, "init")
    elif i < (len(data.b_alpha) - 1):
        save_for_output(eta1_opt, eta2_opt, str(i))
    else:
        save_for_output(eta1_opt, eta2_opt, "final")

    costvec, avg_temperatures = save_and_compute_cost(eta1_opt, eta2_opt, i, True)
    steps_vector.append(i)
    J_tots.append(costvec["tot"])
    J_vals.append(costvec["val"])
    J_pres.append(costvec["pre"])# * data.const_adim_cost_pressure)
    J_thes.append(costvec["hea"])# * data.const_adim_cost_thermal)
    J_penalizations.append(costvec["penalization"])
    J_velocities.append(costvec["velocity"])
    J_intermediate_vels.append(costvec["intermediate_velocity"])
    avg_T_cold_in.append(avg_temperatures[0])
    avg_T_hot_in.append(avg_temperatures[1])
    avg_T_cold_out.append(avg_temperatures[2])
    avg_T_hot_out.append(avg_temperatures[3])
##########################


# Print results
print("------------------ RESULTS--------------------")
print("len(steps_vector) = " + str(len(steps_vector)))

for ii in range(len(steps_vector)):
    if ii == 0:
        init_str = "++++++++  Initial"
    else:
        init_str = "++++++++ Step " + str(ii)

    print(init_str, " : J_tot = ", J_tots[ii], ", J_val = ", J_vals[ii], ", J_pressure = ", J_pres[ii], ", J_thermal = ", J_thes[ii])
    print("              ... : J_penalization = ", J_penalizations[ii], ", J_velocity = ", J_velocities[ii], ", J_intermediate_vel = ", J_intermediate_vels[ii])
    print("              ... : T_cold_in = ",avg_T_cold_in[ii], ", T_hot_in = ",avg_T_hot_in[ii],", T_cold_out = ",avg_T_cold_out[ii],", T_hot_out = ",avg_T_hot_out[ii], "\n")

print("------------------END--------------------")


### Test
