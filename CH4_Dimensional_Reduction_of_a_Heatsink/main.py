# In this script we consider a model based on the sctipt heatsink_7, but we
# try to add the non-linear component of Navier-Stokes.



## Import libraries
from dolfin import *                    # Library for the optimization
from dolfin_adjoint import *            # Library for the optimization
import pyipopt                          # Library for the optimization
import params as data                   # numerical parameters and physical constant values
import os, shutil                       # tools to work with folders
import meshes as meshes               # meshes
import numpy as np
from ufl import tanh

## Interpolation Functions
def I_alpha(gamma):     # I_alpha(0) = 1,   I_alpha(1) = 0
    """Interpolation for the inverse permeability"""
    return Alpha_penalization * (1.0 - gamma) / (1.0 + b_alpha * gamma)

def I_k(gamma):     # I_k(0) = k_s/k_f,     I_k(1) = 1
    """ Interpolation for the thermal conductivity"""
    return (gamma * (data.C_k * (1.0 + b_k) - 1.0) + 1.0) / (data.C_k * (1.0 + b_k * gamma))

def I_h(gamma):     # I_h(0) = h_s/h_f,     I_h(1) = 1
    """ Interpolation for the heat transfer coefficient"""
    return (gamma * (data.C_h * (1.0 + b_h) - 1.0) + 1.0) / (data.C_h * (1.0 + b_h * gamma))




## Setting output path and file names
def set_path_and_names(step):
    if step in range(0,len(data.b_alpha) - 1):
        controls_d = File(data.controls_d+"_"+str(step)+"_guess.pvd")
    elif step == (len(data.b_alpha) - 1):
        controls_d = File(data.controls_d+"_solution"+".pvd")
    elif step == -1:
        controls_d = File(data.controls_d+"_original.pvd")
    else:
        raise Exception("No further precision step defined")

    gamma_d_viz = Function(meshes.A, name = data.gamma_d_viz_name)
    out_file = open(data.output_log_file_name, 'w+')

    return  controls_d, gamma_d_viz, out_file

def save_for_output(gamma, str_iter):
    fFile = HDF5File(MPI.comm_world, data.output_to_be_loaded_folder + "/control_" + str_iter + ".h5", "w")
    fFile.write(gamma,"/gamma")
    fFile.close()
##############################################################################################

## Funzione forward
def forward(gamma, annotation):
    """Solve the forward problem for a given fluid distribution gamma(x)."""
    ### Fluid Stokes problem

    bc_v_symm = DirichletBC(meshes.UPTT.sub(0).sub(1), Constant(0.0), meshes.symm_bound)
    bc_T_f_in = DirichletBC(meshes.UPTT.sub(2), Constant(data.T_in), meshes.inflow_bound)
    bc_T_s_Omf = DirichletBC(meshes.UPTT.sub(3), Constant(50.0), meshes.omega_f)

    bc_complete = [bc_v_symm, bc_T_f_in, bc_T_s_Omf]

    (v, q, S_f, S_s) = TestFunctions(meshes.UPTT)
    uptt0 = Function(meshes.UPTT)
    u, p, T_f, T_s = split(uptt0)

    F_complete = (I_alpha(gamma) * inner(u,v) * dx(2)
                    + inner(grad(u) * u, v) * dx
                    + Rem1 * inner(grad(u), grad(v)) * dx
                    - inner(div(v), p) * dx
                    + inner(Border_strain_inflow, v) * ds(1)
                    + inner(Border_strain_outflow, v) * ds(2)
                    - inner(div(u), q) * dx
                    + Transp_coeff_F2 * gamma * inner(u, grad(T_f)) * S_f * dx
                    + Diff_coeff_F2_f * I_k(gamma) * inner(grad(T_f), grad(S_f)) * dx
                    - Heat_coupling_coeff_F2_f * I_h(gamma) * T_s * S_f * dx(2)
                    + Heat_coupling_coeff_F2_f * I_h(gamma) * T_f * S_f * dx(2)
                    + Diff_coeff_F2_s * inner(grad(T_s), grad(S_s)) * dx(2)
                    + Heat_coupling_coeff_F2_s * I_h(gamma) * T_s * S_s * dx(2)
                    - Heat_coupling_coeff_F2_s * I_h(gamma) * T_f * S_s * dx(2)
                    - Heat_generation_F2 * S_s * dx(2))


    solve(F_complete == 0, uptt0, bcs = bc_complete, annotate = annotation)

    return uptt0

## Cost function
def define_cost_fct(gamma1, u1, p1, T_f1, T_s1, i):
    cost = T_s1 * dx(2)
    return assemble(cost)

def compute_Reynolds(u0):
    integral_inlet_velocity_adim = assemble(inner(u0, versore_x) * ds(1))
    avg_inlet_velocity = integral_inlet_velocity_adim * data.ubar / data.length_y_adim
    return avg_inlet_velocity * data.length_y * data.rho_f / data.mu_f

def save_and_compute_cost(gamma_opt, iteration):
    uptt_opt = forward(gamma_opt, False)
    (u_opt, p_opt, T_f_opt, T_s_tot_opt) = uptt_opt.split(True)
    T_s_d_opt = interpolate(T_s_tot_opt, meshes.A_d)

    gamma_opt.rename("control", "label")
    u_opt.rename("velocity", "label")
    p_opt.rename("pressure", "label")
    T_f_opt.rename("fluid temperature", "label")
    T_s_d_opt.rename("base temperature", "label")
    if iteration == -1:
        gamma_opt_xdmf = XDMFFile(MPI.comm_world, "output/control_initial.xdmf")
        u_opt_xdmf = XDMFFile(MPI.comm_world, "output/velocity_initial.xdmf")
        p_opt_xdmf = XDMFFile(MPI.comm_world, "output/pressure_initial.xdmf")
        T_f_opt_xdmf = XDMFFile(MPI.comm_world, "output/temperature_fluid_initial.xdmf")
        T_s_d_opt_xdmf = XDMFFile(MPI.comm_world, "output/temperature_base_initial.xdmf")
    elif iteration < len(data.b_alpha) - 1:
        gamma_opt_xdmf = XDMFFile(MPI.comm_world, "output/control_guess_"+str(iteration)+".xdmf")
        u_opt_xdmf = XDMFFile(MPI.comm_world, "output/velocity_guess_"+str(iteration)+".xdmf")
        p_opt_xdmf = XDMFFile(MPI.comm_world, "output/pressure_guess_"+str(iteration)+".xdmf")
        T_f_opt_xdmf = XDMFFile(MPI.comm_world, "output/temperature_fluid_guess_"+str(iteration)+".xdmf")
        T_s_d_opt_xdmf = XDMFFile(MPI.comm_world, "output/temperature_base_guess_"+str(iteration)+".xdmf")
    else:
        gamma_opt_xdmf = XDMFFile(MPI.comm_world, "output/control_solution.xdmf")
        u_opt_xdmf = XDMFFile(MPI.comm_world, "output/velocity_solution.xdmf")
        p_opt_xdmf = XDMFFile(MPI.comm_world, "output/pressure_solution.xdmf")
        T_f_opt_xdmf = XDMFFile(MPI.comm_world, "output/temperature_fluid_solution.xdmf")
        T_s_d_opt_xdmf = XDMFFile(MPI.comm_world, "output/temperature_base_solution.xdmf")

    gamma_opt_xdmf.write(gamma_opt)
    u_opt_xdmf.write(u_opt)
    p_opt_xdmf.write(p_opt)
    T_f_opt_xdmf.write(T_f_opt)
    T_s_d_opt_xdmf.write(T_s_d_opt)

    #### Compute the costT_s_tot_opt
    J_val = define_cost_fct(gamma_opt, u_opt, p_opt, T_f_opt, T_s_tot_opt, iteration)
    Re = compute_Reynolds(u_opt)
    return J_val, Re

## Function solving the optimization problem for a given precision level, defined by "iteration"
def solve_optimization(iteration, gamma_opt):
    """This function solves, for a given initial condition gamma_opt, one iteration of the optimization."""
    #### Assigning constant values and resetting tape
    if iteration >= 0:
        b_alpha.assign(data.b_alpha[iteration])
        b_k.assign(data.b_k[iteration])
        b_h.assign(data.b_h[iteration])
        gamma.assign(gamma_opt)
        set_working_tape(Tape())
    else:
        raise Exception("iteration must be an integer >= 0")

    #### Setting path and names
    controls_d, gamma_d_viz, out_file = set_path_and_names(iteration)

    (upTT) = forward(gamma, True)
    (u, p, T_f, T_s) = split(upTT)

    #### Call Back function: used to save each step
    def simple_eval_cb(j, gamma):
        ### These projections are used to save olny the A_d part of the mesh.
        gamma_d_p = Function(meshes.A, name = data.gamma_d_viz_name)
        gamma_d_p = interpolate(gamma, meshes.A_d)
        gamma_d_viz.assign(gamma_d_p)
        ####### gamma_d_viz.assign(gamma) NO!
        controls_d << gamma_d_viz


    #### Defining the optimization problem
    J = define_cost_fct(gamma, u, p, T_f, T_s, iteration)
    m = Control(gamma)
    Jhat = ReducedFunctional(J, m, eval_cb_post=simple_eval_cb)
    ######## Constraints
    #volume_constraint = UFLInequalityConstraint((+max_fluid_frac - 1.0 + gamma)*dx, m)
    omega_f_constraint = UFLInequalityConstraint((-1.0 + gamma)*dx(1), m)

    optim_contraints = [omega_f_constraint]
    ######## problem
    problem = MinimizationProblem(Jhat, bounds = (data.lb, data.ub), constraints = optim_contraints)
    parameters = {'maximum_iterations': data.iterations[iteration], "output_file": data.output_log_file_name}
    solver = IPOPTSolver(problem, parameters = parameters)


    #### Solve problem
    gamma_opt = solver.solve()

    #### Return the computed optimal control
    return gamma_opt

def extension_to_Omega_tot(gamma_pass):
    """ This function receives a scalar CG Function defined on Omega_d and returns a
        CG function on Omega_tot equal to the first in Omega_d and zero in Omega_f."""

    interpolated_gamma = Function(meshes.A, name = data.gamma_viz_name)
    expression_gamma = Expression("g", g = gamma_pass, degree = 2)
    class Extend_gamma(UserExpression):
        def eval(self, values, x):
            if ((x[0] >= data.length_x_f_adim) and (x[0] <= data.length_x_f_adim + data.length_x_d_adim)):
                values[0] = expression_gamma(x)
            else:
                values[0] = 1.0

        def value_shape(self):
            return (1,)

    ic = Extend_gamma(element = meshes.A.ufl_element())
    interpolated_gamma.assign(interpolate(ic, meshes.A))
    return interpolated_gamma


def helmholtz_filter(gamma):
    expression_gamma = Expression("g", g = gamma, degree = 2)
    gamma_tilde = TrialFunction(meshes.A_d)
    eta_tilde = TestFunction(meshes.A_d)
    filtered_gamma = Function(meshes.A_d)

    bc_gamma_Omd_internal = DirichletBC(meshes.A_d, Constant(1.0), meshes.internal_d_bound)
    bc_gamma = [bc_gamma_Omd_internal]

    FHelmholtz = (Filter_radius_squared_adim * inner(grad(gamma_tilde), grad(eta_tilde)) * dx_d
                    + gamma_tilde * eta_tilde * dx_d
                    - expression_gamma * eta_tilde * dx_d)

    solve(lhs(FHelmholtz) == rhs(FHelmholtz), filtered_gamma, bcs=bc_gamma, annotate = False)
    return extension_to_Omega_tot(filtered_gamma)

def smoothed_heaviside(gamma_pass):
    new_gamma = Function(meshes.A, name = data.gamma_viz_name)
    expression_gamma = Expression("g", g = gamma_pass, degree = 2)
    class Heaviside(UserExpression):
        def eval(self, values, x):
            values[0] = (heaviside_tanh_eta_beta + np.tanh(heaviside_beta * (expression_gamma(x) - data.heaviside_eta))) / denominator_heaviside

        def value_shape(self):
            return (1,)

    ic = Heaviside(element = meshes.A.ufl_element())
    new_gamma.assign(interpolate(ic, meshes.A))
    return new_gamma

def update_constants(iteration):
    b_alpha.assign(data.b_alpha[iteration])
    b_k.assign(data.b_k[iteration])
    b_h.assign(data.b_h[iteration])
    heaviside_beta = data.heaviside_beta[iteration]
    denominator_heaviside = np.tanh(data.heaviside_beta[iteration] * data.heaviside_eta) + np.tanh(data.heaviside_beta[iteration] * (1.0 - data.heaviside_eta))
    heaviside_tanh_eta_beta = np.tanh(data.heaviside_beta[iteration] * data.heaviside_eta)

##############################################################################################

## Clear path
data.clear_output()

## Save mesh
meshFile = HDF5File(MPI.comm_world, data.output_to_be_loaded_folder + "/mesh.h5", "w")
meshFile.write(meshes.mesh_tot, "/mesh")
meshFile.close()

## Constants definition
b_alpha = Constant(data.b_alpha[0]) # Convexity param. for inverse permeability
b_k = Constant(data.b_k[0])         # Convexity param. for thermal conuctivity
b_h = Constant(data.b_h[0])         # Convexity param. for heat transfer coefficient

mu_f = Constant(data.mu_f)          # Dynamic viscosity of the fluid
rho_f = Constant(data.rho_f)        # Density of the fluid
nu_f = Constant(mu_f / rho_f)       # Cynematic viscosity of the fluid

max_fluid_frac = Constant(data.max_fluid_frac)  # maximal fluid fraction
min_fluid_frac = Constant(data.min_fluid_frac)  # minimal fluid fraction

versore_x = Constant((1.0, 0.0))

dx = dx(subdomain_data=meshes.domains)
ds = ds(subdomain_data=meshes.sub_domains)
dx_d = dx(subdomain_data=meshes.domains_d)
n = FacetNormal(meshes.mesh_tot)

## Coefficients for the adimensionalized equations
Rem1 = Constant(1/data.Re)  # Re^-1
Alpha_penalization = Constant(data.alpha_s / data.Stokes_pressure_coeff)     # Adimensionalization of the penalization coeff

Transp_coeff_F2 = Constant(data.Transp_coeff_F2 / data.Diff_coeff_F2_f)    # Transport coefficient for heat eq [-]
Diff_coeff_F2_f = Constant(data.Diff_coeff_F2_f / data.Diff_coeff_F2_f)    # diffusion coefficient for T_f [-]
Heat_coupling_coeff_F2_f = Constant(data.Heat_coupling_coeff_F2_f / data.Diff_coeff_F2_f)  # coefficient for the thermal coupling for T_f [-]

Diff_coeff_F2_s = Constant(data.Diff_coeff_F2_s / data.Diff_coeff_F2_s)   # diffusion coefficient for T_s [-]
Heat_coupling_coeff_F2_s = Constant(data.Heat_coupling_coeff_F2_s / data.Diff_coeff_F2_s)      # coefficient for the thermal coupling for T_s [-]
Heat_generation_F2 = Constant(data.Heat_generation_F2 / data.Diff_coeff_F2_s)     # known, constant heat generation term [-]

Border_strain_inflow = Constant((-data.p_in/data.p_in, 0.0))
Border_strain_outflow = Constant((data.p_out/data.p_in, 0.0))

Filter_radius_squared_adim = Constant(data.r_filter ** 2 / data.Lbar**2)
heaviside_beta = data.heaviside_beta[0]
Heaviside_eta = Constant(data.heaviside_eta)
denominator_heaviside = np.tanh(data.heaviside_beta[0] * data.heaviside_eta) + np.tanh(data.heaviside_beta[0] * (1.0 - data.heaviside_eta))
heaviside_tanh_eta_beta = np.tanh(data.heaviside_beta[0] * data.heaviside_eta)


## Initialization of a few vectors
J_vals = []
T_s_avgs = []
steps_vector = []
Res = []

## Print the coeffs of the equations:
print("----------------------------------------------------")
print("Coeffs of the equations:")
print("Re = " + str(data.Re))
print("Transp_coeff_F2 = " + str(data.Transp_coeff_F2 / data.Diff_coeff_F2_f))
print("Diff_coeff_F2_f = " + str(data.Diff_coeff_F2_f / data.Diff_coeff_F2_f))
print("Heat_coupling_coeff_F2_f = " + str(data.Heat_coupling_coeff_F2_f / data.Diff_coeff_F2_f))
print("Diff_coeff_F2_s = " + str(data.Diff_coeff_F2_s / data.Diff_coeff_F2_s))
print("Heat_coupling_coeff_F2_s = " + str(data.Heat_coupling_coeff_F2_s / data.Diff_coeff_F2_s))
print("Heat_generation_F2 = " + str(data.Heat_generation_F2 / data.Diff_coeff_F2_s))
print("Filter_radius_squared_adim = " + str(data.r_filter ** 2 / data.Lbar**2))
print("----------------------------------------------------")

## Computing the initial state
gamma0 = meshes.initial_condition
gamma_opt = gamma0
gamma = gamma0
gamma_opt = helmholtz_filter(gamma_opt)
gamma_opt = smoothed_heaviside(gamma_opt)
save_for_output(gamma_opt, "init")
J_val, Re = save_and_compute_cost(gamma_opt, -1)
steps_vector.append(-1)
J_vals.append(J_val)
T_s_avgs.append(J_val / data.area_d_adim)
Res.append(Re)

## Solving the optimization problem
for i in range(0, len(data.b_alpha)):
    if i > 0:
        update_constants(i)
    gamma_opt = solve_optimization(i, gamma_opt)
    gamma_opt = helmholtz_filter(gamma_opt)
    gamma_opt = smoothed_heaviside(gamma_opt)
    if i < (len(data.b_alpha) - 1):
        save_for_output(gamma_opt, str(i))
    else:
        save_for_output(gamma_opt, "final")
    J_val, Re = save_and_compute_cost(gamma_opt, i)
    steps_vector.append(i)
    J_vals.append(J_val)
    T_s_avgs.append(J_val / data.area_d_adim)
    Res.append(Re)


# Print results
print("------------------SCALED RESULTS--------------------")
print("len(steps_vector) = " + str(len(steps_vector)))
print("+++++++  Initial : J = ", J_vals[0], "T_out_avg = ", T_s_avgs[0], \
    ", R_th = ", (T_s_avgs[0] - data.T_in)/data.dQ_prod, ", Re = ", Res[0])

for ii in range(len(steps_vector)-1):
    print("++++++++ Step ",steps_vector[ii+1]," : J = ", J_vals[ii+1], "T_out_avg = ", T_s_avgs[ii+1], \
        ", R_th = ", (T_s_avgs[ii+1] - data.T_in)/data.dQ_prod, ", Re = ", Res[ii+1])

print("------------------END--------------------")
