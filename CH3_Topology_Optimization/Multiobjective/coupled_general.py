from dolfin import *
from dolfin_adjoint import *
import pyipopt
import params as data
import os, shutil
import coupled_pres as pres
import coupled_temp as temp

# function alpha and kappa
qd = Constant(data.qds[0])
qr = Constant(data.qrs[0])
mu = Constant(data.mu)
rho_fluid = Constant(data.rho_fluid)
nu = Constant(mu / rho_fluid)
V = Constant(data.V)

def alpha(eta):
    """Inverse permeability as a function of eta, equation (40)"""
    return data.alphas + (data.alphaf - data.alphas)\
     * eta * (1 + qd) / (eta + qd)


def kappa(eta):
    return data.ks + (data.kf - data.ks) * eta * (1 + qr) / (eta + qr)

def D(eta):
    return kappa(eta)/(data.rho_fluid * data.Cp * data.ubar)


#####################################################################################
def execute_optimization(define_cost_fct):
    # clearing the output folder
    def clear_output():
        print(os.getcwd())
        current_folder = os.getcwd()
        output_folder = os.getcwd() + '/' + data.output_folder
        print(output_folder)
        if(os.path.isdir(output_folder)):
            shutil.rmtree(output_folder)
        os.mkdir('output')

    # boundary conditions
    class InflowOutflow(UserExpression):
        def eval(self, values, x):
            values[1] = 0.0
            values[0] = 0.0
            gbar = 1.5# maximum inlet velocity (adimensional)

            if (x[0] <= DOLFIN_EPS) and (x[1] > 2.0/5.0) and (x[1] < 3.0/5.0):
                t = (x[1] - 0.5) * 10.0
                values[0] = gbar * (1 - t**2)

        # tells that values are 2D vectors (they are 1D by default)
        def value_shape(self):
            return (2,)

    # PB solution with a given distribution of the material
    def forward(eta):
        """Solve the forward problem for a given fluid distribution eta(x)."""
        (u, p) = TrialFunctions(UP)
        (v, q) = TestFunctions(UP)
        up0 = Function(UP)
        bc_v = DirichletBC(UP.sub(0), InflowOutflow(degree=1), "on_boundary")
        bc_p = DirichletBC(UP.sub(1), Constant(0),
            "on_boundary && x[0] >= 1 - DOLFIN_EPS && x[1] > 2.0/5.0 && x[1] < 3.0/5.0")
        bc_vp = [bc_v, bc_p]

        F1stokes = (1.0 / (data.rho_fluid * data.ubar) * alpha(eta) * inner(u, v) * dx
                    + data.nu / data.ubar * inner(grad(u), grad(v)) * dx
                    - inner(div(v), p) * dx
                    - inner(div(u), q) * dx)
        solve(lhs(F1stokes) == rhs(F1stokes), up0, bcs=bc_vp)
        (u1, p1) = up0.split(True)

        # Fixed point for Navier - Stokes
        if data.NavierStokes == True :
            counter = 0
            err_rel = 1
            while (counter < data.iter_max_NS and err_rel > data.tol_NS):
                F1 = (1.0 / (data.rho_fluid * data.ubar) * alpha(eta) * inner(u, v) * dx
                            + data.nu / data.ubar * inner(grad(u), grad(v)) * dx
                            - inner(div(v), p) * dx
                            - inner(div(u), q) * dx
                            + inner(dot(grad(u), u1), v) * dx)
                solve(lhs(F1) == rhs(F1), up0, bcs=bc_vp)
                (u2, p2) = up0.split(True)
                err_abs = assemble(dot(u1 - u2, u1 - u2) * dx)
                norm_u2 = assemble(dot(u2, u2) * dx)
                err_rel = err_abs / norm_u2
                u1 = u2
                counter += 1
                print("+++++++ counter = ",counter,", err_rel = ", err_rel)
        ##----------
        T = TrialFunction(A)
        S = TestFunction(A)
        T2 = Function(A)

        #F2 = (data.Pe * inner(u1, grad(T)) * S * dx +
        #    kappa(eta) * inner(grad(T), grad(S)) * dx)

        F2 = (inner(u1, grad(T)) * S * dx
                + D(eta) * inner(grad(T), grad(S)) * dx)

        bc_T_in = DirichletBC(A, Constant(data.T_in),
            "on_boundary && x[0] <= DOLFIN_EPS && x[1] > 2.0/5.0 && x[1] < 3.0/5.0")
        bc_T_updown = DirichletBC(A, Constant(data.T_updown),
            "on_boundary && (x[1] <= DOLFIN_EPS || x[1] >= 1.0 - DOLFIN_EPS)")
        bc = [bc_T_in, bc_T_updown]

        solve(lhs(F2) == rhs(F2), T2, bcs=bc)

        return (up0, T2)

    # setting output path and file names
    def set_path_and_names(step):
        if step in range(0,len(data.qrs) - 1):
            controls = File(data.controls+str(step)+"_guess.pvd")
            velocity = File(data.velocity+str(step)+"_guess.pvd")
            pressure = File(data.pressure+str(step)+"_guess.pvd")
            temperature = File(data.temperature+str(step)+"_guess.pvd")
        elif step == (len(data.qrs) - 1):
            controls = File(data.controls+".pvd")
            velocity = File(data.velocity+".pvd")
            pressure = File(data.pressure+".pvd")
            temperature = File(data.temperature+".pvd")
        else:
            raise Exception("No further precision step defined")

        eta_viz = Function(A, name = data.eta_viz_name)
        vel_viz = Function(U, name = data.vel_viz_name)
        pre_viz = Function(A, name = data.pre_viz_name)
        tmp_viz = Function(A, name = data.tmp_viz_name)
        out_file = open(data.output_log_file_name, 'w+')
        return controls, velocity, pressure, temperature,\
         eta_viz, vel_viz, pre_viz, tmp_viz, out_file

    # Volume constraint
    class VolumeConstraint(EqualityConstraint): #InequalityConstraint
        """A class that enforces the volume constraint g(a) = V - a*dx >= 0."""
        def __init__(self, V):
            self.V = float(V)
            self.smass = assemble(TestFunction(A) * Constant(1) * dx)
            self.tmpvec = Function(A)

        def function(self, m):
            print("Evaluting constraint residual")
            self.tmpvec.vector()[:] = m

            # Compute the integral of the control over the domain
            integral = self.smass.inner(self.tmpvec.vector())
            print("Current control integral: ", integral)
            return [self.V - integral]

        def jacobian(self, m):
            print("Computing constraint Jacobian")
            return [-self.smass]

        def output_workspace(self):
            return [0.0]

    # Compute the value of the cost functional
    def compute_J(eta_val, iteration):
        (up_val, T_val)  = forward(eta_val)
        (u_val, p_val) = split(up_val)
        J_val = define_cost_fct(n,eta_val, u_val, p_val, T_val, iteration)
        Jd_val = pres.define_cost_fct(n,eta_val, u_val, p_val, T_val)
        Jr_val = temp.define_cost_fct(n,eta_val, u_val, p_val, T_val)
        return (J_val, Jd_val, Jr_val)


    # A function wrapping the whole problem
    def solve_optimization(iteration, eta_opt, qqd, qqr):
        if iteration >= 0:
            qd.assign(qqd)
            qr.assign(qqr)
            eta.assign(eta_opt)
            set_working_tape(Tape())
        else:
            raise Exception("iteration must be an integer >= 0")

        controls, velocity, pressure, temperature,\
         eta_viz, vel_viz, pre_viz, tmp_viz, out_file = set_path_and_names(iteration)

        (up, T)  = forward(eta)
        (u, p) = split(up)

        # Call Back function: used to save each step
        def eval_cb(j, eta):
            eta_viz.assign(eta)
            (up, T) = forward(eta)
            (u, p) = up.split(True)
            vel_viz.assign(u)
            pre_viz.assign(p)
            tmp_viz.assign(T)
            controls << eta_viz
            velocity << vel_viz
            pressure << pre_viz
            temperature << tmp_viz

        J = define_cost_fct(n,eta, u, p, T, iteration)
        m = Control(eta)
        Jhat = ReducedFunctional(J, m, eval_cb_post=eval_cb)
        # problem
        problem = MinimizationProblem(Jhat, bounds=(data.lb, data.ub), constraints=VolumeConstraint(V))
        parameters = {'maximum_iterations': data.iterations[iteration], "output_file": data.output_log_file_name}
        solver = IPOPTSolver(problem, parameters = parameters)


        # solve problem
        eta_opt = solver.solve()

        # save solution
        if iteration < len(data.qrs) - 1:
            eta_opt_xdmf = XDMFFile(MPI.comm_world, "output/control_solution_guess_"+str(iteration)+".xdmf")
        else:
            eta_opt_xdmf = XDMFFile(MPI.comm_world, "output/control_solution.xdmf")
        eta_opt_xdmf.write(eta_opt)
        return eta_opt

    ###########################
    parameters["std_out_all_processes"] = False

    clear_output()

    #set_function_spaces
    if data.load_mesh == True:
        mesh = Mesh(data.load_name_mesh)
    else:
        mesh = Mesh(RectangleMesh(MPI.comm_world, Point(0.0, 0.0), Point(data.delta, 1.0), data.N, data.N))
    A = FunctionSpace(mesh, "CG", 1)        # control function space
    U = VectorFunctionSpace(mesh, 'CG', 2)
    U_h = VectorElement("CG", mesh.ufl_cell(), 2)
    P_h = FiniteElement("CG", mesh.ufl_cell(), 1)
    UP = FunctionSpace(mesh, MixedElement((U_h, P_h)))
    n = FacetNormal(mesh)
    #define_cost_fct = assemble(define_cost_fct_unassembled)

    etas = []
    prova = []
    if data.load_mesh == True:
        eta0 = Function(A, data.load_name_init)
    else:
        eta0 = interpolate(Constant(float(V)), A)
    eta_opt = eta0
    eta = eta0
    etas.append(eta0)
    prova.append(-1)

    for i in range(0, len(data.qrs)):
        eta_opt1 = solve_optimization(i, eta_opt, data.qds[i], data.qrs[i])
        eta_opt = eta_opt1
        etas.append(eta_opt1)
        prova.append(i)

    # cost evaluation
    etas[0] = interpolate(Constant(float(V)), A)

    J_vals = []
    Jd_vals = []
    Jr_vals = []

    for ie in range(len(etas)):
        (J_val, Jd_val, Jr_val) = compute_J(etas[ie], min(0, prova[ie]))
        J_vals.append(J_val)
        Jd_vals.append(Jd_val)
        Jr_vals.append(Jr_val)

    print("------------------SCALED RESULTS--------------------")
    print("++++++++++++  Initial : Jd = ", Jd_vals[0], "Jr = ", Jr_vals[0])

    for ii in range(len(etas)-1):
        print("+++++++++    Step ",prova[ii+1]," : Jd = ", Jd_vals[ii+1], "Jr = ", Jr_vals[ii+1])


    print("-----------------REALISTIC RESULTS------------------")
    print("++++++++++++  Initial : Jd = ", Jd_vals[0] * data.ubar**3 * data.rho_fluid * data.delta_l,
            "Jr = ", Jr_vals[0] *data.rho_fluid * data.Cp * data.ubar * data.delta_l)
    for ii in range(len(etas)-1):
        print("+++++++++    Step ",prova[ii+1]," : Jd = ", Jd_vals[ii+1]* data.ubar**3 * data.rho_fluid * data.delta_l,
            "Jr = ", Jr_vals[ii+1]*data.rho_fluid * data.Cp * data.ubar * data.delta_l)

    print("----------------------------------------------------")
    File(data.load_name_mesh) << mesh
    File(data.load_name_init) << eta_opt
#
# from dolfin import *
# from dolfin_adjoint import *
# import pyipopt
# import params_coupled as data
# import os, shutil
# import coupled_pres_new2 as pres
# import coupled_temp_new2 as temp
#
# # function alpha and kappa
# qd = Constant(data.qds[0])
# qr = Constant(data.qrs[0])
# mu = Constant(data.mu)
# rho_fluid = Constant(data.rho_fluid)
# nu = Constant(mu / rho_fluid)
# V = Constant(data.V)
#
# def alpha(eta):
#     """Inverse permeability as a function of eta, equation (40)"""
#     return data.alphas + (data.alphaf - data.alphas)\
#      * eta * (1 + qd) / (eta + qd)
#
#
# def kappa(eta):
#     return data.ks + (data.kf - data.ks) * eta * (1 + qr) / (eta + qr)
#
# def D(eta):
#     return kappa(eta)/(data.rho_fluid * data.Cp * data.ubar)
#
#
# #####################################################################################
# def execute_optimization(define_cost_fct):
#     # clearing the output folder
#     def clear_output():
#         print(os.getcwd())
#         current_folder = os.getcwd()
#         output_folder = os.getcwd() + '/' + data.output_folder
#         print(output_folder)
#         if(os.path.isdir(output_folder)):
#             shutil.rmtree(output_folder)
#         os.mkdir('output')
#
#     # boundary conditions
#     class InflowOutflow(UserExpression):
#         def eval(self, values, x):
#             values[1] = 0.0
#             values[0] = 0.0
#             gbar = 1.5# maximum inlet velocity (adimensional)
#
#             if (x[0] <= DOLFIN_EPS) and (x[1] > 2.0/5.0) and (x[1] < 3.0/5.0):
#                 t = (x[1] - 0.5) * 10.0
#                 values[0] = gbar * (1 - t**2)
#
#         # tells that values are 2D vectors (they are 1D by default)
#         def value_shape(self):
#             return (2,)
#
#     # PB solution with a given distribution of the material
#     def forward(eta):
#         """Solve the forward problem for a given fluid distribution eta(x)."""
#         (u, p) = TrialFunctions(UP)
#         (v, q) = TestFunctions(UP)
#         up0 = Function(UP)
#         bc_v = DirichletBC(UP.sub(0), InflowOutflow(degree=1), "on_boundary")
#         bc_p = DirichletBC(UP.sub(1), Constant(0),
#             "on_boundary && x[0] >= 1 - DOLFIN_EPS && x[1] > 2.0/5.0 && x[1] < 3.0/5.0")
#         bc_vp = [bc_v, bc_p]
#
#         F1stokes = (alpha(eta) * inner(u, v) * dx
#                     + data.nu / data.ubar * inner(grad(u), grad(v)) * dx
#                     - inner(div(v), p) * dx
#                     - inner(div(u), q) * dx)
#         solve(lhs(F1stokes) == rhs(F1stokes), up0, bcs=bc_vp)
#         (u1, p1) = up0.split(True)
#
#         # Fixed point for Navier - Stokes
#         if data.NavierStokes == True :
#             counter = 0
#             err_rel = 1
#             while (counter < data.iter_max_NS and err_rel > data.tol_NS):
#                 F1 = (alpha(eta) * inner(u, v) * dx
#                             + data.nu / data.ubar * inner(grad(u), grad(v)) * dx
#                             - inner(div(v), p) * dx
#                             - inner(div(u), q) * dx
#                             + inner(dot(grad(u), u1), v) * dx)
#                 solve(lhs(F1) == rhs(F1), up0, bcs=bc_vp)
#                 (u2, p2) = up0.split(True)
#                 err_abs = assemble(dot(u1 - u2, u1 - u2) * dx)
#                 norm_u2 = assemble(dot(u2, u2) * dx)
#                 err_rel = err_abs / norm_u2
#                 u1 = u2
#                 counter += 1
#                 print("+++++++ counter = ",counter,", err_rel = ", err_rel)
#         ##----------
#         T = TrialFunction(A)
#         S = TestFunction(A)
#         T2 = Function(A)
#
#         #F2 = (data.Pe * inner(u1, grad(T)) * S * dx +
#         #    kappa(eta) * inner(grad(T), grad(S)) * dx)
#
#         F2 = (inner(u1, grad(T)) * S * dx
#                 + D(eta) * inner(grad(T), grad(S)) * dx)
#
#         bc_T_in = DirichletBC(A, Constant(data.T_in),
#             "on_boundary && x[0] <= DOLFIN_EPS && x[1] > 2.0/5.0 && x[1] < 3.0/5.0")
#         bc_T_updown = DirichletBC(A, Constant(data.T_updown),
#             "on_boundary && (x[1] <= DOLFIN_EPS || x[1] >= 1.0 - DOLFIN_EPS)")
#         bc = [bc_T_in, bc_T_updown]
#
#         solve(lhs(F2) == rhs(F2), T2, bcs=bc)
#
#         return (up0, T2)
#
#     # setting output path and file names
#     def set_path_and_names(step):
#         if step in range(0,len(data.qrs) - 1):
#             controls = File(data.controls+str(step)+"_guess.pvd")
#             velocity = File(data.velocity+str(step)+"_guess.pvd")
#             pressure = File(data.pressure+str(step)+"_guess.pvd")
#             temperature = File(data.temperature+str(step)+"_guess.pvd")
#         elif step == (len(data.qrs) - 1):
#             controls = File(data.controls+".pvd")
#             velocity = File(data.velocity+".pvd")
#             pressure = File(data.pressure+".pvd")
#             temperature = File(data.temperature+".pvd")
#         else:
#             raise Exception("No further precision step defined")
#
#         eta_viz = Function(A, name = data.eta_viz_name)
#         vel_viz = Function(U, name = data.vel_viz_name)
#         pre_viz = Function(A, name = data.pre_viz_name)
#         tmp_viz = Function(A, name = data.tmp_viz_name)
#         out_file = open(data.output_log_file_name, 'w+')
#         return controls, velocity, pressure, temperature,\
#          eta_viz, vel_viz, pre_viz, tmp_viz, out_file
#
#     # Volume constraint
#     class VolumeConstraint(EqualityConstraint): #InequalityConstraint
#         """A class that enforces the volume constraint g(a) = V - a*dx >= 0."""
#         def __init__(self, V):
#             self.V = float(V)
#             self.smass = assemble(TestFunction(A) * Constant(1) * dx)
#             self.tmpvec = Function(A)
#
#         def function(self, m):
#             print("Evaluting constraint residual")
#             self.tmpvec.vector()[:] = m
#
#             # Compute the integral of the control over the domain
#             integral = self.smass.inner(self.tmpvec.vector())
#             print("Current control integral: ", integral)
#             return [self.V - integral]
#
#         def jacobian(self, m):
#             print("Computing constraint Jacobian")
#             return [-self.smass]
#
#         def output_workspace(self):
#             return [0.0]
#
#     # Compute the value of the cost functional
#     def compute_J(eta_val, iteration):
#         (up_val, T_val)  = forward(eta_val)
#         (u_val, p_val) = split(up_val)
#         J_val = define_cost_fct(n,eta_val, u_val, p_val, T_val, iteration)
#         Jd_val = pres.define_cost_fct(n,eta_val, u_val, p_val, T_val)
#         Jr_val = temp.define_cost_fct(n,eta_val, u_val, p_val, T_val)
#         return (J_val, Jd_val, Jr_val)
#
#
#     # A function wrapping the whole problem
#     def solve_optimization(iteration, eta_opt, qqd, qqr):
#         if iteration >= 0:
#             qd.assign(qqd)
#             qr.assign(qqr)
#             eta.assign(eta_opt)
#             set_working_tape(Tape())
#         else:
#             raise Exception("iteration must be an integer >= 0")
#
#         controls, velocity, pressure, temperature,\
#          eta_viz, vel_viz, pre_viz, tmp_viz, out_file = set_path_and_names(iteration)
#
#         (up, T)  = forward(eta)
#         (u, p) = split(up)
#
#         # Call Back function: used to save each step
#         def eval_cb(j, eta):
#             eta_viz.assign(eta)
#             (up, T) = forward(eta)
#             (u, p) = up.split(True)
#             vel_viz.assign(u)
#             pre_viz.assign(p)
#             tmp_viz.assign(T)
#             controls << eta_viz
#             velocity << vel_viz
#             pressure << pre_viz
#             temperature << tmp_viz
#
#         J = define_cost_fct(n,eta, u, p, T, iteration)
#         m = Control(eta)
#         Jhat = ReducedFunctional(J, m, eval_cb_post=eval_cb)
#         # problem
#         problem = MinimizationProblem(Jhat, bounds=(data.lb, data.ub), constraints=VolumeConstraint(V))
#         parameters = {'maximum_iterations': data.iterations[iteration], "output_file": data.output_log_file_name}
#         solver = IPOPTSolver(problem, parameters = parameters)
#
#
#         # solve problem
#         eta_opt = solver.solve()
#
#         # save solution
#         if iteration < len(data.qrs) - 1:
#             eta_opt_xdmf = XDMFFile(MPI.comm_world, "output/control_solution_guess_"+str(iteration)+".xdmf")
#         else:
#             eta_opt_xdmf = XDMFFile(MPI.comm_world, "output/control_solution.xdmf")
#         eta_opt_xdmf.write(eta_opt)
#         return eta_opt
#
#     ###########################
#     parameters["std_out_all_processes"] = False
#
#     clear_output()
#
#     #set_function_spaces
#     if data.load_mesh == True:
#         mesh = Mesh(data.load_name_mesh)
#     else:
#         mesh = Mesh(RectangleMesh(MPI.comm_world, Point(0.0, 0.0), Point(data.delta, 1.0), data.N, data.N))
#     A = FunctionSpace(mesh, "CG", 1)        # control function space
#     U = VectorFunctionSpace(mesh, 'CG', 2)  # probably useless
#     U_h = VectorElement("CG", mesh.ufl_cell(), 2)
#     P_h = FiniteElement("CG", mesh.ufl_cell(), 1)
#     UP = FunctionSpace(mesh, MixedElement((U_h, P_h)))
#     n = FacetNormal(mesh)
#     #define_cost_fct = assemble(define_cost_fct_unassembled)
#
#     etas = []
#     prova = []
#     if data.load_mesh == True:
#         eta0 = Function(A, data.load_name_init)
#     else:
#         eta0 = interpolate(Constant(float(V)), A)
#     eta_opt = eta0
#     eta = eta0
#     etas.append(eta0)
#     prova.append(-1)
#
#     for i in range(0, len(data.qrs)):
#         eta_opt1 = solve_optimization(i, eta_opt, data.qds[i], data.qrs[i])
#         eta_opt = eta_opt1
#         etas.append(eta_opt1)
#         prova.append(i)
#
#     # cost evaluation
#     etas[0] = interpolate(Constant(float(V)), A)
#
#     J_vals = []
#     Jd_vals = []
#     Jr_vals = []
#
#     for ie in range(len(etas)):
#         (J_val, Jd_val, Jr_val) = compute_J(etas[ie], min(0, prova[ie]))
#         J_vals.append(J_val)
#         Jd_vals.append(Jd_val)
#         Jr_vals.append(Jr_val)
#
#     print("------------------SCALED RESULTS--------------------")
#     print("++++++++++++  Initial : Jd = ", Jd_vals[0], "Jr = ", Jr_vals[0])
#
#     for ii in range(len(etas)-1):
#         print("+++++++++    Step ",prova[ii+1]," : Jd = ", Jd_vals[ii+1], "Jr = ", Jr_vals[ii+1])
#
#
#     print("-----------------REALISTIC RESULTS------------------")
#     print("++++++++++++  Initial : Jd = ", Jd_vals[0] * data.ubar**3 * data.rho_fluid * data.delta_l,
#             "Jr = ", Jr_vals[0] *data.rho_fluid * data.Cp * data.ubar * data.delta_l)
#     for ii in range(len(etas)-1):
#         print("+++++++++    Step ",prova[ii+1]," : Jd = ", Jd_vals[ii+1]* data.ubar**3 * data.rho_fluid * data.delta_l,
#             "Jr = ", Jr_vals[ii+1]*data.rho_fluid * data.Cp * data.ubar * data.delta_l)
#
#     print("----------------------------------------------------")
#     File(data.load_name_mesh) << mesh
#     File(data.load_name_init) << eta_opt
