from dolfin import *
from dolfin_adjoint import *
import pyipopt
import params as data
import os, shutil

# function alpha and kappa
qd = Constant(data.qds[0])
mu = Constant(data.mu)
rho_fluid = Constant(data.rho_fluid)
nu = Constant(mu / rho_fluid)
V = Constant(data.V)

def alpha(eta):
    """Inverse permeability as a function of eta, equation (40)"""
    return data.alphas + (data.alphaf - data.alphas)\
     * eta * (1 + qd) / (eta + qd)

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
            gbar = 3.0/2.0 #* data.Re * data.nu  # maximum inlet velocity

            if x[0] == 0.0:
                t = (x[1] - 0.5) * 2.0
                values[0] = gbar * (1 - t**2)
            if x[0] == 1.0 and x[1] > 1.0/3.0 and x[1] < 2.0/3.0:
                t = (x[1] - 0.5) * 6.0
                values[0] = gbar * 3.0 *(1 - t**2)

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

        F1stokes = (alpha(eta) * inner(u, v) * dx
            + 1/data.Re * inner(grad(u), grad(v)) * dx
            + inner(grad(p), v) * dx
            + inner(div(u), q) * dx)
        solve(lhs(F1stokes) == rhs(F1stokes), up0, bcs=bc_vp)
        (u1, p1) = up0.split(True)

        # Fixed point for Navier - Stokes
        if data.NavierStokes == True :
            counter = 0
            err_rel = 1
            while (counter < data.iter_max_NS and err_rel > data.tol_NS):
                F1 = (alpha(eta) * inner(u, v) * dx
                    + 1/data.Re * inner(grad(u), grad(v)) * dx
                    + inner(grad(p), v) * dx
                    + inner(div(u), q) * dx
                    + inner(dot(grad(u), u1), v) * dx)
                solve(lhs(F1) == rhs(F1), up0, bcs=bc_vp)
                (u2, p2) = up0.split(True)
                err_abs = assemble(dot(u1 - u2, u1 - u2) * dx)
                norm_u2 = assemble(dot(u2, u2) * dx)
                err_rel = err_abs / norm_u2
                u1 = u2
                counter += 1
                print("+++++++ counter = ",counter,", err_rel = ", err_rel)
        return (up0)

    # setting output path and file names
    def set_path_and_names(step):
        if step in range(0,len(data.qds) - 1):
            controls = File(data.controls+str(step)+"_guess.pvd")
            velocity = File(data.velocity+str(step)+"_guess.pvd")
            pressure = File(data.pressure+str(step)+"_guess.pvd")
        elif step == (len(data.qds) - 1):
            controls = File(data.controls+".pvd")
            velocity = File(data.velocity+".pvd")
            pressure = File(data.pressure+".pvd")
        else:
            raise Exception("No further precision step defined")

        eta_viz = Function(A, name = data.eta_viz_name)
        vel_viz = Function(U, name = data.vel_viz_name)
        pre_viz = Function(A, name = data.pre_viz_name)
        out_file = open(data.output_log_file_name, 'w+')
        return controls, velocity, pressure,\
         eta_viz, vel_viz, pre_viz, out_file

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
        up_val  = forward(eta_val)
        (u_val, p_val) = split(up_val)
        J_val = define_cost_fct(n,eta_val, u_val, p_val, iteration)
        return J_val


    # A function wrapping the whole problem
    def solve_optimization(iteration, eta_opt, qqd):
        if iteration >= 0:
            qd.assign(qqd)
            eta.assign(eta_opt)
            set_working_tape(Tape())
        else:
            raise Exception("iteration must be an integer >= 0")

        controls, velocity, pressure,\
         eta_viz, vel_viz, pre_viz, out_file = set_path_and_names(iteration)

        up  = forward(eta)
        (u, p) = split(up)

        # Call Back function: used to save each step
        def eval_cb(j, eta):
            eta_viz.assign(eta)
            up = forward(eta)
            (u, p) = up.split(True)
            vel_viz.assign(u)
            pre_viz.assign(p)
            controls << eta_viz
            velocity << vel_viz
            pressure << pre_viz

        J = define_cost_fct(n,eta, u, p, iteration)
        m = Control(eta)
        Jhat = ReducedFunctional(J, m, eval_cb_post=eval_cb)
        # problem
        problem = MinimizationProblem(Jhat, bounds=(data.lb, data.ub), constraints=VolumeConstraint(V))
        parameters = {'maximum_iterations': data.iterations[iteration], "output_file": data.output_log_file_name}
        solver = IPOPTSolver(problem, parameters = parameters)


        # solve problem
        eta_opt = solver.solve()

        # save solution
        if iteration < len(data.qds) - 1:
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
    U = VectorFunctionSpace(mesh, 'CG', 2)  # probably useless
    U_h = VectorElement("CG", mesh.ufl_cell(), 2)
    P_h = FiniteElement("CG", mesh.ufl_cell(), 1)
    UP = FunctionSpace(mesh, MixedElement((U_h, P_h)))
    n = FacetNormal(mesh)

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

    for i in range(0, len(data.qds)):
        eta_opt1 = solve_optimization(i, eta_opt, data.qds[i])
        eta_opt = eta_opt1
        etas.append(eta_opt1)
        prova.append(i)

    # cost evaluation
    etas[0] = interpolate(Constant(float(V)), A)

    J_vals = []

    for ie in range(len(etas)):
        J_val = compute_J(etas[ie], min(0, prova[ie]))
        J_vals.append(J_val)

    print("------------------SCALED RESULTS--------------------")
    print("++++++++++++  Initial : Jd = ", J_vals[0])

    for ii in range(len(etas)-1):
        print("+++++++++    Step ",prova[ii+1]," : Jd = ", J_vals[ii+1])


    print("----------------------------------------------------")
    File(data.load_name_mesh) << mesh
    File(data.load_name_init) << eta_opt
