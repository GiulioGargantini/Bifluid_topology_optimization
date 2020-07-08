# This script contains the data about the different meshes composing the domain for the
# ADIMENSIONALIZED problem on Omega_tot

## Import libraries
from dolfin import *                    # Library for the optimization
from dolfin_adjoint import *            # Library for the optimization
import pyipopt                          # Library for the optimization
import params as data                   # numerical parameters and physical constant values
import os, shutil                       # tools to work with folders
import numpy as np                      # Math tools


## Meshes definitions
if data.load_mesh == False:
    mesh_tot = RectangleMesh(MPI.comm_world, Point(0.0, 0.0), Point(data.length_x_tot_adim, data.length_y_adim),  data.Nx_tot, data.Ny)
else:
    mesh_tot = Mesh()
    mesh_file = HDF5File(MPI.comm_world, data.load_name_mesh, "r")
    mesh_file.read(mesh_tot, "mesh", False)
    mesh_file.close()
## Differentiating btw the two subdomains f and d
#### Define the subdomains
class Omega_f(SubDomain):
    def inside(self, x, on_boundary):
        return not (between(x[0], (data.length_x_f_adim, data.length_x_f_adim + data.length_x_d_adim)))

class Omega_d(SubDomain):
    def inside(self, x, on_boundary):
        return (between(x[0], (data.length_x_f_adim, data.length_x_f_adim + data.length_x_d_adim)))

#### Initialize sub-domain instances
omega_f = Omega_f()
omega_d = Omega_d()

domains = MeshFunction("size_t", mesh_tot, mesh_tot.topology().dim())

#### Marking the sub-domains
domains.set_all(0)
omega_f.mark(domains, 1)
omega_d.mark(domains, 2)

####Creating a submesh for Omega_d
mesh_d = SubMesh(mesh_tot, domains, 2)
domains_d = MeshFunction("size_t", mesh_d, mesh_d.topology().dim())
domains_d.set_all(0)

## Vector spaces
A   = FunctionSpace(mesh_tot, "CG", 1)
A_d  = FunctionSpace(mesh_d, "CG", 1)
U = VectorFunctionSpace(mesh_tot, "CG", 2)

parameters['allow_extrapolation'] = True

Uh = VectorElement("CG", mesh_tot.ufl_cell(), 2)
Ah = FiniteElement("CG", mesh_tot.ufl_cell(), 1)

UP = FunctionSpace(mesh_tot, MixedElement((Uh, Ah)))   # Compatible function space for velocity and pressure on Omega_tot
TT = FunctionSpace(mesh_tot, MixedElement((Ah, Ah)))
UPTT = FunctionSpace(mesh_tot, MixedElement((Uh, Ah, Ah, Ah)))

n = FacetNormal(mesh_tot)         # normal facet
n_d = FacetNormal(mesh_d)         # normal facet


## boundaries Omega_tot
class Symm_boundaries(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and ((near(x[1], 0.0)) or (near(x[1], data.length_y/data.Lbar)))

class Inflow_boundaries(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], 0.0)

class Outflow_boundaries(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], data.length_x_tot/data.Lbar)

symm_bound = Symm_boundaries()
inflow_bound = Inflow_boundaries()
outflow_bound = Outflow_boundaries()

#### Marking the boundaries:    symmetries -> 0
                   #            inflow -> 1
                   #            outflow -> 2
sub_domains = MeshFunction("size_t", mesh_tot, mesh_tot.topology().dim() - 1)
symm_bound.mark(sub_domains, 0)
inflow_bound.mark(sub_domains, 1)
outflow_bound.mark(sub_domains, 2)

## boundaries Omega_d
class Internal_boundaries_Omega_d(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and (near(x[0], data.length_x_f/data.Lbar) or near(x[0], (data.length_x_d + data.length_x_f)/data.Lbar))

internal_d_bound = Internal_boundaries_Omega_d()

#### Marking the boundaries:    internal -> 5
                   #            outflow -> 6
sub_domains_d = MeshFunction("size_t", mesh_d, mesh_d.topology().dim() - 1)
internal_d_bound.mark(sub_domains_d, 5)
## Initial condition

###### Uniform initial condition
class class_IC_uniform(UserExpression):
    def eval(self, values, x):
        values[0] = 1.0
        if ((x[0] >= data.length_x_f_adim) and (x[0] <= data.length_x_f_adim + data.length_x_d_adim)):
            values[0] = data.max_fluid_frac

    def value_shape(self):
        return(1,)

###### Central horizontal bar
class class_IC_horizontal_bar(UserExpression):
    def eval(self, values, x):
        values[0] = 1.0
        if ((x[0] >= data.length_x_f_adim) and (x[0] <= data.length_x_f_adim + data.length_x_d_adim) and
         (x[1] >= 0.35 * data.length_y_adim) and (x[1] <= 0.65 * data.length_y_adim)):
            values[0] = 0.0

    def value_shape(self):
        return(1,)

###### Three bubbles
class class_IC_3_bubbles(UserExpression):
    def eval(self, values, x):
        values[0] = 1.0
        if ((x[0] - 0.35 - data.length_x_f_adim)**2 + (x[1] - 0.25)**2 <= 0.15**2):
            values[0] = 0.0
        if ((x[0] - 0.85 - data.length_x_f_adim)**2 + (x[1] - 0.5)**2 <= 0.15**2):
            values[0] = 0.0
        if ((x[0] - 1.35 - data.length_x_f_adim)**2 + (x[1] - 0.75)**2 <= 0.15**2):
            values[0] = 0.0

    def value_shape(self):
        return(1,)

###### Uniform fluid
class class_IC_uniform_fluid(UserExpression):
    def eval(self, values, x):
        values[0] = 1.0

    def value_shape(self):
        return(1,)


###### Uniform solid
class class_IC_uniform_solid(UserExpression):
    def eval(self, values, x):
        values[0] = 1.0
        if ((x[0] >= data.length_x_f_adim) and (x[0] <= data.length_x_f_adim + data.length_x_d_adim)):
            values[0] = 0.0

    def value_shape(self):
        return(1,)

###### Four bubbles
class class_IC_4_bubbles(UserExpression): #29.75859078226726
    def eval(self, values, x):
        values[0] = 1.0
        if ((x[0] - 0.3 - data.length_x_f_adim)**2 + (x[1] - 0.2)**2 <= 0.1**2):
            values[0] = 0.0
        if ((x[0] - 0.7 - data.length_x_f_adim)**2 + (x[1] - 0.4)**2 <= 0.1**2):
            values[0] = 0.0
        if ((x[0] - 1.0 - data.length_x_f_adim)**2 + (x[1] - 0.6)**2 <= 0.1**2):
            values[0] = 0.0
        if ((x[0] - 1.4 - data.length_x_f_adim)**2 + (x[1] - 0.8)**2 <= 0.1**2):
            values[0] = 0.0

    def value_shape(self):
        return(1,)

###### Three bubbles, half outside
class class_IC_3_bubbles_halved(UserExpression):
    def eval(self, values, x):
        values[0] = 1.0
        if ((x[0] - 0.3 - data.length_x_f_adim)**2 + (x[1] - 0.0)**2 <= 0.2**2):
            values[0] = 0.0
        if ((x[0] - 0.85 - data.length_x_f_adim)**2 + (x[1] - 0.5)**2 <= 0.2**2):
            values[0] = 0.0
        if ((x[0] - 1.4 - data.length_x_f_adim)**2 + (x[1] - 1.0)**2 <= 0.2**2):
            values[0] = 0.0

    def value_shape(self):
        return(1,)

###### Four bubbles, half outside
class class_IC_4_bubbles_halved(UserExpression):
    def eval(self, values, x):
        values[0] = 1.0
        if ((x[0] - 0.3 - data.length_x_f_adim)**2 + (x[1] - 0.0)**2 <= 0.15**2):
            values[0] = 0.0
        if ((x[0] - 0.6 - data.length_x_f_adim)**2 + (x[1] - 0.35)**2 <= 0.15**2):
            values[0] = 0.0
        if ((x[0] - 1.1 - data.length_x_f_adim)**2 + (x[1] - 0.65)**2 <= 0.15**2):
            values[0] = 0.0
        if ((x[0] - 1.4 - data.length_x_f_adim)**2 + (x[1] - 1.0)**2 <= 0.15**2):
            values[0] = 0.0

    def value_shape(self):
        return(1,)

### Initialization
initial_condition = Function(A, name = data.gamma_viz_name)
if data.load_mesh == False:
    ic = class_IC_4_bubbles(element=A.ufl_element())
    initial_condition.assign(interpolate(ic, A))
else:
    ic_file = HDF5File(MPI.comm_world, data.load_name_init, "r")
    ic_file.read(initial_condition, "gamma")
