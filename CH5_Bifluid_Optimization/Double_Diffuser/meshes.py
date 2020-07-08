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
    print("***** Generating mesh")
    mesh = RectangleMesh(MPI.comm_world, Point(0.0, 0.0), Point(data.length_x_adim, data.length_y_adim),  data.Nx, data.Ny, diagonal = "right")
else:
    print("***** Importing mesh")
    mesh = Mesh()
    mesh_file = HDF5File(MPI.comm_world, data.load_name_mesh, "r")
    mesh_file.read(mesh, "mesh", False)
    mesh_file.close()

domains = MeshFunction("size_t", mesh, mesh.topology().dim())
domains.set_all(0)


## Vector spaces
A   = FunctionSpace(mesh, "CG", 1)
U = VectorFunctionSpace(mesh, "CG", 2)

parameters['allow_extrapolation'] = True

Uh = VectorElement("CG", mesh.ufl_cell(), 2)
Ah = FiniteElement("CG", mesh.ufl_cell(), 1)

UP = FunctionSpace(mesh, MixedElement((Uh, Ah)))   # Compatible function space for velocity and pressure on Omega
UPT = FunctionSpace(mesh, MixedElement((Uh, Ah, Ah)))   # Compatible function space for velocity, pressure and temperature on Omega
UP2T = FunctionSpace(mesh, MixedElement((Uh, Ah, Uh, Ah, Ah)))   # Compatible function space for velocity, pressure and temperature on Omega
UPT2Q = FunctionSpace(mesh, MixedElement((Uh, Ah, Ah, Ah, Ah)))   # Compatible function space for velocity, pressure, temperature  and filtered Qs on Omega
UPTQ = FunctionSpace(mesh, MixedElement((Uh, Ah, Ah, Ah)))   # Compatible function space for velocity, pressure, temperature  and filtered eta2 on Omega

## Boundaries
class Inflow_cold_boundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and ((x[1] >= data.length_y_wall_hole_in_adim) and (x[1] <= data.length_y_wall_hole_in_adim + data.length_y_hole_in_adim)) and near(x[0], 0.0)

class Inflow_hot_boundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and ((x[1] <= data.length_y_adim - data.length_y_wall_hole_in_adim) and (x[1] >= data.length_y_adim - data.length_y_wall_hole_in_adim - data.length_y_hole_in_adim)) and near(x[0], 0.0)

class Outflow_cold_boundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and ((x[1] >= data.length_y_wall_hole_out_adim) and (x[1] <= data.length_y_wall_hole_out_adim + data.length_y_hole_out_adim)) and near(x[0], data.length_x_adim)

class Outflow_hot_boundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and ((x[1] <= data.length_y_adim - data.length_y_wall_hole_out_adim) and (x[1] >= data.length_y_adim - data.length_y_wall_hole_out_adim - data.length_y_hole_out_adim)) and near(x[0], data.length_x_adim)


inflow_cold_bound = Inflow_cold_boundary()
inflow_hot_bound = Inflow_hot_boundary()
outflow_cold_bound = Outflow_cold_boundary()
outflow_hot_bound = Outflow_hot_boundary()

sub_domains = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
sub_domains.set_all(0)
inflow_cold_bound.mark(sub_domains, 1)
inflow_hot_bound.mark(sub_domains, 2)
outflow_cold_bound.mark(sub_domains, 3)
outflow_hot_bound.mark(sub_domains, 4)

###### Uniform initial condition
class class_IC_uniform_eta1(UserExpression):
    def eval(self, values, x):
        values[0] = data.phi_f1 + data.phi_f2

    def value_shape(self):
        return(1,)

class class_IC_uniform_0_eta1(UserExpression):
    def eval(self, values, x):
        values[0] = 0.0

    def value_shape(self):
        return(1,)

class class_IC_uniform_eta2(UserExpression):
    def eval(self, values, x):
        values[0] = 0.5

    def value_shape(self):
        return(1,)

class class_IC_partition_eta2(UserExpression):
    def eval(self, values, x):
        values[0] = 0.4
        if x[1] > 0.5:
            values[0] = 0.56

    def value_shape(self):
        return(1,)

class class_IC_total_partition_eta2(UserExpression):
    def eval(self, values, x):
        values[0] = 0.0
        if x[1] > 0.5:
            values[0] = 1.0

    def value_shape(self):
        return(1,)


### Initialization

eta1_initial = Function(A, name = data.eta1_viz_name)
eta2_initial = Function(A, name = data.eta2_viz_name)
if data.load_mesh == False:
    ic_eta1 = class_IC_uniform_eta1(element=A.ufl_element())
    ic_eta2 = class_IC_uniform_eta2(element=A.ufl_element())
    eta1_initial.assign(interpolate(ic_eta1, A))
    eta2_initial.assign(interpolate(ic_eta2, A))
else:
    print("**** load path = ", data.load_path)
    ic_eta1_file = HDF5File(MPI.comm_world, data.load_name_init_eta1, "r")
    ic_eta2_file = HDF5File(MPI.comm_world, data.load_name_init_eta2, "r")
    ic_eta1_file.read(eta1_initial, data.eta1_name)
    ic_eta2_file.read(eta2_initial, data.eta2_name)
