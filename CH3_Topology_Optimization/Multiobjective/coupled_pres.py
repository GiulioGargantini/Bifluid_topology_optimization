from coupled_general import *

# Functional definition
def cost_fct(n1,eta1, u1, p1, T1):
    cost = - inner(n1, u1) * (p1 + 0.5 * inner(u1,u1)) * ds
    return cost

def define_cost_fct(n1,eta1, u1, p1, T1):
    cost = cost_fct(n1,eta1, u1, p1, T1)
    return assemble(cost)
#execute_optimization(define_cost_fct)
