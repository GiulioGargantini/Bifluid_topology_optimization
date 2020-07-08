from coupled_general import *
import coupled_pres as pres
import coupled_temp as temp

# Functional definition
def define_cost_fct(n1,eta1, u1, p1, T1, i):
    cost = ((1 - data.w) / data.diff_Jr[i] * temp.cost_fct(n1,eta1, u1, p1, T1) +
         data.w / data.diff_Jd[i] * pres.cost_fct(n1,eta1, u1, p1, T1))
    return assemble(cost)

execute_optimization(define_cost_fct)
