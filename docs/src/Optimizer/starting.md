#

## Solving a basic problem problem
```julia
jumpmodel4 = Model(solver=EAGO_NLPSolver(LBD_func_relax = "NS-STD-OFF",
                                         LBDsolvertype = "LP",
                                         probe_depth = -1,
                                         variable_depth = 1000,
                                         DAG_depth = -1,
                                         STD_RR_depth = -1))
@variable(jumpmodel4, -200 <= x <= -100)
@variable(jumpmodel4, 200 <= y <= 400)
@constraint(jumpmodel4, -500 <= x+2y <= 400)
@NLobjective(jumpmodel4, Min, x*y)
status2 = solve(jumpmodel4)
```
