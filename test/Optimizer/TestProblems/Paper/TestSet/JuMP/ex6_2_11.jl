using JuMP

m = Model()

# ----- Variables ----- #
x_Idx = Any[2, 3, 4]
@variable(m, x[x_Idx])
setlowerbound(x[2], 1.0e-6)
setupperbound(x[2], 1.0)
setlowerbound(x[3], 1.0e-6)
setupperbound(x[3], 1.0)
setlowerbound(x[4], 1.0e-6)
setupperbound(x[4], 1.0)


# ----- Constraints ----- #
@constraint(m, e2, x[2]+x[3]+x[4] == 1.0)


# ----- Objective ----- #
@NLobjective(m, Min, ((15.3261663216011*x[2]+23.2043471859416*x[3]+6.69678129464404*x[4])*log(2.1055*x[2]+3.1878*x[3]+0.92*x[4])+1.04055250396734*x[2]-2.24199441248417*x[3]+3.1618173099828*x[4]+6.4661663216011*x[2]*log(x[2]/(2.1055*x[2]+3.1878*x[3]+0.92*x[4]))+12.2043471859416*x[3]*log(x[3]/(2.1055*x[2]+3.1878*x[3]+0.92*x[4]))+0.696781294644034*x[4]*log(x[4]/(2.1055*x[2]+3.1878*x[3]+0.92*x[4]))+9.86*x[2]*log(x[2]/(1.972*x[2]+2.4*x[3]+1.4*x[4]))+12*x[3]*log(x[3]/(1.972*x[2]+2.4*x[3]+1.4*x[4]))+7*x[4]*log(x[4]/(1.972*x[2]+2.4*x[3]+1.4*x[4]))+(1.972*x[2]+2.4*x[3]+1.4*x[4])*log(1.972*x[2]+2.4*x[3]+1.4*x[4])+1.972*x[2]*log(x[2]/(1.972*x[2]+0.283910843616504*x[3]+3.02002220174195*x[4]))+2.4*x[3]*log(x[3]/(1.45991339466884*x[2]+2.4*x[3]+0.415073537580851*x[4]))+1.4*x[4]*log(x[4]/(0.602183324335333*x[2]+0.115623371371275*x[3]+1.4*x[4]))-17.2981663216011*x[2]*log(x[2])-25.6043471859416*x[3]*log(x[3])-8.09678129464404*x[4]*log(x[4])))

m = m 		 # model get returned when including this script.
