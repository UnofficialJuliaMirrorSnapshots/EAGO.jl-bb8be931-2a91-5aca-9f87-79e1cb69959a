
module Check_Script_Bridge

    using Compat
    using Compat.Test
    using JuMP, Ipopt, EAGO

    function check_node(nd,type,indx,child)
        flag = true
        if nd.nodetype != type
            flag = false
            return flag
        end
        if nd.index != indx
            flag = false
            return flag
        end
        for i in 1:length(child)
            if nd.children[i] != child[i]
                flag = false
                return flag
            end
        end
        return flag
    end

    function f1(x)
        return sin(3.0*x[1]) + x[2]
    end
    tape1 = Tracer.trace_script(f1,2)
    @test check_node(tape1.nd[1],JuMP.Derivatives.VARIABLE, 1, [-1])
    @test check_node(tape1.nd[2],JuMP.Derivatives.VARIABLE, 2, [-1])
    @test check_node(tape1.nd[3],JuMP.Derivatives.VALUE, 1, [-2])
    @test check_node(tape1.nd[4],JuMP.Derivatives.CALL, 3, [3,1])
    @test check_node(tape1.nd[5],JuMP.Derivatives.CALLUNIVAR, 15, [4])
    @test check_node(tape1.nd[6],JuMP.Derivatives.CALL, 1, [5,2])
    @test tape1.const_values[1] == 3.0
    @test tape1.num_valued[1] == false
    @test tape1.num_valued[2] == false
    @test tape1.num_valued[3] == true
    @test tape1.num_valued[4] == false
    @test tape1.num_valued[5] == false
    @test tape1.num_valued[6] == false
    @test tape1.set_trace_count == 6
    @test tape1.const_count == 1

    function f2(x)
        z = abs(x[1])
        y = 1.3
        for i in 1:2
            y += i*z
        end
        return sin(3.0*x[2]) + y
    end
    tape2 = Tracer.trace_script(f2,2)
    @test check_node(tape2.nd[1],JuMP.Derivatives.VARIABLE, 1, [-1])
    @test check_node(tape2.nd[2],JuMP.Derivatives.VARIABLE, 2, [-1])
    @test check_node(tape2.nd[3],JuMP.Derivatives.CALLUNIVAR, 3, [1])
    @test check_node(tape2.nd[4],JuMP.Derivatives.VALUE, 1, [-2])
    @test check_node(tape2.nd[5],JuMP.Derivatives.CALL, 3, [4,3])
    @test check_node(tape2.nd[6],JuMP.Derivatives.VALUE, 2, [-2])
    @test check_node(tape2.nd[7],JuMP.Derivatives.CALL, 1, [6,5])
    @test check_node(tape2.nd[8],JuMP.Derivatives.VALUE, 3, [-2])
    @test check_node(tape2.nd[9],JuMP.Derivatives.CALL, 3, [8,3])
    @test check_node(tape2.nd[10],JuMP.Derivatives.CALL, 1, [7,9])
    @test check_node(tape2.nd[11],JuMP.Derivatives.VALUE, 4, [-2])
    @test check_node(tape2.nd[12],JuMP.Derivatives.CALL, 3, [11,2])
    @test check_node(tape2.nd[13],JuMP.Derivatives.CALLUNIVAR, 15, [12])
    @test check_node(tape2.nd[14],JuMP.Derivatives.CALL, 1, [13,10])
    @test tape2.const_values[1] == 1.0
    @test tape2.const_values[2] == 1.3
    @test tape2.const_values[3] == 2.0
    @test tape2.const_values[4] == 3.0
    @test tape2.num_valued[1] == false
    @test tape2.num_valued[2] == false
    @test tape2.num_valued[3] == false
    @test tape2.num_valued[4] == true
    @test tape2.num_valued[5] == false
    @test tape2.num_valued[6] == true
    @test tape2.num_valued[7] == false
    @test tape2.num_valued[8] == true
    @test tape2.num_valued[9] == false
    @test tape2.num_valued[10] == false
    @test tape2.num_valued[11] == true
    @test tape2.num_valued[12] == false
    @test tape2.num_valued[13] == false
    @test tape2.num_valued[14] == false
    @test tape2.set_trace_count == 14
    @test tape2.const_count == 4

    function f3(x)
        z = abs(x[1])::Float64
        return z
    end
    tape3 = Tracer.trace_script(f3,2)
    @test tape3.const_count == 0
    @test tape3.set_trace_count == 3
    @test check_node(tape3.nd[1],JuMP.Derivatives.VARIABLE, 1, [-1])
    @test check_node(tape3.nd[2],JuMP.Derivatives.VARIABLE, 2, [-1])
    @test check_node(tape3.nd[3],JuMP.Derivatives.CALLUNIVAR, 3, [1])
end

println("begin component trace example...")

f(x) = x[1] + x[2]
xl = [1.0 2.0]
xu = [2.0 3.0]

function f3(x)
    z = zeros(typeof(x[1]),2)
    z[1] = -abs(x[1])::Float64
    z[2] = -x[1] - 1.0 + z[1]
    return z
end
tape3 = Tracer.trace_script(f3,2)

g_plus = x -> f3(x) .+ [i for i in 1:2]
tape_plus = Tracer.trace_script(g_plus ,2)

nodes_g, vars_g = Tracer.get_component_tapes(f3, 2, 2)

opt = EAGO.Optimizer()
xl = [1.0,1.0]
xu = [2.0,2.0]
fobj(x) = x[1]
output = EAGO.solve_script(fobj, xl, xu, opt, g = f3)

using JuMP, Ipopt
m = Model(with_optimizer(Ipopt.Optimizer))
@variable(m, 1 <= x[1:2] <= 2)
@NLobjective(m, Min, x[1])
@NLconstraint(m, -abs(x[1]) <= 0)
@NLconstraint(m, -x[1] - 1.0 - abs(x[1]) <= 0)
JuMP.optimize!(m)

evaluator_from_script = opt.working_upper_optimizer.nlp_data.evaluator
evaluator_from_model = m.moi_backend.optimizer.model.optimizer.nlp_data.evaluator

objective_script = evaluator_from_script.objective
objective_model = evaluator_from_model.objective

constraints_script = evaluator_from_script.constraints
constraints_model = evaluator_from_model.constraints

model_script = evaluator_from_script.m
model_model = evaluator_from_model.m
#output = solve_script(f, xl, xu)

#=
function f2(x)
    z = abs(x[1])
    y = 1.3
    for i in 1:2
        y += i*z
    end
    return sin(3.0*x[2]) + y
end
tape2 = Tracer.trace_script(f2,2)

function f3(x)
    z = abs(x[1])::Float64
    return z
end
tape3 = Tracer.trace_script(f3,2)
=#
#Tracer.child_to_parent!(tape5)
#=
function f6(x)
    z::Int64 = (abs(x[1]) + 3.0*x[2])
    y = 1.3
    for i in 1:3
        y += i*z
    end
    cos(x[1]) + y
    return sin(3.0*x[1]) + y*3.0*x[2] + z
end
tape6 = Tracer.trace_script(f6,2)
=#
#Tracer.child_to_parent!(tape6)


#m = Model(with_optimizer(Ipopt.Optimizer))
#@variable(m,  0 <= x[1:2] <= 1)
#@variable(m, y)
#@NLobjective(m, Min, sin(3.0*x[1]) + 3.9*x[2])
#JuMP.optimize!(m)

#moi_backend = JuMP.backend(m)
#evaluator = moi_backend.optimizer.model.optimizer.nlp_data.evaluator
#objstorage =  evaluator.objective
#+ 3.0*x[2]
