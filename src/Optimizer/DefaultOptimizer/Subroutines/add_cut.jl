"""
    default_cut_condition

Branch-and-cut feature currently under development. Currently, returns false.
"""
default_cut_condition(x::Optimizer) = x.cut_add_flag

#=
function check_cut_tolerance(x::Optimizer, solution::Vector{Float64})
end
=#

"""
    default_add_cut!

Branch-and-Cut under development. Currently does nothing.
"""
function default_add_cut!(x::Optimizer, y::NodeBB)

    xpnt = x.current_lower_info.solution[1:end]
    update_lower_variable_bounds1!(x, y, x.working_relaxed_optimizer)
    x.relax_function!(x, x.working_relaxed_optimizer, y, x.relaxation, xpnt, load = true)
    x.relax_function!(x, x.working_relaxed_optimizer, y, x.relaxation, xpnt, load = false)

    # Optimizes the object
    MOI.optimize!(x.working_relaxed_optimizer)

    # Process output info and save to CurrentUpperInfo object
    termination_status = MOI.get(x.working_relaxed_optimizer, MOI.TerminationStatus())
    result_status_code = MOI.get(x.working_relaxed_optimizer, MOI.PrimalStatus())
    valid_flag, feasible_flag = is_globally_optimal(termination_status, result_status_code)
    last_solution = x.current_lower_info.solution[1:end]

    println("last_solution: $last_solution")

    if valid_flag
        if feasible_flag
            x.current_lower_info.feasibility = true
            x.current_lower_info.value = MOI.get(x.working_relaxed_optimizer, MOI.ObjectiveValue())
            vprimal_solution = MOI.get(x.working_relaxed_optimizer, MOI.VariablePrimal(), x.lower_variables)
            solutions_distinct = ~(last_solution == vprimal_solution)
            x.cut_add_flag = (x.cut_iterations < x.cut_max_iterations) && solutions_distinct
            x.current_lower_info.solution[1:end] = vprimal_solution
            set_dual!(x)
        else
            x.cut_add_flag = false
            x.current_lower_info.feasibility = false
            x.current_lower_info.value = -Inf
        end
    else
        error("Lower problem returned a TerminationStatus = $(termination_status) and
               ResultStatusCode = $(result_status_code). This pair of codes does not
               definitively prove the subproblem to be globally optimal or infeasible.
               The subproblem must be solved to global optimality.")
    end

end
