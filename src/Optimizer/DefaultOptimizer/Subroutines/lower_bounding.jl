"""
    default_lower_bounding!

Constructs and solves the relaxation using the default EAGO relaxation scheme
and optimizer on node `y`.
"""
function default_lower_bounding!(x::Optimizer, y::NodeBB)

    # Copies initial model into working model (if initial model isn't dummy)
    # A dummy model is left iff all terms are relaxed
    if x.use_lower_factory
        factory = x.lower_factory(; x.lower_optimizer_options...)     # Should accept keyword arguments
        x.working_relaxed_optimizer = factory
        MOI.add_variables(x.working_relaxed_optimizer, x.variable_number)
    else
        if (x.initial_relaxed_optimizer != DummyOptimizer()) && (x.obbt_performed_flag)
            x.working_relaxed_optimizer = deepcopy(x.initial_relaxed_optimizer)
        end
    end

    xmid = 0.5*(lower_variable_bounds(y) + upper_variable_bounds(y))
    update_lower_variable_bounds1!(x, y, x.working_relaxed_optimizer)
    x.relax_function!(x, x.working_relaxed_optimizer, y, x.relaxation, xmid, load = true)
    x.relax_function!(x, x.working_relaxed_optimizer, y, x.relaxation, xmid, load = false)

    # Optimizes the object
    MOI.optimize!(x.working_relaxed_optimizer)

    # Process output info and save to CurrentUpperInfo object
    termination_status = MOI.get(x.working_relaxed_optimizer, MOI.TerminationStatus())
    result_status_code = MOI.get(x.working_relaxed_optimizer, MOI.PrimalStatus())
    valid_flag, feasible_flag = is_globally_optimal(termination_status, result_status_code)
    solution = MOI.get(x.working_relaxed_optimizer, MOI.VariablePrimal(), x.lower_variables)

    if valid_flag
        if feasible_flag
            x.current_lower_info.feasibility = true
            x.current_lower_info.value = MOI.get(x.working_relaxed_optimizer, MOI.ObjectiveValue())
            vprimal_solution = MOI.get(x.working_relaxed_optimizer, MOI.VariablePrimal(), x.lower_variables)
            x.current_lower_info.solution[1:end] = vprimal_solution
            x.cut_add_flag = true
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
