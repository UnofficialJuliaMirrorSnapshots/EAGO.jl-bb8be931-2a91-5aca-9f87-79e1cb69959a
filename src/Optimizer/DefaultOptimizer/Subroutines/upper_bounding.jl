"""
    default_upper_bounding!

Constructs and solves the problem locally on on node `y` and saves upper
bounding info to `x.current_upper_info`.
"""
function default_upper_bounding!(x::Optimizer,y::NodeBB)
    #println("start upper bound")
    if is_integer_feasible(x) #&& mod(x.CurrentIterationCount,x.UpperBoundingInterval) == 1
        if x.use_upper_factory
            factory = x.upper_factory()
            x.initial_upper_optimizer = factory
            x.upper_variables = MOI.add_variables(x.initial_upper_optimizer, x.variable_number)
            set_local_nlp!(x)
            x.working_upper_optimizer = x.initial_upper_optimizer
        else
            if x.initial_upper_optimizer != DummyOptimizer()
                x.working_upper_optimizer = deepcopy(x.initial_upper_optimizer)
            end
        end
        update_upper_variable_bounds!(x,y,x.working_upper_optimizer)

        if x.upper_has_node
            set_current_node!(x.working_upper_optimizer.nlp_data.evaluator,y)
        end

        # Optimizes the object
        TT = stdout
        redirect_stdout()
        x.debug1 = x.working_upper_optimizer
        MOI.optimize!(x.working_upper_optimizer)
        redirect_stdout(TT)

        # Process output info and save to CurrentUpperInfo object
        termination_status = MOI.get(x.working_upper_optimizer, MOI.TerminationStatus())
        result_status = MOI.get(x.working_upper_optimizer, MOI.PrimalStatus())
        solution = MOI.get(x.working_upper_optimizer, MOI.VariablePrimal(), x.upper_variables)

        if is_feasible_solution(termination_status, result_status)
            x.current_upper_info.feasibility = true
            mult = (x.optimization_sense == MOI.MIN_SENSE) ? 1.0 : -1.0
            x.current_upper_info.value = mult*MOI.get(x.working_upper_optimizer, MOI.ObjectiveValue())
            x.current_upper_info.solution[1:end] = MOI.get(x.working_upper_optimizer, MOI.VariablePrimal(), x.upper_variables)

        else
            x.current_upper_info.feasibility = false
            x.current_upper_info.value = Inf
        end
    else
        x.current_upper_info.feasibility = false
        x.current_upper_info.value = Inf
    end
    #println("finish upper bound")
end
