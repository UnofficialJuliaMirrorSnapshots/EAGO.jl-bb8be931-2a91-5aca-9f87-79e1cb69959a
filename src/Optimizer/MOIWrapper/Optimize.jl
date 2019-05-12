function triv_function(x) end

linear_solve!(m::Optimizer) = (println("Linear solve to be implemented. Recommend using linear solver such as GLPK directly. EAGO will continue..."))

function MOI.optimize!(m::Optimizer; custom_mod! = triv_function, custom_mod_args = (1,))

    ########### Reformulate DAG using auxilliary variables ###########
    NewVariableSize = length(m.variable_info)
    m.continuous_variable_number = NewVariableSize
    m.variable_number = NewVariableSize

    ########### Set Correct Size for Problem Storage #########
    m.current_lower_info.solution = Float64[0.0 for i=1:NewVariableSize]
    m.current_lower_info.lower_variable_dual = Float64[0.0 for i=1:NewVariableSize]
    m.current_lower_info.upper_variable_dual = Float64[0.0 for i=1:NewVariableSize]
    m.current_upper_info.solution = Float64[0.0 for i=1:NewVariableSize]

    # loads variables into the working model
    m.variable_index_to_storage = Dict{Int,Int}()
    for i=1:NewVariableSize
        m.variable_index_to_storage[i] = i
    end
    m.storage_index_to_variable = ReverseDict(m.variable_index_to_storage)

    # Get various other sizes
    num_nlp_constraints = length(m.nlp_data.constraint_bounds)
    m.continuous_solution = zeros(Float64,NewVariableSize)

    # Sets any unset functions to default values
    set_to_default!(m)

    # Create initial node and add it to the stack
    create_initial_node!(m)

    # Build the JuMP NLP evaluator
    evaluator = m.nlp_data.evaluator
    features = MOI.features_available(evaluator)
    has_hessian = (:Hess in features)
    init_feat = [:Grad]
    #has_hessian && push!(init_feat, :Hess)
    num_nlp_constraints > 0 && push!(init_feat, :Jac)
    MOI.initialize(evaluator,init_feat)

    # Checks for univariate and bivariate quadratics and adds them to specialized storage
    #classify_quadratics!(m)

    # Creates initial nlp evaluator
    m.working_evaluator_block = m.nlp_data
    if ~isa(m.nlp_data.evaluator, EAGO.EmptyNLPEvaluator)
        built_evaluator = build_nlp_evaluator(MC{m.variable_number}, m.nlp_data.evaluator, m, false)
        (m.optimization_sense == MOI.MAX_SENSE) && neg_objective!(built_evaluator)
        m.working_evaluator_block = MOI.NLPBlockData(m.nlp_data.constraint_bounds, built_evaluator, m.nlp_data.has_objective)
    end

    # eliminate redundant expressions & flatten
    m.reform_epigraph_flag && reform_epigraph!(m)
    #m.reform_cse_flag && dag_cse_simplify!(m)
    #m.reform_flatten_flag && dag_flattening!(m)

    m.upper_variables = MOI.add_variables(m.initial_relaxed_optimizer, m.variable_number)
    m.lower_variables = MOI.VariableIndex.(1:m.variable_number)

    ###### OBBT Setup #####
    # Label fixed variables:
    lbd = -Inf
    ubd = Inf
    for i in 1:m.variable_number
        lbd = m.variable_info[i].lower_bound
        ubd = m.variable_info[i].upper_bound
        if (lbd == ubd)
            m.variable_info[i].is_fixed = true
            m.fixed_variable[i] = true
        end
    end

    # Sets terms that OBBT will be performed on & nonlinear variables
    label_nonlinear_variables!(m, evaluator)
    if ~in(true, values(m.nonlinear_variable))
        linear_solve!(m)
    end
    for i=1:NewVariableSize
        if m.nonlinear_variable[i]
            push!(m.obbt_variables, MOI.VariableIndex(i))
        end
    end

    # Relax initial model terms
    relax_model!(m, m.initial_relaxed_optimizer, m.stack[1], m.relaxation, load = true)

    # Runs a customized function if one is provided
    m.custom_mod_flag = (custom_mod! != triv_function)
    if m.custom_mod_flag
        custom_mod!(m, custom_mod_args)
    end

    # if optimizer type is supplied for upper, build factory
    if (~m.use_upper_factory)
        MOI.add_variables(m.initial_upper_optimizer, m.variable_number)
        m.upper_variables = MOI.add_variables(m.working_upper_optimizer, m.variable_number)
        set_local_nlp!(m)
    end

    println("start nlp solve")
    # Runs the branch and bound routine
    solve_nlp!(m)
end
