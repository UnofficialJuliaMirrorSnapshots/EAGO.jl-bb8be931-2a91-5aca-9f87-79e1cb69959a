function contains_optimum!(x::NodeBB, s::String)
#=
    true_opt = -5.508013271485620
    opt = [2.329520197651560, 3.178493073834060]
    flag = true
    for i in 1:length(x)
        if ((x.lower_variable_bounds[i] > opt[i]) ||
           (x.upper_variable_bounds[i] < opt[i]))
           flag = false
           break
        end
    end
    if flag
        println("THIS NODE $(x.id) CONTAINS THE SOLUTION POINT")
        println(s)
    else
        println("THIS NODE $(x.id) DOES NOT CONTAIN THE SOLUTION POINT")
        println(s)
    end
    =#
    return
end

function contains_optimum!(x::Optimizer, s::String, L::Float64, U::Float64)
    #=
    true_opt = -5.508013271485620
    cnode = x._current_node
    opt = [2.329520197651560, 3.178493073834060]
    flag = true
    for i in 1:length(cnode)
        if ((cnode.lower_variable_bounds[i] > opt[i]) ||
           (cnode.upper_variable_bounds[i] < opt[i]))
           flag = false
           break
        end
    end
    if flag
        println("THIS NODE $(cnode.id) CONTAINS THE SOLUTION POINT")
        if ~(L < true_opt)
            println("THIS NODE DOESN'T BOUND PROPERLY")
            for i in 1:length(cnode)
                Li = cnode.lower_variable_bounds[i]
                Ui = cnode.upper_variable_bounds[i]
                println("X[$i] = [$Li, $Ui]")
            end
            println("$L < $true_opt")
            println("iteration #: $(x._iteration_count)")
        end
    end
    =#
    return
end


"""
    node_selection

Selects node with the lowest lower bound in stack.
"""
function node_selection!(t::ExtensionType, x::Optimizer)
    x._node_count -= 1
    x._current_node = popmin!(x._stack)
    #println("NODE SELECTION.... Current Node: $(x._current_node)")
    return
end

"""
    branch_node!

Stores the two nodes to the stack.
"""
function branch_node!(t::ExtensionType, x::Optimizer)

    #println("PERFORMED BRANCH")

    y = x._current_node
    nvar = x._variable_number
    lvbs = y.lower_variable_bounds
    uvbs = y.upper_variable_bounds

    max_pos = 0
    max_val = -Inf
    temp_max = 0.0

    flag = true
    for i in 1:nvar
        @inbounds flag = ~x._fixed_variable[i]
        @inbounds flag &= x.branch_variable[i]
        @inbounds vi = x._variable_info[i]
        if flag
            @inbounds temp_max = uvbs[i] - lvbs[i]
            @inbounds temp_max /= vi.upper_bound - vi.lower_bound
            if temp_max > max_val
                max_pos = i
                max_val = temp_max
            end
        end
    end

    @inbounds lvb = lvbs[max_pos]
    @inbounds uvb = uvbs[max_pos]
    @inbounds lsol = x._lower_solution[max_pos]
    cvx_f = x.branch_cvx_factor
    cvx_g = x.branch_offset
    branch_pnt = cvx_f*lsol + (1.0 - cvx_f)*(lvb + uvb)/2.0
    if branch_pnt < lvb*(1.0 - cvx_g) + cvx_g*uvb
        branch_pnt = (1.0 - cvx_g)*lvb + cvx_g*uvb
    elseif branch_pnt > cvx_g*lvb + (1.0 - cvx_g)*uvb
        branch_pnt = cvx_g*lvb + (1.0 - cvx_g)*uvb
    end
    N1::Interval{Float64} = Interval{Float64}(lvb, branch_pnt)
    N2::Interval{Float64} = Interval{Float64}(branch_pnt, uvb)
    lvb_1 = copy(lvbs)
    uvb_1 = copy(uvbs)
    lvb_2 = copy(lvbs)
    uvb_2 = copy(uvbs)
    @inbounds lvb_1[max_pos] = N1.lo
    @inbounds uvb_1[max_pos] = N1.hi
    @inbounds lvb_2[max_pos] = N2.lo
    @inbounds uvb_2[max_pos] = N2.hi

    lower_bound = max(y.lower_bound, x._lower_objective_value)
    upper_bound = min(y.upper_bound, x._upper_objective_value)
    x._maximum_node_id += 1
    X1 = NodeBB(lvb_1, uvb_1, lower_bound, upper_bound, y.depth + 1, x._maximum_node_id)
    x._maximum_node_id += 1
    X2 = NodeBB(lvb_2, uvb_2, lower_bound, upper_bound, y.depth + 1, x._maximum_node_id)
    #contains_optimum!(X1, "Node X1")
    #contains_optimum!(X2, "Node X2")

    push!(x._stack, X1)
    push!(x._stack, X2)

    x._node_repetitions = 1
    x._node_count += 2

    return
end

"""
    single_storage!

Stores the current node to the stack after updating lower/upper bounds.
"""
function single_storage!(t::ExtensionType, x::Optimizer)
    y = x._current_node
    x._node_repetitions += 1
    x._maximum_node_id += 0
    x._node_count += 1
    y.lower_bound = max(y.lower_bound, x._lower_objective_value)
    y.upper_bound = min(y.upper_bound, x._upper_objective_value)
    y.depth += 1
    push!(x._stack, y)
    return
end

"""
    fathom!

Selects and deletes nodes from stack with lower bounds greater than global
upper bound.
"""
function fathom!(t::ExtensionType, d::Optimizer)
    upper = d._global_upper_bound
    continue_flag = ~isempty(d._stack)
    while continue_flag
        max_node = maximum(d._stack)
        max_check = (max_node.lower_bound > upper)
        if max_check
            popmax!(d._stack)
            d._node_count -= 1
        else
            if ~max_check
                continue_flag = false
            elseif isempty(d._stack)
                continue_flag = false
            end
        end
    end
    return
end

"""
    repeat_check

Checks to see if current node should be reprocessed.
"""
repeat_check(t::ExtensionType, x::Optimizer) = false

function relative_gap(L::Float64, U::Float64)
    gap = Inf
    if (L > -Inf) & (U < Inf)
        gap = abs(U - L)/(max(abs(L),abs(U)))
    end
    return gap
end
function relative_tolerance(L::Float64, U::Float64, tol::Float64)
    return (relative_gap(L, U)  > tol) || ~(L > -Inf)
end

"""
    termination_check

Checks for termination of algorithm due to satisfying absolute or relative
tolerance, infeasibility, or a specified limit, returns a boolean valued true
if algorithm should continue.
"""
function termination_check(t::ExtensionType, x::Optimizer)

    L = x._global_lower_bound
    U = x._global_upper_bound

    if isempty(x._stack)

        if (x._first_solution_node > 0)

            x._termination_status_code = MOI.OPTIMAL
            x._result_status_code = MOI.FEASIBLE_POINT
            (x.verbosity >= 3) && println("Empty Stack: Exhaustive Search Finished")
        else
            x._termination_status_code = MOI.INFEASIBLE
            x._result_status_code = MOI.INFEASIBILITY_CERTIFICATE
            (x.verbosity >= 3) && println("Empty Stack: Infeasible")
        end
    elseif length(x._stack) >= x.node_limit

        x._termination_status_code = MOI.NODE_LIMIT
        x._result_status_code = MOI.UNKNOWN_RESULT_STATUS
        (x.verbosity >= 3) && println("Node Limit Exceeded")

    elseif x._iteration_count >= x.iteration_limit

        x._termination_status_code = MOI.ITERATION_LIMIT
        x._result_status_code = MOI.UNKNOWN_RESULT_STATUS
        (x.verbosity >= 3) && println("Maximum Iteration Exceeded")

    elseif ~relative_tolerance(L, U, x.relative_tolerance)

        x._termination_status_code = MOI.OPTIMAL
        x._result_status_code = MOI.FEASIBLE_POINT
        (x.verbosity >= 3) && println("Relative Tolerance Achieved")

    elseif (U - L) < x.absolute_tolerance
        x._termination_status_code = MOI.OPTIMAL
        x._result_status_code = MOI.FEASIBLE_POINT
        (x.verbosity >= 3) && println("Absolute Tolerance Achieved")

    elseif x._run_time > x.time_limit

        x._termination_status_code = MOI.TIME_LIMIT
        x._result_status_code = MOI.UNKNOWN_RESULT_STATUS
        (x.verbosity >= 3) && println("Time Limit Exceeded")

    else

        return false

    end

    return true
end

"""
    convergence_check

Checks for convergence of algorithm with respect to absolute and/or relative
tolerances.
"""
function convergence_check(t::ExtensionType, x::Optimizer)

  #println("ran convergence check...")
  L = x._lower_objective_value
  U = x._global_upper_bound
  t = (U - L) <= x.absolute_tolerance
  if (U < Inf) & (L > Inf)
      t |= (abs(U - L)/(max(abs(L),abs(U))) <= x.relative_tolerance)
  end
 # println("algorithm converged... $t")
  return t
end

"""
    is_globally_optimal

Takes an `MOI.TerminationStatusCode` and a `MOI.ResultStatusCode` and returns
the tuple `(valid_result::Bool, feasible::Bool)`. The value `valid_result` is
`true` if the pair of codes prove that either the subproblem solution was solved
to global optimality or the subproblem solution is infeasible. The value of
`feasible` is true if the problem is feasible and false if the problem is infeasible.
"""
function is_globally_optimal(t::MOI.TerminationStatusCode, r::MOI.ResultStatusCode)

    feasible = false
    valid_result = false

    if (t == MOI.INFEASIBLE && r == MOI.INFEASIBILITY_CERTIFICATE)
        valid_result = true
    elseif (t == MOI.INFEASIBLE && r == MOI.NO_SOLUTION)
        valid_result = true
    elseif (t == MOI.INFEASIBLE && r == MOI.UNKNOWN_RESULT_STATUS)
        valid_result = true
    elseif (t == MOI.OPTIMAL && r == MOI.FEASIBLE_POINT)
        valid_result = true
        feasible = true
    elseif (t == MOI.INFEASIBLE_OR_UNBOUNDED && r == MOI.NO_SOLUTION)
        valid_result = true
        feasible = false
    end

    return valid_result, feasible
end

"""
    is_feasible_solution

Takes an `MOI.TerminationStatusCode` and a `MOI.ResultStatusCode` and returns `true`
if this corresponds to a solution that is proven to be feasible.
Returns `false` otherwise.
"""
function is_feasible_solution(t::MOI.TerminationStatusCode, r::MOI.ResultStatusCode)

    termination_flag = false
    result_flag = false

    (t == MOI.OPTIMAL) && (termination_flag = true)
    (t == MOI.LOCALLY_SOLVED) && (termination_flag = true)

    # This is default solver specific... the acceptable constraint tolerances
    # are set to the same values as the basic tolerance. As a result, an
    # acceptably solved solution is feasible but non necessarily optimal
    # so it should be treated as a feasible point
    if (t == MOI.ALMOST_LOCALLY_SOLVED) && (r == MOI.NEARLY_FEASIBLE_POINT)
        termination_flag = true
        result_flag = true
    end

    (r == MOI.FEASIBLE_POINT) && (result_flag = true)

    return (termination_flag && result_flag)
end

"""
    set_dual!

Retrieves the lower and upper duals for variable bounds from the
`relaxed_optimizer` and sets the appropriate values in the
`_lower_lvd` and `_lower_uvd` storage fields.
"""
function set_dual!(x::Optimizer)

    opt = x.relaxed_optimizer

    lower_variable_lt_indx = x._lower_variable_lt_indx
    lower_variable_lt = x._lower_variable_lt
    for i in 1:length(lower_variable_lt_indx)
        @inbounds vi = lower_variable_lt_indx[i]
        @inbounds ci_lt = lower_variable_lt[i]
        @inbounds x._lower_uvd[vi] = MOI.get(opt, MOI.ConstraintDual(), ci_lt)
    end

    lower_variable_gt_indx = x._lower_variable_gt_indx
    lower_variable_gt = x._lower_variable_gt
    for i in 1:length(lower_variable_gt_indx)
        @inbounds vi = lower_variable_gt_indx[i]
        @inbounds ci_gt = lower_variable_gt[i]
        @inbounds x._lower_lvd[vi] = MOI.get(opt, MOI.ConstraintDual(), ci_gt)
    end

    return
end

"""
    preprocess!

Runs interval, linear, quadratic contractor methods followed by obbt and a
constraint programming walk up to tolerances specified in
`EAGO.Optimizer` object.
"""
function preprocess!(t::ExtensionType, x::Optimizer)

    # Sets initial feasibility
    feas = true
    rept = 0

    x._initial_volume = prod(upper_variable_bounds(x._current_node) -
                             lower_variable_bounds(x._current_node))

    # runs poor man's LP contractor
    if ((x.lp_depth >= x._iteration_count) & feas)
        feas = lp_bound_tighten(x)
    end

    # runs univariate quadratic contractor
    if ((x.quad_uni_depth >= x._iteration_count) & feas)
        for i = 1:x.quad_uni_reptitions
            feas = univariate_quadratic(x)
            (~feas) && (break)
        end
    end

    x._obbt_performed_flag = false
    if (x.obbt_depth >= x._iteration_count)
        #println("ran obbt... $(x.obbt_depth) >= $(x._iteration_count)")
        if feas
            x._obbt_performed_flag = true
            for i = 1:x.obbt_reptitions
                feas = obbt(x)
                (~feas) && (break)
            end
        end
    end

    if ((x.cp_depth >= x._iteration_count) & feas)
        println("attempted cpwalk!!!")
        feas = cpwalk(x)
    end

    x._final_volume = prod(upper_variable_bounds(x._current_node) -
                           lower_variable_bounds(x._current_node))
    x._preprocess_feasibility = feas

    return
end

"""
    update_relaxed_problem_box!

Updates the relaxed constraint by setting the constraint set of v == x* ,
xL_i <= x_i , and x_i <= xU_i for each such constraint added to the relaxed
    optimizer.
"""
function update_relaxed_problem_box!(x::Optimizer, y::NodeBB)

    opt = x.relaxed_optimizer
    lower_node_bnd = y.lower_variable_bounds
    upper_node_bnd = y.upper_variable_bounds

    lower_variable_et = x._lower_variable_et
    lower_variable_et_indx = x._lower_variable_et_indx
    for i in 1:length(lower_variable_et_indx)
        @inbounds ci = lower_variable_et[i]
        @inbounds ni = lower_variable_et_indx[i]
        @inbounds vb = lower_node_bnd[ni]
        MOI.set(opt, MOI.ConstraintSet(), ci, ET(vb))
    end

    lower_variable_lt = x._lower_variable_lt
    lower_variable_lt_indx = x._lower_variable_lt_indx
    for i in 1:length(lower_variable_lt_indx)
        @inbounds ci = lower_variable_lt[i]
        @inbounds ni = lower_variable_lt_indx[i]
        @inbounds vb = upper_node_bnd[ni]
        MOI.set(opt, MOI.ConstraintSet(), ci, LT(vb))
    end

    lower_variable_gt = x._lower_variable_gt
    lower_variable_gt_indx = x._lower_variable_gt_indx
    for i in 1:length(x._lower_variable_gt_indx)
        @inbounds ci = lower_variable_gt[i]
        @inbounds ni = lower_variable_gt_indx[i]
        @inbounds vb = lower_node_bnd[ni]
        MOI.set(opt, MOI.ConstraintSet(), ci, GT(vb))
    end

    return
end

"""
    interval_bound

Computes the natural interval extension of a ScalarAffineFunction or
ScalarQuadaraticFunction on a node y. Returns the lower bound if flag
is true and the upper bound if flag is false.
"""
function interval_bound(s::SAF, y::NodeBB, flag::Bool)
    val_lo = s.constant
    lo_bnds = y.lower_variable_bounds
    up_bnds = y.upper_variable_bounds
    for term in s.terms
        vi = term.variable_index.value
        coeff = term.coefficient
        if (coeff > 0.0)
            if flag
                @inbounds val_lo += coeff*lo_bnds[vi]
            else
                @inbounds val_lo += coeff*up_bnds[vi]
            end
        else
            if flag
                @inbounds val_lo += coeff*up_bnds[vi]
            else
                @inbounds val_lo += coeff*lo_bnds[vi]
            end
        end
    end
    return val_lo
end
function interval_bound(s::SQF, y::NodeBB, flag::Bool)
    lo_bnds = y.lower_variable_bounds
    up_bnds = y.upper_variable_bounds
    val_intv = Interval(s.constant)
    for term in s.affine_terms
        coeff = term.coefficient
        vi = term.variable_index.value
        @inbounds il1b = lo_bnds[vi]
        @inbounds iu1b = up_bnds[vi]
        val_intv += coeff*Interval(il1b, iu1b)
    end
    for term in s.quadratic_terms
        coeff = term.coefficient
        vi1 = term.variable_index_1.value
        vi2 = term.variable_index_2.value
        @inbounds il1b = lo_bnds[vi1]
        @inbounds iu1b = up_bnds[vi1]
        if vi1 == vi2
            val_intv += coeff*Interval(il1b, iu1b)^2
        else
            @inbounds il2b = lo_bnds[vi2]
            @inbounds iu2b = up_bnds[vi2]
            val_intv += coeff*Interval(il1b, iu1b)*Interval(il2b, iu2b)
        end
    end
    if flag
        return val_intv.lo
    end
    return val_intv.hi
end

"""
    interval_lower_bound!

A fallback lower bounding problem that consists of an natural interval extension
calculation. This is called when the optimizer used to compute the lower bound
does not return a termination and primal status code indicating that it
successfully solved the relaxation to a globally optimal point.
"""
function interval_lower_bound!(x::Optimizer, y::NodeBB)

    feas = true

    d = x._relaxed_evaluator

    if x._objective_is_nlp

        objective_lo = eval_objective_lo(d)
        constraints = d.constraints
        constr_num = d.constraint_number
        constraints_intv_lo = zeros(Float64, constr_num)
        constraints_intv_hi = zeros(Float64, constr_num)
        eval_constraint_lo!(d, constraints_intv_lo)
        eval_constraint_hi!(d, constraints_intv_hi)
        constraints_bnd_lo = d.constraints_lbd
        constrains_bnd_hi = d.constraints_ubd

        for i in 1:d.constraint_number
            @inbounds constraints_intv_li = constraints_bnd_lo[i]
            @inbounds constraints_intv_hi = constraints_bnd_hi[i]
            if (constaints_intv_li > constaints_intv_hi) ||
               (constaints_intv_hi < constaints_intv_li)
                feas = false
                break
            end
        end
    else
        if x._objective_is_sv
            obj_indx = x._objective_sv.variable.value
            @inbounds objective_lo = y.lower_variable_bounds[obj_indx]
        elseif x._objective_is_saf
            objective_lo = interval_bound(x._objective_saf, y, true)
        elseif x._objective_is_sqf
            objective_lo = interval_bound(x._objective_sqf, y, true)
        end
    end

    for (func, set, i) in x._linear_leq_constraints
        (~feas) && break
        if interval_bound(func, y, true) > set.upper
            feas = false
        end
    end
    for (func, set, i) in x._linear_geq_constraints
        (~feas) && break
        if interval_bound(func, y, false) < set.lower
            feas = false
        end
    end
    for (func, set, i) in x._linear_eq_constraints
        (~feas) && break
        if (interval_bound(func, y, true) > set.value) ||
           (interval_bound(func, y, false) < set.value)
            feas = false
        end
    end

    for (func, set, i) in x._quadratic_leq_constraints
        (~feas) && break
        if interval_bound(func, y, true) > set.upper
            feas = false
        end
    end
    for (func, set, i) in x._quadratic_geq_constraints
        (~feas) && break
        if interval_bound(func, y, false) < set.lower
            feas = false
        end
    end
    for (func, set, i) in x._quadratic_eq_constraints
        (~feas) && break
        if (interval_bound(func, y, true) > set.value) ||
           (interval_bound(func, y, false) < set.value)
            feas = false
        end
    end

    x._lower_feasibility = feas
    if feas
        if objective_lo > x._lower_objective_value
            x._lower_objective_value = objective_lo
        end
    else
        x._lower_objective_value = -Inf
    end

    return
end

"""
    lower_problem!

Constructs and solves the relaxation using the default EAGO relaxation scheme
and optimizer on node `y`.
"""
function lower_problem!(t::ExtensionType, x::Optimizer)

    y = x._current_node

    #println("    ")
    #println("    ")
    #println("    ")
    contains_optimum!(y, "Lower Problem Start")

    if ~x._obbt_performed_flag
        x._current_xref = @. 0.5*(y.lower_variable_bounds + y.upper_variable_bounds)
        update_relaxed_problem_box!(x, y)
        relax_problem!(x, x._current_xref, 1)
    end

    relax_objective!(x, x._current_xref)
    if (x.objective_cut_on)
        objective_cut_linear!(x, 1)
    end

    # Optimizes the object
    opt = x.relaxed_optimizer
    MOI.optimize!(opt)

    x._lower_termination_status = MOI.get(opt, MOI.TerminationStatus())
    x._lower_result_status = MOI.get(opt, MOI.PrimalStatus())
    valid_flag, feas_flag = is_globally_optimal(x._lower_termination_status, x._lower_result_status)
    #println("valid_flag: $(valid_flag)")
    #println("feas_flag: $(feas_flag)")

    if valid_flag
        if feas_flag
            x._lower_feasibility = true
            x._lower_objective_value = MOI.get(opt, MOI.ObjectiveValue())
            #println("lower_objective_value: $(x._lower_objective_value)")
            @inbounds x._lower_solution[:] = MOI.get(opt, MOI.VariablePrimal(), x._lower_variable_index)
            x._cut_add_flag = x._lower_feasibility
            set_dual!(x)
        else
            x._cut_add_flag = false
            x._lower_feasibility  = false
            x._lower_objective_value = -Inf
        end
    else
        interval_lower_bound!(x, y)
        #println("lower feasibility: $(x._lower_feasibility)")
        #println("lower objective: $(x._lower_objective_value)")
        x._cut_add_flag = false
    end
    contains_optimum!(y, "Lower Problem End")
    return
end

"""
    cut_update

Updates the internal storage in the optimizer after a valid feasible cut is added.
"""
function cut_update(x::Optimizer)

    contains_optimum!(x._current_node, "Cut Update Start")
    x._cut_feasibility = true

    opt = x.relaxed_optimizer
    obj_val = MOI.get(opt, MOI.ObjectiveValue())
    prior_obj_val = (x._cut_iterations == 2) ? x._lower_objective_value : x._cut_objective_value

    if prior_obj_val < obj_val
        x._cut_objective_value = obj_val
        x._lower_objective_value = obj_val
        x._lower_termination_status = x._cut_termination_status
        x._lower_result_status = x._cut_result_status
        @inbounds x._cut_solution[:] = MOI.get(opt, MOI.VariablePrimal(), x._lower_variable_index)
        copyto!(x._lower_solution, x._cut_solution)
        set_dual!(x)
        x._cut_add_flag = true
    else
        x._cut_add_flag = false
    end
    contains_optimum!(x._current_node, "Cut Update End")

    return
end

"""
    cut_condition

Checks if a cut should be added and computes a new reference point to add the
cut at. If no cut should be added the constraints not modified in place are
deleted from the relaxed optimizer and the solution is compared with the
interval lower bound. The best lower bound is then used.
"""
function cut_condition(t::ExtensionType, x::Optimizer)



    flag = x._cut_add_flag
    flag &= (x._cut_iterations < x.cut_max_iterations)

    if flag
        y = x._current_node
        xprior = x._current_xref
        xsol = (x._cut_iterations > 1) ? x._cut_solution : x._lower_solution
        xnew = (1.0 - x.cut_cvx)*mid(y) + x.cut_cvx*xsol
        if norm((xprior - xnew)./diam(y), 1) > x.cut_tolerance
            x._current_xref = xnew
            flag &= true
        else
            flag &= false
        end
    end

    if ~flag
        # if not further cuts then empty the added cuts
        for i in 2:x._cut_iterations
            for ci in x._quadratic_ci_leq[i]
                MOI.delete(x.relaxed_optimizer, ci)
            end
            for ci in x._quadratic_ci_geq[i]
                MOI.delete(x.relaxed_optimizer, ci)
            end
            for (ci1,ci2) in x._quadratic_ci_eq[i]
                MOI.delete(x.relaxed_optimizer, ci1)
                MOI.delete(x.relaxed_optimizer, ci2)
            end
            for ci in x._lower_nlp_affine[i]
                MOI.delete(x.relaxed_optimizer, ci)
            end
            for ci in x._upper_nlp_affine[i]
                MOI.delete(x.relaxed_optimizer, ci)
            end
        end
        if x._objective_cut_set !== -1
            if ~x._objective_is_sv
                for i in 2:x._cut_iterations
                    ci = x._objective_cut_ci_saf[i]
                    MOI.delete(x.relaxed_optimizer, ci)
                end
            end
        end
    end

    # check to see if interval bound is preferable
    if x._lower_feasibility
        y = x._current_node
        if x._objective_is_nlp
            intv_lo = eval_objective_lo(x._relaxed_evaluator)
        else
            if x._objective_is_sv
                obj_indx = x._objective_sv.variable.value
                @inbounds intv_lo = y.lower_variable_bounds[obj_indx]
            elseif x._objective_is_saf
                intv_lo = interval_bound(x._objective_saf, y, true)
            elseif x._objective_is_sqf
                intv_lo = interval_bound(x._objective_sqf, y, true)
            end
        end
        if (intv_lo > x._lower_objective_value)
            x._lower_objective_value = intv_lo
            fill!(x._lower_lvd, 0.0)
            fill!(x._lower_uvd, 0.0)
        end
    end
    contains_optimum!(x._current_node, "Cut Condition End")
    x._cut_iterations += 1

    return flag
end

"""
    add_cut!

Adds a cut for each constraint and the objective function to the subproblem.
"""
function add_cut!(t::ExtensionType, x::Optimizer)


    #println("x._cut_iterations: $(x._cut_iterations)")
    relax_problem!(x, x._current_xref, x._cut_iterations)
    relax_objective!(x, x._current_xref)
    if x.objective_cut_on
        objective_cut_linear!(x, x._cut_iterations)
    end

    # Optimizes the object
    opt = x.relaxed_optimizer
    MOI.optimize!(opt)

    x._cut_termination_status = MOI.get(opt, MOI.TerminationStatus())
    x._cut_result_status = MOI.get(opt, MOI.PrimalStatus())
    valid_flag, feas_flag = is_globally_optimal(x._cut_termination_status, x._cut_result_status)

    if valid_flag
        if feas_flag
            cut_update(x)
        else
            x._cut_add_flag = false
            x._lower_feasibility  = false
            x._lower_objective_value = -Inf
        end
    else
        x._cut_add_flag = false
    end
    #contains_optimum!(y, "Add Cut End")

    return
end

"""
    default_nlp_heurestic

Default check to see if the upper bounding problem should be run. By default,
The upper bounding problem is run on every node up to depth `upper_bounding_depth`
and is triggered with a probability of `0.5^(depth - upper_bounding_depth)`
afterwards.
"""
function default_nlp_heurestic(x::Optimizer, y::NodeBB)
    bool = false
    bool |= (y.depth <= x.upper_bounding_depth)
    bool |= (rand() < 0.5^(y.depth - x.upper_bounding_depth))
    return bool
end

"""
    solve_local_nlp!

Constructs and solves the problem locally on on node `y` updated the upper
solution informaton in the optimizer.
"""
function solve_local_nlp!(x::Optimizer{S,T}) where {S <: MOI.AbstractOptimizer, T <: MOI.AbstractOptimizer}

    y = x._current_node

    if default_nlp_heurestic(x,y)

        nvar = x._variable_number
        x.upper_optimizer = x.upper_factory()
        upper_optimizer = x.upper_optimizer
        upper_vars = MOI.add_variables(upper_optimizer, nvar)
        lvb = 0.0
        uvb = 0.0
        x0 = 0.0
        for i in 1:nvar
            @inbounds var = x._variable_info[i]
            @inbounds svi = upper_vars[i]
            @inbounds sv = MOI.SingleVariable(upper_vars[i])
            if var.is_integer
            else
                @inbounds lvb = y.lower_variable_bounds[i]
                @inbounds uvb = y.upper_variable_bounds[i]
                if var.is_fixed
                    MOI.add_constraint(upper_optimizer, sv, ET(lvb))
                elseif var.has_lower_bound
                    if var.has_upper_bound
                        MOI.add_constraint(upper_optimizer, sv, LT(uvb))
                        MOI.add_constraint(upper_optimizer, sv, GT(lvb))
                    else
                        MOI.add_constraint(upper_optimizer, sv, GT(lvb))
                    end
                elseif var.has_upper_bound
                    MOI.add_constraint(upper_optimizer, sv, LT(uvb))
                end
                x0 = @. 0.5*(lvb + uvb)
                MOI.set(upper_optimizer, MOI.VariablePrimalStart(), svi, x0)
            end
        end

        # Add linear and quadratic constraints to model
        for (func, set) in x._linear_leq_constraints
             MOI.add_constraint(upper_optimizer, func, set)
        end
        for (func, set) in x._linear_geq_constraints
            MOI.add_constraint(upper_optimizer, func, set)
        end
        for (func, set) in x._linear_eq_constraints
            MOI.add_constraint(upper_optimizer, func, set)
        end

        for (func, set) in x._quadratic_leq_constraints
            MOI.add_constraint(upper_optimizer, func, set)
        end
        for (func, set) in x._quadratic_geq_constraints
            MOI.add_constraint(upper_optimizer, func, set)
        end
        for (func, set) in x._quadratic_eq_constraints
            MOI.add_constraint(upper_optimizer, func, set)
        end

        # Add nonlinear evaluation block
        MOI.set(upper_optimizer, MOI.NLPBlock(), x._nlp_data)

        MOI.set(upper_optimizer, MOI.ObjectiveSense(), MOI.MIN_SENSE)
        if x._objective_is_sv
            MOI.set(upper_optimizer, MOI.ObjectiveFunction{SV}(), x._objective_sv)
        elseif x._objective_is_saf
            MOI.set(upper_optimizer, MOI.ObjectiveFunction{SAF}(), x._objective_saf)
        elseif x._objective_is_sqf
            MOI.set(upper_optimizer, MOI.ObjectiveFunction{SQF}(), x._objective_sqf)
        end

        # Optimizes the object
        MOI.optimize!(upper_optimizer)

        # Process output info and save to CurrentUpperInfo object
        x._upper_termination_status = MOI.get(upper_optimizer, MOI.TerminationStatus())
        x._upper_result_status = MOI.get(upper_optimizer, MOI.PrimalStatus())

        if is_feasible_solution(x._upper_termination_status, x._upper_result_status)
            x._upper_feasibility = true
            value = MOI.get(upper_optimizer, MOI.ObjectiveValue())
            x._upper_objective_value = value  + 1.0E-6
            x._best_upper_value = min(value, x._best_upper_value)
            x._upper_solution[1:end] = MOI.get(upper_optimizer, MOI.VariablePrimal(), upper_vars)
        else
            x._upper_feasibility = false
            x._upper_objective_value = Inf
        end
    else
        x._upper_feasibility = false
        x._upper_objective_value = Inf
    end
    return
end

"""
    upper_problem!

Default upper bounding problem which simply calls `solve_local_nlp!`.
"""
function upper_problem!(t::ExtensionType, x::Optimizer)

    solve_local_nlp!(x)
    return
end


"""
    postprocess!

Perfoms duality-based bound tightening on the `y`.
"""
function postprocess!(t::ExtensionType, x::Optimizer)

    if (x.dbbt_depth > x._iteration_count)
        variable_dbbt!(x._current_node, x._lower_lvd, x._lower_uvd,
                       x._lower_objective_value, x._global_upper_bound,
                       x._variable_number)
    end

    x._postprocess_feasibility = true

    return
end

"""
    optimize_hook!

Provides a hook for extensions to EAGO as opposed to standard global, local,
or linear solvers.
"""
function optimize_hook!(t::ExtensionType, x::Optimizer)
    return
end
