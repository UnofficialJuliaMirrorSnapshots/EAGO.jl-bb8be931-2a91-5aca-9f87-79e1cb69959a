"""
    midpoint_affine!

Constructs a midpoint affine bound over node `n` by evaluating the relaxations
of the objective and subgradients and their respective (sub)gradients at the
midpoint of the domain of `n`. The resulting affine objective and constraints
are added to the `trg` optimizer.
"""
function midpoint_affine!(src::Optimizer,trg,n::NodeBB,r)
    ngrad = src.variable_number
    nx =  src.state_variables
    np = ngrad - nx
    var = src.upper_variables

    src.working_evaluator_block.evaluator.current_node = n
    midx = (n.upper_variable_bounds + n.lower_variable_bounds)/2.0

    # Add objective
    if src.working_evaluator_block.has_objective
        # Calculates relaxation and subgradient
        df = zeros(Float64, np)
        f = MOI.eval_objective(src.working_evaluator_block.evaluator, midx)
        MOI.eval_objective_gradient(src.working_evaluator_block.evaluator, df, midx)

        # Add objective relaxation to model
        saf_const = f
        for i in (1+nx):ngrad
            grad_c = df[i-nx]
            midx_c = midx[i]
            saf_const -= midx_c*grad_c
        end
        saf = MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(df, var[(1+nx):ngrad]), saf_const)
        #println("objective cut: saf $saf")
        MOI.set(trg, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(), saf)
        # Add objective cut if nonlinear (if bound is finite)
        if src.global_upper_bound < Inf
            cut_saf = MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(df, var[(1+nx):ngrad]), 0.0)
            set = MOI.LessThan(src.global_upper_bound-saf_const)
            MOI.add_constraint(trg, cut_saf, set)
        end
    end

    # Add other affine constraints
    if length(src.working_evaluator_block.constraint_bounds) > 0
        leng = length(src.working_evaluator_block.constraint_bounds)
        g = zeros(Float64,leng)
        dg = zeros(Float64,leng,np)

        g_cc = zeros(Float64,leng)
        dg_cc = zeros(Float64,leng,np)

        MOI.eval_constraint(src.working_evaluator_block.evaluator, g, midx)
        MOI.eval_constraint_jacobian(src.working_evaluator_block.evaluator, dg, midx)

        eval_constraint_cc(src.working_evaluator_block.evaluator, g_cc, midx)
        eval_constraint_cc_grad(src.working_evaluator_block.evaluator, dg_cc, midx)

        for (j,bns) in enumerate(src.working_evaluator_block.constraint_bounds)
            if bns.upper != Inf
                constant = g[j]
                for i in 1:np
                    constant -= midx[i+nx]*dg[j,i]
                end
                saf = MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(dg[j,:], var[(1+nx):ngrad]), 0.0)
                set = MOI.LessThan(bns.upper-constant)
                MOI.add_constraint(trg, saf, set)
            end
            if bns.lower > -Inf
                constant = g_cc[j]
                for i in 1:np
                    constant -= midx[i+nx]*dg_cc[j,i]
                end
                saf = MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(-dg_cc[j,:], var[(1+nx):ngrad]), 0.0)
                set = MOI.LessThan(constant)
                MOI.add_constraint(trg, saf, set)
            end
        end
    end
end
