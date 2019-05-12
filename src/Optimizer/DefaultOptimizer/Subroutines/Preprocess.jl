"""
    default_preprocess!

Runs interval, linear, and quadratic contractor methods up to tolerances
specified in `EAGO.Optimizer` object.
"""
function default_preprocess!(x::Optimizer,y::NodeBB)

    #println("start preprocess")
    # Sets initial feasibility
    feas = true; rept = 0

    # runs poor man's LP contractor
    if (x.poor_man_lp_depth > x.current_iteration_count)
        for i in 1:x.poor_man_lp_reptitions
            feas = poor_man_lp(x,y)
            (~feas) && (break)
        end
    end
    #println("finished poor man lp")

    # runs univariate quadratic contractor
    if ((x.univariate_quadratic_depth > x.current_iteration_count) && feas)
        for i in 1:x.univariate_quadratic_reptitions
            feas = univariate_quadratic(x,y)
            (~feas) && (break)
        end
    end
    #println("finished quadratic univ")

    if ((x.obbt_depth > x.current_iteration_count) && feas)
        for i in 1:x.obbt_reptitions
            feas = obbt(x,y)
            (~feas) && (break)
        end
    end
    #println("obbt")

    if ((x.cp_depth > x.current_iteration_count) && feas)
        for i in 1:x.cp_reptitions
            feas = cpwalk(x,y)
            (~feas) && (break)
        end
    end

    println("obbt feas results: $feas at iteration $(x.current_iteration_count)")
    x.current_preprocess_info.feasibility = feas
    #println("end preprocess")
end
