module EAGO

    import MathOptInterface

    using Printf
    using SparseArrays
    using LinearAlgebra

    using JuMP

    # To drop CPLEX from support in favor Clp
    using Ipopt, GLPK
    using DiffRules, ForwardDiff, ReverseDiff, Calculus
    using Reexport, StaticArrays

    import IntervalArithmetic: +, -, *, /, convert, in, isempty, one, zero,
                               real, eps, max, min, abs, exp,
                               expm1, log, log2, log10, log1p, sqrt,
                               sin, cos, tan, min, max, sec, csc, cot, step,
                               sign, dist, mid, pow, Interval, sinh, cosh, ∩,
                               IntervalBox, bisect, isdisjoint
                               # faster versions of: ^, inv, exp2, exp10, tanh,
                               # asinh, acosh, atanh


    const MOI = MathOptInterface
    const MOIU = MOI.Utilities

    include("mccormick_library/mccormick.jl")
    using .McCormick

    import Base: eltype, copy, length

    # Add storage types for EAGO optimizers
    export NodeBB, get_history, get_lower_bound, get_upper_bound, get_lower_time,
           get_upper_time, get_preprocess_time, get_postprocess_time, get_lower_bound, get_solution_time,
           get_iteration_number, get_node_count, get_absolute_gap, get_relative_gap
    include("optimizer/branch_bound/node_bb.jl")
    include("optimizer/branch_bound/subproblem_info.jl")
    include("relaxations/relax_scheme.jl")
    include("optimizer/moi_wrapper/optimizer.jl")

    # Routines for (branch-and-bound/cut)
    include("optimizer/branch_bound/branch_bound.jl")

    # Routines for loading relax models
    include("relaxations/relax_model.jl")

    # MOI wrappers and basic optimization routines
    include("optimizer/moi_wrapper/moi_wrapper.jl")

    # Domain reduction subroutines
    include("optimizer/domain_reduction/domain_reduction.jl")         # special character warning

    # Default subroutines for optimizers
    include("optimizer/default_optimizer/default_optimizer.jl")

    # Adds the parametric interval methods
    export parametric_interval_params, param_intv_contractor
    include("parametric_interval/parametric_interval.jl")

    # Solve for implicit function optimization
    export ImplicitLowerEvaluator, build_lower_evaluator!,
           ImplicitUpperEvaluator, MidPointUpperEvaluator,
           build_implicitupperevaluator!, build_midpointupperevaluator!, solve_implicit
    include("optimizer/implicit_optimizer/implicit_optimizer.jl")

    # Import the script solving utilities
    include("script/script_module.jl")

    export solve_script
    include("script/solve_script.jl")

    # Routines for solving SIPs
    export SIP_Options, SIP_Result, explicit_sip_solve, implicit_sip_solve
    include("semiinfinite/semi_infinite.jl")
end
