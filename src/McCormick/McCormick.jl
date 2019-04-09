@reexport module McCormick

using StaticArrays, CommonSubexpressions, DiffRules, BenchmarkTools, LinearAlgebra

import Base: +, -, *, /, convert, in, isempty, one, zero, real, eps, max, min,
             abs, inv, exp, exp2, exp10, expm1, log, log2, log10, log1p, acosh,
             sqrt, sin, cos, tan, min, max, sec, csc, cot, ^, step, sign, intersect,
             promote_rule

import IntervalArithmetic: dist, mid, pow, +, -, *, /, convert, in, isempty,
                           one, zero, real, eps, max, min, abs, exp,
                           expm1, log, log2, log10, log1p, sqrt, ^,
                           sin, cos, tan, min, max, sec, csc, cot, step,
                           sign, dist, mid, pow, Interval, interval, sinh, cosh,
                           ∩, IntervalBox, pi_interval, bisect, isdisjoint, length

# Export forward operators
export MC, cc, cv, Intv, lo, hi,  cc_grad, cv_grad, cnst, +, -, *, /, convert,
       one, zero, dist, real, eps, mid, exp, exp2, exp10, expm1, log, log2,
       log10, log1p, acosh, sqrt, sin, cos, tan, min, max, sec, csc, cot, ^,
       abs, step, sign, pow, in, isempty, intersect, length
       #acos, asin, atan, sinh, cosh, tanh, asinh, atanh, inv, sqr

# Export inplace operators
export plus!, mult!, min!, max!, minus!, div!, exp!, exp2!, exp10!, expm1!,
       log!, log2!, log10!, log1p!, sin!, cos!, tan!, asin!, acos!, atan!,
       sinh!, cosh!, tanh!, asinh!, acosh!, atanh!, abs!, sqr!, sqrt!, pow!

export seed_gradient, IntervalType, set_mc_differentiability!, set_multivar_refine!,
       set_tolerance!, set_iterations!, MC_param

# Export reverse operators
export plus_rev, mul_rev, min_rev, max_rev, minus_rev, div_rev, exp_rev,
       exp2_rev, exp10_rev, expm1_rev, log_rev, log2_rev, log10_rev,
       log1p_rev, sin_rev, cos_rev, tan_rev, asin_rev, acos_rev, atan_rev,
       sinh_rev, cosh_rev, tanh_rev, asinh_rev, acosh_rev, atanh_rev,
       abs_rev, sqr_rev, sqrt_rev, pow_rev

# Export utility operators
#=
export grad, zgrad, ∩, mid3, MC_param, mid_grad, seed_g, line_seg, dline_seg,
       outer_rnd, cut, set_valid_check, set_subgrad_refine, set_multivar_refine,
       set_outer_rnd, tighten_subgrad, set_iterations, set_tolerance,
       default_options, value, mincv, maxcc, promote_rule
=#
export mc_opts, gen_expansion_params, gen_expansion_params!, implicit_relax_h,
       implicit_relax_h!, implicit_relax_f, implicit_relax_fg

include("convexity_rules.jl")

include("./Utilities/constants.jl")
include("Utilities/inner_utilities.jl")
include("Utilities/fast_intervals.jl")

include("OperatorLibrary/type.jl")

include("Utilities/api_utilities.jl")
include("Utilities/root_finding.jl")

include("OperatorLibrary/ForwardOperators/forward.jl")
include("OperatorLibrary/ReverseOperators/reverse.jl")

include("ImplicitRoutines/implicit.jl")

end
