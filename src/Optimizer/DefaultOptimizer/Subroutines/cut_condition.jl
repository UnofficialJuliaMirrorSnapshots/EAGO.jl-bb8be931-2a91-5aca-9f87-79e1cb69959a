"""
    default_cut_condition

Branch-and-cut feature currently under development. Currently, returns false.
"""
default_cut_condition(x::Optimizer) = false
#CutContConditions(x::Optimizer) = (x.CutIterations < x.MaxCutIterations) ? false : CheckTightCut(x)
