"""
    exp_rev

Reverse McCormick operator for `exp`.
"""
function exp_rev(y::MC, x::MC)
    y = y ∩ IntervalType(0.0, ∞)
    x = x ∩ log(y)
    y,x
end

"""
    exp2_rev

Reverse McCormick operator for `exp2`.
"""
function exp2_rev(y::MC, x::MC)
    y = y ∩ IntervalType(0.0, ∞)
    x = x ∩ log2(y)
    y,x
end

"""
    exp10_rev

Reverse McCormick operator for `exp10`.
"""
function exp10_rev(y::MC, x::MC)
    y = y ∩ IntervalType(0.0, ∞)
    x = x ∩ log10(y)
    y,x
end

"""
    expm1_rev

Reverse McCormick operator for `expm1`.
"""
function expm1_rev(y::MC, x::MC)
    y = y ∩ IntervalType(-1.0, ∞)
    x = x ∩ log1p(y)
    y,x
end

"""
    log_rev

Reverse McCormick operator for `log`.
"""
function log_rev(y::MC, x::MC)
    x = x ∩ exp(y)
    y,x
end

"""
    log2_rev

Reverse McCormick operator for `log2`.
"""
function log2_rev(y::MC, x::MC)
    x = x ∩ exp2(y)
    y,x
end

"""
    log10_rev

Reverse McCormick operator for `log10`.
"""
function log10_rev(y::MC, x::MC)
    x = x ∩ exp10(y)
    y,x
end

"""
    log1p_rev

Reverse McCormick operator for `log1p`.
"""
function log1p_rev(y::MC, x::MC)
    x = x ∩ expm1(y)
    y,x
end
