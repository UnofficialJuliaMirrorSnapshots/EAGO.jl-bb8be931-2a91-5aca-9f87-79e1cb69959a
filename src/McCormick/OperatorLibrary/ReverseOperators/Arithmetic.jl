"""
    plus_rev

Creates reverse McCormick contractor for `a` = `b` +`c`
"""
function plus_rev(a::MC, b::MC, c::MC)  # a = b + c
    b = b ∩ (a - c)
    c = c ∩ (a - b)
    a,b,c
end
plus_rev(a,b,c) = plus_rev(promote(a,b,c)...)

"""
    minus_rev

Creates reverse McCormick contractor for `a` = `b`- `c`
"""
function minus_rev(a::MC, b::MC, c::MC)  # a = b - c
    b = b ∩ (a + c)
    c = c ∩ (b - a)
    a,b,c
end
minus_rev(a::MC, b::MC) = (b = -a; return (a, b))     # a = -b
minus_rev(a,b,c) = minus_rev(promote(a,b,c)...)
minus_rev(a,b) = minus_rev(promote(a,b)...)

"""
    mul_rev

Creates reverse McCormick contractor for `a` = `b`*`c`
"""
function mul_rev(a::MC, b::MC, c::MC)  # a = b * c
    temp1 = a / b
    temp2 = a / c
    ((0.0 ∉ a.Intv) ||  (0.0 ∉ b.Intv)) && (c = c ∩ (a / b))
    ((0.0 ∉ a.Intv) ||  (0.0 ∉ c.Intv)) && (b = b ∩ (a / c))
    a,b,c
end
mul_rev(a::MC{N},b::MC{N},c::Float64) where N = mul_rev(a,b,MC{N}(c))
mul_rev(a::MC{N},b::Float64,c::MC{N}) where N = mul_rev(a,MC{N}(b),c)

"""
    div_rev

Creates reverse McCormick contractor for `a` = `b`/`c`
"""
function div_rev(a::MC, b::MC, c::MC)  # a = b / c
    b = b ∩ (a * c)
    c = c ∩ (b / a)
    a,b,c
end
div_rev(a,b,c) = div_rev(promote(a,b,c)...)

"""
    inv_rev

Creates reverse McCormick contractor for `a` = `inv(b)`
"""
function inv_rev(a::MC, b::MC)  # a = inv(b)
    b = b ∩ inv(a)
    a,b
end
inv_rev(a,b) = inv_rev(promote(a,b)...)

"""
    pow_rev

Creates reverse McCormick contractor for `a` = `b`^`c`
"""
function pow_rev(a::MC, b::MC, c::MC)  # a = b^c
    b = b ∩ (a^(inv(c)))
    c = c ∩ (log(a) / log(b))
    a,b,c
end
pow_rev(a, b, c) = pow_rev(promote(a, b, c)...)


"""
    sqrt_rev

Creates reverse McCormick contractor for `a` = `sqrt(b)`
"""
function sqrt_rev(a::MC, b::MC)  # a = sqrt(b)
    b = b ∩ (a^2)
    a,b
end

"""
    sqr_rev

Creates reverse McCormick contractor for `a` = `sqrt(b)`
"""
sqr_rev(f, x)  = pow_rev(f,x,2)


"""
    abs_rev!

Creates reverse McCormick contractor for `a` = `abs(b)`
"""
abs_rev(a::MC, b::MC) = (a,b)
#=
function abs_rev!(y::MC{N}, x::MC{N}) where N   # y = abs(x); refine x

    y_new = y ∩ (0..∞)

    x1 = y_new ∩ x
    x2 = -(y_new ∩ (-x))

    cv =
    cc =
    cv_grad =
    cc_grad =
    Intv = hull(Intv(x1),Intv(x2))


    y = MC{N}(cv, cc, Intv, cv_grad, cc_grad, y.cnst)

    return
end
=#
