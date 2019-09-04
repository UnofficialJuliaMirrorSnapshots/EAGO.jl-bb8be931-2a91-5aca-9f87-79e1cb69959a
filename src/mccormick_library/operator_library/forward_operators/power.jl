# UPDATED CONVEX/CONCAVE RELAXATIONS, NEED TO UPDATE ALL MC OBJECTS....

# defines square operator
@inline sqr(x::Float64) = x*x
@inline cv_sqr_NS(x::Float64, xL::Float64, xU::Float64) = x^2
@inline dcv_sqr_NS(x::Float64, xL::Float64, xU::Float64) = 2.0*x
@inline cc_sqr(x::Float64, xL::Float64, xU::Float64) = (xU > xL) ? xL^2 + (xL + xU)*(x - xL) : xU^2
@inline dcc_sqr(x::Float64, xL::Float64, xU::Float64) = (xU > xL) ? (xL + xU) : 0.0
@inline function cv_sqr(x::Float64, xL::Float64, xU::Float64)
    (0.0 <= xL || xU <= 0.0) && return x^2
	((xL < 0.0) && (0.0 <= xU) && (0.0 <= x)) && return (x^3)/xU
	return (x^3)/xL
end
@inline function dcv_sqr(x::Float64, xL::Float64, xU::Float64)
    (0.0 <= xL || xU <= 0.0) && return 2.0*x
	((xL < 0.0) && (0.0 <= xU) && (0.0 <= x)) && (3.0*x^2)/xU
	return (3.0*x^2)/xL
end
@inline function sqr_kernel(x::MC{N}, y::Interval{Float64}) where N
    eps_max = abs(x.Intv.hi) > abs(x.Intv.lo) ?  x.Intv.hi : x.Intv.lo
	if (x.Intv.lo < 0.0 < x.Intv.hi)
		eps_min = 0.0
	else
		eps_min = abs(x.Intv.hi) > abs(x.Intv.lo) ?  x.Intv.lo : x.Intv.hi
	end
	midcc, cc_id = mid3(x.cc, x.cv, eps_max)
	midcv, cv_id = mid3(x.cc, x.cv, eps_min)
	cc = cc_sqr(midcc, x.Intv.lo, x.Intv.hi)
	dcc = dcc_sqr(midcc, x.Intv.lo, x.Intv.hi)
	if (MC_param.mu >= 1)
	   cv = cv_sqr(midcv, x.Intv.lo, x.Intv.hi)
	   dcv = dcv_sqr(midcv, x.Intv.lo, x.Intv.hi)
	   gdcc1 = dcc_sqr(x.cv, x.Intv.lo, x.Intv.hi)
	   gdcv1 = dcv_sqr(x.cv, x.Intv.lo, x.Intv.hi)
	   gdcc2 = dcc_sqr(x.cc, x.Intv.lo, x.Intv.hi)
	   gdcv2 = dcv_sqr(x.cc, x.Intv.lo, x.Intv.hi)
	   cv_grad = max(0.0, gdcv1)*x.cv_grad + min(0.0, gdcv2)*x.cc_grad
	   cc_grad = min(0.0, gdcc1)*x.cv_grad + max(0.0, gdcc2)*x.cc_grad
	 else
	   cv = cv_sqr_NS(midcv, x.Intv.lo, x.Intv.hi)
	   dcv = dcv_sqr_NS(midcv, x.Intv.lo, x.Intv.hi)
	   cc_grad = mid_grad(x.cc_grad, x.cv_grad, cc_id)*dcc
	   cv_grad = mid_grad(x.cc_grad, x.cv_grad, cv_id)*dcv
	   cv, cc, cv_grad, cc_grad = cut(y.lo, y.hi, cv, cc, cv_grad, cc_grad)
	 end
	 return MC{N}(cv, cc, y, cv_grad, cc_grad, x.cnst)
end
@inline sqr(x::MC) = sqr_kernel(x, (x.Intv)^2)

# convex/concave relaxation (Khan 3.1-3.2) of integer powers of 1/x for positive reals
@inline pow_deriv(x::Float64, n::Int) = n*x^(n-1)
@inline cv_npp_or_pow4(x::Float64, xL::Float64, xU::Float64, n::Integer) = x^n, n*x^(n-1)
@inline cc_npp_or_pow4(x::Float64, xL::Float64, xU::Float64, n::Integer) = dline_seg(^, pow_deriv, x, xL, xU, n)
# convex/concave relaxation of integer powers of 1/x for negative reals
@inline function cv_negpowneg(x::Float64, xL::Float64, xU::Float64, n::Integer)
  isodd(n) && (return dline_seg(^, pow_deriv, x, xL, xU, n))
  return x^n, n*x^(n-1)
end
@inline function cc_negpowneg(x::Float64, xL::Float64, xU::Float64, n::Integer)
  isodd(n) && (return x^n,n*x^(n-1))
  return dline_seg(^, pow_deriv, x, xL, xU, n)
end
# convex/concave relaxation of odd powers
@inline function cv_powodd(x::Float64, xL::Float64, xU::Float64, n::Integer)
    (xU <= 0.0) && (return dline_seg(^, powodd_deriv, x, xL, xU, n))
    (0.0 <= xL) && (return x^n, n*x^(n - 1))
    val = (xL^n)*(xU - x)/(xU - xL) + (max(0.0, x))^n
    dval = -(xL^n)/(xU - xL) + n*(max(0.0, x))^(n-1)
    return val, dval
end
@inline function cc_powodd(x::Float64, xL::Float64, xU::Float64, n::Integer)
    (xU <= 0.0) && (return x^n, n*x^(n - 1))
    (0.0 <= xL) && (return dline_seg(^, powodd_deriv, x, xL, xU, n))
    val = (xU^n)*(x - xL)/(xU - xL) + (min(0.0, x))^n
    dval = (xU^n)/(xU - xL) + n*(min(0.0, x))^(n-1)
    return val, dval
end

@inline function npp_or_pow4(x::MC{N}, c::Integer, y::Interval{Float64}) where N
  if (x.Intv.hi < 0.0)
    eps_min = x.Intv.hi
    eps_max = x.Intv.lo
  elseif (x.Intv.lo > 0.0)
    eps_min = x.Intv.lo
    eps_max = x.Intv.hi
  else
    eps_min = 0.0
    eps_max = (abs(x.Intv.lo) >= abs(x.Intv.hi)) ? x.Intv.lo : x.Intv.hi
  end
  midcc, cc_id = mid3(x.cc, x.cv, eps_max)
  midcv, cv_id = mid3(x.cc, x.cv, eps_min)
  cc, dcc = cc_npp_or_pow4(midcc, x.Intv.lo, x.Intv.hi, c)
  cv, dcv = cv_npp_or_pow4(midcv, x.Intv.lo, x.Intv.hi, c)
  if (MC_param.mu >= 1)
    gcc1, gdcc1 = cc_npp_or_pow4(x.cv, x.Intv.lo, x.Intv.hi, c)
    gcv1, gdcv1 = cv_npp_or_pow4(x.cv, x.Intv.lo, x.Intv.hi, c)
    gcc2, gdcc2 = cc_npp_or_pow4(x.cc, x.Intv.lo, x.Intv.hi, c)
    gcv2, gdcv2 = cv_npp_or_pow4(x.cc, x.Intv.lo, x.Intv.hi, c)
    cv_grad = max(0.0, gdcv1)*x.cv_grad + min(0.0, gdcv2)*x.cc_grad
    cc_grad = min(0.0, gdcc1)*x.cv_grad + max(0.0, gdcc2)*x.cc_grad
  else
    cc_grad = mid_grad(x.cc_grad, x.cv_grad, cc_id)*dcc
    cv_grad = mid_grad(x.cc_grad, x.cv_grad, cv_id)*dcv
		cv, cc, cv_grad, cc_grad = cut(y.lo, y.hi, cv, cc, cv_grad, cc_grad)
  end
  return MC{N}(cv, cc, y, cv_grad, cc_grad, x.cnst)
end
@inline function pos_odd(x::MC{N}, c::Integer, y::Interval{Float64}) where N
    midcc, cc_id = mid3(x.cc, x.cv, x.Intv.hi)
    midcv, cv_id = mid3(x.cc, x.cv, x.Intv.lo)
    cc, dcc = cc_powodd(midcc, x.Intv.lo, x.Intv.hi, c)
    cv, dcv = cv_powodd(midcv, x.Intv.lo, x.Intv.hi, c)
    if (MC_param.mu >= 1)
        gcc1, gdcc1 = cc_powodd(x.cv, x.Intv.lo, x.Intv.hi, c)
        gcv1, gdcv1 = cv_powodd(x.cv, x.Intv.lo, x.Intv.hi, c)
        gcc2, gdcc2 = cc_powodd(x.cc, x.Intv.lo, x.Intv.hi, c)
        gcv2, gdcv2 = cv_powodd(x.cc, x.Intv.lo, x.Intv.hi, c)
        cv_grad = max(0.0, gdcv1)*x.cv_grad + min(0.0, gdcv2)*x.cc_grad
        cc_grad = min(0.0, gdcc1)*x.cv_grad + max(0.0, gdcc2)*x.cc_grad
    else
        cc_grad = mid_grad(x.cc_grad, x.cv_grad, cc_id)*dcc
        cv_grad = mid_grad(x.cc_grad, x.cv_grad, cv_id)*dcv
				cv, cc, cv_grad, cc_grad = cut(y.lo, y.hi, cv, cc, cv_grad, cc_grad)
    end
    return MC{N}(cv, cc, y, cv_grad, cc_grad, x.cnst)
end

@inline function neg_powneg_odd(x::MC{N}, c::Integer, y::Interval{Float64}) where {N}
  xL = x.Intv.lo
  xU = x.Intv.hi
  eps_max = xU
  eps_min = xL
  if (MC_param.mu >= 1)
    midcc, cc_id = mid3(x.cc, x.cv, eps_max)
    midcv, cv_id = mid3(x.cc, x.cv, eps_min)
    cc, dcc = cc_negpowneg(midcc, xL, xU, c)
    cv, dcv = cv_negpowneg(midcv, xL, xU, c)
    gcc1, gdcc1 = cc_negpowneg(x.cv, xL, xU, c)
    gcv1, gdcv1 = cv_negpowneg(x.cv, xL, xU, c)
    gcc2, gdcc2 = cc_negpowneg(x.cc, xL, xU, c)
    gcv2, gdcv2 = cv_negpowneg(x.cc, xL, xU, c)
    cv_grad = max(0.0, gdcv1)*x.cv_grad + min(0.0, gdcv2)*x.cc_grad
    cc_grad = min(0.0, gdcc1)*x.cv_grad + max(0.0, gdcc2)*x.cc_grad
  else
    if (xL < x.cv)
      cc = x.cv^c
      cc_grad = (c*x.cv^(c-1))*x.cv_grad
    elseif (xL > x.cc)
      cc = x.cc^c
      cc_grad = (c*x.cc^(c-1))*x.cc_grad
    else
      cc = xL^c
      cc_grad = zeros(SVector{N,Float64})
    end
    if (xU == xL)
      cv = xLc
      cv_grad = zeros(SVector{N,Float64})
    else
      dcv = (xU^c - xL^c)/(xU - xL) # function decreasing
      if (xU < x.cv)
        cv = xUc + dcv*(x.cv - xL)
        cv_grad = dcv*x.cv_grad
      elseif (xU > x.cc)
        cv = xUc + dcv*(x.cc - xL)
        cv_grad = dcv*x.cc_grad
      else
        cv = xUc + dcv*(xU - xL)
        cv_grad = zeros(SVector{N,Float64})
      end
    end
		cv, cc, cv_grad, cc_grad = cut(y.lo, y.hi, cv, cc, cv_grad, cc_grad)
  end
	return MC{N}(cv, cc, y, cv_grad, cc_grad, x.cnst)
end
@inline function neg_powneg_even(x::MC{N}, c::Integer, y::Interval{Float64}) where {N}
  midcc, cc_id = mid3(x.cc, x.cv, x.Intv.hi)
  midcv, cv_id = mid3(x.cc, x.cv, x.Intv.lo)
  cc, dcc = cc_negpowneg(midcc, x.Intv.lo, x.Intv.hi, c)
  cv, dcv = cv_negpowneg(midcv, x.Intv.lo, x.Intv.hi, c)
  if (MC_param.mu >= 1)
    gcc1, gdcc1 = cc_negpowneg(x.cv, x.Intv.lo, x.Intv.hi, c)
    gcv1, gdcv1 = cv_negpowneg(x.cv, x.Intv.lo, x.Intv.hi, c)
    gcc2, gdcc2 = cc_negpowneg(x.cc, x.Intv.lo, x.Intv.hi, c)
    gcv2, gdcv2 = cv_negpowneg(x.cc, x.Intv.lo, x.Intv.hi, c)
    cv_grad = max(0.0, gdcv1)*x.cv_grad + min(0.0, gdcv2)*x.cc_grad
    cc_grad = min(0.0, gdcc1)*x.cv_grad + max(0.0, gdcc2)*x.cc_grad
  else
    cc_grad = mid_grad(x.cc_grad, x.cv_grad, cc_id)*dcc
    cv_grad = mid_grad(x.cc_grad, x.cv_grad, cv_id)*dcv
		cv, cc, cv_grad, cc_grad = cut(y.lo, y.hi, cv, cc, cv_grad, cc_grad)
  end
  return MC{N}(cv, cc, y, cv_grad, cc_grad, x.cnst)
end

@inline function pow_kernel(x::MC, c::Q, y::Interval{Float64}) where {Q<:Integer}
  (c == 0) && (return one(x))
	(c == 1) && (return x)
  if (c > 0)
    (c == 2) && (return sqr_kernel(x, y))
    isodd(c) && (return pos_odd(x, c, y))
    return npp_or_pow4(x, c, y)
  else
    if (x.Intv.hi < 0.0)
      isodd(c) && (return neg_powneg_odd(x, c, y))
      return neg_powneg_even(x, c, y)
    end
		(x.Intv.lo > 0.0) && (return npp_or_pow4(x, c, y))
  end
end
@inline function pow(x::MC, c::Q) where {Q<:Integer}
	if (x.Intv.lo <= 0.0 <= x.Intv.hi) && (c < 0)
		error("Function unbounded on this domain")
	end
	return pow_kernel(x, c, pow(x.Intv, c))
end
@inline (^)(x::MC, c::Q) where {Q <: Integer} = pow(x, c)

# Power of MC to float
@inline cv_flt_pow_1(x::Float64, xL::Float64, xU::Float64, n::Float64) = dline_seg(^, pow_deriv, x, xL, xU, n)
@inline cc_flt_pow_1(x::Float64, xL::Float64, xU::Float64, n::Float64) = x^n, n*x^(n-1)
@inline function flt_pow_1(x::MC{N}, c::Float64, y::Interval{Float64}) where N
	midcc, cc_id = mid3(x.cc, x.cv, x.Intv.hi)
	midcv, cv_id = mid3(x.cc, x.cv, x.Intv.lo)
	cc, dcc = cc_flt_pow_1(midcc, x.Intv.lo, x.Intv.hi, c)
	cv, dcv = cv_flt_pow_1(midcv, x.Intv.lo, x.Intv.hi, c)
  if (MC_param.mu >= 1)
    gcc1, gdcc1 = cc_flt_pow_1(x.cv ,x.Intv.lo, x.Intv.hi, c)
    gcv1, gdcv1 = cv_flt_pow_1(x.cv, x.Intv.lo, x.Intv.hi, c)
    gcc2, gdcc2 = cc_flt_pow_1(x.cc, x.Intv.lo, x.Intv.hi, c)
    gcv2, gdcv2 = cv_flt_pow_1(x.cc, x.Intv.lo, x.Intv.hi, c)
    cv_grad = max(0.0, gdcv1)*x.cv_grad + min(0.0, gdcv2)*x.cc_grad
    cc_grad = min(0.0, gdcc1)*x.cv_grad + max(0.0, gdcc2)*x.cc_grad
  else
    cc_grad = mid_grad(x.cc_grad, x.cv_grad, cc_id)*dcc
    cv_grad = mid_grad(x.cc_grad, x.cv_grad, cv_id)*dcv
    cv, cc, cv_grad, cc_grad = cut(y.lo, y.hi, cv, cc, cv_grad, cc_grad)
  end
  return MC{N}(cv, cc, y, cv_grad, cc_grad, x.cnst)
end
@inline function  (^)(x::MC{N}, c::Float64, y::Interval{Float64}) where N
    isinteger(c) && (return pow_kernel(x, Int(c), y))
    ((x.Intv.lo >= 0) && (0.0 < c < 1.0)) && (return flt_pow_1(x, c, y))
		z = exp(c*log(x))
    return MC{N}(z.cv, z.cc, y, z.cv_grad, z.cc_grad, x.cnst)
end
@inline (^)(x::MC, c::Float32, y::Interval{Float64}) = (^)(x, Float32(c), y)
@inline (^)(x::MC, c::Float16, y::Interval{Float64}) = (^)(x, Float16(c), y)
@inline (^)(x::MC, c::Float64) = (^)(x, c, x.Intv^c)
@inline (^)(x::MC, c::Float32) = x^Float64(c) # DONE
@inline (^)(x::MC, c::Float16) = x^Float64(c) # DONE
@inline (^)(x::MC, c::MC) = exp(c*log(x)) # DONE (no kernel)
@inline pow(x::MC, c::Q) where {Q <: AbstractFloat} = x^c

# Define powers to MC of floating point number
@inline function pow(b::Float64, x::MC) # DONE (no kernel)
	(b <= 0.0) && error("Relaxations of a^x where a<=0 not currently defined in library.
		                   Functions of this type may prevent convergences in global
			                 optimization algorithm as they may be discontinuous.")
	exp(x*log(b))
end
@inline ^(b::Float64, x::MC) = pow(b, x) # DONE (no kernel)

########### Defines inverse
@inline function inv_kernel(x::MC, y::Interval{Float64})
  (x.Intv.lo <= 0.0 <= x.Intv.hi) && error("Function unbounded on domain: $(x.Intv)")
  (x.Intv.hi < 0.0) && (return neg_powneg_odd_kernel(x, -1, y))
  (x.Intv.lo > 0.0) && (return neg_powpos_kernel(x, -1, y))
end
@inline inv_kernel(x::MC) = inv_kernel(x, (x.Intv)^(-1))
