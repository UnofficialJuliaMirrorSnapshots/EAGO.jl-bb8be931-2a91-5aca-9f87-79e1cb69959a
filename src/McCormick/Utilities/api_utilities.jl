"""
    grad(x::MC{N},j::Int) where N

sets convex and concave (sub)gradients of length `n` of `x` to be `1` at index `j`
"""
function grad(x::MC{N},j::Int) where N
  sv_grad::SVector{N,Float64} = seed_gradient(T,j,N)
  return MC{N}(x.cc,x.cv,sv_grad,sv_grad,x.Intv,x.cnst)
end

"""
    zgrad(x::SMCg{N,T},n::Int64) where {N,T}

sets convex and concave (sub)gradients of length `n` to be zero
"""
function zgrad(x::MC{N}) where N
  grad::SVector{N,Float64} = zeros(SVector{N,Float64})
  return MC{N}(x.cc,x.cv,grad,grad,x.Intv,x.cnst)
end

Intv(x::MC) = x.Intv
lo(x::MC) = x.Intv.lo
hi(x::MC) = x.Intv.hi
cc(x::MC) = x.cc
cv(x::MC) = x.cv
cc_grad(x::MC) = x.cc_grad
cv_grad(x::MC) = x.cv_grad
cnst(x::MC) = x.cnst
length(x::MC) = length(x.cc_grad)

"""
    set_mc_differentiability!
Set differentiability of relaxations used.
"""
function set_mc_differentiability!(val::Integer)
  diff_relax = val > 0
  if (diff_relax > 0)
    MC_param.mu = val
  elseif val == 0
    MC_param.mu = 0
  else
    error("Differentiability must be an integer input greater than or equal to zero.")
  end
end

"""
    set_multivar_refine!
Specifies whether the tigher but more expensive multvivaiate relaxation should be used.
"""
function set_multivar_refine!(yn::Bool)
  @assert tol > 0.0
  MC_param.multivar_refine = yn
end

"""
    set_multivar_refine!
Specifies number of iterations to be used in envelope root finding routines.
"""
function set_iterations!(k::Int)
  @assert tol > 0
  MC_param.env_max_int = k
end

"""
    set_tolerance!
Specifies absolute tolerance for in envelope root finding routines.
"""
function set_tolerance!(k::Float64)
  @assert tol > 0.0
  MC_param.env_tol = k
end
