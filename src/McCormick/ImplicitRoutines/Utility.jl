function precondition_mc(h::Function,
                          hj::Function,
                          z_mc::Vector{MC{N}},
                          aff_mc::Vector{MC{N}},
                          p_mc::Vector{MC{N}},
                          opt::mc_opts) where {N}
    if (opt.linear_algebra == :DenseBand)
        H,J = mc_denseband_precondition!(h,hj,z_mc,aff_mc,p_mc,opt)
    elseif (opt.linear_algebra == :DenseBlockDiag)
        H,J = mc_denseblockdiag_precondition!(h,hj,z_mc,aff_mc,p_mc,opt)
    elseif (opt.linear_algebra == :Dense)
        H,J = mc_dense_precondition!(h,hj,z_mc,aff_mc,p_mc,opt)
    else
        error("Unsupported Linear Algebra Style")
    end
    return H,J
end

function mc_dense_precondition!(h::Function,
                             hj::Function,
                             z_mc::Vector{MC{N}},
                             aff_mc::Vector{MC{N}},
                             p_mc::Vector{MC{N}},
                             opt::mc_opts) where N

    H::Vector{MC{N}} = h(z_mc,p_mc)
    J = hj(aff_mc,p_mc)
    Y = [mid(J[i,j].Intv) for i=1:opt.nx, j=1:opt.nx]
    if (opt.nx == 1)
        YH::Vector{MC{N}} = H/Y[1]
        YJ = J/Y[1]
    else
        F = lu(Y)
        YH = F\H
        YJ = F\J
    end
    return YH,YJ
end

function mc_precondition_1!(h!::Function, hj!::Function, z_mc, aff_mc, p_mc, J, H, Y, xp_mc, flt_param::Vector{Float64}, precond::Bool)
  h!(H, z_mc, xp_mc, p_mc, flt_param)
  hj!(J, aff_mc, xp_mc, p_mc, flt_param)
  if precond
    Y[1,1] = mid(J[1,1].Intv)
    H[1] /= Y[1,1]
    J[1,1] /= Y[1,1]
  end
end

function mc_denseband_precondition!(h!::Function, hj!::Function, z_mc, aff_mc, p_mc, J, H, Y, xp_mc, flt_param::Vector{Float64}, precond::Bool, nx::Int)

  h!(H, z_mc, xp_mc, p_mc, flt_param)
  hj!(J, aff_mc, xp_mc, p_mc, flt_param)

  if precond
  #  for i in 1:nx
  #    for j in 1:nx
  #        Y[i,j] = mid(J[i,j].Intv)
  #      end
  #    end


    #F = lu(Y)
    Y[:,:] = mid.(Intv.(J))
    YInv = inv(Y)

    #H[:] = F\H
    #J[:,:] = F\J
    H[:] = YInv*H
    J[:,:] = YInv*J
  end
end

"""
    smooth_cut

An operator that cuts the `x_mc` object using the `x_mc_int bounds` in a
differentiable fashion.
"""
function smooth_cut(x_mc::MC{N},x_mc_int::MC{N}) where N
  t_cv::MC{N} = x_mc + max(0.0, x_mc_int-x_mc)
  t_cc::MC{N} = x_mc + min(0.0, x_mc-x_mc_int)
  return MC{N}(t_cv.cv, t_cc.cc, (x_mc.Intv ∩ x_mc_int.Intv),
               t_cv.cv_grad, t_cc.cc_grad, (t_cv.cnst && t_cc.cnst))
end

"""
    final_cut

An operator that cuts the `x_mc` object using the `x_mc_int bounds` in a
differentiable or nonsmooth fashion as specified by the `MC_param.mu flag`.
"""
function final_cut(x_mc::MC{N},x_mc_int::MC{N}) where N
  if (MC_param.mu < 1)
    Intv = x_mc.Intv ∩ x_mc_int.Intv
    if (x_mc.cc < x_mc_int.cc)
      cc = x_mc.cc
      cc_grad::SVector{N,Float64} = x_mc.cc_grad
    else
      cc = x_mc_int.cc
      cc_grad = x_mc_int.cc_grad
    end
    if (x_mc.cv > x_mc_int.cv)
      cv = x_mc.cv
      cv_grad::SVector{N,Float64} = x_mc.cv_grad
    else
      cv = x_mc_int.cv
      cv_grad = x_mc_int.cv_grad
    end
    x_out::MC{N} = MC{N}(cv,cc,(x_mc.Intv ∩ x_mc_int.Intv),cv_grad,cc_grad,x_mc.cnst)
  else
    x_out = smooth_cut(x_mc,x_mc_int)
  end
  return x_out
end

"""
    rnd_out_z_intv

Rounds the interval of the `z_mct` vector elements out by `epsvi`.
"""
function rnd_out_z_intv(z_mct::Vector{MC{N}},epsvi::Float64) where N
  return [MC{N}(z_mct[i].cv, z_mct[i].cc, IntervalType(z_mct[i].Intv.lo-epsvi, z_mct[i].Intv.hi+epsvi),
                z_mct[i].cv_grad, z_mct[i].cc_grad, z_mct[i].cnst) for i=1:length(z_mct)]
end

"""
    rnd_out_z_all

Rounds the interval and relaxation bounds of the `z_mct` vector elements out by `epsvi`.
"""
function rnd_out_z_all(z_mct::Vector{MC{N}},epsvi::Float64) where {N}
  return [MC{N}(z_mct[i].cv-epsvi, z_mct[i].cc+epsvi, IntervalType(z_mct[i].Intv.lo-epsvi, z_mct[i].Intv.hi+epsvi),
                z_mct[i].cv_grad, z_mct[i].cc_grad, z_mct[i].cnst) for i=1:length(z_mct)]
end

"""
    rnd_out_h_all

Rounds the interval bounds of the `z_mct` and `Y_mct` elements out by `epsvi`.
"""
function rnd_out_h_all(z_mct::Vector{MC{N}},Y_mct::Array{MC{N},2},epsvi::Float64) where N
  temp1::Vector{MC{N}} = [MC{N}(z_mct[i].cv-epsvi, z_mct[i].cc+epsvi, IntervalType(z_mct[i].Intv.lo-epsvi, z_mct[i].Intv.hi+epsvi),
                                z_mct[i].cv_grad, z_mct[i].cc_grad, z_mct[i].cnst) for i=1:length(z_mct)]
  temp2::Array{MC{N},2} = [MC{N}(Y_mct[i,j].cv-epsvi, Y_mct[i,j].cc+epsvi,
                                 Y_mct[i,j].cv_grad, Y_mct[i,j].cc_grad, IntervalType(Y_mct[i,j].Intv.lo-epsvi, Y_mct[i,j].Intv.hi+epsvi),
                                 Y_mct[i,j].cnst) for i=1:length(z_mct), j=1:length(z_mct)]
  return temp1,temp2
end

"""
    rnd_out_h_intv

Rounds the interval and relaxation bounds of the `z_mct` and `Y_mct` elements out by `epsvi`.
"""
function rnd_out_h_intv(z_mct::Vector{MC{N}},Y_mct::Array{MC{N},2},epsvi::Float64) where N
  temp1::Vector{MC{N}} = [MC{N}(z_mct[i].cv, z_mct[i].cc, IntervalType(z_mct[i].Intv.lo-epsvi, z_mct[i].Intv.hi+epsvi),
                                z_mct[i].cv_grad, z_mct[i].cc_grad, z_mct[i].cnst) for i=1:length(z_mct)]
  temp2::Array{MC{N},2} = [MC{N}(Y_mct[i,j].cv, Y_mct[i,j].cc, IntervalType(Y_mct[i,j].Intv.lo-epsvi, Y_mct[i,j].Intv.hi+epsvi),
                                 Y_mct[i,j].cv_grad, Y_mct[i,j].cc_grad, Y_mct[i,j].cnst) for i=1:length(z_mct), j=1:length(z_mct)]
  return temp1,temp2
end
