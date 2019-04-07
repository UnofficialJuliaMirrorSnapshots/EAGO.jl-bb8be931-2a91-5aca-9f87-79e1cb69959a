function implicit_llp(xbar::Vector{Float64}, sip_storage::SIPResult,  ProblemStorage::SIPProblemStorage)
    # load keyword arguments
    np = ProblemStorage.np
    ny = ProblemStorage.ny

    f = ProblemStorage.f; gSIP = ProblemStorage.gSIP
    h = ProblemStorage.h; hj = ProblemStorage.hj

    y_l = ProblemStorage.y_l; y_u = ProblemStorage.y_u
    p_l = ProblemStorage.p_l; p_u = ProblemStorage.p_u

    upper_style = ProblemStorage.upper
    optimizer = ProblemStorage.opts.lower_level_optimizer()

    # defines discretization of SIP constraint, state variable, state jacobian
    g_llp(y,p) = -gSIP(sip_storage.x_bar, y, p)
    h_llp(out,y,p) = h(out, sip_storage.x_bar, y, p)
    hj_llp(out,y,p) = hj(out, sip_storage.x_bar, y, p)

    var, optimizer = solve_implicit(g_llp, h_llp, y_l, y_u, p_l, p_u, optimizer,
                                    hj_llp, nothing; upper_sym = upper_style)

    termination_status = MOI.get(optimizer, MOI.TerminationStatus())
    result_status = MOI.get(optimizer, MOI.PrimalStatus())
    valid_result, is_feasible = is_globally_optimal(termination_status, result_status)

    sip_storage.lower_level_time += MOI.get(optimizer, MOI.SolveTime())
    if valid_result
      if is_feasible
        INNg2 = -MOI.get(optimizer, MOI.ObjectiveValue())
        sip_storage.p_bar[:] = MOI.get(optimizer, MOI.VariablePrimal(), var[(ny+1):(np+ny)])
      end
    else
        error("Upper problem did not solve to global optimality.")
    end

    return INNg2, is_feasible
end

function implicit_lbp(lower_disc_set::Vector{Vector{Float64}}, sip_storage::SIPResult, ProblemStorage::SIPProblemStorage)

    println("lower_disc_set: $lower_disc_set")
    ng = length(lower_disc_set);

    # load keyword arguments
    nx = ProblemStorage.nx; np = ProblemStorage.np;
    ny = ProblemStorage.ny

    f = ProblemStorage.f; gSIP = ProblemStorage.gSIP
    h = ProblemStorage.h; hj = ProblemStorage.hj

    x_l = ProblemStorage.x_l; x_u = ProblemStorage.x_u
    y_l = ProblemStorage.y_l; y_u = ProblemStorage.y_u

    upper_style = ProblemStorage.upper
    optimizer = ProblemStorage.opts.lower_problem_optimizer()

    # reformates the start variables to reflect new dimension
    y_l_lbp = Float64[]
    y_u_lbp = Float64[]
    for element in lower_disc_set
        append!(y_l_lbp, y_l)
        append!(y_u_lbp, y_u)
    end

    # defines discretization of SIP constraint, state variable, state jacobian
    function g(y,x)
        lo_indx::Int = 0
        hi_indx::Int = 0
        gSIP_out = zeros(typeof(x[1]),ng)
        for (indx, element) in enumerate(lower_disc_set)
            lo_indx = 1+indx*(ny-1); hi_indx = indx*ny
            gSIP_out[indx] = gSIP(x, y[lo_indx:hi_indx], element)
        end
        return gSIP_out
     end
     function h_lbp!(h_out, y, x)
        lo_indx::Int = 0
        hi_indx::Int = 0
        for (indx, element) in enumerate(lower_disc_set)
            lo_indx = 1+ny*(indx-1)
            hi_indx = indx*ny
            h(view(h_out, lo_indx:hi_indx), x, y[lo_indx:hi_indx], element)
        end
      end
      function hj_lbp!(hj_out, y, x)
        lo_indx::Int = 0
        hi_indx::Int = 0
        for (indx, element) in enumerate(lower_disc_set)
            lo_indx = 1+ny*(indx-1)
            hi_indx = indx*ny
            hj(view(hj_out, lo_indx:hi_indx, lo_indx:hi_indx), x, y[lo_indx:hi_indx], element)
        end
      end

      var, optimizer = solve_implicit(f, h_lbp!, y_l_lbp, y_u_lbp, x_l, x_u,
                                      optimizer, hj_lbp!, g; upper_sym = upper_style)

      termination_status = MOI.get(optimizer, MOI.TerminationStatus())
      result_status = MOI.get(optimizer, MOI.PrimalStatus())
      valid_result, is_feasible = is_globally_optimal(termination_status, result_status)

      sip_storage.lower_bounding_time += MOI.get(optimizer, MOI.SolveTime())
      if valid_result
          if is_feasible
              sip_storage.lower_bound = MOI.get(optimizer, MOI.ObjectiveValue())
              sip_storage.x_bar[:] = MOI.get(optimizer, MOI.VariablePrimal(),var[(ny*ng+1):(ny*ng+nx)])
          end
       else
          error("Lower problem did not solve to global optimality.")
       end

       return is_feasible
end

function implicit_ubp(upper_disc_set::Vector{Vector{Float64}}, eps_g::Float64, sip_storage::SIPResult,  ProblemStorage::SIPProblemStorage)

    println("upper_disc_set: $upper_disc_set")
    ng = length(upper_disc_set)

    # load keyword arguments
    nx = ProblemStorage.nx
    np = ProblemStorage.np
    ny = ProblemStorage.ny

    gSIP = ProblemStorage.gSIP
    h = ProblemStorage.h
    hj = ProblemStorage.hj

    x_l = ProblemStorage.x_l; x_u = ProblemStorage.x_u
    y_l = ProblemStorage.y_l; y_u = ProblemStorage.y_u

    upper_style = ProblemStorage.upper
    optimizer = ProblemStorage.opts.upper_problem_optimizer()

    # reformates the start variables to reflect new dimension
    y_l_ubp = Float64[]
    y_u_ubp = Float64[]
    for element in upper_disc_set
        append!(y_l_ubp, y_l)
        append!(y_u_ubp, y_u)
    end

    # defines discretization of SIP constraint, state variable, state jacobian
    function g(y,x)
        lo_indx::Int = 0
        hi_indx::Int = 0
        gSIP_out = zeros(typeof(x[1]),ng)
        for (indx, element) in enumerate(upper_disc_set)
            lo_indx = 1+ny*(indx-1)
            hi_indx = indx*ny
            gSIP_out[indx] = gSIP(x, y[lo_indx:hi_indx], element) + eps_g
        end
        return gSIP_out
    end
    function h_ubp!(h_out,y,x)
        lo_indx::Int = 0
        hi_indx::Int = 0
        for (indx, element) in enumerate(upper_disc_set)
            lo_indx = 1+ny*(indx-1)
            hi_indx = indx*ny
            h(view(h_out, lo_indx:hi_indx), x, y[lo_indx:hi_indx], element)
        end
    end
    function hj_ubp!(hj_out,y,x)
        lo_indx::Int = 0
        hi_indx::Int = 0
        for (indx, element) in enumerate(upper_disc_set)
            lo_indx = 1+ny*(indx-1)
            hi_indx = indx*ny
            hj(view(hj_out, lo_indx:hi_indx, lo_indx:hi_indx), x, y[lo_indx:hi_indx], element)
        end
    end

    var, optimizer = solve_implicit(ProblemStorage.f, h_ubp!, y_l_ubp, y_u_ubp, x_l, x_u,
                                  optimizer, hj_ubp!, g; upper_sym = upper_style)

    termination_status = MOI.get(optimizer, MOI.TerminationStatus())
    result_status = MOI.get(optimizer, MOI.PrimalStatus())
    valid_result, is_feasible = is_globally_optimal(termination_status, result_status)

    sip_storage.upper_bounding_time += MOI.get(optimizer, MOI.SolveTime())
    if valid_result
      if is_feasible
        sip_storage.upper_bound = MOI.get(optimizer, MOI.ObjectiveValue())
        sip_storage.x_bar[:] = MOI.get(optimizer, MOI.VariablePrimal(),var[(ny*ng+1):(ny*ng+nx)])
      end
    else
        error("Upper problem did not solve to global optimality.")
    end

    return is_feasible
end

function implicit_sip_solve(f::Function, gSIP::Function, h::Function,
                            x_l::Vector{Float64}, x_u::Vector{Float64},
                            y_l::Vector{Float64}, y_u::Vector{Float64},
                            p_l::Vector{Float64}, p_u::Vector{Float64};
                            opts = nothing,  hj = nothing, upper_style = :MidPointUpperEvaluator)

    @assert length(p_l) == length(p_u)
    @assert length(y_l) == length(y_u)
    @assert length(x_l) == length(x_u)

    xp_l = vcat(p_l, x_l)
    xp_u = vcat(p_u, x_u)

    n_p = length(p_l)
    n_y = length(y_l)
    n_x = length(x_l)
    n_xp = n_x + n_p

    if opts == nothing
        opts = SIPOptions()
    end

    fxp = (x,p) -> f(p)

    ProblemInfo = SIPProblemStorage(fxp, gSIP, x_l, x_u, p_l, p_u, n_p, n_x,
                                    opts, h, hj, y_l, y_u, n_y, upper_style,
                                    xp_l, xp_u, n_xp)

    # Seed with mid(P) to avoid having to call explicit solver initially
    push!(ProblemInfo.opts.lower_disc_set, (p_l + p_u)/2.0, p_l, p_u)
    push!(ProblemInfo.opts.upper_disc_set, (p_l + p_u)/2.0, p_l, p_u)

    sip_sto = core_sip_routine(implicit_llp, implicit_llp, implicit_lbp,
                               implicit_ubp, set_xpbar, ProblemInfo)
    return sip_sto
end
