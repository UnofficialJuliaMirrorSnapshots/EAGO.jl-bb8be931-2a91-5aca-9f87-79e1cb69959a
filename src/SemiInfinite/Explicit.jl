function set_xpbar(ProblemStorage::SIPProblemStorage)
  xbar = (ProblemStorage.x_u + ProblemStorage.x_l)/2.0
  pbar = (ProblemStorage.p_u + ProblemStorage.p_l)/2.0
  return xbar, pbar, ProblemStorage.nx, ProblemStorage.np
end

function explicit_llp(xbar::Vector{Float64}, sip_storage::SIPResult, ProblemStorage::SIPProblemStorage)

  g(p) = ProblemStorage.gSIP(xbar,p)
  model = Model(with_optimizer(ProblemStorage.opts.lower_level_optimizer))
  vars, model = solve_script(ProblemStorage.gSIP, ProblemStorage.p_l, ProblemStorage.p_u, model)

  termination_status = JuMP.termination_status(model)
  result_status = JuMP.primal_status(model)
  valid_result, is_feasible = is_globally_optimal(termination_status, result_status)

  sip_storage.lower_level_time += MOI.get(model, MOI.SolveTime())

  if valid_result
      if is_feasible
        INNg2 = -JuMP.objective_value(model)
        sip_storage.p_bar[:] = JuMP.value.(vars)
      end
  else
    error("Lower level problem.")
  end

  return INNg2, is_feasible
end

# should be done
function explicit_lbp(lower_disc_set::Vector{Vector{Float64}}, sip_storage::SIPResult, ProblemStorage::SIPProblemStorage)
  ng = length(lower_disc_set)
  gL = Float64[-Inf for i=1:ng]
  gU = Float64[0.0 for i=1:ng]

  # defines discretization of SIP constraint
  g(x) = ProblemStorage.gSIP.(x,lower_disc_set)

  # create JuMP model
  model = Model(with_optimizer(ProblemStorage.opts.lower_problem_optimizer))
  vars, model = solve_script(ProblemStorage.f,
                             ProblemStorage.x_l, ProblemStorage.x_u,
                             model, g = g, gL = gL, gU = gU)

  termination_status = JuMP.termination_status(model)
  result_status = JuMP.primal_status(model)
  valid_result, is_feasible = is_globally_optimal(termination_status, result_status)

  sip_storage.lower_bounding_time += MOI.get(model, MOI.SolveTime())

  if valid_result
      if is_feasible
        sip_storage.lower_bound = JuMP.objective_value(model)
        sip_storage.x_bar[:] = JuMP.value.(vars)
      end
  else
    error("Lower problem did not solve to global optimality.")
  end

  return is_feasible
end

function explicit_ubp(upper_disc_set::Vector{Vector{Float64}}, eps_g::Float64, sip_storage::SIPResult, ProblemStorage::SIPProblemStorage)
  ng = length(upper_disc_set)
  gL = Float64[-Inf for i=1:ng]
  gU = Float64[-eps_g for i=1:ng]

  # defines discretization of SIP constraint
  g(x) = ProblemStorage.gSIP.(x, upper_disc_set)

  # create JuMP model
  model = Model(with_optimizer(ProblemStorage.opts.upper_problem_optimizer))
  vars, model = solve_script(ProblemStorage.f,
                             ProblemStorage.x_l, ProblemStorage.x_u,
                             model, g = g, gL = gL, gU = gU)

  termination_status = JuMP.termination_status(model)
  result_status = JuMP.primal_status(model)
  valid_result, is_feasible = is_globally_optimal(termination_status, result_status)

  sip_storage.upper_bounding_time += MOI.get(model, MOI.SolveTime())

  if valid_result
      if is_feasible
        sip_storage.upper_bound = JuMP.objective_value(model)
        sip_storage.x_bar[:] = JuMP.value.(vars)
      end
  else
    error("Upper problem did not solve to global optimality.")
  end

  return is_feasible
end

"""
    explicit_sip_solve

Solves a semi-infinite program via the algorithm presented in Mitsos2011 using
the EAGOGlobalSolver to solve the lower bounding problem, lower level problem,
and the upper bounding problem. The options for the algorithm and the global
solvers utilized are set by manipulating a SIPopt containing the options info.
Inputs:
* `f::Function`: Objective in the decision variable. Takes a single argument
                 vector that must be untyped.
* `gSIP::Function`: The semi-infinite constraint. Takes two arguments: the first
                    being a vector containing the decision variable and the
                    second being a vector containing the uncertainity
                    variables. The function must be untyped.
* `SIPopt::SIP_opts`: Option type containing problem information
Returns: A SIP_result composite type containing solution information.
"""
function explicit_sip_solve(f::Function, gSIP::Function, x_l::Vector{Float64},
                            x_u::Vector{Float64}, p_l::Vector{Float64},
                            p_u::Vector{Float64}; opts = nothing)

  @assert length(p_l) == length(p_u)
  @assert length(x_l) == length(x_u)
  n_p = length(p_l)
  n_x = length(x_l)

  if opts == nothing
      opts = SIPOptions()
  end

  ProblemStorage = SIPProblemStorage(f, gSIP, x_l, x_u, p_l, p_u, n_p, n_x, opts,
                                     nothing, nothing, Float64[], Float64[], 0,
                                     :nothing, Float64[], Float64[], 0)

  sip_sto = core_sip_routine(explicit_llp, explicit_llp, explicit_lbp,
                             explicit_ubp, set_xpbar, ProblemStorage)
  return sip_sto
end
