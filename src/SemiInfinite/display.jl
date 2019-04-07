"""
    print_int!(SIPopt,k_int,lbd,ubd,eps,r)
--------------------------------------------------------------------------------
Description:
Prints current iteration statistics.
--------------------------------------------------------------------------------
Inputs:
SIPopt   SIP_opts - Options object
k_int    Int64   - Iteration Number
lbd      Float64 - Lower Bound
ubd      Float64 - Upper Bound
eps      Float64 - Restriction size
r        Float64 - Restriction Adjustment
--------------------------------------------------------------------------------
Returns:
No returned value but prints the iteration, lower bound, upper bound,
restriction value, r value, absolute ratio
and relative ratio if the verbosity is set to "Normal".
--------------------------------------------------------------------------------
"""
@inline function print_int!(SIPopt::SIPOptions, k_int::Int, lbd::Float64, ubd::Float64, eps::Float64, r::Float64)
  if (SIPopt.verbosity == "Normal")
    # prints header line every hdr_intv times
    if (mod(k_int,SIPopt.header_interval)==0||k_int==1)
      println("Iteration   LBD    UBD     eps      r     Absolute_Gap    Absolute_Ratio")
    end
    # prints iteration summary every prnt_intv times
    if (mod(k_int,SIPopt.print_interval)==0)
      ptr_arr_temp = [k_int lbd ubd eps r (ubd-lbd) (lbd/ubd)]
      ptr_arr1 = join([Printf.@sprintf("%6u",x) for x in ptr_arr_temp[1]], ",   ")
      ptr_arr2 = join([Printf.@sprintf("%3.7f",x) for x in ptr_arr_temp[2:5]], ",     ")
      ptr_arr3 = join([Printf.@sprintf("%6u",x) for x in ptr_arr_temp[6:7]], ",")
      println(string(ptr_arr1,",      ",ptr_arr2,",      ",ptr_arr3))
    end
  end
end

function print_llp1!(SIPopt::SIPOptions, INNg1::Float64, pbar::Vector{Float64}, feas::Bool)
  if (SIPopt.verbosity == "Full" || SIPopt.verbosity == "Normal")
    println("solved INN #1: ",INNg1," ",pbar," ",feas)
  end
end

function print_llp2!(SIPopt::SIPOptions, INNg2::Float64, pbar::Vector{Float64}, feas::Bool)
  if (SIPopt.verbosity == "Full" || SIPopt.verbosity == "Normal")
    println("solved INN #2: ",INNg2," ",pbar," ",feas)
  end
end

function print_lbp!(SIPopt::SIPOptions, LBDg::Float64, xbar::Vector{Float64}, feas::Bool)
  if (SIPopt.verbosity == "Full" || SIPopt.verbosity == "Normal")
    println("solved LBD: ",LBDg," ",xbar," ",feas)
  end
end

function print_ubp!(SIPopt::SIPOptions, UBD_temp::Float64, xbar::Vector{Float64}, feas::Bool)
  if (SIPopt.verbosity == "Full" || SIPopt.verbosity == "Normal")
    println("solved UBD: ",UBD_temp," ",xbar," ",feas)
  end
end
