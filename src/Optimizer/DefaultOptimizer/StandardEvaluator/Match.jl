"""
    dag_cse_simplify!

"""
function dag_cse_simplify!(d::NLPEvaluator) where T<:Real
end

"""
    flatten_function!
"""
function flatten_function!(y)
    for k in length(y.nd):-1:1
        @inbounds nod = y.nd[k]
        if nod.nodetype == CALL
            op = nod.index
            if (op == 3) # Is multiply?
                # flatten exp(x)exp(y) = exp(x+y)
                #         negative entropy forms: xlog(x), xlog2(x), xlog10(x), (x-1)*log1p(x-1)
                if (op == 12) # Is exp?
                elseif (op == 13) # Is exp2?
                elseif (op == 14) # Is expm1? (exp10 not supported in JuMP)
                end
                # Are two or more children exp(...)
            elseif (op == 4)
                # flatten a^log(x) = x^log(a)
                # flatten (x^a)^b = x^(a*b)
                # flatten (a^x)^b = (a^b)^x
            end

        elseif nod.nodetype == CALLUNIVAR
            op = nod.index
            # flatten log(xy) = log(x) + log(y)
            # flatten log(a^x) = x*log(a)
            if (op == 8)  # Is log?
                cop = #
                if (cop == 3) # Is multiply?
                elseif (cop == 4) # Is power?
                end
            elseif (op == 9)  # Is log10?
            elseif (op == 10) # Is log2?
            elseif (op == 11) # Is log1p?
            end
        end
    end
end
# potentially use hyperbolic as well and product of powers versus power of products
# hyperbolic sinh(x) = (exp(x) - exp(-x))/2
#            sinh(x) = (exp(2x) - 1)/(2exp(x))
#            sinh(x) = (1 - exp(-2x))/(2exp(-x))
# hyperbolic cosh(x) = (exp(x) + exp(-x))/2
#            cosh(x) = (exp(2x) + 1)/(2exp(x))
#            cosh(x) = (1 + exp(-2x))/(2exp(-x))
# (x_1*x_2*...x_n)^a = x_1^a...x_n^a
# potentially use disaggregation
"""
    dag_flattening!
"""
function dag_flattening!(d::NLPEvaluator) where T<:Real

    # flatten objective
    flatten_function!(d.objective)

    # flatten constraints
    for i in length(d.constraints)
        flatten_function!(d.constraints[i])
    end

    # flatten subexpression
    for i in length(d.subexpressions)
        flatten_function!(d.subexpressions[i])
    end

end
