"""
    set_value_construct

Construct the set-valued relaxation of variable `i` of `N` in `node` at `x_values`.
"""
function set_value_construct(i::Int,N::Int,x_values::Vector{Float64},node::NodeBB)
    @inbounds xval = x_values[i]
    @inbounds intv = IntervalType(node.lower_variable_bounds[i],node.upper_variable_bounds[i])
    seed = seed_gradient(Float64,i,N)
    @inbounds MC{N}(xval, xval,intv,seed,seed,false)
end

"""
    set_value_post

Post process set_value operator. By default, performs the affine interval cut on
a MC structure.
"""
function set_value_post(x_values::Vector{Float64},val::MC{N},node::NodeBB) where N
    lower = val.cv
    upper = val.cc
    for i in 1:N
        @inbounds cv_val = val.cv_grad[i]
        @inbounds cc_val = val.cc_grad[i]
        if (cv_val > 0.0)
            @inbounds lower += cv_val*(node.lower_variable_bounds[i]-x_values[i])
        else
            @inbounds lower += cv_val*(node.upper_variable_bounds[i]-x_values[i])
        end
        if (cc_val > 0.0)
            @inbounds upper += cc_val*(node.upper_variable_bounds[i]-x_values[i])
        else
            @inbounds upper += cc_val*(node.lower_variable_bounds[i]-x_values[i])
        end
    end
    lower = max(lower,lo(val))
    upper = min(upper,hi(val))
    return MC{N}(val.cv, val.cc, IntervalType(lower,upper), val.cv_grad, val.cc_grad, val.cnst)
end
#set_value_post(x_values::Vector{Float64},val::MC{N},node::NodeBB) where N = val

id_to_operator = Dict(value => key for (key, value) in JuMP.univariate_operator_to_id)
function print_evaluator_struct(k, node)
    op = node.index
    op_label = id_to_operator[op]
    println("k: $k, op: $op_label")
end

function print_call!(sym, k, children_idx)
    print_string = "k: $k, node type: $sym( "
    for i in children_idx
        print_string = print_string * "node[$i], "
    end
    print_string = print_string * ")"
    println(print_string)
end

const PRINT_EVAL = false

function forward_eval(setstorage::Vector{T}, numberstorage::Vector{Float64}, numvalued::Vector{Bool},
                      nd::AbstractVector{JuMP.NodeData}, adj, const_values, parameter_values, current_node::NodeBB,
                      x_values::Vector{Float64}, subexpr_values_flt, subexpr_values_set, subexpression_isnum::Vector{Bool}, user_input_buffer;
                      user_operators::JuMP._Derivatives.UserOperatorRegistry=JuMP._Derivatives.UserOperatorRegistry()) where T

    @assert length(numberstorage) >= length(nd)
    @assert length(setstorage) >= length(nd)
    @assert length(numvalued) >= length(nd)

    children_arr = rowvals(adj)
    N = length(x_values)

    for k in length(nd):-1:1
        @inbounds nod = nd[k]
        op = nod.index
        if nod.nodetype == JuMP._Derivatives.VARIABLE
            PRINT_EVAL && println("k: $k, node type: Variable[$op]")
            PRINT_EVAL && println("x_MC{$k}, $(set_value_construct(nod.index,N,x_values,current_node))")
            setstorage[k] = set_value_construct(nod.index,N,x_values,current_node)
            numvalued[k] = false
        elseif nod.nodetype == JuMP._Derivatives.VALUE
            PRINT_EVAL && println("k: $k, node type: Constant = $(const_values[nod.index])")
            @inbounds numberstorage[k] = const_values[nod.index]
            numvalued[k] = true
        elseif nod.nodetype == JuMP._Derivatives.SUBEXPRESSION
            PRINT_EVAL && println("k: $k, node type: Subexpression at $(nod.index)")
            @inbounds isnum = subexpression_isnum[nod.index]
            if isnum
                @inbounds numberstorage[k] = subexpr_values_flt[nod.index]
            else
                @inbounds setstorage[k] = subexpr_values_set[nod.index]
            end
            PRINT_EVAL && println("result number: $(subexpr_values_flt[nod.index])")
            PRINT_EVAL && println("result set: $(subexpr_values_set[nod.index])")
            PRINT_EVAL && println("is numeric: $(isnum)")
            numvalued[k] = isnum
        elseif nod.nodetype == JuMP._Derivatives.PARAMETER
            PRINT_EVAL && println("k: $k, node type: Parameter at $(nod.index) = $(parameter_values[nod.index])")
            @inbounds numberstorage[k] = parameter_values[nod.index]
            numvalued[k] = true
        elseif nod.nodetype == JuMP._Derivatives.CALL
            op = nod.index
            @inbounds children_idx = nzrange(adj,k)
            n_children = length(children_idx)
            if op == 1 # :+
                tmp_sum = 0.0
                isnum = true
                chdset = true
                for c_idx in children_idx
                    @inbounds ix = children_arr[c_idx]
                    @inbounds chdset = numvalued[ix]
                    if (chdset)
                        @inbounds tmp_sum += numberstorage[ix]
                    else
                        @inbounds tmp_sum += setstorage[ix]
                    end
                    @inbounds isnum &= chdset
                end
                numvalued[k] = isnum
                if (isnum)
                    numberstorage[k] = tmp_sum
                else
                    setstorage[k] = tmp_sum
                end
                PRINT_EVAL && print_call!(:+, k, children_idx)
                PRINT_EVAL && println("result: $tmp_sum")
            elseif op == 2 # :-
                child1 = first(children_idx)
                @assert n_children == 2
                @inbounds ix1 = children_arr[child1]
                @inbounds ix2 = children_arr[child1+1]
                @inbounds chdset1 = numvalued[ix1]
                @inbounds chdset2 = numvalued[ix2]
                @inbounds isnum = chdset1
                @inbounds isnum &= chdset2
                if chdset1
                    @inbounds tmp_sub = numberstorage[ix1]
                else
                    @inbounds tmp_sub = setstorage[ix1]
                end
                if chdset2
                    @inbounds tmp_sub -= numberstorage[ix2]
                else
                    @inbounds tmp_sub -= setstorage[ix2]
                end
                PRINT_EVAL && print_call!(:-, k, children_idx)
                PRINT_EVAL && println("result: $tmp_sub")
                numvalued[k] = isnum
                if (isnum)
                    numberstorage[k] = tmp_sub
                else
                    setstorage[k] = set_value_post(x_values, tmp_sub, current_node)
                end
            elseif op == 3 # :*
                tmp_prod = 1.0
                isnum = true
                chdset = true
                for c_idx in children_idx
                    @inbounds chdset = numvalued[children_arr[c_idx]]
                    isnum &= chdset
                    if (chdset)
                        @inbounds tmp_prod *= numberstorage[children_arr[c_idx]]
                        else
                        @inbounds tmp_prod = tmp_prod*setstorage[children_arr[c_idx]]
                    end
                end
                PRINT_EVAL && print_call!(:*, k, children_idx)
                PRINT_EVAL && println("result: $tmp_prod")
                if (isnum)
                    numberstorage[k] = tmp_prod
                else
                    setstorage[k] = set_value_post(x_values, tmp_prod, current_node)
                end
                numvalued[k] = isnum
            elseif op == 4 # :^
                @assert n_children == 2
                idx1 = first(children_idx)
                idx2 = last(children_idx)
                @inbounds ix1 = children_arr[idx1]
                @inbounds ix2 = children_arr[idx2]
                @inbounds chdset1 = numvalued[ix1]
                @inbounds chdset2 = numvalued[ix2]
                if chdset1
                    @inbounds base = numberstorage[ix1]
                else
                    @inbounds base = setstorage[ix1]
                end
                if chdset2
                    @inbounds exponent = numberstorage[ix2]
                else
                    @inbounds exponent = setstorage[ix2]
                end
                if exponent == 1
                    if chdset1
                        @inbounds numberstorage[k] = base
                    else
                        @inbounds setstorage[k] = base
                    end
                else
                    if chdset1 || chdset2
                        setstorage[k] = set_value_post(x_values, pow(base,exponent), current_node)
                    else
                        numberstorage[k] = pow(base,exponent)
                    end
                    PRINT_EVAL && print_call!(:^, k, children_idx)
                    PRINT_EVAL && println("result: $(pow(base,exponent))")
                end
                numvalued[k] = ~(chdset1 || chdset2)
            elseif op == 5 # :/
                @assert n_children == 2
                idx1 = first(children_idx)
                idx2 = last(children_idx)
                @inbounds ix1 = children_arr[idx1]
                @inbounds ix2 = children_arr[idx2]
                @inbounds chdset1 = numvalued[ix1]
                @inbounds chdset2 = numvalued[ix2]
                if chdset1
                    @inbounds numerator = numberstorage[ix1]
                else
                    @inbounds numerator = setstorage[ix1]
                end
                if chdset2
                    @inbounds denominator = numberstorage[ix2]
                else
                    @inbounds denominator = setstorage[ix2]
                end
                if chdset1 && chdset2
                    numberstorage[k] = numerator/denominator
                else
                    setstorage[k] = set_value_post(x_values, numerator/denominator, current_node)
                end
                PRINT_EVAL && print_call!(:/, k, children_idx)
                PRINT_EVAL && println("result: $(numerator/denominator)")
                numvalued[k] = chdset1 && chdset2
            elseif op == 6 # ifelse
                PRINT_EVAL && print_call!(:ifelse, k, children_idx)
                @assert n_children == 3
                idx1 = first(children_idx)
                @inbounds chdset1 = numvalued[idx1]
                if chdset1
                    @inbounds condition = setstorage[children_arr[idx1]]
                else
                    @inbounds condition = numberstorage[children_arr[idx1]]
                end
                @inbounds chdset2 = numvalued[children_arr[idx1+1]]
                @inbounds chdset3 = numvalued[children_arr[idx1+2]]
                if chdset2
                    @inbounds lhs = setstorage[children_arr[idx1+1]]
                else
                    @inbounds lhs = numberstorage[children_arr[idx1+1]]
                end
                if chdset3
                    @inbounds rhs = setstorage[children_arr[idx1+2]]
                else
                    @inbounds rhs = numberstorage[children_arr[idx1+2]]
                end
                error("IF ELSE TO BE IMPLEMENTED SHORTLY")
                #storage[k] = set_value_post(x_values, ifelse(condition == 1, lhs, rhs), current_node)
            elseif op >= JuMP._Derivatives.USER_OPERATOR_ID_START
                op_sym = id_to_operator[op]
                evaluator = user_operators.multivariate_operator_evaluator[op - JuMP._Derivatives.USER_OPERATOR_ID_START+1]
                f_input = view(user_input_buffer, 1:n_children)
                r = 1
                isnum = true
                for c_idx in children_idx
                    @inbounds ix = children_arr[c_idx]
                    @inbounds chdset = numvalued[ix]
                    isnum &= chdset
                    if chdset
                        @inbounds f_input[r] = setstorage[ix]
                    else
                        @inbounds f_input[r] = numberstorage[ix]
                    end
                    r += 1
                end
                fval = MOI.eval_objective(evaluator, f_input)
                PRINT_EVAL && print_call!(op_sym, k, children_idx)
                PRINT_EVAL && println("result: $(fval)")
                if isnum
                    numberstorage[k] = fval
                else
                    setstorage[k] = set_value_post(x_values, fval, current_node)
                end
                numvalued[k] = isnum
            else
                error("Unsupported operation $(operators[op])")
            end
        elseif nod.nodetype == JuMP._Derivatives.CALLUNIVAR # univariate function
            @inbounds child_idx = children_arr[adj.colptr[k]]
            @inbounds chdset = numvalued[child_idx]
            if chdset
                @inbounds child_val = numberstorage[child_idx]
            else
                @inbounds child_val = setstorage[child_idx]
            end
            if op >= JuMP._Derivatives.USER_UNIVAR_OPERATOR_ID_START
                userop = op - JuMP._Derivatives.USER_UNIVAR_OPERATOR_ID_START + 1
                @inbounds f = user_operators.univariate_operator_f[userop]
                fval = f(child_val)
            else
                fval = eval_univariate_set(op, child_val)
            end
            if chdset
                @inbounds numberstorage[k] = fval
            else
                @inbounds setstorage[k] = set_value_post(x_values, fval, current_node)
            end
            PRINT_EVAL && print_call!(id_to_operator[op], k, child_idx)
            PRINT_EVAL && println("result: $(fval)")
            numvalued[k] = chdset
        elseif nod.nodetype == JuMP._Derivatives.COMPARISON
            op = nod.index
            @inbounds children_idx = nzrange(adj,k)
            n_children = length(children_idx)
            result = true
            for r in 1:n_children-1
                @inbounds ix1 = children_arr[children_idx[r]]
                @inbounds ix2 = children_arr[children_idx[r+1]]
                @inbounds isnum1 = numvalued[ix1]
                @inbounds isnum2 = numvalued[ix2]
                if isnum1
                    @inbounds cval_lhs = numberstorage[ix1]
                else
                    @inbounds cval_lhs = setstorage[ix1]
                end
                if isnum2
                    @inbounds cval_rhs = numberstorage[ix2]
                else
                    @inbounds cval_rhs = setstorage[ix2]
                end
                if op == 1
                    result &= cval_lhs <= cval_rhs
                elseif op == 2
                    result &= cval_lhs == cval_rhs
                elseif op == 3
                    result &= cval_lhs >= cval_rhs
                elseif op == 4
                    result &= cval_lhs < cval_rhs
                elseif op == 5
                    result &= cval_lhs > cval_rhs
                end
            end
            numberstorage[k] = result
        elseif nod.nodetype == JuMP._Derivatives.LOGIC
            op = nod.index
            @inbounds children_idx = nzrange(adj,k)
            ix1 = children_arr[first(children_idx)]
            ix2 = children_arr[last(children_idx)]
            @inbounds isnum1 = numvalued[ix1]
            @inbounds isnum2 = numvalued[ix2]
            if isnum1
                cval_lhs = (numberstorage[ix1] == 1)
            else
                cval_lhs = (setstorage[ix1] == 1)
            end
            if isnum2
                cval_rhs = (numberstorage[ix2] == 1)
            else
                cval_rhs = (setstorage[ix2] == 1)
            end
            if op == 1
                numberstorage[k] = cval_lhs && cval_rhs
            elseif op == 2
                numberstorage[k] = cval_lhs || cval_rhs
            end
        else
            error("Unrecognized node type $(nod.nodetype).")
        end
    end
    if numvalued[1]
        return numberstorage[1]
    else
        return setstorage[1]
    end
end

function forward_eval_all(d::Evaluator,x)
    subexpr_values_flt = d.subexpression_values_flt
    subexpr_values_set = d.subexpression_values_set
    user_operators = d.m.nlp_data.user_operators::JuMP._Derivatives.UserOperatorRegistry
    user_input_buffer = d.jac_storage

    PRINT_EVAL && println("START TO EVALUATE SUBEXPRESSIONS")
    for (ind, k) in enumerate(reverse(d.subexpression_order))
        PRINT_EVAL && println("SUBEXPRESSION NUMBER: #$(ind)")
        ex = d.subexpressions[k]
        temp = forward_eval(ex.setstorage, ex.numberstorage, ex.numvalued,
                                         ex.nd, ex.adj, ex.const_values,
                                         d.parameter_values, d.current_node,
                                         x, subexpr_values_flt, subexpr_values_set, d.subexpression_isnum, user_input_buffer,
                                         user_operators=user_operators)
        PRINT_EVAL && println("START TO EVALUATE SUBEXPRESSIONS")
        d.subexpression_isnum[ind] = ex.numvalued[1]
        if d.subexpression_isnum[ind]
            d.subexpression_values_flt[k] = temp
        else
            d.subexpression_values_set[k] = temp
        end
    end

    PRINT_EVAL && println("START TO EVALUATE OBJECTIVE")
    if d.has_nlobj
        ex = d.objective
        forward_eval(ex.setstorage, ex.numberstorage, ex.numvalued,
                     ex.nd, ex.adj, ex.const_values,
                     d.parameter_values, d.current_node,
                     x, subexpr_values_flt, subexpr_values_set, d.subexpression_isnum, user_input_buffer,
                     user_operators=user_operators)
    end

    PRINT_EVAL && println("START TO EVALUATE CONSTRAINTS")
    for (ind,ex) in enumerate(d.constraints)
        PRINT_EVAL && println("CONSTRAINT NUMBER: #$(ind)")
        forward_eval(ex.setstorage, ex.numberstorage, ex.numvalued,
                     ex.nd, ex.adj, ex.const_values,
                     d.parameter_values, d.current_node,
                     x, subexpr_values_flt, subexpr_values_set, d.subexpression_isnum, user_input_buffer,
                     user_operators=user_operators)
    end
end

# maximum number to perform reverse operation on associative term by summing and evaluating pairs
# remaining terms not reversed
const MAX_ASSOCIATIVE_REVERSE = 4
function reverse_eval(setstorage::Vector{T}, numberstorage, numvalued,
                      nd::Vector{JuMP.NodeData}, adj, x_values) where T

    @assert length(setstorage) >= length(nd)
    @assert length(numberstorage) >= length(nd)
    @assert length(numvalued) >= length(nd)

    children_arr = rowvals(adj)
    N = length(x_values)

    for k in 2:length(nd)
        @inbounds nod = nd[k]
        if (nod.nodetype == JuMP._Derivatives.VALUE ||
            nod.nodetype == JuMP._Derivatives.LOGIC ||
            nod.nodetype == JuMP._Derivatives.COMPARISON ||
            nod.nodetype == JuMP._Derivatives.PARAMETER ||
            nod.nodetype == JuMP._Derivatives.EXTRA ||
            nod.nodetype == JuMP._Derivatives.VARIABLE)
            continue
        elseif nod.nodetype == JuMP._Derivatives.SUBEXPRESSION
            @inbounds isnum = subexpression_isnum[nod.index]
            if ~isnum
                @inbounds subexpr_values_set[nod.index] = setstorage[k]
            end
        elseif numvalued[k]
            continue
        elseif (nod.nodetype == JuMP._Derivatives.CALL)
            op = nod.index
            parent_index = nod.parent
            @inbounds children_idx = nzrange(adj,k)
            @inbounds parent_value = setstorage[parent_index]
            n_children = length(children_idx)
            if (op >= JuMP._Derivatives.USER_OPERATOR_ID_START)
                continue
            elseif (op == 1) # :+
                PRINT_EVAL && print_call!(:plus_rev, k, children_idx)
                tmp_sum = 0.0
                tmp_hold = 0.0
                chdset = true
                count = 0
                for c_idx in children_idx
                    if (count < MAX_ASSOCIATIVE_REVERSE)
                        if ~numvalued[ci_idx]
                            tmp_sum = 0.0; tmp_hold = 0.0
                            for cin_idx in children_idx
                                if (cin_idx != c_idx)
                                    @inbounds ix = children_arr[cin_idx]
                                    @inbounds chdset = numvalued[cin_idx]
                                    if (chdset)
                                        @inbounds tmp_sum += numberstorage[ix]
                                    else
                                        @inbounds tmp_sum += setstorage[ix]
                                    end
                                end
                            end
                            @inbounds ix = children_arr[c_idx]
                            @inbounds chdset = numvalued[c_idx]
                            if (chdset)
                                @inbounds tmp_hold += numberstorage[ix]
                            else
                                @inbounds tmp_hold += setstorage[ix]
                            end
                            pnew, xhold, xsum = plus_rev(parent_value, tmp_hold, tmp_sum)
                            setstorage[parent_index] = pnew
                            setstorage[ix] = xhold
                            count += 1
                        end
                    else
                        break
                    end
                end
                PRINT_EVAL && println("pnew: $pnew, xnew: $xnew, ynew: $ynew")
            elseif (op == 2) # :-
                PRINT_EVAL && print_call!(:minus_rev, k, children_idx)
                child1 = first(children_idx)
                @assert n_children == 2
                @inbounds ix1 = children_arr[child1]
                @inbounds ix2 = children_arr[child1+1]
                @inbounds chdset1 = numvalued[ix1]
                @inbounds chdset2 = numvalued[ix2]
                if chdset1
                    pnew, xnew, ynew = minus_rev(parent_value, numberstorage[ix1], setstorage[ix2])
                elseif chdset2
                    pnew, xnew, ynew = minus_rev(parent_value, setstorage[ix1], numberstorage[ix2])
                else
                    pnew, xnew, ynew = minus_rev(parent_value, setstorage[ix1], setstorage[ix2])
                end
                setstorage[parent_index] = pnew
                if ~chdset1
                    setstorage[chdset1] = xnew
                end
                if ~chdset2
                    setstorage[chdset2] = ynew
                end
                PRINT_EVAL && println("pnew: $pnew, xnew: $xnew, ynew: $ynew")
            elseif (op == 3) # :*
                PRINT_EVAL && print_call!(:mul_rev, k, children_idx)
                tmp_sum = 1.0
                tmp_hold = 1.0
                chdset = true
                count = 0
                for c_idx in children_idx
                    if (count < MAX_ASSOCIATIVE_REVERSE)
                        if ~numvalued[ci_idx]
                            tmp_sum = 1.0; tmp_hold = 1.0
                            for cin_idx in children_idx
                                if (cin_idx != c_idx)
                                    @inbounds ix = children_arr[cin_idx]
                                    @inbounds chdset = numvalued[cin_idx]
                                    if (chdset)
                                        @inbounds tmp_sum *= numberstorage[ix]
                                    else
                                        @inbounds tmp_sum *= setstorage[ix]
                                    end
                                end
                            end
                            @inbounds ix = children_arr[c_idx]
                            @inbounds chdset = numvalued[c_idx]
                            if (chdset)
                                @inbounds tmp_hold *= numberstorage[ix]
                            else
                                @inbounds tmp_hold *= setstorage[ix]
                            end
                            pnew, xhold, xsum = mul_rev(parent_value, tmp_hold, tmp_sum)
                            setstorage[parent_index] = pnew
                            setstorage[ix] = xhold
                            count += 1
                        end
                    else
                        break
                    end
                end
                PRINT_EVAL && println("pnew: $pnew, xnew: $xnew, ynew: $ynew")
            elseif (op == 4) # :^
                PRINT_EVAL && print_call!(:pow_rev, k, children_idx)
                child1 = first(children_idx)
                @assert n_children == 2
                @inbounds ix1 = children_arr[child1]
                @inbounds ix2 = children_arr[child1+1]
                @inbounds chdset1 = numvalued[ix1]
                @inbounds chdset2 = numvalued[ix2]
                if chdset1
                    pnew, xnew, ynew = pow_rev(parent_value, numberstorage[ix1], setstorage[ix2])
                elseif chdset2
                    pnew, xnew, ynew = pow_rev(parent_value, setstorage[ix1], numberstorage[ix2])
                else
                    pnew, xnew, ynew = pow_rev(parent_value, setstorage[ix1], setstorage[ix2])
                end
                setstorage[parent_index] = pnew
                if ~chdset1
                    setstorage[chdset1] = xnew
                end
                if ~chdset2
                    setstorage[chdset2] = ynew
                end
                PRINT_EVAL && println("pnew: $pnew, xnew: $xnew, ynew: $ynew")
            elseif (op == 5) # :/
                PRINT_EVAL && print_call!(:div_rev, k, children_idx)
                child1 = first(children_idx)
                @assert n_children == 2
                @inbounds ix1 = children_arr[child1]
                @inbounds ix2 = children_arr[child1+1]
                @inbounds chdset1 = numvalued[ix1]
                @inbounds chdset2 = numvalued[ix2]
                if chdset1
                    pnew, xnew, ynew = div_rev(parent_value, numberstorage[ix1], setstorage[ix2])
                elseif chdset2
                    pnew, xnew, ynew = div_rev(parent_value, setstorage[ix1], numberstorage[ix2])
                else
                    pnew, xnew, ynew = div_rev(parent_value, setstorage[ix1], setstorage[ix2])
                end
                setstorage[parent_index] = pnew
                if ~chdset1
                    setstorage[chdset1] = xnew
                end
                if ~chdset2
                    setstorage[chdset2] = ynew
                end
                PRINT_EVAL && println("pnew: $pnew, xnew: $xnew, ynew: $ynew")
            elseif (op == 6) # ifelse
                continue
            end
            #=
            PRINT_EVAL && println("result: $tmp_sum")
            =#
        elseif (nod.nodetype == JuMP._Derivatives.CALLUNIVAR) # assumes that child is set-valued and thus parent is set-valued
            op = nod.index
            @inbounds child_value = setstorage[k]
            @inbounds parent_value = setstorage[nod.parent]
            pnew, cnew = eval_univariate_set_reverse(op, parent_value, child_val)
            @inbounds setstorage[k] = cnew
            @inbounds setstorage[nod.parent] = pnew
            PRINT_EVAL && print_call!(univariate_operators_rev[op], k, children_idx)
        end
    end
    return nothing
end

# looks good
function reverse_eval_all(d::Evaluator,x)
    if d.has_nlobj
        # Cut Objective at upper bound
        ex = d.objective
        ex.setstorage[1] = x.setstorage[1] ∩ IntervalType(-Inf,d.objective_ubd)
        reverse_eval(ex.setstorage, ex.numberstorage, ex.numvalued, ex.nd, ex.adj, x)
    end
    for i in 1:length(d.constraints)
        # Cut constraints on constraint bounds & reverse
        ex = d.constraints[i]
        ex.setstorage[1] = x.setstorage[1] ∩ IntervalType(d.constraints_lbd[i], d.constraints_ubd[i])
        reverse_eval(ex.setstorage, ex.numberstorage, ex.numvalued, ex.nd, ex.adj, x)
    end
    for k in 1:length(d.subexpression_order)
        ex = d.subexpressions[d.subexpression_order[k]]
        ex.setstorage[1] = subexpression_values_set[d.subexpression_order[k]] # TODO MAKE SURE INDEX ON SUBEXPRESSION_VALUES_SET CORRECT
        reverse_eval(ex.setstorage, ex.numberstorage, ex.numvalued, ex.nd, ex.adj, x)
    end
    copyto!(d.last_x,x)
end

"""
    forward_reverse_pass(d::Evaluator,x)

Performs a `d.fw_repeats` forward passes of the set-value evaluator each followed
by a reverse pass if `d.has_reverse` as long as the node between passes differs
by more that `d.fw_atol` at each iteration.
"""
function forward_reverse_pass(d::Evaluator,x)
    #if ~same_box(d.current_node, d.last_node, 0.0)
#        d.last_node = d.current_node
        if (d.last_x != x)
            if d.has_reverse
                for i in d.fw_repeats
                    forward_eval_all(d,x)
                    reverse_eval_all(d,x)
                    # if node on reverse is same... forward reversing
                    same_box(d.current_node, get_node(d), d.fw_atol) && break
                end
            else
                forward_eval_all(d,x)
            end
        end
#    end
end
