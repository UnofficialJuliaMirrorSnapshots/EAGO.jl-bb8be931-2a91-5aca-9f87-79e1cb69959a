function check_valid_safety!()
end

function is_safe_linear_cut(x::Optimizer,a::Vector{Float64},b::Float64)
    (abs(b) > x.cut_safety_constant) && (return false)
    lena = length(a)
    for i in 1:lena
        if a[i] != 0.0
            ~(x.cut_safety_lower <= abs(a[i]) <= x.cut_safety_upper) && (return false)
            for j in 1:lena
                if a[j] != 0.0
                    ~(x.cut_safety_lower <= abs(a[i]/a[j]) <= x.cut_safety_upper) && (return false)
                end
            end
        end
    end
    return true
end

"""
    default_add_cut!

Branch-and-Cut under development. Currently does nothing.
"""
function default_add_cut!(x::Optimizer)
end
