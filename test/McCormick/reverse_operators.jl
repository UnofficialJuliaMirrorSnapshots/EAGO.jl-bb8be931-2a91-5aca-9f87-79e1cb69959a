using EAGO, StaticArrays

function MC_1_is_equal(y, x, tol)
    bool1 = isapprox(y.cc,x.cc,atol=tol)
    bool2 = isapprox(y.cv,x.cv,atol=tol)
    bool3 = isapprox(y.cv_grad[1], x.cv_grad[1], atol=tol)
    bool4 = isapprox(y.cc_grad[1], x.cc_grad[1], atol=tol)
    bool5 = isapprox(y.Intv.lo, x.Intv.lo, atol=tol)
    bool6 = isapprox(y.Intv.hi, x.Intv.hi, atol=tol)
    return (bool1 && bool2 && bool3 && bool4 && bool5 && bool6)
end

a = MC{1}(1.0,EAGO.IntervalType(0.4,3.0),1)
a1 = MC{1}(-7.0,EAGO.IntervalType(-12.0,-4.0),1)
b = MC{1}(EAGO.IntervalType(-10.0,-1.0))
c = MC{1}(2.0,EAGO.IntervalType(1.1,4.5),1)
aout1, bout1, cout1 = mul_rev(a,b,c)
aout2, bout2, cout2 = div_rev(a,b,c)


a0 = MC{1}(7.0,EAGO.IntervalType(4.5,12.0),1)
b0 = MC{1}(EAGO.IntervalType(6.0,9.0))
c0 = MC{1}(5.0,EAGO.IntervalType(4.1,9.5),1)
aout3, bout3, cout3 = pow_rev(a0,b0,c0)

#=
@testset "Reverse Arithmetic Operators" begin
    tol = 1E-3
    a = MC{1}(1.0,EAGO.IntervalType(0.4,3.0),1)
    b = MC{1}(EAGO.IntervalType(-10.0,-1.0))
    c = MC{1}(2.0,EAGO.IntervalType(1.1,4.5),1)

    aout, bout, cout = plus_rev(a,b,c)
    a1_cv_grad = SVector{1,Float64}(1.0)
    a1_cc_grad = SVector{1,Float64}(1.0)
    a1 = MC{1}(1.0, 1.0, EAGO.IntervalType(0.4,3.0), a1_cv_grad, a1_cc_grad, false)
    b1_cv_grad = SVector{1,Float64}(0.0)
    b1_cc_grad = SVector{1,Float64}(0.0)
    b1 = MC{1}(-1.0, -1.0, EAGO.IntervalType(-4.1, -1.0), b1_cv_grad, b1_cc_grad, false)
    c1_cv_grad = SVector{1,Float64}(1.0)
    c1_cc_grad = SVector{1,Float64}(1.0)
    c1 = MC{1}(2.0, 2.0, EAGO.IntervalType(1.4, 4.5), c1_cv_grad, c1_cc_grad, false)
    @test MC_1_is_equal(aout, a1, tol)
    @test MC_1_is_equal(bout, b1, tol)
    @test MC_1_is_equal(cout, c1, tol)

    #aout, bout, cout = plus_rev(a,b,1.0)
    #@test MC_1_is_equal(a, x, tol)
    #@test MC_1_is_equal(b, x, tol)


    a_alt = MC{1}(-7.0,EAGO.IntervalType(-12.0,-4.0),1)
    aout1, bout1, cout1 = minus_rev(a, b, c)
    aout2, bout2, cout2 = minus_rev(a_alt, b, c)

    b1_cv_grad = SVector{1,Float64}(2.0)
    b1_cc_grad = SVector{1,Float64}(0.0)
    c1_cv_grad = SVector{1,Float64}(1.0)
    c1_cc_grad = SVector{1,Float64}(-1.0)
    a1 = MC{1}(-7.0,EAGO.IntervalType(-12.0,-4.0),1)
    b1 = MC{1}(3.0, -1.0, EAGO.IntervalType(Inf,-Inf), b1_cv_grad, b1_cc_grad, false)
    c1 = MC{1}(2.0, -2.0, EAGO.IntervalType(Inf,-Inf), b1_cv_grad, b1_cc_grad, false)

    b1_cv_grad = SVector{1,Float64}(2.0)
    b1_cc_grad = SVector{1,Float64}(2.0)
    c1_cv_grad = SVector{1,Float64}(1.0)
    c1_cc_grad = SVector{1,Float64}(1.0)
    a2 = MC{1}(-7.0,EAGO.IntervalType(-12.0,-4.0),1)
    b2 = MC{1}(-5.0, -5.0, EAGO.IntervalType(-10.0, -1.0), b1_cv_grad, b1_cc_grad, false)
    c2 = MC{1}(2.0, 2.0, EAGO.IntervalType(1.1, 4.5), c1_cv_grad, c1_cc_grad, false)

    @test MC_1_is_equal(aout1, a1, tol)
    @test MC_1_is_equal(bout1, b1, tol)
    @test MC_1_is_equal(cout1, c1, tol)
    @test MC_1_is_equal(aout2, a2, tol)
    @test MC_1_is_equal(bout2, b2, tol)
    @test MC_1_is_equal(cout2, c2, tol)

    #minus_rev(a::MC, b::MC)
    #aout, bout, cout = minus_rev(a,b,c)
    #minus_rev(a,b)

    aout, bout, cout = mul_rev(a::MC, b::MC, c::MC)
    a1 = MC{1}()
    b1 = MC{1}()
    c1 = MC{1}()
    @test MC_1_is_equal(aout, a1, tol)
    @test MC_1_is_equal(bout, b1, tol)
    @test MC_1_is_equal(cout, c1, tol)
    mul_rev(a,b,c)

    aout, bout, cout = div_rev(a::MC, b::MC, c::MC)
    a1 = MC{1}()
    b1 = MC{1}()
    c1 = MC{1}()
    @test MC_1_is_equal(aout, a1, tol)
    @test MC_1_is_equal(bout, b1, tol)
    @test MC_1_is_equal(cout, c1, tol)
    div_rev(a,b,c)

    inv_rev(a::MC, b::MC)
    inv_rev(a,b)

    aout, bout, cout = pow_rev(a::MC, b::MC, c::MC)
    a1 = MC{1}()
    b1 = MC{1}()
    c1 = MC{1}()
    @test MC_1_is_equal(aout, a1, tol)
    @test MC_1_is_equal(bout, b1, tol)
    @test MC_1_is_equal(cout, c1, tol)

    pow_rev(a, b, c)

    aout, bout = sqrt_rev(a::MC, b::MC)

    sqr_rev(f, x)

    #abs_rev!(a::MC, b::MC)  TO ADD
    =#
#end
#=
@testset "Reverse Exponential Operators" begin

    tol = 1E-3
    x = MC{1}()
    y = MC{1}()

    yout, xout = exp_rev(y,x)
    y1 = MC{1}()
    x1 = MC{1}()
    @test MC_1_is_equal(yout, y1, tol)
    @test MC_1_is_equal(xout, x1, tol)

    yout, xout = exp2_rev(y,x)
    y1 = MC{1}()
    x1 = MC{1}()
    @test MC_1_is_equal(yout, y1, tol)
    @test MC_1_is_equal(xout, x1, tol)

    yout, xout = exp10_rev(y,x)
    y1 = MC{1}()
    x1 = MC{1}()
    @test MC_1_is_equal(yout, y1, tol)
    @test MC_1_is_equal(xout, x1, tol)

    yout, xout = expm1_rev(y,x)
    y1 = MC{1}()
    x1 = MC{1}()
    @test MC_1_is_equal(yout, y1, tol)
    @test MC_1_is_equal(xout, x1, tol)

    yout, xout = log_rev(y,x)
    y1 = MC{1}()
    x1 = MC{1}()
    @test MC_1_is_equal(yout, y1, tol)
    @test MC_1_is_equal(xout, x1, tol)

    yout, xout = log2_rev(y,x)
    y1 = MC{1}()
    x1 = MC{1}()
    @test MC_1_is_equal(yout, y1, tol)
    @test MC_1_is_equal(xout, x1, tol)

    yout, xout = log10_rev(y,x)
    y1 = MC{1}()
    x1 = MC{1}()
    @test MC_1_is_equal(yout, y1, tol)
    @test MC_1_is_equal(xout, x1, tol)

    yout, xout = log1p_rev(y,x)
    y1 = MC{1}()
    x1 = MC{1}()
    @test MC_1_is_equal(yout, y1, tol)
    @test MC_1_is_equal(xout, x1, tol)
end

@testset "Reverse Extrema" begin
    # max_rev(a::MC, b::MC, c::MC) TODO
    # max_rev(a,b,c)

    # min_rev(a::MC, b::MC, c::MC) TODO
    # min_rev(a,b,c)
end

@testset "Reverse Hyperbolic" begin
    yout, xout = sinh_rev(y,x)
    y1 = MC{1}()
    x1 = MC{1}()
    @test MC_1_is_equal(yout, y1, tol)
    @test MC_1_is_equal(xout, x1, tol)

    yout, xout = cosh_rev(y,x)
    y1 = MC{1}()
    x1 = MC{1}()
    @test MC_1_is_equal(yout, y1, tol)
    @test MC_1_is_equal(xout, x1, tol)

    yout, xout = tanh_rev(y,x)
    y1 = MC{1}()
    x1 = MC{1}()
    @test MC_1_is_equal(yout, y1, tol)
    @test MC_1_is_equal(xout, x1, tol)
end

@testset "Reverse Inverse Hyperbolic" begin
    yout, xout = asinh_rev(y,x)
    y1 = MC{1}()
    x1 = MC{1}()
    @test MC_1_is_equal(yout, y1, tol)
    @test MC_1_is_equal(xout, x1, tol)

    yout, xout = acosh_rev(y,x)
    y1 = MC{1}()
    x1 = MC{1}()
    @test MC_1_is_equal(yout, y1, tol)
    @test MC_1_is_equal(xout, x1, tol)

    yout, xout = atanh_rev(y,x)
    y1 = MC{1}()
    x1 = MC{1}()
    @test MC_1_is_equal(yout, y1, tol)
    @test MC_1_is_equal(xout, x1, tol)
end

@testset "Reverse Inverse Hyperbolic" begin
    asin_rev(y,x)
    y1 = MC{1}()
    x1 = MC{1}()
    @test MC_1_is_equal(yout, y1, tol)
    @test MC_1_is_equal(xout, x1, tol)

    acos_rev(y,x)
    y1 = MC{1}()
    x1 = MC{1}()
    @test MC_1_is_equal(yout, y1, tol)
    @test MC_1_is_equal(xout, x1, tol)

    atan_rev(y,x)
    y1 = MC{1}()
    x1 = MC{1}()
    @test MC_1_is_equal(yout, y1, tol)
    @test MC_1_is_equal(xout, x1, tol)
end

@testset "Reverse Trignometric" begin
    #sin_rev(a::MC, b::MC) TODO
    #cos_rev(a::MC, b::MC) TODO
    #tan_rev(a::MC, b::MC) TODO
end
=#
