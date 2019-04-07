module ParamIntvPrecond_Tests

using Compat
using Compat.Test
using IntervalArithmetic
using EAGO, LinearAlgebra

@testset "Preconditioners" begin
    opt1 = parametric_interval_params(:None,:Newton,1E-30,1E-6,Int(2),Int(2),100)
    function h(out,x,p)
         out[1] = p[1]
     end
    function hj(out,x,p)
        out[1,1] = p[1]
    end
    X = [IntervalType(1.0,2.0),IntervalType(1.0,2.0)]
    P = [IntervalType(1.0,2.0),IntervalType(1.0,3.0)]

    X1,P2 = EAGO.precondition(h,hj,X,P,opt1)
    @test X == X
    @test P == P

    opt2 = parametric_interval_params(:XYZYZYYA,:Newton,1E-30,1E-6,Int(2),Int(2),100)
    @test_throws ErrorException EAGO.precondition(h,hj,X,P,opt2)
end
#=
@testset "Preconditioner Utilities" begin
    A = speye(spzeros(Float64,9,9))
    A[2,3] = 2.1
    A[8,7] = 3.1
    kl,ku = EAGO.sparse_bandwidth(A)
    @test kl == 1
    @test ku == 1
end
=#
end
