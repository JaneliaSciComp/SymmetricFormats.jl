using PackedArrays, Test, LinearAlgebra

A = collect(reshape(1:9.0,3,3))

@testset "setindex!" begin
    APL = SymmetricPacked(A, :L)
    APL[1,1]=3
    @test APL[1,1] == 3
    @test_throws ArgumentError APL[1,3]=3
end

@testset "mul!" begin
    for uplo in [:U, :L]
        y = Float64[1,2,3]
        AP = SymmetricPacked(A, uplo)
        x = Float64[4,5,6]
        α = 0.5
        β = 0.5

        y_julia = α*AP*x + β*y
        y_blas = copy(y);  mul!(y_blas, AP, x, α, β)
        @test isapprox(y_julia, y_blas)
    end
end

VERSION<v"1.8.0-DEV.1049" && include("blas.jl")
