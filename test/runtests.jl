using SymmetricFormats, Test, LinearAlgebra

A = collect(reshape(1:9.0,3,3))

@testset "setindex!" begin
    APL = SymmetricPacked(A, :L)
    APL[1,1]=3
    @test APL[1,1] == 3
    @test_throws ArgumentError APL[1,3]=3
end

@testset "read/write upper" begin
    APU = SymmetricPacked(A, :U, Val(:RW))

    @test APU[1,1] == A[1,1]
    @test APU[1,2] == A[1,2]
    @test APU[1,3] == A[1,3]
    @test APU[2,1] == A[1,2]
    @test APU[2,2] == A[2,2]
    @test APU[2,3] == A[2,3]
    @test APU[3,1] == A[1,3]
    @test APU[3,2] == A[2,3]
    @test APU[3,3] == A[3,3]

    APU[1,2] = 0
    @test APU[2,1] == 0
end

@testset "read/write lower" begin
    APL = SymmetricPacked(A, :L, Val(:RW))

    @test APL[1,1] == A[1,1]
    @test APL[1,2] == A[2,1]
    @test APL[1,3] == A[3,1]
    @test APL[2,1] == A[2,1]
    @test APL[2,2] == A[2,2]
    @test APL[2,3] == A[3,2]
    @test APL[3,1] == A[3,1]
    @test APL[3,2] == A[3,2]
    @test APL[3,3] == A[3,3]

    APL[1,2] = 0
    @test APL[2,1] == 0
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

@testset "packedsize" begin
    @test packedsize(A) == 6
    AP = SymmetricPacked(A)
    @test packedsize(AP) == 6
end

@testset "vector constructor" begin
    AP = SymmetricPacked(A)
    BP = SymmetricPacked(AP.tri)
    @test AP == BP
end

VERSION<v"1.8.0-DEV.1049" && include("blas.jl")
