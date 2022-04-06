using SymmetricFormats, Test, LinearAlgebra

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

@testset "packedsize" begin
    @test packedsize(A) == 6
    AP = SymmetricPacked(A)
    @test packedsize(AP) == 6
end

@testset "vector constructor" begin
    AP = SymmetricPacked(A)
    BP = SymmetricPacked(AP.tri)
    @test AP == BP
    @test parent(BP) == AP.tri
end

A3 = collect(reshape(1:45.0,3,3,5))

@testset "batched" begin
    BAP = BatchedSymmetricPacked(A3, :L)
    BAP[1,1,1]=3
    @test BAP[1,1,1] == 3
    @test_throws ArgumentError BAP[1,3,1]=3
    @test packedsize(A3) == 30
    @test packedsize(BAP) == 30
    VBP = BatchedSymmetricPacked(BAP.tri, :L)
    @test VBP == BAP

    BA = BatchedSymmetric(A3, :L)
    BA[1,1,1]=3
    @test BA[1,1,1] == 3
    @test_throws ArgumentError BA[1,3,1]=3

    BA = BatchedMatrix(A3)
    BA[1,1,1]=3
    @test BA[1,1,1] == 3

    BA = BatchedVector(A3[:,1,:])
    BA[1,1]=3
    @test BA[1,1] == 3
end

VERSION<v"1.8.0-DEV.1049" && include("blas.jl")
