using PackedArrays, Test, LinearAlgebra

A = collect(reshape(1:9.0,3,3))

@testset "setindex!" begin
    APL = SymmetricPacked(A, :L)
    APL[1,1]=3
    @test APL[1,1] == 3
    @test_throws ArgumentError APL[1,3]=3
end

VERSION<v"1.8.0-DEV.1049" && include("blas.jl")
