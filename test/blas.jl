using SymmetricFormats, Test, LinearAlgebra

n=10
for elty in (Float32, Float64)
    @testset "spr! $elty" begin
        α = rand(elty)
        M = rand(elty, n, n)
        AL = Symmetric(M, :L)
        AU = Symmetric(M, :U)
        x = rand(elty, n)

        ALP_result_julia_lower = SymmetricFormats.pack(α*x*x' + AL, :L)
        ALP_result_blas_lower = SymmetricFormats.pack(AL, :L)
        SymmetricFormats.spr!('L', α, x, ALP_result_blas_lower)
        @test ALP_result_julia_lower ≈ ALP_result_blas_lower

        AUP_result_julia_upper = SymmetricFormats.pack(α*x*x' + AU, :U)
        AUP_result_blas_upper = SymmetricFormats.pack(AU, :U)
        SymmetricFormats.spr!('U', α, x, AUP_result_blas_upper)
        @test AUP_result_julia_upper ≈ AUP_result_blas_upper
    end
end
