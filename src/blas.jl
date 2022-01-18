import LinearAlgebra.BLAS: libblastrampoline, @blasfunc, BlasInt, BlasReal

export spr!

for (fname, elty) in ((:dspr_, :Float64),
                      (:sspr_, :Float32))
    @eval begin
        function spr!(uplo::AbstractChar,
                      n::Integer,
                      α::$elty,
                      x::Union{Ptr{$elty}, AbstractArray{$elty}},
                      incx::Integer,
                      AP::Union{Ptr{$elty}, AbstractArray{$elty}})

            ccall((@blasfunc($fname), libblastrampoline), Cvoid,
                  (Ref{UInt8},     # uplo,
                   Ref{BlasInt},   # n,
                   Ref{$elty},     # α,
                   Ptr{$elty},     # x,
                   Ref{BlasInt},   # incx,
                   Ptr{$elty},     # AP,
                   Clong),         # length of uplo
                  uplo,
                  n,
                  α,
                  x,
                  incx,
                  AP,
                  1)
            return AP
        end
    end
end

function spr!(uplo::AbstractChar,
              α::Real, x::Union{DenseArray{T}, AbstractVector{T}},
              AP::Union{DenseArray{T}, AbstractVector{T}}) where {T <: BlasReal}
    require_one_based_indexing(AP, x)
    N = length(x)
    if 2*length(AP) < N*(N + 1)
        throw(DimensionMismatch("Packed symmetric matrix A has size smaller than length(x) = $(N)."))
    end
    return spr!(uplo, N, convert(T, α), x, stride(x, 1), AP)
end

"""
    spr!(uplo, α, x, AP)

Update matrix `A` as `α*A*x*x'`, where `A` is a symmetric matrix provided
in packed format `AP` and `x` is a vector.

With `uplo = 'U'`, the array AP must contain the upper triangular part of the
symmetric matrix packed sequentially, column by column, so that `AP[1]`
contains `A[1, 1]`, `AP[2]` and `AP[3]` contain `A[1, 2]` and `A[2, 2]`
respectively, and so on.

With `uplo = 'L'`, the array AP must contain the lower triangular part of the
symmetric matrix packed sequentially, column by column, so that `AP[1]`
contains `A[1, 1]`, `AP[2]` and `AP[3]` contain `A[2, 1]` and `A[3, 1]`
respectively, and so on.

The scalar input `α` must be real.

The array inputs `x` and `AP` must all be of `Float32` or `Float64` type.
Return the updated `A`.
"""
spr!
