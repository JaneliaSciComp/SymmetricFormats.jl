module SymmetricFormats

import Base: require_one_based_indexing, size, convert, unsafe_convert
import Base: getindex, setindex!, copy
import LinearAlgebra: checksquare, char_uplo
import LinearAlgebra: mul!, BLAS, BlasFloat, generic_matvecmul!, MulAddMul

export SymmetricPacked, packedsize

struct SymmetricPacked{T,S<:AbstractVecOrMat{<:T},R} <: AbstractMatrix{T}
    tri::Vector{T}
    n::Int
    uplo::Char

    function SymmetricPacked{T,S,R}(tri, n, uplo) where {T,S<:AbstractVecOrMat{<:T},R}
        require_one_based_indexing(tri)
        uplo=='U' || uplo=='L' || throw(ArgumentError("uplo must be either 'U' (upper) or 'L' (lower)"))
        new{T,S,R}(tri, n, uplo)
    end
end

function pack(A::AbstractMatrix{T}, uplo::Symbol) where {T}
    n = size(A,1)
    AP = Vector{T}(undef, (n*(n+1))>>1)
    k = 1
    for j in 1:n
        for i in (uplo==:L ? (j:n) : (1:j))
            AP[k] = A[i,j]
            k += 1
        end
    end
    return AP
end

"""
    SymmetricPacked(A, uplo=:U, offdiag=Val(:RO))

Construct a `Symmetric` matrix in packed form of the upper (if `uplo = :U`)
or lower (if `uplo = :L`) triangle of the matrix `A`.  `offdiag` specifies
whether elements not on the diagaonal can be set (if `:RW`) or not (if `:RO`).

# Examples
```jldoctest
julia> A = [1 0 2 0 3; 0 4 0 5 0; 6 0 7 0 8; 0 9 0 1 0; 2 0 3 0 4]
5×5 Matrix{Int64}:
 1  0  2  0  3
 0  4  0  5  0
 6  0  7  0  8
 0  9  0  1  0
 2  0  3  0  4

julia> AP = SymmetricPacked(A)
5×5 SymmetricPacked{Int64, Matrix{Int64}}:
 1  0  2  0  3
 0  4  0  5  0
 2  0  7  0  8
 0  5  0  1  0
 3  0  8  0  4

julia> Base.summarysize(A)
240

julia> Base.summarysize(AP)
184
```
"""
function SymmetricPacked(A::AbstractMatrix{T}, uplo::Symbol=:U, offdiag=Val(:RO)) where {T}
    n = checksquare(A)
    SymmetricPacked{T,typeof(A),offdiag}(pack(A, uplo), n, char_uplo(uplo))
end

function SymmetricPacked(x::SymmetricPacked{T,S,R}) where{T,S,R}
    SymmetricPacked{T,S,R}(T.(x.tri), x.n, x.uplo)
end

function SymmetricPacked(V::AbstractVector{T}, uplo::Symbol=:U, offdiag=Val(:RO)) where {T}
    n = (sqrt(1+8*length(V))-1)/2
    isinteger(n) || throw(DimensionMismatch("length of vector does not corresond to the number of elements in the triangle of a square matrix"))
    SymmetricPacked{T,typeof(V),offdiag}(V, round(Int, n), char_uplo(uplo))
end

checksquare(x::SymmetricPacked) = x.n

convert(::Type{SymmetricPacked{T,S,R}}, x::SymmetricPacked) where {T,S,R} = SymmetricPacked{T,S}(T.(x.tri), x.n, x.uplo)

unsafe_convert(::Type{Ptr{T}}, A::SymmetricPacked{T,S,R}) where {T,S,R} = Base.unsafe_convert(Ptr{T}, A.tri)

size(A::SymmetricPacked) = (A.n,A.n)

function size(A::SymmetricPacked, d::Integer)
    d<1 && throw(ArgumentError("dimension must be ≥ 1, got $d"))
    d<=2 ? A.n : 1
end

"""
    packedsize(A)

Return the number of elements in the triangle of a square matrix.
"""
packedsize(A::SymmetricPacked) = length(A.tri)

function packedsize(A::AbstractArray)
    n = checksquare(A)
    (n*(n+1))>>1
end

@inline function getindex(A::SymmetricPacked, i::Int, j::Int)
    @boundscheck checkbounds(A, i, j)
    if A.uplo=='U'
        i,j = minmax(i,j)
        @inbounds r = A.tri[i+(j*(j-1))>>1]
    else
        j,i = minmax(i,j)
        @inbounds r = A.tri[i+((2*A.n-j)*(j-1))>>1]
    end
    return r
end

function _setindex!(A::SymmetricPacked, v, i::Int, j::Int)
    @boundscheck checkbounds(A, i, j)
    if A.uplo=='U'
        i,j = minmax(i,j)
        @inbounds A.tri[i+(j*(j-1))>>1] = v
    else
        j,i = minmax(i,j)
        @inbounds A.tri[i+((2*A.n-j)*(j-1))>>1] = v
    end
    return v
end

function setindex!(A::SymmetricPacked{T,S,Val(:RO)}, v, i::Int, j::Int) where {T,S}
    i!=j && throw(ArgumentError("Cannot set a non-diagonal index in a symmetric matrix"))
    _setindex!(A, v, i, j)
end

setindex!(A::SymmetricPacked{T,S,Val(:RW)}, v, i::Int, j::Int) where {T,S} = _setindex!(A, v, i, j)

function copy(A::SymmetricPacked{T,S}) where {T,S}
    B = copy(A.tri)
    SymmetricPacked{T,S}(B, A.n, A.uplo)
end

@inline function mul!(y::StridedVector{T},
                      AP::SymmetricPacked{T,<:StridedMatrix,R},
                      x::StridedVector{T},
                      α::Number, β::Number) where {T<:BlasFloat,R}
    alpha, beta = promote(α, β, zero(T))
    if alpha isa Union{Bool,T} && beta isa Union{Bool,T}
        BLAS.spmv!(AP.uplo, alpha, AP.tri, x, beta, y)
    else
        generic_matvecmul!(y, 'N', AP, x, MulAddMul(alpha, beta))
    end
end

VERSION<v"1.8.0-DEV.1049" && include("blas.jl")

end # module
