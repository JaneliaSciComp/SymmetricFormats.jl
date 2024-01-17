module SymmetricFormats

import Base: require_one_based_indexing, size, convert, unsafe_convert
import Base: getindex, setindex!, copy
import LinearAlgebra: checksquare, char_uplo
import LinearAlgebra: mul!, BLAS, BlasFloat, generic_matvecmul!, MulAddMul

export SymmetricPacked, packedsize
export BatchedSymmetricPacked, BatchedSymmetric, BatchedMatrix, BatchedVector

struct SymmetricPacked{T,S<:AbstractVecOrMat{<:T}} <: AbstractMatrix{T}
    tri::Vector{T}
    n::Int
    uplo::Char

    function SymmetricPacked{T,S}(tri, n, uplo) where {T,S<:AbstractVecOrMat{<:T}}
        require_one_based_indexing(tri)
        uplo=='U' || uplo=='L' || throw(ArgumentError("uplo must be either 'U' (upper) or 'L' (lower)"))
        new{T,S}(tri, n, uplo)
    end
end

struct BatchedSymmetricPacked{T,S<:AbstractArray{<:T}} <: AbstractArray{T,3}
    tri::Matrix{T}
    n::Int
    uplo::Char

    function BatchedSymmetricPacked{T,S}(tri, n, uplo) where {T,S<:AbstractArray{<:T}}
        require_one_based_indexing(tri)
        uplo=='U' || uplo=='L' || throw(ArgumentError("uplo must be either 'U' (upper) or 'L' (lower)"))
        new{T,S}(tri, n, uplo)
    end
end

struct BatchedSymmetric{T,S<:AbstractArray{<:T}} <: AbstractArray{T,3}
    data::S
    uplo::Char

    function BatchedSymmetric{T,S}(data, uplo) where {T,S<:AbstractArray{<:T}}
        require_one_based_indexing(data)
        new{T,S}(data, uplo)
    end
end

struct BatchedMatrix{T} <: AbstractArray{T,3}
    data::AbstractArray{T,3}

    function BatchedMatrix{T}(data) where {T}
        require_one_based_indexing(data)
        new{T}(data)
    end
end

struct BatchedVector{T} <: AbstractMatrix{T}
    data::AbstractArray{T,2}

    function BatchedVector{T}(data) where {T}
        require_one_based_indexing(data)
        new{T}(data)
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

function pack(A::AbstractArray{T}, uplo::Symbol) where {T}
    n,_,p = size(A)
    AP = Matrix{T}(undef, (n*(n+1))>>1, p)
    for k in 1:p
        l = 1
        for j in 1:n
            for i in (uplo==:L ? (j:n) : (1:j))
                AP[l,k] = A[i,j,k]
                l += 1
            end
        end
    end
    return AP
end

"""
    SymmetricPacked(A, uplo=:U)

Construct a `Symmetric` matrix in packed form of the upper (if `uplo = :U`)
or lower (if `uplo = :L`) triangle of the matrix `A`.

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
function SymmetricPacked(A::AbstractMatrix{T}, uplo::Symbol=:U) where {T}
    n = checksquare(A)
    SymmetricPacked{T,typeof(A)}(pack(A, uplo), n, char_uplo(uplo))
end

function SymmetricPacked(x::SymmetricPacked{T,S}) where{T,S}
    SymmetricPacked{T,S}(T.(x.tri), x.n, x.uplo)
end

function SymmetricPacked(V::AbstractVector{T}, uplo::Symbol=:U) where {T}
    n = (sqrt(1+8*length(V))-1)/2
    isinteger(n) || throw(DimensionMismatch("length of vector does not corresond to the number of elements in the triangle of a square matrix"))
    SymmetricPacked{T,typeof(V)}(V, round(Int, n), char_uplo(uplo))
end

"""
    BatchedSymmetricPacked(A, uplo=:U)

Construct a batch of `Symmetric` matrices in packed form of the upper (if
`uplo = :U`) or lower (if `uplo = :L`) triangle of slices in the array
`A[:,:,i]`.
```
"""
function BatchedSymmetricPacked(A::AbstractArray{T,3}, uplo::Symbol=:U) where {T}
    n = checksquare(A)
    BatchedSymmetricPacked{T,typeof(A)}(pack(A, uplo), n, char_uplo(uplo))
end

function BatchedSymmetricPacked(x::BatchedSymmetricPacked{T,S}) where{T,S}
    BatchedSymmetricPacked{T,S}(T.(x.tri), x.n, x.uplo)
end

function BatchedSymmetricPacked(M::AbstractMatrix{T}, uplo::Symbol=:U) where {T}
    n = (sqrt(1+8*size(M,1))-1)/2
    isinteger(n) || throw(DimensionMismatch("length of first dimension of matrix does not corresond to the number of elements in the triangle of a square matrix"))
    BatchedSymmetricPacked{T,typeof(M)}(M, round(Int, n), char_uplo(uplo))
end

"""
    BatchedSymmetric(A, uplo=:U)

Construct a batch of `Symmetric` matrices of the upper (if
`uplo = :U`) or lower (if `uplo = :L`) triangle of slices in the array
`A[:,:,i]`.
```
"""
function BatchedSymmetric(A::AbstractArray{T,3}, uplo::Symbol=:U) where {T}
    n = checksquare(A)
    BatchedSymmetric{T,typeof(A)}(A, char_uplo(uplo))
end

function BatchedSymmetric(x::BatchedSymmetric{T,S}) where{T,S}
    BatchedSymmetric{T,S}(T.(x.data), x.uplo)
end

"""
    BatchedMatrix(A)

Construct a batch of matrices from an array A with three dimensions.
``
"""
function BatchedMatrix(A::AbstractArray{T,3}) where {T}
    BatchedMatrix{T}(A)
end

"""
    BatchedVector(A)

Construct a batch of vectors from an array A with two dimensions.
```
"""
function BatchedVector(A::AbstractMatrix{T}) where {T}
    BatchedVector{T}(A)
end

checksquare(x::SymmetricPacked) = x.n

function checksquare(A::Array{T,3}) where T
    m,n,p = size(A)
    m == n || throw(DimensionMismatch("slices in the last dimension of array are not square: dimensions are $(size(A))"))
    m
end

convert(::Type{SymmetricPacked{T,S}}, x::SymmetricPacked) where {T,S} = SymmetricPacked{T,S}(T.(x.tri), x.n, x.uplo)

unsafe_convert(::Type{Ptr{T}}, A::SymmetricPacked{T,S}) where {T,S} = Base.unsafe_convert(Ptr{T}, A.tri)

parent(A::Union{SymmetricPacked,BatchedSymmetricPacked}) = A.tri
parent(A::Union{BatchedSymmetric,BatchedMatrix,BatchedVector}) = A.data

size(A::SymmetricPacked) = (A.n, A.n)

function size(A::SymmetricPacked, d::Integer)
    d<1 && throw(ArgumentError("dimension must be ≥ 1, got $d"))
    d<=2 ? A.n : 1
end

size(A::BatchedSymmetricPacked) = (A.n, A.n, size(A.tri,2))

function size(A::BatchedSymmetricPacked, d::Integer)
    d<1 && throw(ArgumentError("dimension must be ≥ 1, got $d"))
    d<=2 && return A.n
    d==3 && return size(A.tri,3)
    1
end

size(A::Union{BatchedSymmetric,BatchedMatrix,BatchedVector}) = size(A.data)
size(A::Union{BatchedSymmetric,BatchedMatrix,BatchedVector}, d::Integer) = size(A.data, d)

"""
    packedsize(A)

Return the number of elements in the triangle of a square matrix.
"""
packedsize(A::SymmetricPacked) = length(A.tri)
packedsize(A::Union{SymmetricPacked,BatchedSymmetricPacked}) = length(A.tri)

function packedsize(A::AbstractMatrix)
    n = checksquare(A)
    (n*(n+1))>>1
end

function packedsize(A::AbstractArray{T,3}) where T
    n = checksquare(A)
    ((n*(n+1))>>1)*size(A,3)
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

@inline function getindex(A::BatchedSymmetricPacked, i::Int, j::Int, k::Int)
    @boundscheck checkbounds(A, i, j, k)
    if A.uplo=='U'
        i,j = minmax(i,j)
        @inbounds r = A.tri[i+(j*(j-1))>>1, k]
    else
        j,i = minmax(i,j)
        @inbounds r = A.tri[i+((2*A.n-j)*(j-1))>>1, k]
    end
    return r
end

@inline function getindex(A::BatchedSymmetric, i::Int, j::Int, k::Int)
    @boundscheck checkbounds(A, i, j, k)
    if A.uplo=='U'
        i,j = minmax(i,j)
    else
        j,i = minmax(i,j)
    end
    @inbounds r = A.data[i, j, k]
    return r
end

@inline function getindex(A::BatchedMatrix, i::Int, j::Int, k::Int)
    @boundscheck checkbounds(A, i, j, k)
    return @inbounds A.data[i, j, k]
end

@inline function getindex(A::BatchedVector, i::Int, j::Int)
    @boundscheck checkbounds(A, i, j)
    return @inbounds A.data[i, j]
end

function setindex!(A::SymmetricPacked, v, i::Int, j::Int)
    i!=j && throw(ArgumentError("Cannot set a non-diagonal index in a symmetric matrix"))
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

function setindex!(A::BatchedSymmetricPacked, v, i::Int, j::Int, k::Int)
    i!=j && throw(ArgumentError("Cannot set a non-diagonal index in a symmetric matrix"))
    @boundscheck checkbounds(A, i, j, k)
    if A.uplo=='U'
        i,j = minmax(i,j)
        @inbounds A.tri[i+(j*(j-1))>>1, k] = v
    else
        j,i = minmax(i,j)
        @inbounds A.tri[i+((2*A.n-j)*(j-1))>>1, k] = v
    end
    return v
end

function setindex!(A::BatchedSymmetric, v, i::Int, j::Int, k::Int)
    i!=j && throw(ArgumentError("Cannot set a non-diagonal index in a symmetric matrix"))
    @boundscheck checkbounds(A, i, j, k)
    if A.uplo=='U'
        i,j = minmax(i,j)
    else
        j,i = minmax(i,j)
    end
    @inbounds A.data[i, j, k] = v
    return v
end

function setindex!(A::BatchedMatrix, v, i::Int, j::Int, k::Int)
    @boundscheck checkbounds(A, i, j, k)
    @inbounds A.data[i, j, k] = v
    return v
end

function setindex!(A::BatchedVector, v, i::Int, j::Int)
    @boundscheck checkbounds(A, i, j)
    @inbounds A.data[i, j] = v
    return v
end

function copy(A::SymmetricPacked{T,S}) where {T,S}
    B = copy(A.tri)
    SymmetricPacked{T,S}(B, A.n, A.uplo)
end

@inline function mul!(y::StridedVector{T},
                      AP::SymmetricPacked{T,<:StridedMatrix},
                      x::StridedVector{T},
                      α::Number, β::Number) where {T<:BlasFloat}
    alpha, beta = promote(α, β, zero(T))
    if alpha isa Union{Bool,T} && beta isa Union{Bool,T}
        BLAS.spmv!(AP.uplo, alpha, AP.tri, x, beta, y)
    else
        generic_matvecmul!(y, 'N', AP, x, MulAddMul(alpha, beta))
    end
end

VERSION<v"1.8.0-DEV.1049" && include("blas.jl")

end # module
