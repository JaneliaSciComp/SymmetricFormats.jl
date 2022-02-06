SymmetricFormats.jl defines the `SymmetricPacked` type which
[concatenates](http://www.netlib.org/lapack/lug/node123.html) the columns
of either the upper or lower triangle into a vector thereby using only a
little more than half the memory.

Other formats to efficiently store symmetric matrices, which
are not (yet) implemented here, include recursive packed (RP;
Andersen et al 2001, 2002), square block packed (SBP; Gustavson
2003), block packed hybrid (BPH; Andersen et al 2005, Gustavson et
al 2007), and rectangular full packed (RFP; Gustavson et al 2009).  See
[GenericLinearAlgebra.jl](https://github.com/JuliaLinearAlgebra/GenericLinearAlgebra.jl)
for an Julian implementation of RFP.

```
julia> using SymmetricFormats

julia> A = rand(5,5)
5×5 Matrix{Float64}:
 0.364856   0.736451    0.704095  0.842738  0.207618
 0.0797498  0.63549     0.861638  0.834968  0.0660772
 0.152769   0.00271082  0.131074  0.727387  0.0995039
 0.201612   0.0429332   0.682401  0.96513   0.616451
 0.283603   0.818754    0.76548   0.284176  0.369353

julia> AP = SymmetricPacked(A)  # defaults to :U -- the upper triangle
5×5 SymmetricPacked{Float64, Matrix{Float64}}:
 0.364856  0.736451   0.704095   0.842738  0.207618
 0.736451  0.63549    0.861638   0.834968  0.0660772
 0.704095  0.861638   0.131074   0.727387  0.0995039
 0.842738  0.834968   0.727387   0.96513   0.616451
 0.207618  0.0660772  0.0995039  0.616451  0.369353

julia> AP.tri
15-element Vector{Float64}:
 0.364855947737336
 0.7364505269442262
 0.63549015394607
 0.7040945131597321
 0.8616375176332387
 0.13107440168239382
 0.8427381656704127
 0.8349681382276347
 0.7273865854008484
 0.9651301546768161
 0.20761807799000453
 0.066077207744069
 0.09950392155913346
 0.6164510864085191
 0.3693525034730706

julia> packedsize(AP)
15

julia> Base.summarysize(A)
240

julia> Base.summarysize(AP)
184
```

Many BLAS routines input this packed storage format:

```
julia> using LinearAlgebra

julia> y = rand(5)
5-element Vector{Float64}:
 0.47717272627749296
 0.11563714323328722
 0.7096777607286059
 0.2632097580818683
 0.28688074092456195

julia> x = rand(5)
5-element Vector{Float64}:
 0.14292999002604434
 0.8641487402596497
 0.914860119103189
 0.06250499506408325
 0.14202984533447915

julia> α = 0.5
0.5

julia> β = 0.5
0.5

julia> BLAS.spmv!(AP.uplo, α, AP.tri, x, β, copy(y))  # directly call BLAS
5-element Vector{Float64}:
 0.946017838465631
 0.809954221322169
 0.867204761074523
 0.9592679333345454
 0.27783932357851576

julia> mul!(copy(y), AP, x, α, β)  # or use the equivalent julian interface
5-element Vector{Float64}:
 0.946017838465631
 0.809954221322169
 0.867204761074523
 0.9592679333345454
 0.27783932357851576
```
