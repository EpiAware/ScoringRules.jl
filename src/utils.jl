# Small numerical helpers shared across scoring-rule implementations.

# Standard normal pdf and cdf, written out to avoid constructing a `Normal`
# object in hot inner loops.
@inline _norm_pdf(x::Real) = exp(-x^2 / 2) / sqrt(2 * oftype(float(x), pi))
@inline _norm_cdf(x::Real) = erfc(-x / sqrt(oftype(float(x), 2))) / 2
@inline _norm_logpdf(x::Real) = -(x^2 + log(2 * oftype(float(x), pi))) / 2

# The floating-point type to compute in. `float` needs a numeric type, and an
# array declared `Any` (as one read from a table often is) has none, and there
# its elements decide.
_float_type(A::AbstractArray{<:Number}) = float(eltype(A))
_float_type(A::AbstractArray) = float(mapreduce(typeof, promote_type, A; init = Bool))
