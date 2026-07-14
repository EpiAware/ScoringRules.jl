# Small numerical helpers shared across scoring-rule implementations.

# Standard normal pdf and cdf, written out to avoid constructing a `Normal`
# object in hot inner loops.
@inline _norm_pdf(x::Real) = exp(-x^2 / 2) / sqrt(2 * oftype(float(x), pi))
@inline _norm_cdf(x::Real) = erfc(-x / sqrt(oftype(float(x), 2))) / 2
@inline _norm_logpdf(x::Real) = -(x^2 + log(2 * oftype(float(x), pi))) / 2

"""
Check that all supplied vectors either share the same length or are scalars,
returning the common broadcast length. Errors otherwise.
"""
function _check_common_length(vectors...)
    n = 1
    for v in vectors
        len = length(v)
        if len != 1
            if n == 1
                n = len
            elseif len != n
                throw(DimensionMismatch(
                    "forecast parameter vectors must share a common length; " *
                    "got lengths $(map(length, vectors))"))
            end
        end
    end
    return n
end
