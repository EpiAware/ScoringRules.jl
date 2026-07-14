# Shared helper for the R-comparison tests: loads the committed reference CSVs
# produced by test/references/generate_references.R and exposes each column as a
# `Vector{Float64}` (parsing "Inf"/"-Inf" that R writes for unbounded support).
@testmodule References begin
    using DelimitedFiles: readdlm

    const DATA_DIR = joinpath(@__DIR__, "references", "data")

    _tof(x::Number) = Float64(x)
    function _tof(x::AbstractString)
        s = strip(strip(x, '"'))
        (s == "NA" || s == "NaN") && return NaN
        return parse(Float64, s)
    end

    "Load reference table `name` as a `Dict{String, Vector{Float64}}` plus row count."
    function load(name::AbstractString)
        raw, header = readdlm(joinpath(DATA_DIR, name * ".csv"), ',',
            Any; header = true)
        names = strip.(replace.(vec(header), '"' => ""))
        cols = Dict{String, Vector{Float64}}()
        for (j, nm) in enumerate(names)
            cols[String(nm)] = _tof.(raw[:, j])
        end
        return cols, size(raw, 1)
    end
end
