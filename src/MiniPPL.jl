module MiniPPL

using Reexport

@reexport using Distributions
@reexport using LinearAlgebra
@reexport using FillArrays
@reexport using Statistics
@reexport using Random
@reexport using StructArrays
@reexport using PosteriorStats

using LogDensityProblems, LogDensityProblemsAD
using TransformVariables: TransformVariables, dimension, transform, as, corr_cholesky_factor, asℝ₊, asℝ, as𝕀
using TransformedLogDensities
using DynamicHMC: mcmc_with_warmup, pool_posterior_matrices, stack_posterior_matrices, ProgressMeterReport
using DynamicHMC.Diagnostics: summarize_tree_statistics
using OhMyThreads: tcollect

export AbstractModel
export outcome_model, prior, transformation, logjoint

export sample, summarize
export as_array, as_namedtuples, as_structarray

include("interface.jl")
include("transformations.jl")
include("macro-model.jl")
include("sampling.jl")

end
