
# MiniPPL.jl

`MiniPPL` is a minimal probabilistic programming language written in Julia. It is a light wrapper around the functionality provided by `Distributions.jl` and the `LogDensityProblems.jl` ecosystem (which comprises constraint transformations via `TransformVariables.jl`, AD compatibility via `LogDensityProblemsAD.jl`, and MCMC sampling via the NUTS implementation of `DynamicHMC.jl`).

Here's an exampe of how to define a model via the `@model` macro:

```julia
@model LogisticRegression(y, X) begin
  a ~ Normal(0, 1)
  b ~ product_distribution(Fill(Normal(0, 1), size(X, 2)))
  m = a .+ X * b
  y ~ product_distribution(BernoulliLogit.(m))
end
```

With this model, it is then easy to build an instance with the provided signature, e.g. using simulated data:

```julia
n, k = 100, 2
y, X = rand(Bool, n), randn(n, k)
model = LogisticRegression(y, X)
```

The macro above has generated a parametric version of the model struct alongside methods for the interface functions `prior(model)`, `outcome_model(model)`, `logjoint(model)`, and `transformation(model)`. Here's how to get the prior:

```julia
julia> prior(model)
ProductNamedTupleDistribution{(:a, :b)}(
a: Normal{Float64}(μ=0.0, σ=1.0)
b: MvNormal{Float64, PDMats.ScalMat{Float64}, Fill{Float64, 1, Tuple{Base.OneTo{Int64}}}}(
dim: 2
μ: Fill(0.0, 2)
Σ: [1.0 0.0; 0.0 1.0]
)

)
```

The result is a `Distributions.ProductNamedTupleDistribution`, from which we can easily draw samples:

```julia
julia> prior(model) |> rand
(a = 0.8071154589151536, b = [-1.4544289043621876, -1.115299813251272])
```

Similarly, the `outcome_model(model)` returns a function that can be called with a `NamedTuple` of parameter values to also return a `ProductNamedTupleDistribution`:

```julia
julia> params = rand(prior(model))
(a = -0.30495405471306214, b = [-0.1276177673872644, -0.9955625333234345])

julia> outcome_model(model)(params)
ProductNamedTupleDistribution{(:y,)}(
y: Product{Discrete, BernoulliLogit{Float64}, Vector{BernoulliLogit{Float64}}}(
v: BernoulliLogit{Float64}[BernoulliLogit{Float64}(logitp=-0.10258975036185453), ...]
)

)
```

To get prior predictive samples we just call `rand` (or `mean`) again:

```julia
julia> outcome_model(model)(params) |> rand
(y = Bool[1, 0, 1, 1, 0, 0, 0, 0, 1, 0  …  0, 1, 0, 0, 1, 0, 0, 0, 0, 0],)
```


This makes for a very convenient, julia-native Bayesian workflow.

To get samples from the posterior distribution, use `sample(model, iter, chains)`:

```julia
julia> post = sample(model, 1000, 4)
Posterior with 1000 iterations and 4 chains
  Parameters: a, b
```

`MiniPPL.jl` reexports `PosteriorStats.jl`, which comes with a convenient summarize function (along many other useful things):

```julia
julia> summarize(post)
SummaryStats
          mean    std  eti89            ess_tail  ess_bulk  rhat  mcse_mean  mcse_std
 a      0.210   0.203  -0.112 .. 0.535      2995      4189  1.00     0.0031    0.0031
 b[1]  -0.069   0.222  -0.422 .. 0.279      2838      3645  1.00     0.0037    0.0033
 b[2]   0.0008  0.204  -0.329 .. 0.326      2991      3060  1.00     0.0037    0.0028
```

If you want the samples in various useful formats, there are some helpers such as `as_array`, `as_namedtuples` or `as_structarray`:

```julia
julia> as_structarray(post)
4000-element StructArray(::Vector{Float64}, ::Vector{Vector{Float64}}) with eltype @NamedTuple{a::Float64, b::Vector{Float64}}:
 (a = -0.15115540332083433, b = [0.27640317239759377, 0.013210663164596959])
 (a = 0.34306030610208577, b = [-0.1504619093944744, 0.15451145180111744])
 (a = -0.23615479036717046, b = [0.28222119351735686, 0.04619496161886755])
 (a = -0.2899572350458107, b = [0.3365093624721501, -0.004838164631454839])
 (a = 0.35496234739736887, b = [-0.263154302798704, -0.04024488918849862])
 (a = -0.2811050994468089, b = [0.3382536766110389, -0.0247311152960821])
 ...
```

