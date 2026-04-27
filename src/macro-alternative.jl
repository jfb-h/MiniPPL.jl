# struct LinearRegression{S, T} <: AbstractModel
#     y::S
#     X::T
# end
#
# @prior LinearRegression begin
#     a ~ Normal(0, 1)
#     b ~ MvNormal(Zeros(size(X, 2)), I)
#     s ~ Exponential(1)
# end
#
# @outcome LinearRegression begin
#     m = a .+ X * b
#     y ~ MvNormal(m, I * s)
# end

# macro prior(model, body)
# end
