abstract type AbstractModel end

function prior(::AbstractModel) end
function outcome_model(::AbstractModel) end
function logjoint(::AbstractModel) end

transformation(model::AbstractModel) = as(map(tv_transform, prior(model).dists))

# function predict(model::AbstractModel, samples)
#     return map(outcome_model(model), samples)
# end
