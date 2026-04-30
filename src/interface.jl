abstract type AbstractModel end

function prior(m::AbstractModel)
    error("not implemented for $(typeof(m))")
end
function outcome_model(m::AbstractModel)
    error("not implemented for $(typeof(m))")
end
function logjoint(m::AbstractModel)
    error("not implemented for $(typeof(m))")
end

(m::AbstractModel)(params::NamedTuple) = outcome_model(m)(params)

transformation(model::AbstractModel) = as(map(tv_transform, prior(model).dists))
