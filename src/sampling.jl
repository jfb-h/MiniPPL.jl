struct Posterior{T, S}
    transformation::S
    data::T
end

function param_names(post::Posterior; expand = true)
    # This might be a hacky use of TransformVariables internals
    ks = keys(getfield(post.transformation, :inner))
    expand || (return ks)
    # This only works for 1-dimensional containers right now
    ps = mapreduce(vcat, ks) do k
        dim = dimension(getproperty(post.transformation, k))
        dim == 1 ? string(k) : map(i -> "$(string(k))[$i]", 1:dim)
    end
    return ps
end

function Base.show(io::IO, p::Posterior)
    n_iter = size(first(p.data).posterior_matrix, 2)
    n_chain = length(p.data)
    println(io, "Posterior with $n_iter iterations and $n_chain chains")
    println(io, "  Parameters: $(join(param_names(p; expand = false), ", "))")
    return
end

function sample(model, iter, chains; backend = :ForwardDiff, reporter = ProgressMeterReport())
    trans = transformation(model)
    P = TransformedLogDensity(trans, logjoint(model))
    ∇P = ADgradient(backend, P)
    res = tcollect(mcmc_with_warmup(Random.default_rng(), ∇P, iter; reporter) for _ in 1:chains)
    return Posterior(trans, res)
end

function as_array(post::Posterior; stack_or_pool = :stack)
    if stack_or_pool == :stack
        a = stack_posterior_matrices(post.data)
        a = stack(eachslice(a; dims = 2); dims = 2) do m
            nts = transform.(post.transformation, eachrow(m))
            stack(nt -> reduce(vcat, nt), nts; dims = 1)
        end
    elseif stack_or_pool == :pool
        a = pool_posterior_matrices(post.data)
        nts = transform.(post.transformation, eachcol(a))
        stack(nt -> reduce(vcat, nt), nts; dims = 2)
    else
        error("Unknown option $stack_or_pool, choose one of :stack or :pool")
    end
    return a
end

function as_namedtuples(post::Posterior)
    a = pool_posterior_matrices(post.data)
    return transform.(post.transformation, eachcol(a))
end

function as_structarray(post::Posterior)
    return StructArray(as_namedtuples(post))
end

function tree_statistics(post::Posterior)
    return foreach(c -> println(summarize_tree_statistics(c.tree_statistics)), post.data)
end

function PosteriorStats.summarize(post::Posterior)
    a = as_array(post; stack_or_pool = :stack)
    var_names = [string(p) for p in param_names(post)]
    return PosteriorStats.summarize(a; var_names)
end
