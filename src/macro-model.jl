export @model

macro model(header, body)
    @assert header isa Expr && header.head == :call "@model: expected ModelName(fields...) header"

    model_name = header.args[1]::Symbol
    data_fields = Symbol.(header.args[2:end])
    data_set = Set(data_fields)

    clean(stmts) = filter(s -> !(s isa LineNumberNode), stmts)
    is_tilde(e) = e isa Expr && e.head == :call && e.args[1] == :~

    prior_pairs = Pair{Symbol, Any}[]
    outcome_pairs = Pair{Symbol, Any}[]
    outcome_body_stmts = Expr[]

    for stmt in clean(body.args)
        if is_tilde(stmt)
            lhs, rhs = stmt.args[2]::Symbol, stmt.args[3]
            if lhs in data_set
                push!(outcome_pairs, lhs => rhs)
            else
                push!(prior_pairs, lhs => rhs)
            end
        elseif stmt isa Expr
            push!(outcome_body_stmts, stmt)
        end
    end

    isempty(prior_pairs)   && error("@model: no prior parameters defined")
    isempty(outcome_pairs) && error("@model: no outcome variables defined")

    prior_names = first.(prior_pairs)
    type_params = [Symbol(:_T_, f) for f in data_fields]

    struct_def = Expr(
        :struct, false,
        Expr(:<:, Expr(:curly, model_name, type_params...), :AbstractModel),
        Expr(:block, [Expr(:(::), f, tp) for (f, tp) in zip(data_fields, type_params)]...)
    )

    prior_nt = Expr(:tuple, Expr(:parameters, [Expr(:kw, n, d) for (n, d) in prior_pairs]...))
    outcome_nt = Expr(:tuple, Expr(:parameters, [Expr(:kw, n, d) for (n, d) in outcome_pairs]...))

    data_destruct = Expr(:(=), Expr(:tuple, Expr(:parameters, data_fields...)), :model)
    params_destruct = Expr(:(=), Expr(:tuple, Expr(:parameters, prior_names...)), :params)
    data_nt = Expr(:tuple, Expr(:parameters, data_fields...))
    obs_nt = Expr(:tuple, Expr(:parameters, [Expr(:kw, f, :(model.$f)) for f in data_fields]...))

    inner_fn_body = Expr(
        :block,
        params_destruct,
        outcome_body_stmts...,
        :(product_distribution($outcome_nt))
    )

    return esc(
        quote
            export $model_name
            $struct_def

            function MiniPPL.prior(model::$model_name)
                $data_destruct
                return product_distribution($prior_nt)
            end

            function MiniPPL.outcome_model(model::$model_name)
                $data_destruct
                return function (params)
                    $inner_fn_body
                end
            end

            function MiniPPL.logjoint(model::$model_name)
                pri = prior(model)
                lik = outcome_model(model)
                return function (params)
                    return logpdf(pri, params) + logpdf(lik(params), $obs_nt)
                end
            end

            #transformation(model::$model_name) = as(map(tv_transform, prior(model).dists))

            nothing
        end
    )
end
