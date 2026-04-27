using MiniPPL

@model LinearRegression(y, X) begin
    a ~ Normal(0, 1)
    b ~ MvNormal(Zeros(size(X, 2)), I)
    s ~ Exponential(1)
    m = a .+ X * b
    y ~ MvNormal(m, I * s)
end

begin
    # Simulate fake data
    n, k = 100, 3
    y, X = randn(n), randn(n, k)

    tmp = LinearRegression(y, X)
    pri = prior(tmp) |> rand
    out = pri |> outcome_model(tmp) |> rand

    # Define and fit model
    model = LinearRegression(out.y, X)
    post = sample(model, 1000, 4)
end

@model MultilevelLogisticRegression(y, X, g) begin
    a ~ Normal(0, 1)
    b ~ MvNormal(Zeros(size(X, 2)), I)
    s ~ Exponential(1)
    d ~ product_distribution(Fill(Normal(0, 1), maximum(g)))
    m = a .+ X * b .+ d[g]
    y ~ product_distribution(BernoulliLogit.(m))
end

let
    n, k, l = 1000, 2, 8
    y, X, g = rand(Bool, n), randn(n, k), rand(1:l, n)

    tmp = MultilevelLogisticRegression(y, X, g)
    pri = prior(tmp) |> rand
    out = pri |> outcome_model(tmp) |> rand

    mod = MultilevelLogisticRegression(out.y, X, g)

    post = sample(mod, 1000) |> StructArray
end
