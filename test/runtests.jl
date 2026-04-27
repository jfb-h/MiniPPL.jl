using Test
using MiniPPL
using Distributions
using FillArrays
using LinearAlgebra

@testset "ProbabilityModels" begin
    @model LinearRegression(y, X) begin
        a ~ Normal(0, 1)
        b ~ MvNormal(Zeros(size(X, 2)), I)
        s ~ Exponential(1)
        m = a .+ X * b
        y ~ MvNormal(m, I * s)
    end

    @testset "struct generation" begin
        @test LinearRegression <: AbstractModel
        @test fieldnames(LinearRegression) == (:y, :X)

        y, X = randn(10), randn(10, 2)
        model = LinearRegression(y, X)
        @test model.y === y
        @test model.X === X
    end

    @testset "prior" begin
        y, X = randn(10), randn(10, 2)
        model = LinearRegression(y, X)
        pri = prior(model)

        @test pri isa Distributions.ProductNamedTupleDistribution
        p = rand(pri)
        @test keys(p) == (:a, :b, :s)
        @test p.a isa Float64
        @test length(p.b) == 2
        @test p.s isa Float64
    end

    @testset "outcome_model" begin
        y, X = randn(10), randn(10, 2)
        model = LinearRegression(y, X)
        p = rand(prior(model))
        lik = outcome_model(model)(p)

        @test lik isa Distributions.ProductNamedTupleDistribution
        s = rand(lik)
        @test keys(s) == (:y,)
        @test length(s.y) == 10
    end

    @testset "log-joint" begin
        y, X = randn(10), randn(10, 2)
        model = LinearRegression(y, X)
        p = rand(prior(model))

        lj = logjoint(model)
        @test lj(p) isa Float64
        @test isfinite(lj(p))
        @test lj(p) ≈ logpdf(prior(model), p) + logpdf(outcome_model(model)(p), (; y))
    end

    @testset "dynamic dimensions" begin
        for k in [1, 3, 10]
            y, X = randn(20), randn(20, k)
            model = LinearRegression(y, X)
            p = rand(prior(model))
            @test length(p.b) == k
            @test isfinite(logjoint(model)(p))
        end
    end

    @testset "multiple data fields and outcome_model variables" begin
        @model PairedModel(y1, y2) begin
            mu ~ Normal(0, 1)
            sigma ~ Exponential(1)
            y1 ~ Normal(mu, sigma)
            y2 ~ Normal(mu, sigma)
        end

        y1, y2 = randn(), randn()
        model = PairedModel(y1, y2)

        p = rand(prior(model))
        @test keys(p) == (:mu, :sigma)

        s = rand(outcome_model(model)(p))
        @test keys(s) == (:y1, :y2)
        @test s.y1 isa Float64

        @test isfinite(logjoint(model)(p))
    end

    @testset "intermediate computations in outcome_model" begin
        @model ScaledRegression(y, X, z) begin
            a ~ Normal(0, 1)
            b ~ Normal(0, 1)
            mu = a * z .+ X * fill(b, size(X, 2))
            y ~ MvNormal(mu, I)
        end

        n = 20
        y, X, z = randn(n), randn(n, 2), 2.5
        model = ScaledRegression(y, X, z)
        p = rand(prior(model))
        @test isfinite(logjoint(model)(p))
    end

    @testset "error conditions" begin
        @test_throws ErrorException @macroexpand @model NoOutcome(y) begin
            a ~ Normal(0, 1)
        end

        @test_throws ErrorException @macroexpand @model NoPrior(y) begin
            y ~ Normal(0, 1)
        end
    end

end
