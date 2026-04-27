export tv_transform

# Unbounded scalars
tv_transform(::Normal) = asℝ
tv_transform(::Cauchy) = asℝ
tv_transform(::TDist) = asℝ
tv_transform(::Laplace) = asℝ
tv_transform(::Logistic) = asℝ
tv_transform(::Gumbel) = asℝ

# Positive scalars
tv_transform(::Exponential) = asℝ₊
tv_transform(::Gamma) = asℝ₊
tv_transform(::InverseGamma) = asℝ₊
tv_transform(::LogNormal) = asℝ₊
tv_transform(::Weibull) = asℝ₊
tv_transform(::Chi) = asℝ₊
tv_transform(::Chisq) = asℝ₊

# Unit interval
tv_transform(::Beta) = as𝕀

# Bounded / truncated
tv_transform(d::Uniform) = as(Real, d.a, d.b)
tv_transform(d::Truncated) = as(Real, minimum(d), maximum(d))

# Unconstrained vectors
tv_transform(d::MvNormal) = as(Vector, length(d))
tv_transform(d::MvTDist) = as(Vector, length(d))

# Homogeneous product distributions (e.g. product_distribution(Fill(Exponential(1), n)))
tv_transform(d::Product{Continuous}) = as(Array, tv_transform(first(d.v)), length(d))

# Simplex
tv_transform(d::Dirichlet) = UnitSimplex(length(d))

# Correlation matrix (Cholesky factor)
tv_transform(d::LKJCholesky) = corr_cholesky_factor(d.d)
tv_transform(d::LKJ) = corr_cholesky_factor(d.d)

# Fallback
tv_transform(d) = error("no tv_transform defined for $(typeof(d))")
