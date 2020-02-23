#=
Hotelling's T-square

=#

using Distributions
using LinearAlgebra


struct Hotelling{T<:Real}
    mu_hat::Matrix
    sigma_hat::Matrix
    a_th::T
end

function Hotelling(data::Union{AbstractMatrix{T},AbstractVector{T}}, alpha::T) where T<:Real
    if ndims(data) > 2
        throw(ArgumentError("The dimension of data must be 1 or 2"))
    end

    M = size(data, 1)  # number of dimensions

    # compute threshhold
    chisq = Chisq(M)  # Chi spuared distribution with `dim` degrees of freedom
    a_th = quantile(chisq, alpha)

    # MLE estimator of mean
    mu_hat = mean(data, dims=2)

    # MLE estimator of variance-covariance matrix
    dif = data .- mu_hat
    sigma_hat = mean(dif*transpose(dif), dims=2)

    return Hotelling(mu_hat, sigma_hat, a_th)
end

function Detection(ht::Hotelling, new_data::Union{AbstractMatrix, T}) where T<:Real
    a = transpose(new_data .- ht.mu_hat) * inv(ht.sigma_hat) * (new_data .- mu_hat)
    return a > ht.a_th
end