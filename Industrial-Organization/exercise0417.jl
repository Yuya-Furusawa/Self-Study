#=
2019/04/17
Exercise in IO class
Julia code

Estimate the following equation
y_i = \beta_0 + \beta_1^{x1_i} + \beta_2 * x2 + \epsilon
=#

using CSV
using Optim


data = CSV.read("DataProblem201904.csv", header=false, delim=',')

function nlog(theta::Vector{S}) where S<:Real
    y = data[1]
    x1 = data[2]
    x2 = data[3]
    n = length(y)
    
    total = 0
    for i in 1:n
        total += (y[i] - theta[1]-  theta[2]^x1[i] - theta[3]*x2[i])^2
    end
    return total
end

theta = ones(3)
result = optimize(nlog, theta)
Optim.minimizer(result)

#=
Result

Results of Optimization Algorithm
 * Algorithm: Nelder-Mead
 * Starting Point: [1.0,1.0,1.0]
 * Minimizer: [1.0000041777679232,1.9999919426477555, ...]
 * Minimum: 1.815850e-08
 * Iterations: 88
 * Convergence: true
   *  √(Σ(yᵢ-ȳ)²)/n < 1.0e-08: true
   * Reached Maximum Number of Iterations: false
 * Objective Calls: 167

3-element Array{Float64,1}:
1.0000041777679232
1.9999919426477555
2.9999926446795557
=#