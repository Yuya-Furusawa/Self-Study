# 2019/04/17
# Exercise in IO class
# R code

# Estimate the following equation
# y_i = \beta_0 + \beta_1^{x1_i} + \beta_2 * x2 + \epsilon

data <- read.csv("DataProblem201904.csv", header=FALSE)
f <- function (data){
  y <- data[,1]
  x1 <- data[,2]
  x2 <- data[,3]
  n <- length(y)
  
  return(function(theta){
    total = 0
    for (i in 1:n){
      total <- total + (y[i] - theta[1] - theta[2]^x1[i] - theta[3]*x2[i] )^2
    }
    return(total)
  })
}

optim(par=c(1,1,1), fn=f(data))


# Results

# $par
# [1] 0.9999121 2.0002117 3.0000444

# $value
# [1] 4.423116e-06

# $counts
# function gradient 
# 122       NA 

# $convergence
# [1] 0

# $message
# NULL