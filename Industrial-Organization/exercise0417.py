"""
2019/04/17
Exercise in IO class
Python code

Estimate the following equation
y_i = \beta_0 + \beta_1^{x1_i} + \beta_2 * x2 + \epsilon
"""

import pandas as pd
import numpy as np
from scipy.optimize import fmin


def nlog(theta):
	data = pd.read_csv('DataProblem201904.csv', sep=',', header=None) #data
	y = np.asarray(data[0])
	x1 = np.asarray(data[1])
	x2 = np.asarray(data[2])
	n = len(y)

	total = 0
	for i in range(n):
		total += (y[i] - theta[0] - theta[1]**x1[i] - theta[2]*x2[i])**2
	return total

theta = np.ones(3) #initial values
fmin(nlog, theta)

"""
Result:

Optimization terminated successfully.
        Current function value: 0.000001
        Iterations: 80
        Function evaluations: 147
[0.9999538  2.00003254 3.00002048]
"""