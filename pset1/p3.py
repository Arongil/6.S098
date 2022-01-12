import numpy as np
import cvxpy as cp

'''
Formulate:
    a) 1/x + 1/y <= 1, x >= 0, y >= 0
    b) xy >= 1, x >= 0, y >= 0
    c) (x + y)^2 / sqrt(y) <= x - y + 5, y >= 0
    d) x + z <= 1 + sqrt(xy - z^2), x >= 0, y >= 0
'''

# a

c = cp.Variable(2)
constraints = [
    c >= 0,
    cp.harmonic_mean(c) >= 2
]
objective = cp.Minimize(0)
prob = cp.Problem(objective, constraints)
print(f'a) Is DCP? {prob.is_dcp()}')

# b -- reformulate as 1/xy >= 1

c = cp.Variable(2)
constraints = [
    c >= 0,
    cp.inv_prod(c) <= 1
]
objective = cp.Minimize(0)
prob = cp.Problem(objective, constraints)
print(f'b) Is DCP? {prob.is_dcp()}')

# c

x = cp.Variable()
y = cp.Variable()
constraints = [
    y >= 0,
    cp.quad_over_lin(x + y, cp.sqrt(y)) <= x - y + 5
]
objective = cp.Minimize(0)
prob = cp.Problem(objective, constraints)
print(f'c) Is DCP? {prob.is_dcp()}')

# d

x = cp.Variable()
y = cp.Variable()
z = cp.Variable()
constraints = [
    x >= 0,
    y >= 0,
    x + z <= 1 + cp.geo_mean(cp.hstack([
        cp.geo_mean(cp.hstack([x, y])) + z,
        cp.geo_mean(cp.hstack([x, y])) - z
    ]))
]
objective = cp.Minimize(0)
prob = cp.Problem(objective, constraints)
print(f'd) Is DCP? {prob.is_dcp()}')
