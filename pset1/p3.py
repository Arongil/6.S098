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


# d

x = cp.Variable()
y = cp.Variable()
z = cp.Variable()
u = cp.Variable()
constraints = [
    x >= 0,
    y >= 0,
    cp.square(y) <= cp.quad_over_lin(x, u),
    x + z <= 1 + cp.sqrt(u - cp.square(z))
]
objective = cp.Minimize(0)
prob = cp.Problem(objective, constraints)
prob.solve()
