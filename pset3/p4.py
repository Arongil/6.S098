import numpy as np
import cvxpy as cp

m = 20
n = 10

p_max = 10*np.ones(m)
alpha = np.ones(m)
beta = np.ones(m)

np.random.seed(3)
R = np.round(np.random.random((m, n)))

p = cp.Variable(m)
c = cp.Variable(m)
f = cp.Variable(n)
constraints = [
    R @ f <= c,
    cp.multiply(1/beta, cp.exp(c / alpha) - 1) <= p,
    p <= p_max,
    p >= 0,
    c >= 0
]
def solve_given_lambda(l):
    objective = cp.Minimize(l*cp.sum(p) - (1 - l)*cp.sum(cp.sqrt(f)))
    problem = cp.Problem(objective, constraints)
    problem.solve(warm_start = True)
    return np.sum(p.value), np.sum(np.sqrt(f.value)) # power, utility

power = []
utility = []
steps = 100
for i in range(steps + 1):
    a, b = solve_given_lambda(i/steps)
    power.append(a)
    utility.append(b)

# plot it!
import matplotlib.pyplot as plt

plt.plot(power, utility, label = 'power-utility curve')

plt.xlabel('power')
plt.ylabel('utility')
plt.title('Optimal Power vs. Utility for Varying Preferences')
plt.legend()

plt.savefig('p4_plot.png')
