import numpy as np
import cvxpy as cp

np.random.seed(10)
(m, n) = (30, 10)
A = np.random.rand(m, n); A = np.asmatrix(A)
b = np.random.rand(m, 1); b = np.asmatrix(b)
c_nom = np.ones((n, 1)) + np.random.rand(n, 1); c_nom = np.asmatrix(c_nom)

F = np.vstack((-np.identity(n), np.identity(n), -np.ones(n), np.ones(n)))
g = np.vstack((-0.75 * c_nom, 1.25 * c_nom, -0.9*np.sum(c_nom), 1.1*np.sum(c_nom)))

y = cp.Variable((2*n + 2, 1))
x = cp.Variable((n, 1))
objective = cp.Minimize(g.T @ y)
constraints = [
    F.T @ y >= x,
    y >= 0,
    A @ x >= b
]
prob = cp.Problem(objective, constraints)
m = prob.solve()

print(f'Worst case cost f(x) for robust optimal: {m}')
print(f'Nominal cost for robust optimal: {c_nom.T @ x.value}')
print(f'The robust optimal x is\n{x.value}')

x = cp.Variable((n, 1))
objective = cp.Minimize(c_nom.T @ x)
constraints = [A @ x >= b]
nominal_prob = cp.Problem(objective, constraints)
m = nominal_prob.solve()

print(f'Worst case cost f(x) for nominal optimal: {m}')
print(f'Nominal cost for nominal optimal: {c_nom.T @ x.value}')
print(f'The nominal optimal x is\n{x.value}')
