import numpy as np
import cvxpy as cp

n = 5 # participants
m = 5 # outcomes

p = np.array([0.5, 0.6, 0.6, 0.6, 0.2])
q = np.array([10, 5, 5, 20, 10])
S = np.array([
    [1, 1, 0, 0, 0], # {1, 2}
    [0, 0, 0, 1, 0], # {4}
    [1, 0, 0, 1, 1], # {1, 4, 5}
    [0, 1, 0, 0, 1], # {2, 5}
    [0, 0, 1, 0, 0]  # {3}
]).T

# c  = [p1 p2 ... pn -1]
c = np.zeros(n + 1)
for i in range(n):
    c[i] = p[i]
c[-1] = -1

# x = [x1 x2 ... xn  t]
x = cp.Variable(n + 1)

#     [ I_n   0 ]
# A = [         ]
#     [ S^T  -1 ]
A = np.zeros((m + n, n + 1))
for i in range(n):
    A[i, i] = 1
for i in range(m):
    A[n + i, -1] = -1
for i in range(m):
    for j in range (n):
        A[n + i, j] = S[i, j]

b = np.zeros(n + m)
for i in range(n):
    b[i] = q[i]

print(f'c = {c}')
print(f'b = {b}')
print()
print(f'A = {A}')

objective = cp.Maximize(c.T @ x)
constraints = [A @ x <= b, x >= 0]
prob = cp.Problem(objective, constraints)
prob.solve()
print('\nSolving maximization problem...\n')
print(f'x = {x.value}')
print(f'Resultant maximum worst case profit is {c.T @ x.value}')

print()
print(f'Ax = {A @ x.value}')

# Dual problem
y = cp.Variable(n + m)
objective = cp.Minimize(b.T @ y)
constraints = [A.T @ y >= c, y >= 0]
dual = cp.Problem(objective, constraints)
dual.solve()

readable_y = []
for i in y.value:
    if i < 1e-8:
        readable_y.append(0)
    else:
        readable_y.append(round(i, 2))
print('\nSolving dual minimization problem...\n')
print(f'y = {readable_y}')
print()
print(f'A^T = {A.T}')
print()
print(f'A^T y = {A.T @ y.value}')

extractor = np.zeros(n + m)
for i in range(n, n + m):
    extractor[i] = 1
print()
print(f'sum of final m entries of y is {extractor.T @ y.value}')
pi = np.zeros(m)
for i in range(m):
    pi[i] = y.value[n + i]
print(f'pi = {pi}')
x_without_t = x.value[:-1]
expected_profit = x_without_t.T @ (p - S.T @ pi)
print(f'Expected profit = {expected_profit}')
print()
print(f'Fair prices are {S.T @ pi}')

x_star = cp.Variable(m)
objective = cp.Maximize(x_star.T @ (p - S.T @ pi))
constraints = [ x_star >= 0, x_star <= q]
expected_value_prob = cp.Problem(objective, constraints)
expected_value_prob.solve()
print()
print(f'Given pi, the optimal expected value is {x_star.value.T @ (p - S.T @ pi)} with:\nx = {x_star.value}')
