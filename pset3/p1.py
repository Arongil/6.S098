import numpy as np
import cvxpy as cp

import numpy as np

np.random.seed(1)
N = 100

# create an increasing input signal
xtrue = np.zeros(N)
xtrue[0:40] = 0.1
xtrue[49] = 2
xtrue[69:80] = 0.15
xtrue[79] = 1
xtrue = np.cumsum(xtrue)

# pass the increasing input through a moving-average filter
# and add Gaussian noise
h = np.array([1, -0.85, 0.7, -0.3])
k = h.size
yhat = np.convolve(h,xtrue)
y = yhat[0:-3].reshape(N, 1) + np.random.randn(N, 1)

xtrue = np.asmatrix(xtrue.reshape(N,1))
y = np.array(y).reshape(N,)

# We solve:      argmin_x     ||Ax - y||^2
#                subject to   x >= 0
#                             x nondecreasing

n = N

#     [ h(1) 0 ...
#     [ h(2) h(1) 0 ...
# A = [ h(3) h(2) h(1) 0 ...
#     [ h(4) h(3) h(2) h(1) 0 ...
#     [ 0    h(4) h(3) h(2) h(1) 0 ...
A = np.zeros((n, n))
for i in range(n):
    for j in range(max(0, i - 3), i + 1):
        A[i, j] = h[i - j]

#     [ -1  1  0  0 ...
#     [  0 -1  1  0 ...
# B = [  0  0 -1  1 ...
#     [  0  0  0 -1 ...
#     [  0  0  0  0 ...
B = np.zeros((n - 1, n))
for i in range(n - 1):
    B[i, i] = -1
    B[i, i + 1] = 1

# solve nondecreasing version of the problem
x = cp.Variable(n)
objective = cp.Minimize(cp.square(cp.norm(A @ x - y, 2)))
constraints = [
    x >= 0,
    B @ x >= 0
]
problem = cp.Problem(objective, constraints)
problem.solve()

# solve free version of the problem
x_free = cp.Variable(n)
objective = cp.Minimize(cp.square(cp.norm(A @ x_free - y, 2)))
constraints = [
    x_free >= 0
]
problem = cp.Problem(objective, constraints)
problem.solve()

# plot it!
import matplotlib.pyplot as plt

r = range(N)

plt.plot(r, xtrue, label = 'ground truth')
plt.plot(r, x.value, label = 'estimate')

plt.xlabel('t')
plt.title('Nondecreasing Maximum Likelihood Approximation')
plt.legend()

plt.savefig('p1_plot_nondecreasing.png')
plt.clf()

plt.plot(r, xtrue, label = 'ground truth')
plt.plot(r, x_free.value, label = 'estimate')

plt.xlabel('t')
plt.title('Free Maximum Likelihood Approximation')
plt.legend()
plt.savefig('p1_plot_free.png')
