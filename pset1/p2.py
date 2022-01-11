import cvxpy as cp
import numpy as np

A = np.array(np.mat(
'0  1  0  0  0  0  1  0  0  0  1  1  1  0  0  1  0  1  1  0  1  1  0  1  0  1  1  1  0  1;\
 1  0  0  0  0  0  1  1  0  0  1  0  1  0  0  1  1  0  0  0  0  0  0  0  1  0  0  1  1  0;\
 0  0  0  1  1  1  0  0  0  0  1  1  0  1  0  1  0  0  0  1  1  0  1  0  1  1  0  1  1  0;\
 0  0  1  0  0  1  0  0  1  0  0  0  1  1  0  0  0  1  1  1  1  0  0  1  0  0  0  1  0  1;\
 0  0  1  0  0  1  0  0  1  1  0  0  0  0  0  1  0  0  1  0  0  0  0  1  1  0  1  0  0  1;\
 0  0  1  1  1  0  0  1  0  1  1  1  1  0  1  0  1  1  1  1  1  1  1  1  1  1  0  1  1  1;\
 1  1  0  0  0  0  0  1  1  0  0  1  0  0  0  1  0  0  1  0  0  1  0  0  0  0  1  0  1  1;\
 0  1  0  0  0  1  1  0  0  1  1  0  0  1  0  1  0  0  1  0  0  1  1  1  0  1  0  1  0  0;\
 0  0  0  1  1  0  1  0  0  1  1  0  0  1  1  1  1  1  1  0  1  0  0  1  0  0  0  1  0  1;\
 0  0  0  0  1  1  0  1  1  0  0  0  1  1  0  1  0  0  0  0  0  1  0  1  0  0  0  0  0  0;\
 1  1  1  0  0  1  0  1  1  0  0  1  0  0  0  0  1  0  0  0  1  1  0  1  0  1  1  1  0  1;\
 1  0  1  0  0  1  1  0  0  0  1  0  0  0  0  0  1  1  0  0  0  0  0  0  1  0  0  1  1  0;\
 1  1  0  1  0  1  0  0  0  1  0  0  0  1  1  1  0  0  0  0  1  0  1  0  1  1  0  1  1  0;\
 0  0  1  1  0  0  0  1  1  1  0  0  1  0  0  1  0  0  1  0  1  0  0  1  0  0  0  1  0  1;\
 0  0  0  0  0  1  0  0  1  0  0  0  1  0  0  1  0  0  1  1  0  0  0  1  1  0  1  0  0  1;\
 1  1  1  0  1  0  1  1  1  1  0  0  1  1  1  0  0  1  0  1  1  1  1  1  1  1  0  1  1  1;\
 0  1  0  0  0  1  0  0  1  0  1  1  0  0  0  0  0  1  1  0  0  1  0  0  0  0  1  0  1  1;\
 1  0  0  1  0  1  0  0  1  0  0  1  0  0  0  1  1  0  0  1  0  1  1  1  0  1  0  1  0  0;\
 1  0  0  1  1  1  1  1  1  0  0  0  0  1  1  0  1  0  0  1  1  0  0  1  0  0  0  1  0  1;\
 0  0  1  1  0  1  0  0  0  0  0  0  0  0  1  1  0  1  1  0  0  1  0  1  0  0  0  0  0  0;\
 1  0  1  1  0  1  0  0  1  0  1  0  1  1  0  1  0  0  1  0  0  1  0  0  0  0  1  0  0  0;\
 1  0  0  0  0  1  1  1  0  1  1  0  0  0  0  1  1  1  0  1  1  0  0  0  0  0  1  1  0  0;\
 0  0  1  0  0  1  0  1  0  0  0  0  1  0  0  1  0  1  0  0  0  0  0  1  1  1  0  0  0  0;\
 1  0  0  1  1  1  0  1  1  1  1  0  0  1  1  1  0  1  1  1  0  0  1  0  0  1  0  0  1  0;\
 0  1  1  0  1  1  0  0  0  0  0  1  1  0  1  1  0  0  0  0  0  0  1  0  0  1  0  0  1  1;\
 1  0  1  0  0  1  0  1  0  0  1  0  1  0  0  1  0  1  0  0  0  0  1  1  1  0  0  1  0  1;\
 1  0  0  0  1  0  1  0  0  0  1  0  0  0  1  0  1  0  0  0  1  1  0  0  0  0  0  1  1  0;\
 1  1  1  1  0  1  0  1  1  0  1  1  1  1  0  1  0  1  1  0  0  1  0  0  0  1  1  0  0  1;\
 0  1  1  0  0  1  1  0  0  0  0  1  1  0  0  1  1  0  0  0  0  0  0  1  1  0  1  0  0  1;\
 1  0  0  1  1  1  1  0  1  0  1  0  0  1  1  1  1  0  1  0  0  0  0  0  1  1  0  1  1  0'))

B = np.array(np.mat(
'0  1  0  1  1  1  1  1  1  1  0  1  1  0  1  1  1  0  1  0  1  1  1  1  1  1  0  1  0  1;\
 1  0  1  0  0  0  1  0  1  0  0  0  0  1  0  1  0  1  1  1  0  0  1  1  1  0  0  1  0  0;\
 0  1  0  0  1  0  1  0  1  0  0  0  0  0  0  1  0  1  0  1  1  0  0  0  0  0  0  1  0  1;\
 1  0  0  0  1  0  0  0  1  0  1  1  0  0  0  1  1  1  1  1  1  1  0  1  0  0  0  1  1  0;\
 1  0  1  1  0  0  0  1  0  0  1  0  0  1  0  0  1  0  1  0  0  1  0  0  0  1  0  0  0  0;\
 1  0  0  0  0  0  1  1  1  1  0  1  1  1  1  0  0  0  1  0  1  0  0  0  0  0  0  1  1  1;\
 1  1  1  0  0  1  0  0  1  0  1  0  1  0  1  1  1  1  0  0  0  1  0  1  0  0  1  1  0  0;\
 1  0  0  0  1  1  0  0  0  1  0  0  1  1  1  0  1  0  0  0  1  0  0  0  0  1  1  0  0  1;\
 1  1  1  1  0  1  1  0  0  0  1  1  0  1  0  0  0  1  0  0  0  0  0  0  0  1  0  0  1  0;\
 1  0  0  0  0  1  0  1  0  0  0  0  1  1  0  0  0  0  0  0  0  1  1  1  0  1  0  0  0  0;\
 0  0  0  1  1  0  1  0  1  0  0  1  0  1  0  1  1  0  0  1  1  1  1  0  1  0  0  1  1  0;\
 1  0  0  1  0  1  0  0  1  0  1  0  0  0  0  0  1  0  1  0  0  1  1  0  0  1  0  1  1  0;\
 1  0  0  0  0  1  1  1  0  1  0  0  0  1  0  0  1  1  0  0  0  1  1  1  0  1  0  1  0  0;\
 0  1  0  0  1  1  0  1  1  1  1  0  1  0  0  0  1  1  1  1  1  1  1  1  1  1  1  1  1  1;\
 1  0  0  0  0  1  1  1  0  0  0  0  0  0  0  1  0  1  0  1  0  0  1  0  0  0  0  1  0  1;\
 1  1  1  1  0  0  1  0  0  0  1  0  0  0  1  0  1  0  0  0  0  0  1  0  0  0  1  0  0  1;\
 1  0  0  1  1  0  1  1  0  0  1  1  1  1  0  1  0  1  0  1  1  0  0  0  0  0  0  1  1  1;\
 0  1  1  1  0  0  1  0  1  0  0  0  1  1  1  0  1  0  0  1  0  1  1  0  0  1  1  1  0  0;\
 1  1  0  1  1  1  0  0  0  0  0  1  0  1  0  0  0  0  0  0  0  1  1  0  0  0  0  0  0  0;\
 0  1  1  1  0  0  0  0  0  0  1  0  0  1  1  0  1  1  0  0  0  0  0  1  0  0  1  0  0  1;\
 1  0  1  1  0  1  0  1  0  0  1  0  0  1  0  0  1  0  0  0  0  1  0  0  1  0  0  0  0  0;\
 1  0  0  1  1  0  1  0  0  1  1  1  1  1  0  0  0  1  1  0  1  0  1  1  1  0  0  0  1  1;\
 1  1  0  0  0  0  0  0  0  1  1  1  1  1  1  1  0  1  1  0  0  1  0  0  0  0  0  1  0  0;\
 1  1  0  1  0  0  1  0  0  1  0  0  1  1  0  0  0  0  0  1  0  1  0  0  1  0  1  1  1  0;\
 1  1  0  0  0  0  0  0  0  0  1  0  0  1  0  0  0  0  0  0  1  1  0  1  0  1  0  0  1  0;\
 1  0  0  0  1  0  0  1  1  1  0  1  1  1  0  0  0  1  0  0  0  0  0  0  1  0  1  1  1  1;\
 0  0  0  0  0  0  1  1  0  0  0  0  0  1  0  1  0  1  0  1  0  0  0  1  0  1  0  1  0  1;\
 1  1  1  1  0  1  1  0  0  0  1  1  1  1  1  0  1  1  0  0  0  0  1  1  0  1  1  0  1  0;\
 0  0  0  1  0  1  0  0  1  0  1  1  0  1  0  0  1  0  0  0  0  1  0  1  1  1  0  1  0  0;\
 1  0  1  0  0  1  0  1  0  0  0  0  0  1  1  1  1  0  0  1  0  1  0  0  0  1  1  0  0  0'))

'''
Set of linear equalities and inequalities necessary and sufficient for P to be a permutation matrix such that PAP^T = B, where we may assume P_ij = 0 or 1 always:
    1) P @ 1 == 1, i.e. [1, ..., 1] is an eigenvector of P with eigenvalue 1.
    2) P.T @ 1 == 1.
Condition (1) ensures we have exactly one 1 per row. Condition (2) ensures we have exactly one 1 per column. Together, P satisfies (1) and (2) iff P is a permutation matrix. Finally, we require so that P is the permutation matrix we want:
    3) P @ A @ P.T == B

And it is possible that we need to treat P like a vector in \R^{n^2}. In this case, we would replace conditions (1) and (2) with n constraints each, which encode the column-by-column equalities we are looking for.
    - Update: we can use cp.Variable((n, n)) to create a matrix variable! Just what we need.

Lastly, I propose switching the equalities in (1) and (2) to inequalities. This will help us when we relax the boolean constraint that P_ij = 0 or 1. We need an additional constraint each, namely P @ 1 >= 0 and P.T @ 1 >= 0. Eeeergh, we need to rethink how to reformulate these constraints as inequalities ... this doesn't quite work ...
'''

num_rows, num_cols = A.shape
n = num_rows # should be the same because A and B are square

P = cp.Variable((n, n))
ones = np.ones(n)
constraints = [
    P >= 0,
    P @ ones == 1,
    P.T @ ones == 1,
    P @ A == B @ P
]

# Add a random linear objective on the elements of P to coax values near 0 or 1
W = np.random.normal(size=(n, n))
objective = cp.Minimize(cp.pnorm(cp.multiply(W, P), 1))

prob = cp.Problem(objective, constraints)
prob.solve()


m = 0 # used to find which entry is furthest from integer
rounded = P.value
for i in range(n):
    for j in range(n):
        x = rounded[i][j]
        rounded[i][j] = round(x, 2)
        if x*(1-x) > m:
            m = x*(1-x)

print(rounded)
print(f'\nProof of closeness to integers: max of p_ij (1 - p_ij) is {m}')

v = np.zeros(n)
for i in range(0, n):
    v[i] = i + 1

print('\nHere is the permutation vector:')
print(P.value @ v)

