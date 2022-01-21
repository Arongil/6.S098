'''
Branch and Bound is an algorithm to solve an integer programming problem.

Standard form:     maximize     c^T x
                   subject to   Ax <= b
                                x >= 0
                                x_i is integer

One key idea is that we can relax the integer constraint to obtain an associated LP.
The optimal solution to an associated LP is an upper bound to the integer problem.

The steps are:
    1) Solve the associated relaxed LP to find an upper bound z.
    2) If z is integer, we are done. Otherwise, select the first fractional component z_i of z. Create two new children in the tree. One has an extra constraint x_i <= floor(z_i), the other x_i >= ceil(z_i).
    3) Continue expanding down the tree, pruning infeasible children, until we find an all-integer solution. This solution gives us a lower bound l.
    4) Now expand all other branches, pruning infeasible children or those whose associated LP has an optimal solution worse than l. Update l as we find better all-integer solutions.
    5) Return the best integer solution.
Remark: rather than a binary tree, we could use a queue.
'''

import numpy as np
import cvxpy as cp

import queue
import time

# Input:  n by 1 vector c, k by n matrix A, k by 1 vector b
# Output: n by 1 vector x with integer components which maximizes c^T x
def branch_and_bound(c, A, b):
    # Example element in the queue:
    #   [ (3, 4), (5, -2) ]
    # It means x_3 <= 4, x_5 >= 2 are additional constraints for this child.
    q = queue.LifoQueue()
    l = -1 # lower bound for integer solutions
    int_best = np.zeros(c.size) # current best integer solution

    q.put([]) # initialize with the fully relaxed LP

    while not q.empty():
        node = q.get()
        x, value = solve_relaxed_LP(c, A, b, node)
        if x is None or value <= l: # infeasible or relaxed optimum worse than l
            continue
        # if any non-integer entries, create two child nodes for the first
        children = get_additional_constraints(x)
        if len(children) == 0: # all entries are integer!
            l = value
            int_best = x
        else:
            # combine old constraints with the two new ones
            q.put(node + [children[0]])
            q.put(node + [children[1]])

    return np.rint(int_best)

def is_int(x, tolerance=1e-8):
    return abs(x - int(x)) < tolerance

def get_additional_constraints(x):
    non_integer = []
    for index, value in enumerate(x):
        if not is_int(value):
            non_integer.append((index, np.floor(value))) # x_index <= floor(value)
            non_integer.append((index, -np.ceil(value))) # x_index >= ceil(value)
            break
    return non_integer

def solve_relaxed_LP(c, A, b, additional_constraints):
    x = cp.Variable(c.size)
    objective = cp.Maximize(c.T @ x)
    constraints = [
        A @ x <= b,
        x >= 0
    ]
    for index, bound in additional_constraints:
        if bound >= 0:
            constraints.append(x[index] <= bound)
        else:
            constraints.append(x[index] >= -bound)
    problem = cp.Problem(objective, constraints)
    l = problem.solve()
    if problem.status == 'infeasible':
        return None, None
    return x.value, l

'''
c = np.array([5, 8])
A = np.array([
    [1, 1],
    [5, 9]
])
b = np.array([6, 45])

# solve:     max c^T x
#            st  Ax <= b
#                x >= 0
#                x integer
best_int_solution = branch_and_bound(c, A, b)
print(best_int_solution)
print(c.T @ best_int_solution)
'''

# knapsack problem (constraint to take only one of each item max)
# items: apple, banana, Raisin Bran, Fiber One, chicken, beef, milk, wine
budget = 40
value = np.array([3,   9, 5, 6, 8, 2, 10,  1]) # how we value the items
costs = np.array([1, 0.5, 5, 5, 9, 12, 5, 30])
A = np.vstack((costs, np.identity(8)))
b = np.hstack((budget, 5*np.ones(8)))

start = time.time()
best_int_solution = branch_and_bound(value, A, b)
end = time.time()

print(f'The best solution is to buy {best_int_solution}.\nThe cost is ${costs.T @ best_int_solution}.\nRuntime is {round(end - start, 3)} seconds.')
