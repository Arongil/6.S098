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
def branch_and_bound(c, A, b, target = 1e-8):
    # Example element in the queue:
    #   [ (3, 4), (5, -2) ]
    # It means x_3 <= 4, x_5 >= 2 are additional constraints for this child.
    q = queue.LifoQueue()
    l = -1e8 # lower bound for integer solutions
    int_best = np.zeros(c.size) # current best integer solution

    q.put([]) # initialize with the fully relaxed LP

    # assume we can do better than half the fully relaxed optimum
    x, value = solve_relaxed_LP(c, A, b, [])
    if value is None:
        print('Infeasible!')
        return

    lp_count = 0
    while not q.empty():
        lp_count += 1
        node = q.get()
        x, value = solve_relaxed_LP(c, A, b, node)
        if x is None or value <= l: # infeasible or relaxed optimum worse than l
            continue
        # if any non-integer entries, create two child nodes for the first
        children = get_additional_constraints(x)
        if len(children) == 0: # all entries are integer!
            l = value
            int_best = x
            print(f'New best {round(l, 2)} found! {q.qsize()} children remain. {lp_count} LPs solved so far.')
            if l >= target: # we have found a satisfactory solution
                print("Yay! Hit target.")
                break
        else:
            # combine old constraints with the two new ones
            q.put(node + [children[0]])
            q.put(node + [children[1]])

    return np.rint(int_best)

def is_int(x, tolerance=1e-8):
    return abs(x - int(x)) < tolerance

def most_infeasible(x):
    # find the fractional entry in x closest to 0.5
    index = np.argmin(np.abs(x - np.floor(x) - 0.5))
    if is_int(x[index]): # all integer
        return None
    return index

def get_additional_constraints(x):
    # Apply "most infeasible branching" heuristic.
    # We choose the variable with fractional part closest to 0.5
    index = most_infeasible(x)
    value = x[index]
    if index is None:
        return []
    # The constraints stand for:
    #   1) x_index <= floor(value)
    #   2) x_index >= ceil(value)
    return [(index, np.floor(value)), (index, -np.ceil(value))]

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
# Toy example for testing
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
