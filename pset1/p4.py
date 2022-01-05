import cvxpy as cp
import numpy as np

def generate_np_sign_arrays(n):
    arrs = []
    for i in range(2**n):
        binary = np.binary_repr(i, n)
        arrs.append(np.zeros(n))
        for j in range(n):
            arrs[-1][j] = int(binary[j])*2 - 1
    return arrs

def option1(n, x):
    signs = generate_np_sign_arrays(n)
    constraints = []
    for sign in signs:
        constraints.append(sign @ x <= 1)
    return constraints

def option2(n, x):
    constraints = []
    y = cp.Variable(n)
    for i in range(n):
        constraints += [
            x[i] >= -y[i],
            x[i] <= y[i]
        ]
    constraints += [ cp.sum(y) <= 1 ]
    return constraints

def option3(n, x):
    constraints = [ cp.norm(x, 1) <= 1 ]
    return constraints

def clock_a_problem(prob, iters=100):
    time = 0
    for i in range(iters):
        prob.solve()
        time += prob.solver_stats.solve_time
    return time/iters

def run_the_tests():
    # There are 3 constraint options and 3 values of n, so 9 tests total

    n_values = [2, 5, 10]
    constraint_generators = {
        "exponential": option1,
        "linear": option2,
        "built-in": option3
    }

    for n in n_values:
        for (name, generate_constraints) in constraint_generators.items():
            c = np.ones(n)
            x = cp.Variable(n)
            objective = cp.Maximize(c.T@x)
            constraints = generate_constraints(n, x)
            prob = cp.Problem(objective, constraints)
            avg_time = round(clock_a_problem(prob), 6)
            print(f'n = {n} and the {name} constraints: avg time {avg_time}')

run_the_tests()
