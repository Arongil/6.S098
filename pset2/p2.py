import numpy as np
import cvxpy as cp

h = 1 # time step
drag = 0.05 # friction
B_max = 0.5 # max braking
F_max = 4 # max thrust
S = 0.8 # max thrust change per time
V_to = 40 # takeoff velocity
L = 300 # length of runway

def prettify(x):
    return round(x, 2)

def prettify_list(arr):
    output = ''
    for i in arr:
        output += str(prettify(i)) + ', '
    return output[:-2]

# Idea: we will solve many convex optimization problems.
# For i = 1, 2, 3, ..., we will add a constraint that says v_i >= takeoff velocity.
# Then for whichever i the problem is first feasible, that is the min takeoff time.
def formulate_problem(takeoff_time):
    n = takeoff_time + 1
    p = cp.Variable(n)
    v = cp.Variable(n)
    f = cp.Variable(n)
    b = cp.Variable(n)
    constraints = [
        p[0] == 0,
        v[0] == 0,
        f[0] == 0,
        b[0] == 0,
        b >= 0,
        f >= 0,
        p[takeoff_time] <= L,
        v[takeoff_time] >= V_to,
    ]
    for t in range(n):
        constraints.append(f[t] <= F_max)
        constraints.append(b[t] <= B_max)
        if t != n - 1:
            constraints.append(p[t + 1] == p[t] + h * v[t])
            constraints.append(v[t + 1] == (1 - drag)*v[t] + h*(f[t] - b[t]))
            constraints.append(cp.abs(f[t + 1] - f[t]) <= S)

    objective = cp.Minimize(0)
    problem = cp.Problem(objective, constraints)
    problem.solve()
    if problem.status != 'infeasible':
        print(f'{prettify(takeoff_time)}\n{prettify(p.value[-1])}\n{prettify_list(p.value)}\n{prettify_list(v.value)}\n{prettify_list(f.value)}\n{prettify_list(b.value)}')
        return True
    return False

min_time = 1
while formulate_problem(min_time) == False:
    min_time += 1
