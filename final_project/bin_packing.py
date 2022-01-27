import numpy as np
import branch_and_bound
import time

from ortools.linear_solver import pywraplp

# starter code from https://developers.google.com/optimization/bin/bin_packing
np.random.seed(1)
def create_data_model(n = 100):
    """Create the data for the example."""
    data = {}
    bin_capacity = 100
    weights = np.floor(np.maximum(5*np.ones(n), np.random.normal(30, 20, size = n)))
    data['weights'] = weights
    data['items'] = list(range(len(weights)))
    data['bins'] = data['items']
    data['bin_capacity'] = bin_capacity
    return data


def benchmark_standard(n = 100, data = None):
    if data is None:
        data = create_data_model(n)

    # Create the mip solver with the SCIP backend.
    solver = pywraplp.Solver.CreateSolver('SCIP')

    # Variables
    # x[i, j] = 1 if item i is packed in bin j.
    x = {}
    for i in data['items']:
        for j in data['bins']:
            x[(i, j)] = solver.IntVar(0, 1, 'x_%i_%i' % (i, j))

    # y[j] = 1 if bin j is used.
    y = {}
    for j in data['bins']:
        y[j] = solver.IntVar(0, 1, 'y[%i]' % j)

    # Constraints
    # Each item must be in exactly one bin.
    for i in data['items']:
        solver.Add(sum(x[i, j] for j in data['bins']) == 1)

    # The amount packed in each bin cannot exceed its capacity.
    for j in data['bins']:
        solver.Add(
            sum(x[(i, j)] * data['weights'][i] for i in data['items']) <= y[j] *
            data['bin_capacity'])

    # Objective: minimize the number of bins used.
    solver.Minimize(solver.Sum([y[j] for j in data['bins']]))

    status = solver.Solve()

    if status == pywraplp.Solver.OPTIMAL:
        num_bins = 0.
        for j in data['bins']:
            if y[j].solution_value() == 1:
                bin_items = []
                bin_weight = 0
                for i in data['items']:
                    if x[i, j].solution_value() > 0:
                        bin_items.append(i)
                        bin_weight += data['weights'][i]
                if bin_weight > 0:
                    num_bins += 1
        print('Number of bins used:', num_bins)
        print('Time = ', solver.WallTime(), ' milliseconds')
        return solver.WallTime()
    else:
        print('The problem does not have an optimal solution.')

def benchmark_homemade(n = 10, data = None, target = True):
    if data is None:
        data = create_data_model(n)
    else:
        n = len(data['weights'])

    # For n items and therefore n possible bins, we need n^2 + n variables.
    # The first n^2 represent whether item i is in bin j.
    # In base n, the ones digit is j and the n's digit is i.
    # The final n represent whether bin j is used.
    # We have 2*n constraints.
    # The first n are to ensure each item is in exactly one bin.
    # The second n are to ensure no bin is packed beyond its capacity.
    # The objective is to minimize the number of bins.
    # Recall the problem is
    #                            max c^T x
    #                            st  Ax <= b
    #                                x >= 0
    #                                x integer

    c = np.hstack((np.zeros(n*n), -np.ones(n))) # final n variables are the bins
    b = np.hstack((-np.ones(n), np.ones(n), np.zeros(n)))

    A = np.zeros((3*n, n*n + n))
    # Each item must be in exactly one bin. Two inequality constraints = equality
    for item in range(n):
        for bin_num in range(n):
            A[item][n*item + bin_num] = -1
            A[n + item][n*item + bin_num] = 1

    # Each bin must not exceed capacity.
    for bin_num in range(n):
        for item in range(n):
            A[2*n + bin_num][item*n + bin_num] = data['weights'][item]
            A[2*n + bin_num][n*n + bin_num] = -data['bin_capacity']

    # Avoid unnecessary search once we find a best possible packing.
    best_possible = -np.ceil((sum(weight for weight in data['weights']))/100)
    best_possible -= 1e-4 # account for floating point errors

    if target == False:
        best_possible = n

    start = time.time()
    best = branch_and_bound.branch_and_bound(c, A, b, target = best_possible)
    end = time.time()

    print(f'\nNum bins is {-c.T @ best}.')
    print(f'Took {round((end-start)*1000)} milliseconds.')
    return round((end-start)*1000)

# data = create_data_model(n = 15)
# benchmark_standard(data = data)
# print('\n-~-~-~-\n')
# benchmark_homemade(data = data)

# plot it!
import matplotlib.pyplot as plt

def graph_results():
    standard = []
    homemade = []
    for n in range(1, 7):
        standard.append(benchmark_standard(n))
        homemade.append(benchmark_homemade(n, target = False))

    r = range(1, 7)

    plt.plot(r, standard, label = 'SCIP')
    plt.plot(r, homemade, label = 'ours')

    plt.xlabel('# objects')
    plt.ylabel('time')
    plt.title('Benchmark SCIP vs. Ours')
    plt.legend()

    plt.savefig('benchmark_no_target.png')
    plt.clf()

    standard = []
    homemade = []
    for n in range(1, 15):
        standard.append(benchmark_standard(n))
        homemade.append(benchmark_homemade(n, target = True))

    r = range(1, 15)

    plt.plot(r, standard, label = 'SCIP')
    plt.plot(r, homemade, label = 'ours')

    plt.xlabel('# objects')
    plt.ylabel('time')
    plt.title('Benchmark SCIP vs. Ours (with target)')
    plt.legend()

    plt.savefig('benchmark_yes_target.png')

graph_results()
