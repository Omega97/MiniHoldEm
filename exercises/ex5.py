"""
In computational intelligence (CI), an evolutionary algorithm (EA) is a
subset of evolutionary computation, a generic population-based metaheuristic
optimization algorithm. An EA uses mechanisms inspired by biological evolution,
such as reproduction, mutation, recombination, and selection. Candidate solutions
to the optimization problem play the role of individuals in a population, and the
fitness function determines the quality of the solutions (see also loss function).
Evolution of the population then takes place after the repeated application of
the above operators.

In evolutionary computation, differential evolution (DE) is a method that
optimizes a problem by iteratively trying to improve a candidate solution with
regard to a given measure of quality. Such methods are commonly known as
metaheuristics as they make few or no assumptions about the optimized problem
and can search very large spaces of candidate solutions. However, metaheuristics
such as DE do not guarantee an optimal solution is ever found.
"""
import numpy as np
import random
import math
import matplotlib.pyplot as plt


def griewank(x):
    """Griewank function"""
    sum_sq = sum(xi**2 for xi in x)
    prod_cos = math.prod(math.cos(xi / math.sqrt(i + 1)) for i, xi in enumerate(x))
    return 1 + (1/4000) * sum_sq - prod_cos


def rastrigin(x):
    """Rastrigin function"""
    n = len(x)
    sum_sq = sum(xi**2 - (10 * math.cos(2 * math.pi * xi)) for xi in x)
    return 10 * n + sum_sq


def schaffer(x):
    """Schaffer function"""
    n = len(x)
    result = 0
    for i in range(n-1):
        term = (x[i]**2 + x[i+1]**2)
        result += (term**0.25) * (math.sin(50 * (term**0.1))**2 + 1)
    return result


def rosenbrock(x):
    """Rosenbrock function"""
    n = len(x)
    result = 0
    for i in range(n - 1):
        result += 100 * (x[i+1] - x[i]**2)**2 + (1 - x[i])**2
    return result


def plot_function(pop, func, val_range, title):
    """function to plot a given benchmark function and a collection of points on it"""
    ax = plt.figure(figsize=(20,10)).add_subplot(projection='3d')
    ax.set_title(title)
    xvals = np.linspace(val_range[0], val_range[1], 10*(val_range[1]-val_range[0]))
    yvals = np.linspace(val_range[0], val_range[1], 10*(val_range[1]-val_range[0]))
    xx, yy = np.meshgrid(xvals, yvals)
    z = np.zeros_like(xx)
    for i in range(xx.shape[0]):
        for j in range(xx.shape[1]):
            input_vector = [xx[i, j], yy[i, j]]
            z[i, j] = func(input_vector)
    ax.plot_surface(xx, yy, z, antialiased=True, alpha=0.2)
    x_pop = [p[0] for p in pop]
    y_pop = [p[1] for p in pop]
    z_pop = [func(p) for p in pop]
    ax.scatter(x_pop, y_pop, z_pop, c="red")
    return ax


# Implement now different versions of Differential Mutation (e.g., DE/rand/1, DE/best/1, etc...)

def donor_de_rand_1(pop, current, best, F=2):
    """generate a donor vector using the DE/rand/1 strategy"""
    # CODE HERE
    a, b, c = random.sample(pop, 3)
    donor = a + F * (b - c)
    return donor


def donor_de_best_1(pop, current, best, F=2):
    # CODE HERE
    ...


def donor_de_curr2best_1(pop, current, best, F=2):
    # CODE HERE
    pass


def donor_de_rand_2(pop, current, best, F=2):
    # CODE HERE
    pass


def donor_de_rand2best_1(pop, current, best, F=2):
    # CODE HERE
    pass


# Implement now a function to get the trial vector and one for the selection.

def trial_vector(x, v, p_cr = 0.5):
    # CODE HERE
    pass


def selection(x, u, fit):
    # CODE HERE
    if fit(u) < fit(x):
        return u
    else:
        return x


# Exploit the previous functions to implement a generation function.

def generation(pop, fit, dm, best):
    next_gen = []
    for i in range(0, len(pop)):
        # CODE HERE
        ...
    return next_gen


# We can now define differential evolution. The function returns the
# last population, the history of all generations and the best individual.


def differential_evolution(n_gens, pop_size, search_space, fit, dm):
    pop = []
    hist = []
    n = len(search_space)

    # initialize the population
    for _ in range(0, pop_size):
        # CODE HERE
        pop.append([random.uniform(search_space[i][0], search_space[i][1]) for i in range(n)])


    hist.append(pop)
    best = min(pop, key=fit)

    for _ in range(0, n_gens):
        # CODE HERE
        ...


    return pop, hist, best


# Try all the methods defined abow on the different benchmark functions and see how they work.

np.random.seed(0)
random.seed(0)

n_gen = 100
pop_size = 20

f = rastrigin
search_space = [[-10, 10], [-10, 10]]
plot_interval = [-10, 10]
last_rand1, hist_rand1, best_rand1 = differential_evolution(n_gen, pop_size, search_space, f, donor_de_rand_1)
last_best1, hist_best1, best_best1 = differential_evolution(n_gen, pop_size, search_space, f, donor_de_best_1)
last_curr2best1, hist_curr2best1, best_curr2best1 = differential_evolution(n_gen, pop_size, search_space, f,
                                                                           donor_de_curr2best_1)
last_rand2, hist_rand2, best_rand2 = differential_evolution(n_gen, pop_size, search_space, f, donor_de_rand_2)
last_rand2best1, hist_rand2best1, best_rand2best1 = differential_evolution(n_gen, pop_size, search_space, f,
                                                                           donor_de_rand2best_1)

print(f"DE/rand/1: {f(best_rand1)}")
print(f"DE/best/1: {f(best_best1)}")
print(f"DE/current-to-best/1: {f(best_curr2best1)}")
print(f"DE/rand/2: {f(best_rand2)}")
print(f"DE/rand-to-best/1: {f(best_rand2best1)}")
plot_function(last_rand1, f, plot_interval, "DE/rand/1")
plot_function(last_best1, f, plot_interval, "DE/best/1")
plot_function(last_curr2best1, f, plot_interval, "DE/current-to-best/1")
plot_function(last_rand2, f, plot_interval, "DE/rand/2")
plot_function(last_rand2best1, f, plot_interval, "DE/rand-to-best/1")


def plot_history(history, fit, tit):
    best_hist = []
    for pop in history:
        best_hist.append(fit(min(pop, key=fit)))
    plt.plot(best_hist)
    plt.title(tit)
    plt.ylabel('Fitness')
    plt.xlabel('Generation')
    plt.show()


plot_history(hist_rand1, f, "DE/rand/1")
plot_history(hist_best1, f, "DE/best/1")
plot_history(hist_curr2best1, f, "DE/current-to-best/1")
plot_history(hist_rand2, f, "DE/rand/2")
plot_history(hist_rand2best1, f, "DE/rand-to-best/1")
