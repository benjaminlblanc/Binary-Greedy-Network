import matplotlib.pyplot as plt
from scipy.optimize import minimize, LinearConstraint
from time import time

import numpy as np
import itertools
import math

def multinomial(lst):
    res, i = 1, 1
    for a in lst:
        for j in range(1,a+1):
            res *= i
            res //= j
            i += 1
    return res


def elem(n, weights, l):
    possible_values = []
    for j in range(len(weights)):
        val = min(math.floor(l / (weights[j] + 1e-20)), n)
        if j == 0:
            possible_values.append(list(range(n + 1)))
        elif l == weights[j]:
            possible_values.append(list(range(2)))
        else:
            possible_values.append(list(range(val + 1)))
    possible = []
    for elements in itertools.product(*possible_values):
        if np.sum(np.array(elements) * np.array(weights)) <= l and np.sum(elements) == n:
            possible.append(elements)
    return possible

def const_1(p, elements, weights, delta):
    tot = 0
    for element in elements:
        curr = math.log10(multinomial(element))
        for j in range(1, len(weights) + 1):
            curr += element[-j] * math.log10(np.abs(p[-j] + 1e-20))
        curr = 10 ** curr
        tot += curr
    tot -= delta
    return tot

def const_2(p):
    return np.sum(p) - 1

def obj(p, w):
    """objective function, to be solved."""
    return -np.sum(p * w)

def opt(n, l, b, delta, vals, upper):
    precision = len(vals)
    weights = [0]
    weights_obj = [b]
    vals = np.flip(vals)
    for i in range(precision + 1 - 1):
        val = vals[i]
        weights.insert(1, val + 1e-5)
        weights_obj.insert(0, val)

    if not upper:
        weights_obj.insert(0, 0)
        precision += 1
        weights = weights_obj
    m = len(weights)
    elements = elem(n, weights, l)
    cons = [{'type': 'ineq', 'fun': const_1, 'args': (elements, weights, delta)},
            LinearConstraint(np.eye(m), np.zeros(m), np.ones(m)),
            {'type': 'eq', 'fun': const_2}]
    initial_guess = np.zeros(m) + 1 / m  # initial guess can be anything
    #initial_guess[0], initial_guess[1] = delta ** (1/n), 1 - delta ** (1/n)

    t_init = time()
    print("Computing...")
    result = minimize(obj, initial_guess, args=weights_obj, constraints=cons)
    print(result.x)
    print(const_1(result.x, elements, weights, delta))
    print(weights)
    print(weights_obj)
    bnd = sum(weights_obj * result.x)
    if upper:
        print(f"Upper bound: {round(bnd, 4)} (took {round(time() - t_init, 2)} sec. to compute).\n")
    else:
        print(f"Lower bound: {round(bnd, 4)} (took {round(time() - t_init, 2)} sec. to compute).\n")
    return bnd

opt(n = 10,
    l = 1,
    b = 1,
    delta = 0.05,
    vals = [0.1, 0.2, 0.4, 0.6, 0.7],
    upper = True)