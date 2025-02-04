from scipy.optimize import minimize, LinearConstraint
from time import time
from utils import log_stirling_approximation, sup_bin

import numpy as np
import itertools
import math


def lower_test_bound(m, d):
    return 1 - math.exp(-math.log(1 / d) / m)

def multinomial(lst, n):
    assert np.sum(lst) == n
    res = log_stirling_approximation(n)
    for a in lst:
        res -= log_stirling_approximation(a)
    return math.exp(res)


def K(n, l, weights):
    possible_values = []
    for j in range(len(weights)):
        if j == 0:
            possible_values.append(list(range(n + 1)))
        elif l == weights[j]:
            possible_values.append(list(range(2)))
        else:
            val = min(math.floor(l / (weights[j] + 1e-10)), n)
            possible_values.append(list(range(val + 1)))

    elements = []
    for element in itertools.product(*possible_values):
        if np.sum(np.array(element) * np.array(weights)) <= l and np.sum(element) == n:
            elements.append(element)
    return elements

def K_jac(combinations, i):
    elements = []
    for element in combinations:
        if element[i] > 0:
            elements.append(element)
    return elements

def F(p, n, combinations, m):
    tot = 0
    for elements in combinations:
        curr = math.log10(multinomial(elements, n))
        for j in range(m):
            curr += elements[j] * math.log10(np.abs(p[j] + 1e-20))
        curr = 10 ** curr
        tot += curr
    return tot

def gradient_F(p, n, combinations):
    """
    Compute the gradient of F with respect to each parameter.

    Parameters:
    - p: Current probability vector
    - n: Total number of samples
    - combinations: Possible combinations satisfying risk constraint
    - delta: Risk threshold

    Returns:
    - Gradient vector representing sensitivity of F to each parameter
    """
    # Ensure probabilities sum to 1 and are positive
    p = np.maximum(p, 1e-10)
    p = p / np.sum(p)
    m = len(p)

    # Initialize gradient vector
    gradient = np.zeros(m)

    # Iterate through each parameter
    for i in range(m):
        # Compute sensitivity for this parameter
        param_sensitivity = 0

        # Iterate through all valid combinations
        for k in combinations[i]:
            # Compute probability of this combination
            try:
                # Log-probability to avoid numerical issues
                log_prob = (
                        log_stirling_approximation(n) -  # Multinomial coefficient
                        sum(log_stirling_approximation(k_j) for k_j in k)  # Individual coefficients
                )

                # Add log probabilities of each parameter
                for j in range(m):
                    # Careful handling of zero probabilities
                    log_prob += k[j] * (np.log(p[j]) if p[j] > 0 else -np.inf)

                # Probability of the combination
                combination_prob = np.exp(log_prob)

                # Derivative of log-probability with respect to parameter i
                # This is k[i]/p[i] when p[i] > 0
                if p[i] > 0:
                    log_prob_derivative = k[i] / p[i] if k[i] > 0 else 0
                else:
                    log_prob_derivative = 0

                # Accumulate gradient component
                param_sensitivity += combination_prob * log_prob_derivative

            except Exception as e:
                # Log any unexpected computational errors
                print(f"Error processing combination {k}: {e}")

        gradient[i] = param_sensitivity

    return gradient

def equality_const(p):
    return np.sum(p[:-1]) - 1

def inequality_const(p, weights, alpha, tol):
    return np.dot(p[:-1], weights) + tol * np.dot(weights, weights) * p[-1] - alpha

def obj(p):
    """objective function, to be solved."""
    return -p[-1]

def opt(n, l, a, b, x_weights, delta, tol):
    assert b == x_weights[-1]
    assert a < x_weights[0]
    print(f"'a' value: {a}.")
    F_weights = np.array(x_weights[:-1].copy()) + 1e-5
    F_weights = np.insert(F_weights, 0, a)
    m = len(F_weights)
    lower = np.zeros(m + 1)
    upper = np.ones(m + 1)
    upper[-1] = np.inf
    alpha = 0
    constraints = [LinearConstraint(np.eye(m + 1), lower, upper),
                   {'type': 'eq', 'fun': inequality_const, 'args': (np.ones(m), 1, 0)},
                   {'type': 'ineq', 'fun': inequality_const, 'args': (x_weights, alpha, -1)}]
    for i in range(m):
        a_1 = np.zeros(m)
        a_1[i] = 1
        constraints.append({'type': 'ineq', 'fun': inequality_const, 'args': (a_1, -1, -1)})

        a_2 = np.zeros(m)
        for j in range(i + 1):
            a_2[j] = 1
        if i < m - 1:
            constraints.append({'type': 'ineq', 'fun': inequality_const, 'args': (a_2.copy(), round(sup_bin(n, min(l // F_weights[i + 1], n), delta), 8) - 1, 0)})
            print(a_2, 1 - round(sup_bin(n, min(l // F_weights[i + 1], n), delta), 8))
        else:
            constraints.append({'type': 'ineq', 'fun': inequality_const, 'args': (a_2, -1, 1)})
    t_init = time()
    combinations = K(n, l, F_weights)
    jac_combinations = []
    for i in range(m):
        jac_combinations.append(K_jac(combinations, i))
    print(f"Took {round(time() - t_init, 2)} sec. to compute combinations...")

    t_init = time()
    alphas = [0]
    while True:
        initial_guess = np.zeros(m + 1) + 1 / m  # initial guess can be anything
        result = minimize(obj, initial_guess, constraints=constraints)
        if F(result.x[:-1], n, combinations, m) > delta:
            alpha = np.dot(result.x[:-1], x_weights)
            print(f"Current bound value: {alpha}...")
            weights = result.x[:-1].copy()
            constraints[2] = {'type': 'ineq', 'fun': inequality_const, 'args': (x_weights, alpha, -1)}
        else:
            jac = gradient_F(result.x[:-1], n, jac_combinations)
            jac = jac.copy()
            if np.sqrt(np.dot(jac, jac)) < 1e-10:
                jac /= np.sqrt(np.dot(jac, jac))
            constraints.append({'type': 'ineq', 'fun': inequality_const, 'args': (jac, np.dot(jac, result.x[:-1]), -1)})

        alphas.append(alpha)
        if len(alphas) > 100:
            if alphas[-1] - alphas[-100] <= tol:
                break
    print(f"Upper bound: {round(alphas[-1], 4)} (took {round(time() - t_init, 2)} sec. to compute).")
    print(f"Weight values: {weights}")
    return alphas[-1]

n = 10
l = 1
delta = 0.05
a = 0#lower_test_bound(1000, delta)
b = 1
x_weights = [0.1, 0.2, 0.4, 0.6, 0.7, 1]
tol = 1e-5

opt(n, l, a, b, x_weights, delta, tol)