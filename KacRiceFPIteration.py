"""
Code to compute the annealed complexity, written by Ian Joffe
From formula (31) of Landscape Complexity for the Empirical Risk of Generalized Linear Models
    by Maillard, Ben Arous, and Biroli
Please see attached report for details 
"""

import numpy as np
import matplotlib.pyplot as plt

mc_iterations = 100
alpha = 20
eps = 1e-3

# define phi as the quadratic for phase retrieval
def phi(x):
    return x**2
def dphi(x):
    return 2*x
def ddphi(x):
    return 2

# monte carlo integral for the numerator of <func>
def integral1_mc(func, lambda_0, lambda_1, g_re, g_im):
    total = 0
    # Given my limited computation, I need to use the same seed for each MC integral
    # or else I can't compare them or manually get their derivatives
    np.random.seed(0)
    for i in range(mc_iterations):
        x = np.random.normal(0,1)
        Dx = func(x)
        total += Dx * np.exp(
            -1/alpha * (
                lambda_0*phi(x) + lambda_1*dphi(x)**2 + g_re*x*dphi(x)
            ) +
            np.log(
                np.sqrt(
                    (alpha + ddphi(x)*g_re)**2 + 
                    (alpha + ddphi(x)*g_im)**2
                )
            )
        )
    return total / mc_iterations

# monte carlo integral for the denominator of <func>
def integral2_mc(func, lambda_0, lambda_1, g_re, g_im):
    total = 0
    np.random.seed(0)
    for i in range(mc_iterations):
        x = np.random.normal(0,1)
        Dx = func(x)
        total += x * np.exp(
            -1/alpha * (
                lambda_0*phi(x) + lambda_1*dphi(x)**2 + g_re*x*dphi(x)
            ) +
            np.log(
                np.sqrt(
                    (alpha + ddphi(x)*g_re)**2 + 
                    (alpha + ddphi(x)*g_im)**2
                )
            )
        )
    return total / mc_iterations

# monte carlo for the integral in the annealed complexity
def integral3_mc(func, lambda_0, lambda_1, g_re, g_im):
    lambda_2 = g_re

    total = 0
    np.random.seed(0)
    for i in range(mc_iterations):
        x = np.random.normal(0,1)
        Dx = func(x)
        total += Dx * np.exp(
            -1/alpha * (
                lambda_0*phi(x) + lambda_1*dphi(x)**2 + lambda_2*x*dphi(x)
            ) +
            np.log(
                np.sqrt(
                    (alpha + ddphi(x)*g_re)**2 + 
                    (alpha + ddphi(x)*g_im)**2
                )
            )
        )
    return total / mc_iterations

# formula for the annealed complexity
def annealed_complexity(func, l, lambda_0, lambda_1, g_re, g_im):
    return 1/2*(-1 + np.log(alpha) - 2*alpha*np.log(alpha)) + \
    lambda_0 * l + \
    (1 + np.log(2))/2 + \
    1/2 * np.log(lambda_1) - \
    np.log(np.sqrt(g_re**2 + g_im**2)) + eps*g_im + \
    alpha*np.log(integral3_mc(func, lambda_0, lambda_1, g_re, g_im))

# perform fixed point iteration on func (a function of one variable), with initial parameter value init
def fixed_point_update(func, init, iterations=100):
    # ensure the algorithm will converge by checking its derivative at init
    dfunc_init = np.abs((func(init + eps) - func(init - eps)) / (2*eps))
    assert dfunc_init < 1, "Improper initialization for fixed point iteration, df|x=" + str(dfunc_init)
    param = init
    for j in range(iterations):
        param = func(param) + param
    return param

def update_l(l_init, lambda_0, lambda_1, g_re, g_im, iterations):
    # The paper recommends treating l as fixed while iterating over other variables
    # This updates l according only to its formula, not iteratively
    # print(l_init)
    return integral1_mc(phi, lambda_0, lambda_1, g_re, g_im) / integral2_mc(phi, lambda_0, lambda_1, g_re, g_im)

def update_extr(l, lambda_0_init, lambda_1_init, g_re_init, g_im_init, iterations):
    # use fixed point iteration to update all variables other than l, with l fixed
    lambda_0_opt, lambda_1_opt, g_re_opt, g_im_opt = lambda_0_init, lambda_1_init, g_re_init, g_im_init
    for i in range(iterations):
        # print("Lambda_0 ", lambda_0_opt)
        lambda_0_opt = fixed_point_update(
            lambda lambda_0: integral1_mc(phi, lambda_0, lambda_1_opt, g_re_opt, g_im_opt) / integral2_mc(phi, lambda_0, lambda_1_opt, g_re_opt, g_im_opt) - l,
            lambda_0_opt
        )
        # print("Lambda_0 Updated ", lambda_0_opt)
        # print("Lambda_1 ", lambda_1_opt)
        lambda_1_opt = fixed_point_update(
            lambda lambda_1: integral1_mc(lambda x: dphi(x)**2, lambda_0_opt, lambda_1, g_re_opt, g_im_opt) / integral2_mc(lambda x: dphi(x)**2, lambda_0_opt, lambda_1, g_re_opt, g_im_opt) - 1/(2*lambda_1),
            lambda_1_opt
        )
        # print("Lambda_1 Updated ", lambda_1_opt)
        # print("g_re ", g_re_opt)
        g_re_opt = fixed_point_update(
            lambda g_re: integral1_mc(
                lambda x: x*dphi(x) - (alpha*ddphi(x)*(alpha + ddphi(x)*g_re))/((alpha + ddphi(x)*g_re)**2 + (alpha + ddphi(x)*g_im_opt)**2),
                lambda_0_opt, lambda_1_opt, g_re, g_im_opt
            ) / integral2_mc(
                lambda x: x*dphi(x) - (alpha*ddphi(x)*(alpha + ddphi(x)*g_re))/((alpha + ddphi(x)*g_re)**2 + (alpha + ddphi(x)*g_im_opt)**2),
                lambda_0_opt, lambda_1_opt, g_re, g_im_opt
            ) * (-1)/(g_re / (g_re**2 + g_im_opt**2)) - 1, 
            g_re_opt
        )
        # print("g_re Updated ", g_re_opt)
        # print("g_im ", g_im_opt)
        g_im_opt = fixed_point_update(
            lambda g_im: -integral1_mc(
                lambda x: (alpha*ddphi(x)**2 * g_im) / ((alpha + ddphi(x)*g_re_opt)**2 + (alpha + ddphi(x)*g_im)**2),
                lambda_0_opt, lambda_1_opt, g_re_opt, g_im
            ) / integral2_mc(
                lambda x: (alpha*ddphi(x)**2 * g_im) / ((alpha + ddphi(x)*g_re_opt)**2 + (alpha + ddphi(x)*g_im)**2),
                lambda_0_opt, lambda_1_opt, g_re_opt, g_im
            ) + g_im/(g_im**2 + g_re_opt**2) - eps,
            g_im_opt
        )
        # print("g_im Updated ", g_im_opt)
    return lambda_0_opt, lambda_1_opt, g_re_opt, g_im_opt

def solve_optimization(l_main, lambda_0_main, lambda_1_main, g_re_main, g_im_main, l_iterations, extr_iterations, total_iterations):
    # get the optimized annealed complexity for the initializations given as parameters
    all_complexities = [annealed_complexity(phi, l_main, lambda_0_main, lambda_1_main, g_re_main, g_im_main)] # used to draw optimization trajectory curve
    for iter in range(total_iterations):
        lambda_0_main, lambda_1_main, g_re_main, g_im_main = update_extr(l_main, lambda_0_main, lambda_1_main, g_re_main, g_im_main, extr_iterations)
        l_main = update_l(l_main, lambda_0_main, lambda_1_main, g_re_main, g_im_main, l_iterations)
        all_complexities.append(annealed_complexity(phi, l_main, lambda_0_main, lambda_1_main, g_re_main, g_im_main))
    return annealed_complexity(phi, l_main, lambda_0_main, lambda_1_main, g_re_main, g_im_main)

# These initializations are not especially principled, 
# but when you use them the FP iteration algorithm gets the necessary derivatives with magnitidue < 1
# and the resulting annealed complexity matches the results of the paper, so I believe they are valid.
# In the future I hope to further explore the space of initial parameters to confirm my results. 
solve_optimization(1, 100, 100, 100, 1, l_iterations=1, extr_iterations=1, total_iterations=10000)