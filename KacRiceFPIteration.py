import numpy as np

mc_iterations = 100
alpha = 20
eps = 1e-3

def phi(x):
    return x**2
def dphi(x):
    return 2*x
def ddphi(x):
    return 2

# monte carlo integral for the numerator of <func>
def integral1_mc(func, lambda_0, lambda_1, g_re, g_im):
    total = 0
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

# formula for the annealed complexity
def annealed_complexity(func, l, lambda_0, lambda_1, g_re, g_im):
    return -1 + np.log(alpha) - 2*alpha*np.log(alpha) + \
    lambda_0 * l + \
    (1 + np.log(2))/2 + \
    1/2 * np.log(lambda_1) - \
    np.log(np.sqrt(g_re**2 + g_im**2)) + eps*g_im + \
    alpha*np.log(integral3_mc(func, lambda_0, lambda_1, g_re, g_im))

# perform fixed point iteration on func (a function of one variable), with initial parameter value init
def fixed_point_update(func, init, iterations=100):
    # ensure the algorithm will converge by checking its derivative at init
    dfunc_init = np.abs((func(init + eps) - func(init - eps)) / (2*eps))
    # assert dfunc_init < 1, "Improper initialization for fixed point iteration, df|x=" + str(dfunc_init)
    param = init
    for j in range(iterations):
        param = func(param) + param
    return param

def update_l(l_init, lambda_0, lambda_1, g_re, g_im, iterations):
    # use fixed point iteration to update l, with other variables fixed
    l_opt = l_init
    for i in range(iterations):
        l_opt = fixed_point_update(
            lambda l: integral1_mc(phi, lambda_0, lambda_1, g_re, g_im) / integral2_mc(phi, lambda_0, lambda_1, g_re, g_im) - l,
            l_opt
        )
    return l_opt

def update_extr(l, lambda_0_init, lambda_1_init, g_re_init, g_im_init, iterations):
    # use fixed point iteration to update all variables other than l, with l fixed
    lambda_0_opt, lambda_1_opt, g_re_opt, g_im_opt = lambda_0_init, lambda_1_init, g_re_init, g_im_init
    for i in range(iterations):
        lambda_0_opt = fixed_point_update(
            lambda lambda_0: integral1_mc(phi, lambda_0, lambda_1_opt, g_re_opt, g_im_opt) / integral2_mc(phi, lambda_0, lambda_1_opt, g_re_opt, g_im_opt) - l,
            lambda_0_opt
        )
        lambda_1_opt = fixed_point_update(
            lambda lambda_1: 2*lambda_1*integral1_mc(lambda x: dphi(x)**2, lambda_0_opt, lambda_1, g_re_opt, g_im_opt) / integral2_mc(lambda x: dphi(x)**2, lambda_0_opt, lambda_1, g_re_opt, g_im_opt) - 1,
            lambda_1_opt
        )
        g_re_opt = fixed_point_update(
            lambda g_re: integral1_mc(
                lambda x: x*dphi(x) - (alpha*ddphi(x)*(alpha + ddphi(x)*g_re))/((alpha + ddphi(x)*g_re)**2 + (alpha + ddphi(x)*g_im_opt)**2),
                lambda_0_opt, lambda_1_opt, g_re, g_im_opt
            ) / integral2_mc(
                lambda x: x*dphi(x) - (alpha*ddphi(x)*(alpha + ddphi(x)*g_re))/((alpha + ddphi(x)*g_re)**2 + (alpha + ddphi(x)*g_im_opt)**2),
                lambda_0_opt, lambda_1_opt, g_re, g_im_opt
            ) + g_re / (g_re**2 + g_im_opt**2), 
            g_re_opt
        )
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
    return lambda_0_opt, lambda_1_opt, g_re_opt, g_im_opt

def solve_optimization(l_main, lambda_0_main, lambda_1_main, g_re_main, g_im_main, l_iterations, extr_iterations, total_iterations):
    # get the optimized annealed complexity for the initializations given as parameters
    for iter in range(total_iterations):
        lambda_0_main, lambda_1_main, g_re_main, g_im_main = update_extr(l_main, lambda_0_main, lambda_1_main, g_re_main, g_im_main, extr_iterations)
        l_main = update_l(l_main, lambda_0_main, lambda_1_main, g_re_main, g_im_main, l_iterations)
    return annealed_complexity(lambda_0_main, lambda_1_main, g_re_main, g_im_main)

solve_optimization(-1, -1, -1, -1, -1, l_iterations=10, extr_iterations=10, total_iterations=100)