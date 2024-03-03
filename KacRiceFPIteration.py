import numpy as np

mc_iterations = 1000
l = 1
alpha = 1

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

# monte carlo for the denominator of <func>
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
