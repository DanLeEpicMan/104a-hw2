'''
Daniel Naylor
5094024
10/20/2022
'''

import math
import matplotlib.pyplot as plt
import numpy as np

from typing import Tuple, List, Callable

def f(x: float) -> float:
    '''
    Our function f for parts a and b
    '''
    return 1/(1 + x**2)

def uniform_lambda(*, n: int, j: int) -> int:
    '''
    Computes a lambda assuming that the grid is uniform (all points are equally spaced).

    `(-1)^j n choose j`
    '''
    return (-1)**j * math.comb(n, j)

def cos_lambda(*, n: int, j: int) -> float:
    '''
    Computes a lambda assuming it's of the cosine form shown in the homework question 3b remark.
    '''
    scalar = 0.5 if j==0 or j==n else 1
    return scalar * (-1)**j

def create_poly_interp(
        data: List[Tuple[float, float]],
        *,
        lambda_formula: Callable[[int, int], int | float]
    ) -> Callable[[float], float]:
    '''
    Returns a polynomial interpolation based on the
    given dataset using Barycentric Weights.

    `lambda_formula` refers to the formula (function) that 
    should be used to compute lambda values.

    The returned callable will act as a mathematical function
    that is defined everywhere except at the datapoints themselves.
    '''
    def wrap(x: float):
        top_sum = 0.0
        bottom_sum = 0.0
        n = len(data) - 1
        for j in range(n + 1):
            # if trying to compute over a given node point, return its value
            if x == data[j][0]: return data[j][1] 
            current_lambda = lambda_formula(n=n, j=j)
            top_sum += (current_lambda * data[j][1])/(x - data[j][0])
            bottom_sum += current_lambda/(x - data[j][0])

        return top_sum/bottom_sum

    return wrap


# --------------------------------------------
#                    Part A
# --------------------------------------------

# all datasets are lists of the form
# j: (xj, fj)
# where j is the jth element in the list

def create_dataset_for_a(*, n: int, func: Callable[[float], float]) -> List[Tuple[float, float]]:
    '''
    Creates a suitable dataset of n + 1 (nodes)
    for question 3a depending on the given n
    and function
    '''
    return [
        (
            -5 + j * (10/n), 
            func(-5 + j * (10/n))
        ) 
        for j in range(n + 1) # j = 0, ..., n
    ]

P_a1 = create_poly_interp( # polynomial approximation for 3a n=4
    data=create_dataset_for_a(n=4, func=f),
    lambda_formula=uniform_lambda
)

P_a2 = create_poly_interp(
    data=create_dataset_for_a(n=8, func=f),
    lambda_formula=uniform_lambda
)

P_a3 = create_poly_interp(
    data=create_dataset_for_a(n=12, func=f),
    lambda_formula=uniform_lambda
)

# --------------------------------------------
#                    Part B
# --------------------------------------------

def create_cos_nodes(*, n: int, func: Callable[[float], float]) -> List[Tuple[float, float]]:
    '''
    Creates nodes using the cosine formula given in part b
    '''
    return [
        (
            5 * math.cos(math.pi * j / n),
            func(5 * math.cos(math.pi * j / n))
        )
        for j in range(n+1)
    ]

P_b1 = create_poly_interp(
    data=create_cos_nodes(n=4, func=f),
    lambda_formula=cos_lambda
)

P_b2 = create_poly_interp(
    data=create_cos_nodes(n=8, func=f),
    lambda_formula=cos_lambda
)

P_b3 = create_poly_interp(
    data=create_cos_nodes(n=12, func=f),
    lambda_formula=cos_lambda
)

P_b4 = create_poly_interp(
    data=create_cos_nodes(n=100, func=f),
    lambda_formula=cos_lambda
)

# --------------------------------------------
#                    Part C
# --------------------------------------------

def f_c(x: float) -> float:
    '''
    Our function f for part c
    '''
    return math.exp(-x**2 / 5)


P_c1 = create_poly_interp( # polynomial approximation for 3a n=4
    data=create_dataset_for_a(n=4, func=f_c),
    lambda_formula=uniform_lambda
)

P_c2 = create_poly_interp(
    data=create_dataset_for_a(n=8, func=f_c),
    lambda_formula=uniform_lambda
)

P_c3 = create_poly_interp(
    data=create_dataset_for_a(n=12, func=f_c),
    lambda_formula=uniform_lambda
)

# ---------------------------------------------------
#             graphing all the functions
#    (i only included the code for graphing part c)
#  (the code for parts a and b are almost identical)
#        (all graphs are attached to this PDF)
# ---------------------------------------------------

domain = np.linspace(-5, 5, 100)

graph_of_P_c1 = np.array([
    P_c1(x)
    for x in domain
])

graph_of_P_c2 = np.array([
    P_c2(x)
    for x in domain
])

graph_of_P_c3 = np.array([
    P_c3(x)
    for x in domain
])

graph_of_f_c = np.array([
    f_c(x)
    for x in domain
])

# setting up the graphs...
fig = plt.figure()

ax_a = fig.add_subplot(1, 1, 1)
ax_a.spines['left'].set_position('center')
ax_a.spines['bottom'].set_position('zero')
ax_a.spines['right'].set_color('none')
ax_a.spines['top'].set_color('none')
ax_a.xaxis.set_ticks_position('bottom')
ax_a.yaxis.set_ticks_position('left')

# plot the functions
plt.plot(domain, graph_of_P_c1, 'b', label='P_4')
plt.plot(domain, graph_of_P_c2, 'c', label='P_8')
plt.plot(domain, graph_of_P_c3, 'r', label='P_12')
plt.plot(domain, graph_of_f_c, 'g', label='e^(-x**2 / 5)')

plt.legend(loc='upper left')

# show the plot
plt.show()
