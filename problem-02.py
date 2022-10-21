'''
Daniel Naylor
5094024
10/20/2022
'''

import math
from typing import Tuple, List, Callable


stored_lambdas: dict[Tuple[int, int], float] = {
     # to allow for efficient recursion 
     # entries are of the form 
     # (n, j): lambda value
     (0, 0): 1.0
}

data_points: List[Tuple[float, float]] = [
    # all relevant data, stored in the form
    # j: (x_j, f_j)
    
    (0.00, 0.0000),
    (0.25, 0.7071),
    (0.50, 1.0000),
    (0.75, 0.7071),
    (1.25, -0.7071),
    (1.50, -1.0000)
]

def compute_lambda_direct(*, n: int, j: int, data: List[Tuple[float, float]] = data_points) -> float:
    '''
    Computes a Barycentric weight (lambda) directly.
    1/product(xj - xk) for j != k

    `data` is the dataset that these lambdas will be computed from.
    '''
    prod = 1.0
    for k in range(n+1): # 0, 1, ..., n
        if k==j: continue
        prod *= (data[j][0] - data[k][0])

    return 1.0/prod

def compute_lambda(*, n: int, j: int, data: List[Tuple[float, float]] = data_points) -> float:
    '''
    Computes a Barycentric weight (lambda) recursively-ish.
    This uses `stored_lambdas` (dynamic programming) for efficiency purposes.
    '''
    if (n, j) in stored_lambdas: 
        return stored_lambdas[(n, j)]
    elif (n - 1, j) in stored_lambdas:
        stored_lambdas[(n, j)] = stored_lambdas[(n-1, j)]/(data[j][0] - data[n][0])
        return stored_lambdas[(n, j)]
    else:
        stored_lambdas[(n, j)] = compute_lambda_direct(n=n, j=j, data=data)
        return stored_lambdas[(n, j)]

def create_poly_interp(data: List[Tuple[float, float]] = data_points) -> Callable[[float], float]:
    '''
    Returns a polynomial interpolation based on the
    given dataset using Barycentric Weights.

    The returned callable will act as a mathematical function
    that is defined everywhere except at the datapoints themselves.
    '''
    def wrap(x: float):
        top_sum = 0.0
        bottom_sum = 0.0
        n = len(data) - 1
        for j in range(n + 1):
            current_lambda = compute_lambda(n=n, j=j, data=data)
            top_sum += (current_lambda * data[j][1])/(x - data[j][0])
            bottom_sum += current_lambda/(x - data[j][0])

        return top_sum/bottom_sum

    return wrap

P_5 = create_poly_interp(data=data_points)

print(P_5(2))

# Output: 0.8519999999999989