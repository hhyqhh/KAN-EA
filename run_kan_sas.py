from problem.LZG import LZG01
from pymoo.optimize import minimize

from alg.ueda import KAN_SAS


if __name__=='__main__':

    problem = LZG01(n_var=5)

    algorithm = KAN_SAS(pop_size=50)


    res = minimize(problem,
                   algorithm,
                   ('n_evals', 300),
                   verbose=True)
