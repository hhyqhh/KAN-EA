from pymoo.optimize import minimize

from problem.LZG import LZG01
from model.KAN_surrogate import KAN_Regressor,KAN_Classifier
from alg.pre_selection import KAN_SPS

problem = LZG01(n_var=5)

model = KAN_Regressor()   # USE KAN_Regressor or KAN_Classifier
# model = KAN_Classifier()  
    
algorithm = KAN_SPS(problem=problem, 
                    pop_size=50, 
                    n_offsprings=50, 
                    reproduction_type="CoDE", 
                    trial_vectors_num=3,
                    model=model
                    )

res = minimize(problem,
                algorithm,
                ('n_evals', 2000),
                verbose=True)