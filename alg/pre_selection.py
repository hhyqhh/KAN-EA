from pymoo.algorithms.base.genetic import GeneticAlgorithm
from pymoo.operators.sampling.lhs import LHS
from pymoo.util.display.single import SingleObjectiveOutput
from pymoo.algorithms.soo.nonconvex.ga import FitnessSurvival
from pymoo.core.replacement import ImprovementReplacement
from pymoo.core.population import Population
from util.reproduction import get_reproduction
import numpy as np
from functools import reduce
import copy

def get_label(ys, rate=0.5, split_bound=None):
    """
    ys: shape (n,1)
    """
    if split_bound is None:
        ys_t = copy.deepcopy(ys)
        ys_t = ys_t.flatten()
        ys_t = np.sort(ys_t)
        split_bound = ys_t[int(len(ys_t) * rate)]

    l = np.zeros_like(ys)
    l[ys <= split_bound] = 1
    # l[ys > split_bound] = -1     # 1: positive, -1: negative
    index = np.where(l == 1)[0]
    return l.flatten().astype(int), index, split_bound



class KAN_SPS(GeneticAlgorithm):
    def __init__(self,
                pop_size=50,
                sampling=LHS(),
                reproduction=None,
                reproduction_type=None,
                model=None,    
                output=SingleObjectiveOutput(),
                trial_vectors_num=3,
                rate = 0.5,
                **kwargs
                ):
        super().__init__(pop_size=pop_size,
                            sampling=sampling,
                            output=output,
                            **kwargs)

        self.model = model
        self.trial_vectors_num = trial_vectors_num

        self.rate = rate


        if reproduction is None:
            if reproduction_type is None:
                raise Exception("reproduction is None")
            else:
                self.reproduction = get_reproduction(reproduction_type)
                self.reproduction_type = reproduction_type
        else:
            self.reproduction = reproduction
            if reproduction_type is not None:
                self.reproduction_type = reproduction_type
            else:
                raise Exception("reproduction_type is None")
        

    def _initialize_advance(self, infills=None, **kwargs):
        if self.reproduction_type == "VWH":
            self.reproduction.eda.init(
                D=self.problem.n_var,
                LB=self.problem.xl * np.ones(shape=self.problem.n_var),
                UB=self.problem.xu * np.ones(shape=self.problem.n_var)
            )


    def next(self):

        # get the infill solutions
        infills = self.infill()

        # call the advance with them after evaluation
        if infills is not None:
            if self.n_gen > 1:
                infills = self.preselection_by_KAN(infills)

            self.evaluator.eval(self.problem, infills, algorithm=self)
            self.advance(infills=infills)

        # if the algorithm does not follow the infill-advance scheme just call advance
        else:
            self.advance()



    def preselection_by_KAN(self, trials):
        trial_vectors_num = self.trial_vectors_num
        index = trials.get('index')

        trials_ = [trials[i:i + trial_vectors_num] for i in range(0, len(trials), trial_vectors_num)]

        scores = np.array([self.model.predict(trial.get("X")) for trial in trials_]).reshape(-1, trial_vectors_num)
        index_ = np.array(index).reshape(-1, trial_vectors_num)

        if self.model.__class__.__name__ == "KAN_Regressor":
            I_selected = np.argmin(scores, axis=1)
        if self.model.__class__.__name__ == "KAN_Classifier":
            I_selected = np.argmax(scores, axis=1)
        I = np.full(fill_value=False, shape=trials.shape)
        for i in range(self.n_offsprings):
            I[index_[i, I_selected[i]]] = True

        return trials[I]
    

    def _advance(self, infills=None, **kwargs):
        assert infills is not None, "This algorithms uses the AskAndTell interface thus infills must to be provided."

        I = range(self.n_offsprings)
        infills.set('index', I)

        self.pop[I] = ImprovementReplacement().do(self.problem, self.pop[I], infills)

        FitnessSurvival().do(self.problem, self.pop, return_indices=True)


    def _infill(self):

        Xs, ys = self.pop.get("X"), self.pop.get("F")

        ys = ys.flatten()

        # use current population to train the surrogate model

        if self.model.__class__.__name__ == "KAN_Regressor":
            self.model.fit(Xs, ys)
        if self.model.__class__.__name__ == "KAN_Classifier":
            
            ls, _, _ = get_label(ys, rate=self.rate)
            self.model.fit(Xs, ls)

        if self.reproduction_type == "CoDE":
            # CoDE operator can generate multiple offspring 
            self.reproduction.trial_vectors_num = self.trial_vectors_num
            infills = self.reproduction.do(self.problem, self.pop, self.n_offsprings, algorithm=self)
        else:
            # Generate multiple offspring by repeatedly executing operators 
            _infills = []
            for _ in range(self.trial_vectors_num):
                _infills.append(self.reproduction.do(self.problem, self.pop, self.n_offsprings, algorithm=self))

            infills_list = [item for sublist in zip(*_infills) for item in sublist]
            infills = reduce(lambda x, y: Population.merge(x, y), infills_list)


        index = np.arange(len(infills))
        infills.set("index", index)
        return infills
    
    def next(self):

        # get the infill solutions
        infills = self.infill()

        # call the advance with them after evaluation
        if infills is not None:
            if self.n_gen > 1:
                infills = self.preselection_by_KAN(infills)

            self.evaluator.eval(self.problem, infills, algorithm=self)
            self.advance(infills=infills)

        # if the algorithm does not follow the infill-advance scheme just call advance
        else:
            self.advance()




if __name__ == '__main__':

    from pymoo.optimize import minimize
    from problem.LZG import LZG01
    from model.KAN_surrogate import KAN_Regressor,KAN_Classifier

    problem = LZG01(n_var=5)

    model = KAN_Regressor()   # USE KAN_Regressor or KAN_Classifier
    # model = KAN_Classifier()  
      
    algorithm = KAN_SPS(problem=problem, pop_size=50, n_offsprings=50, reproduction_type="CoDE", trial_vectors_num=3,model=model)

    res = minimize(problem,
                   algorithm,
                   ('n_evals', 2000),
                   verbose=True, output=SingleObjectiveOutput(), )
    print("Best solution found: \nX = %s\nF = %s\nCV=%s" % (res.X, res.F, res.CV))