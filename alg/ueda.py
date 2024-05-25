from pymoo.algorithms.base.genetic import GeneticAlgorithm
from pymoo.algorithms.soo.nonconvex.ga import FitnessSurvival
from pymoo.core.population import Population
from pymoo.operators.sampling.lhs import LHS
from pymoo.util.display.single import SingleObjectiveOutput

from util.reproduction import VWH_Local_Reproduction_unevaluate


import numpy as np
import copy


from model.KAN_surrogate import KAN_Regressor,KAN_Classifier

class UEDA(GeneticAlgorithm):
    def __init__(self,
                 pop_size=50,
                 tao=100,  # yo
                 sampling=LHS(),
                 reproduction=VWH_Local_Reproduction_unevaluate(),
                 output=SingleObjectiveOutput(),
                 surrogate=None,
                 **kwargs):
        super().__init__(pop_size=pop_size,
                         sampling=sampling,
                         output=output,
                         survival=FitnessSurvival(),
                         **kwargs)
        self.reproduction = reproduction

        # all solutions that have been evaluated so far
        self.archive_eva = Population()
        self.current_sored_Xs = None
        self.tao = tao

        self.surrogate = surrogate


    def _initialize_advance(self, infills=None, **kwargs):
        # init the eda model
        self.reproduction.eda.init(
            D=self.problem.n_var,
            LB=self.problem.xl * np.ones(shape=self.problem.n_var),
            UB=self.problem.xu * np.ones(shape=self.problem.n_var)
        )

        # 将初始化种群保存至Archive
        self.archive_eva = Population.merge(self.pop, self.archive_eva)

        self.unevaluated_pop = copy.deepcopy(self.pop.get('X'))

        if self.surrogate is None:
            raise Exception("surrogate model is None")

    def _infill(self):
        # get current population
        t_xs, t_ys = self.get_raw_training_data()

        # train surrogate model
        self.training_surrogete_model(t_xs, t_ys)

        infills = self.reproduction.do(
            self.problem,
            self.pop,
            self.n_offsprings,
            algorithm=self,
            unevaluated_pop=self.unevaluated_pop
        )

        # surrogete assisted selection
        x_best, unevaluated_pop = self.surrogate_assisted_selection(infills)
        self.unevaluated_pop = unevaluated_pop

        infills = Population.new(X=x_best)
        return infills
    

    def _advance(self, infills=None, **kwargs):
        if infills is not None:
            self.archive_eva = Population.merge(self.archive_eva, infills)

        self.pop = self.survival.do(
            self.problem, self.archive_eva, n_survive=self.pop_size, algorithm=self, **kwargs)
        


    def surrogate_assisted_selection(self, pop):
        Xs = pop.get('X')
        ys_pre = self.surrogate.predict(Xs)

        # 选择最优的解
        sorted_ind = np.argsort(ys_pre.flatten())
        X_best = copy.deepcopy(Xs[sorted_ind[0], :]).reshape(1, -1)

        # 选择unevaluated_pop
        selected_decs = copy.deepcopy(
            Xs)[sorted_ind[:int(self.pop_size / 2)], :]

        # selected = ys_pre.flatten() > 0
        # selected_decs = copy.deepcopy(Xs)[selected, :]
        #
        # if selected_decs.shape[0] > 25:
        #     r_index = np.random.permutation(selected_decs.shape[0])
        #     selected_decs = selected_decs[r_index[:int(self.pop_size / 2)], :]

        return X_best, selected_decs
    

    def get_raw_training_data(self):
        """
        从 archive 中选择tao 个解返回
        """
        t_xs, t_ys = self.archive_eva.get("X"), self.archive_eva.get("F")
        if len(self.archive_eva) <= self.tao:
            return t_xs, t_ys.flatten()
        else:
            t = copy.deepcopy(t_ys).flatten()
            index = t.argsort()
            return t_xs[index[: self.tao], :], t_ys[index[: self.tao], :].flatten()

    def training_surrogete_model(self, Xs, ys):
        self.surrogate.fit(Xs, ys)




class KAN_SAS(UEDA):
    def __init__(self, 
                pop_size=50, 
                tao=50, 
                rate = 0.3,
                sampling=LHS(), 
                output=SingleObjectiveOutput(),
                reproduction=VWH_Local_Reproduction_unevaluate(),
                **kwargs):
        
        self.m1 = KAN_Regressor()
        self.m2 = KAN_Classifier()


        self.rate = rate


        super().__init__(pop_size=pop_size, 
                    tao=tao, 
                    sampling=sampling, 
                    output=output, 
                    reproduction=reproduction, 
                    surrogate=None, 
                    **kwargs)
        

    def _initialize_advance(self, infills=None, **kwargs):
        # init the eda model
        self.reproduction.eda.init(
            D=self.problem.n_var,
            LB=self.problem.xl * np.ones(shape=self.problem.n_var),
            UB=self.problem.xu * np.ones(shape=self.problem.n_var)
        )

        # 将初始化种群保存至Archive
        self.archive_eva = Population.merge(self.pop, self.archive_eva)

        self.unevaluated_pop = copy.deepcopy(self.pop.get('X'))







    def training_surrogete_model(self, Xs, ys):
        # print("training_surrogete_model, Xs shape:", Xs.shape)
        # self.surrogate.fit(Xs, ys)
        ls,_,_ = self.get_label(ys,self.rate)

        self.m1.fit(Xs,ys)
        self.m2.fit(Xs,ls)





    def get_label(self, ys, rate=0.3, split_bound=None):
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
        return l, index, split_bound
    



    def surrogate_assisted_selection(self, pop):
        Xs = pop.get('X')
        ys_pre = self.m1.predict(Xs)
        ls_pre = self.m2.predict(Xs)


        # 选择最优的解
        sorted_ind = np.argsort(ys_pre.flatten())
        X_best = copy.deepcopy(Xs[sorted_ind[0], :]).reshape(1, -1)

        # 选择unevaluated_pop

        mask = ls_pre == 1
        selected_decs = Xs[mask, :]
        if selected_decs.shape[0] > self.pop_size / 2:
            r_index = np.random.permutation(selected_decs.shape[0])
            selected_decs = selected_decs[r_index[:int(self.pop_size / 2)], :]



        return X_best, selected_decs
    



if __name__=='__main__':
    from problem.LZG import LZG01, LZG02, LZG03, LZG04
    from pymoo.optimize import minimize

    problem = LZG01(n_var=10)

    algorithm = KAN_SAS(pop_size=50)


    res = minimize(problem,
                   algorithm,
                   ('n_evals', 300),
                   verbose=True)
    print("Best solution found: \nX = %s\nF = %s\nCV=%s" % (res.X, res.F, res.CV))