import os,sys;sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
import os
from copy import copy
import numpy as np
import random as rand
import pickle
import itertools
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.axes as axes

import fswig_hklgen as H
import hkl_model as Mod
import sxtal_model as S

import  bumps.names  as bumps
import bumps.fitters as fitter
import bumps.lsqerror as lsqerr
from bumps.formatnum import format_uncertainty_pm

from tensorforce.environments import Environment


#Tensorforce Environment representation of
#the pycrysfml 'game'

class PycrysfmlEnvironment(Environment):

    def __init__(self,  observedFile, infoFile, backgFile=None, sxtal=True):

        if sxtal:

            #Read data
            self.spaceGroup, self.crystalCell, self.atomList = H.readInfo(infoFile)

            # return wavelength, refList, sfs2, error, two-theta, and four-circle parameters
            wavelength, refList, sfs2, error = S.readIntFile(observedFile, kind="int", cell=self.crystalCell)
            self.wavelength = wavelength
            self.refList = refList
            self.sfs2 = sfs2
            self.error = error
            self.tt = [H.twoTheta(H.calcS(self.crystalCell, ref.hkl), wavelength) for ref in refList]
            self.backg = None
            self.exclusions = []

        #TODO: else, for powder data

        self.reset()

    def reset(self):

        #Make a cell
        cell = Mod.makeCell(self.crystalCell, self.spaceGroup.xtalSystem)

        #TODO: make model thru tensorforce, not here
        #Define a model
        self.model = S.Model([], [], self.backg, self.wavelength, self.spaceGroup, cell,
                    [self.atomList], self.exclusions,
                    scale=0.06298, error=[],  extinction=[0.0001054])

        #Set a range on the x value of the first atom in the model
        self.model.atomListModel.atomModels[0].z.value = 0.3
        self.model.atomListModel.atomModels[0].z.range(0,0.5)
#        self.model.atomListModel.atomModels[0].B.range(0, 5)
#        self.model.atomListModel.atomModels[1].B.range(0,5)
#        self.model.atomListModel.atomModels[2].B.range(0,5)
#        self.model.atomListModel.atomModels[3].z.range(0,0.5)
#        self.model.atomListModel.atomModels[3].B.range(0,5)
#        self.model.atomListModel.atomModels[4].B.range(0,5)
#        self.model.atomListModel.atomModels[5].x.range(0,0.5)
#        self.model.atomListModel.atomModels[5].y.range(0,0.5)
#        self.model.atomListModel.atomModels[5].z.range(0,0.5)
#        self.model.atomListModel.atomModels[5].B.range(0,5)

        #TODO: clean up excess vars
        self.visited = []
        self.observed = []
        self.remainingActions = []
        for i in range(len(self.refList)):
            self.remainingActions.append(i)

        self.totReward = 0
        self.prevChisq = None
        self.step = 0

        self.state = np.zeros(len(self.refList))
        self.stateList = []

        return self.state

    def fit(self, model):

        #Create a problem from the model with bumps,
        #then fit and solve it
        problem = bumps.FitProblem(model)
#        print("before: ", lsqerr.stderr(problem.cov()))
        fitted = fitter.LevenbergMarquardtFit(problem)
        x, dx = fitted.solve()
#        print(problem.chisq())
#        print("after", lsqerr.stderr(problem.cov()))
        return x, dx, lsqerr.stderr(problem.cov())

    def execute(self, actions):

        self.step += 1

        if self.state[actions] == 1:

#            print(self.refList[actions.item()].hkl, self.step)
            self.totReward -= 0.15
            return self.state, (self.step > 300), -0.15  #stop only if step > 200
        else:
            self.state[actions] = 1



        #No repeats
        self.visited.append(self.refList[actions.item()])
        self.remainingActions.remove(actions.item())

        #Find the data for this hkl value and add it to the model
        self.model.refList = H.ReflectionList(self.visited)
        self.model._set_reflections()

        self.model.error.append(self.error[actions])
        self.model.tt = np.append(self.model.tt, [self.tt[actions]])

        self.observed.append(self.sfs2[actions])
        self.model._set_observations(self.observed)
        self.model.update()

        reward = -0.1

        #Need more data than parameters, have to wait to the second step to fit
        if len(self.visited) > 11:

            x, dx, chisq = self.fit(self.model)
#            print(x, chisq)

#            problem = bumps.FitProblem(self.model)
#            stderr = lsqerr.stderr(problem.cov())

            if (self.prevChisq != None and chisq < self.prevChisq):
                reward = 0.1

            self.prevChisq = chisq

#            print(x,"\n", stderr,"\n", reward, "\n")

        self.totReward += reward

        if (self.prevChisq != None and self.step > 50 and chisq < 50):
            return self.state, True, 0.5
        if (len(self.remainingActions) == 0 or self.step > 300):
#            print(self.model.atomListModel.atomModels[0].z.value, self.prevChisq, self.totReward, self.step)
            terminal = True
        else:
            terminal = False

#        self.stateList.append(self.state.copy())
#        fig = mpl.pyplot.pcolor(self.stateList, cmap="RdBu" )
#        mpl.pyplot.savefig("state_space.png")


        return self.state, terminal, reward

    @property
    def states(self):
        return dict(shape=self.state.shape, type='float')

    @property
    def actions(self):

        #TODO limit to remaining options (no repeats)
        #TODO set up to have the hkls, so it can be generalized
        return dict(num_actions=len(self.refList), type='int')

#    @actions.setter
#    def actions(self, value):
#        self._actions = value


    #_______________________________________________
    #TODO: read, understand, improve for my purposes
    @staticmethod
    def from_spec(spec,     kwargs):
        """
        Creates an environment from a specification dict.
        """
        env = tensorforce.util.get_object(
            obj=spec,
            predefined_objects=tensorforce.environments.environments,
            kwargs=kwargs
        )
        assert isinstance(env, Environment)
        return env

