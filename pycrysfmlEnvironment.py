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

import fswig_hklgen as H
import hkl_model as Mod
import sxtal_model as S

import  bumps.names  as bumps
import bumps.fitters as fitter
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

        self.state = np.zeros(len(refList))
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

        #TODO: clean up excess vars
        self.visited = []
        self.observed = []
        self.remainingActions = []
        for i in range(len(self.refList)):
            self.remainingActions.append(i)

        self.totReward = 0
        self.prevChiSq = None

        return self.state

    def fit(self, model):

        #Create a problem from the model with bumps,
        #then fit and solve it
        problem = bumps.FitProblem(model)
        monitor = fitter.StepMonitor(problem, open("sxtalFitMonitor.txt","w"))

        fitted = fitter.LevenbergMarquardtFit(problem)
        x, dx = fitted.solve(monitors=[monitor])

        return x, dx, problem.chisq()

    def execute(self, actions):

        #TODO check action type, assuming index of action list
#        print(actions)
        print("actions: " + str(actions))
        actionIndex = self.remainingActions[actions]
        print("index " + str(actionIndex))

#        if (self.state[actions] == 1):
 #           return self.state, False, -10

        self.state[actionIndex] = 1

#        if not (actions in self.remainingActions):
#            return self.state, True, -10

 #       print(actions)
  #      print(self.actions)
#        print(type(actions.item()))
   #     print(self.remainingActions)

        #No repeats
       
        self.visited.append(self.refList[actionIndex])
        self.remainingActions.remove(actionIndex)

        #Find the data for this hkl value and add it to the model
        self.model.refList = H.ReflectionList(self.visited)
        self.model._set_reflections()

        self.model.error.append(self.error[actionIndex])
        self.model.tt = np.append(self.model.tt, [self.tt[actionIndex]])

        self.observed.append(self.sfs2[actionIndex])
        self.model._set_observations(self.observed)
        self.model.update()

        reward = 0
        #Need more data than parameters, have to wait to the second step to fit
        if len(self.visited) > 1:

            x, dx, chisq = self.fit(self.model)

            reward -= 1
            if (self.prevChiSq != None and chisq < self.prevChiSq):
                reward += 1.5

            self.prevChiSq = chisq

        self.totReward += reward

        if (len(self.remainingActions) == 0):
            terminal = True
        else:
            terminal = False
 #       print("finished exec")
        return self.state, terminal, reward

    @property
    def states(self):
        return dict(shape=self.state.shape, type='float')

    @property
    def actions(self):

        #TODO limit to remaining options (no repeats)
        #TODO set up to have the hkls, so it can be generalized
        return dict(num_actions=len(self.remainingActions), names = self.remainingActions, type='int')

    @actions.setter
    def actions(self, value):
        self._actions = value


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

