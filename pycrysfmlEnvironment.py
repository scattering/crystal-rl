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
        self.prevStderr = None
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
#        print(self.step, len(self.remainingActions))
        if ((len(self.remainingActions) == 0) or (self.step > 200)):
            print(self.model.atomListModel.atomModels[0].z.value, self.prevStderr, self.totReward, self.step)
            return self.state, True, -1

        else:
            terminal = False



        #TODO check action type, assuming index of action list
#        print(actions)
#        print("actions: " + str(actions))
#        actionIndex = self.remainingActions[actions]
#        print("index " + str(actionIndex))

#        if (self.state[actions] == 1):
 #           return self.state, False, -10
#        print("______________________")
#        print(actions)

        if self.state[actions] == 1:
#            print(self.state[actions], self.step)
            return self.state, False, -2  #stop only if step > 200
        else:
            self.state[actions] = 1


#        print(actions)
  #      print(self.actions)
#        print(type(actions.item()))
   #     print(self.remainingActions)

#        print (self.refList[actions.item()].hkl)
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

        reward = -1
        #Need more data than parameters, have to wait to the second step to fit
        if len(self.visited) > 11:
#            print(np.where(self.state==1))
#            print("model dets")
#            print(self.model.atomListModel.atomModels[0].x.value)
#            print(self.model.atomListModel.atomModels[0].y.value)
#            print(self.model.atomListModel.atomModels[0].z.value)

            x, dx, stderr = self.fit(self.model)
#            print(x, dx)

            if ((not self.prevStderr is None) and stderr[0] < self.prevStderr[0]):
                reward += 2
		print(x, stderr)
#                file = open("deepQ_fit_data.txt","a")
#                file.write(str(indicies)+"\n")
#                file.write(str(x[0])+"\t")
#                file.write(str(dx)+"\t")
#                file.write(str(chisq)+"\n")
#                file.close()

            self.prevStderr = stderr

#            print(x,"\n", stderr,"\n", reward, "\n")

        self.totReward += reward


#        if ((not self.prevStderr is None) and self.step > 50 and stderr[10] < 5):
#         return self.state, True, 5
#        elif (len(self.remainingActions) == 0 or self.step > 200):
#            terminal = True
#        else:
#            terminal = False


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

