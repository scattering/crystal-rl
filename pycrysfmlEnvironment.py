import os,sys;sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))))
import os
from copy import copy
import numpy as np
import random as rand
import pickle
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

        DATAPATH = os.path.dirname(os.path.abspath(__file__))

        if sxtal:

            #Read data
            self.spaceGroup, self.crystalCell, self.atomList = H.readInfo(infoFile)

            # return wavelength, refList, sfs2, error, two-theta, and four-circle parameters
            wavelength, refList, sfs2, error = S.readIntFile(observedFile, kind="int", cell=crystalCell)
            self.wavelength = wavelength
            self.actions = refList
            self.sfs2 = sfs2
            self.error = error
            self.tt = [H.twoTheta(H.calcS(crystalCell, ref.hkl), wavelength) for ref in refList]
            self.backg = None
            self.exclusions = []

        #TODO: else, for powder data


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
        for i in range(len(self.actions)):
            self.remainingActions.append(i)

        self.totReward = 0
        self.stateIndex = 0

    def execute(self, action):

        #TODO check action type, assuming index of action list

        #No repeats
        self.remainingRefs.remove(action)
        self.visited.append(self.actions(action))

        #Find the data for this hkl value and add it to the model
        model.refList = H.ReflectionList(visited)
        model._set_reflections()

        model.error.append(error[actionIndex])
        model.tt = np.append(model.tt, [tt[actionIndex]])

        observed.append(sfs2[actionIndex])
        model._set_observations(observed)
        model.update()

        #Need more data than parameters, have to wait to the second step to fit
        if step > 0:
            x, dx, chisq = fit(model)

            reward -= 1
            if (prevX2 != None and chisq < prevX2):
                reward += 1.5

            qtable[stateIndex, actionIndex] =  qtable[stateIndex, actionIndex] + \
                                               alpha*(reward + gamma*(np.max(qtable[stateIndex,:])) - \
                                               qtable[stateIndex, actionIndex])
            prevX2 = chisq


       #TODO        stateIndex = actionIndex+1  #shifted up one for states, so that the first index is no data
        self.totReward += reward





        """
        Executes action, observes next state(s) and reward.
        Args:
            actions: Actions to execute.
        Returns:
            (Dict of) next state(s), boolean indicating terminal, and reward signal.
        """
        raise NotImplementedError

    @property
    def states(self):
        """
        Return the state space. Might include subdicts if multiple states are available simultaneously.
        Returns: dict of state properties (shape and type).
        """
        raise NotImplementedError

    @property
    def actions(self):

        #TODO limit to remaining options (no repeats)
        #TODO set up to have the hkls, so it can be generalized
        return dict(num_actions=len(self.actions), type='int')


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
