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
            wavelength, refList, sfs2, error = S.readIntFile(observedFile, kind="int", cell=crystalCell)
            self.wavelength = wavelength
            self.actions = refList
            self.sfs2 = sfs2
            self.error = error
            self.tt = [H.twoTheta(H.calcS(crystalCell, ref.hkl), wavelength) for ref in refList]
            self.backg = None
            self.exclusions = []

        #TODO: else, for powder data

        self.state = []
        for i in range(len(self.actions)):
     
            for subset in itertools.combinations(self.actions, i):
                self.append(subset)

        print(len(self.state))


        reset()

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
        self.prevChiSq = None

        return self.state

    def execute(self, action):

        #TODO check action type, assuming index of action list

        #No repeats
        self.remainingRefs.remove(action)
        self.visited.append(self.actions(action))

        #Find the data for this hkl value and add it to the model
        self.model.refList = H.ReflectionList(visited)
        self.model._set_reflections()

        self.model.error.append(self.error[actionIndex])
        self.model.tt = np.append(self.model.tt, [self.tt[actionIndex]])

        self.observed.append(sfs2[actionIndex])
        self.model._set_observations(observed)
        self.model.update()

        #Need more data than parameters, have to wait to the second step to fit
        if len(visited) > 0:

            x, dx, chisq = fit(self.model)

            reward = -1
            if (self.prevChiSq != None and chisq < prevChiSq):
                reward += 1.5

            self.prevChiSq = chisq

        self.totReward += reward

        if (len(self.remaininRefs) == 0):
            terminal = True
        else:
            terminal = False

        return states(), terminal, reward

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




DATAPATH = os.path.dirname(os.path.abspath(__file__))
observedFile = os.path.join(DATAPATH,r"prnio.int")
infoFile = os.path.join(DATAPATH,r"prnio.cfl")
__init__(observedFile, infoFile)
