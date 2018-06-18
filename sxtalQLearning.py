import os,sys;sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
import os
from copy import copy
import numpy as np
import random as rand

import fswig_hklgen as H
import hkl_model as Mod
import sxtal_model as S

import  bumps.names  as bumps
import bumps.fitters as fitter
from bumps.formatnum import format_uncertainty_pm

#Simple Q learning algorithm to optimize a single parameter
#Will determine the optimal order of measurements to make
#to optimize the given parameter

np.seterr(divide="ignore",invalid="ignore")

#Set data files
DATAPATH = os.path.dirname(os.path.abspath(__file__))
backgFile = None
observedFile = os.path.join(DATAPATH,r"../prnio.int")
infoFile = os.path.join(DATAPATH,r"../prnio.cfl")

#Read data
spaceGroup, crystalCell, atomList = H.readInfo(infoFile)
# return wavelength, refList, sfs2, error, two-theta, and four-circle parameters
wavelength, refList, sfs2, error = S.readIntFile(observedFile, kind="int", cell=crystalCell)
tt = [H.twoTheta(H.calcS(crystalCell, ref.hkl), wavelength) for ref in refList]
backg = None
exclusions = []

def setInitParams():

    print("Setting parameters...")

    #Make a cell
    cell = Mod.makeCell(crystalCell, spaceGroup.xtalSystem)

    print(error)

    #Define a model
    m = S.Model([], [], backg, wavelength, spaceGroup, cell,
                [atomList], exclusions,
                scale=0.06298, hkls=refList,error=[],  extinction=[0.0001054])

    #Set a range on the x value of the first atom in the model
#    m.atomListModel.atomModels[0].z = .8
    m.atomListModel.atomModels[0].z.range(0.4,1)

    return m


def fit(model):

    print("Fitting problem...")

    #Crate a problem from the model with bumps,
    #then fit and solve it
    problem = bumps.FitProblem(model)
    fitted = fitter.MPFit(problem)
    x, dx = fitted.solve()

    print(problem.labels())
    print(problem.chisq())
    problem.model_update()
    model.update()

    print(problem.chisq())
    return x, dx

#---------------------------------------
#Q learning methods
#---------------------------------------

def learn():

    #Q params
    epsilon = 1
    minEps = 0.01
    epsDecriment = 0.99

    alpha = .01
    gamma = .9

    maxEpisodes = 1

    maxSteps = len(refList)

    qtable = np.zeros([len(refList), len(refList)])    #qtable(state, action)

    for episode in range(maxEpisodes):

        model = setInitParams()
        state = 0
        prevDx = None

        remainingRefs = []
        for i in range(len(refList)):
            remainingRefs.append(i)

        for step in range(maxSteps):

	    reward = 0

            guess = rand.random()
            if (guess < epsilon):
                #Explore: choose a random action from the posibilities
                actionIndex = rand.choice(remainingRefs)
                action = refList[actionIndex]
                print(action)
            else:
                #Exploit: choose best option, based on qtable
                qValue = 0
                for hklIndex in remainingRefs:
                    if (qtable[refList.index(state), hklIndex] > qValue):
                        qValue = qtable[refList.index(state), hklIndex]
                        action = refList[hklIndex]

            #No repeats
#            print(action, refList.index(action))
#            del remainingRefs[refList.index(action)]

            print(action)
            print(refList[0].hkl)
#            print(refList.reflections())
            #Find the data for this hkl value and add it to the model
            refsIter = refList.__iter__()
            while True:
                try:
                    reflection = refList.next()
                    if (reflection.hkl == action.hkl):
                        print("adding ref")
#                        model.reflections.append(reflection)

#                        model.reflections.set_reflection_list_nref(model.reflections.nref+1)
#                        model.reflections.__setitem__(model.reflections.nref-1, reflection)
                        model.error.append(error[refsIter.index])
                        model.observed = np.append(model.observed, [sfs2[refsIter.index]])
                        model.tt = np.append(model.tt, [tt[refsIter.index]])
                        model.update()     #may not be necessary
                        break

                except StopIteration:
                    break



#            for reflection in refList:        #TODO, should this be adding hkls not refs?
#                if (reflection.hkl == action.hkl):
#                    print("adding ref")
#                    model.reflections.append(reflection)
#                    model.error.append(error[refList.index(reflection)])
#                    model.observed.append(sf2s[refList.index(reflection)])
#                    model.tt.append(tt[refList.index(reflection)])
#                    model.update()     #may not be necessary
#                    break

            print(model.numpoints())
            print(model.reflections)
            print(model.error)
            print("_____________________")

	    if (step > 1):        #TODO not necessary
                x, dx = fit(model)

   	        reward -= 1
                if (prevDx != None and dx < prevDx):
                    reward += 1

                qtable[referenceHkls.index(state), referenceHkls.index(action)] =  qtable[referenceHkls.index(state), referenceHkls.index(action)] + \
                                                                                   alpha*(reward + gamma*(np.max(qtable[referenceHkls.index(state),:])) - \
                                                                                   qtable[referenceHkls.index(state), referenceHkls.index(action)])
                prevDx = dx

            state = action

            if (len(remainingRefs) == 0):
                break

        #Decriment epsilon to explote more as the model learns
        epsilon = epsilon*epsDecriment
        if (epsilon < minEps):
           epsilon = minEps



learn()
