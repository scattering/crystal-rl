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
                scale=0.06298, error=[],  extinction=[0.0001054])

    #Set a range on the x value of the first atom in the model
    m.atomListModel.atomModels[0].z.range(0.3,0.4)

    return m

def fit(model):

    print("Fitting problem...")

    #Crate a problem from the model with bumps,
    #then fit and solve it
    problem = bumps.FitProblem(model)
#    monitor = fitter.StepMonitor(problem, open("sxtalFitMonitor.txt","w"))

    fitted = fitter.MPFit(problem)
   # x, dx = fitted.solve()
    x, dx = fitted.solve()
    print(problem.getp())
    print(problem.labels())
    print(fitted)
    print(x, dx)
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

        visited = []
        observed = []

        for step in range(maxSteps):

	    reward = 0

            guess = rand.random()
            if (guess < epsilon):
                #Explore: choose a random action from the posibilities
                actionIndex = rand.choice(remainingRefs)
                action = refList[actionIndex]

            else:
                #Exploit: choose best option, based on qtable
                qValue = 0
                for actionIndex in remainingRefs:
                    if (qtable[refList.index(state), hklIndex] > qValue):
                        qValue = qtable[refList.index(state), hklIndex]
                        action = refList[actionIndex]

            #No repeats
            remainingRefs.remove(actionIndex)

            print(action.hkl)
            visited.append(action)

            #Find the data for this hkl value and add it to the model
            print("adding ref")

#                        if model.refList is None:
                            #newList = H.ReflectionList()
                            #newList.set_reflection_list_nref(0)
                            #H.funcs.alloc_reflist_array(newList)
            model.refList = H.ReflectionList(visited)
            print("made reflist")
 #                       else:
  #                          model.refList.append(reflection)

            model._set_reflections()

            model.error.append(error[actionIndex])
#                        model.observed = np.append(model.observed, [sfs2[refsIter.index]])
            model.tt = np.append(model.tt, [tt[actionIndex]])

            model.update()

            observed.append(sfs2[actionIndex])
            model._set_observations(observed)
#                        model.reflections.set_reflections_list_nref(model.reflections.nref()+1)
 #                       model.reflections[reflection]

            model.update()     #may not be necessary
#                      break

 #               except StopIteration:
  #                  break



#            for reflection in refList:        #TODO, should this be adding hkls not refs?
#                if (reflection.hkl == action.hkl):
#                    print("adding ref")
#                    model.reflections.append(reflection)
#                    model.error.append(error[refList.index(reflection)])
#                    model.observed.append(sf2s[refList.index(reflection)])
#                    model.tt.append(tt[refList.index(reflection)])
#                    model.update()     #may not be necessary
#                    break

            print "points", model.numpoints()

	    if (step > 0):        #TODO not necessary
                x, dx = fit(model)
                print(model.error)
   	        reward -= 1
                if (prevDx != None and dx < prevDx):
                    reward += 1

   #             refsIter = refList.__iter__()
    #            while True:
     #               try:
      #                  reflection = refsIter.next()
       #                 if reflection.hkl == action.hkl:
        #                    actionIndex = refsIter.index
         #               if reflection.hkl == state.hkl:
          #                  stateIndex = refsIter.index
           #         except StopIteration:
            #           break

                qtable[stateIndex, actionIndex] =  qtable[stateIndex, actionIndex] + \
                                                   alpha*(reward + gamma*(np.max(qtable[stateIndex,:])) - \
                                                   qtable[stateIndex, actionIndex])
                prevDx = dx

            state = action
            stateIndex = actionIndex
            print("_______________________")
            if (len(remainingRefs) == 0):
                break

        #Decriment epsilon to explote more as the model learns
        epsilon = epsilon*epsDecriment
        if (epsilon < minEps):
           epsilon = minEps

        print(qtable)

learn()
