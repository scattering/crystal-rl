import os,sys;sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
import os
from copy import copy
import numpy as np
import random as rand
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

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

#    print("Setting parameters...")

    #Make a cell
    cell = Mod.makeCell(crystalCell, spaceGroup.xtalSystem)

    #Define a model
    m = S.Model([], [], backg, wavelength, spaceGroup, cell,
                [atomList], exclusions,
                scale=0.06298, error=[],  extinction=[0.0001054])

    #Set a range on the x value of the first atom in the model
#    m.atomListModel.atomModels[0].z.fixed = False
#    m.atomListModel.atomModels[0].z.fittable = True
    m.atomListModel.atomModels[0].z.value = 0.3
    m.atomListModel.atomModels[0].z.range(0,0.5)
#    fit(m)

    return m

def fit(model):

    #Create a problem from the model with bumps,
    #then fit and solve it
    problem = bumps.FitProblem(model)
    monitor = fitter.StepMonitor(problem, open("sxtalFitMonitor.txt","w"))

    fitted = fitter.LevenbergMarquardtFit(problem)
    x, dx = fitted.solve(monitors=[monitor])
#    print(problem.getp())
#    print(x, dx)
#    problem.model_update()
#    model.update()

#    print(problem.chisq())
    return x, dx, problem.chisq()

#---------------------------------------
#Q learning methods
#---------------------------------------

def learn():

    #Q params
    epsilon = 1
    minEps = 0.01
    epsDecriment = 0.95

    alpha = .01
    gamma = .9

    maxEpisodes = 100
    maxSteps = len(refList)

    qtable = np.zeros([len(refList)+1, len(refList)])    #qtable(state, action), first index of state is no data

    for episode in range(maxEpisodes):

        model = setInitParams()
        state = 0
        prevX2 = None

        remainingRefs = []
        for i in range(len(refList)):
            remainingRefs.append(i)

        visited = []
        observed = []
        totReward = 0
        stateIndex = 0

        for step in range(maxSteps):

	    reward = 0

            guess = rand.random()
            if (guess < epsilon):
                #Explore: choose a random action from the posibilities
                actionIndex = rand.choice(remainingRefs)
                action = refList[actionIndex]

            else:
                #Exploit: choose best option, based on qtable
                qValue = float('-inf')
                for actionIndex in remainingRefs:
                    if (qtable[stateIndex, actionIndex] > qValue):
                        qValue = qtable[stateIndex, actionIndex]
                        action = refList[actionIndex]
                        break

            #No repeats
            remainingRefs.remove(actionIndex)
            visited.append(action)

            #Find the data for this hkl value and add it to the model
            model.refList = H.ReflectionList(visited)
            model._set_reflections()

            model.error.append(error[actionIndex])
            model.tt = np.append(model.tt, [tt[actionIndex]])

            observed.append(sfs2[actionIndex])
            model._set_observations(observed)
            model.update()

#            for i in range(len(model.observed)):
 #               print(model.reflections[i].hkl , model.observed[i], model.error[i])

            if step > 0:
                x, dx, chisq = fit(model)
                reward -= 1
                if (prevX2 != None and chisq < prevX2):
                    reward += 1.5

                qtable[stateIndex, actionIndex] =  qtable[stateIndex, actionIndex] + \
                                                   alpha*(reward + gamma*(np.max(qtable[stateIndex,:])) - \
                                                   qtable[stateIndex, actionIndex])
                prevX2 = chisq

            state = action
            stateIndex = actionIndex+1  #shifted up one for states, so that the first index is no data
            totReward += reward
#            print(model.nllf())
#            print("_______________________")
            if (prevX2 != None and step > 50 and  chisq < 49):
                break

        #Decriment epsilon to explote more as the model learns
        epsilon = epsilon*epsDecriment
        if (epsilon < minEps):
           epsilon = minEps

        print(x, chisq, totReward, step)
        if ((episode % 10) == 0):
            file = open("qtable.txt", "w")
            file.write("episode: " + str(episode))
            for stateList in qtable:
                file.write(str(stateList[:]))
            file.close()

#        for i in range(len(model.observed)):
#            print(model.reflections[i].hkl , model.observed[i], model.error[i])

#if __name__ == "__main__":
    # program run normally
 #   learn()
#else:
    # called using bumps
#    cell = Mod.makeCell(crystalCell, spaceGroup.xtalSystem)

#    m = S.Model(tt, sfs2, backg, wavelength, spaceGroup, cell,
#            [atomList], exclusions,
#            scale=0.06298,hkls=refList, error=error,  extinction=[0.0001054])

#    Set a range on the x value of the first atom in the model
#    m.atomListModel.atomModels[0].z.range(0, 1)
#    m.atomListModel.atomModels[0].z.value = 0.5

#    problem = bumps.FitProblem(m)


learn()


def printChi2():

	cell = Mod.makeCell(crystalCell, spaceGroup.xtalSystem)
	
	#Define a model
	m = S.Model(tt, sfs2, backg, wavelength, spaceGroup, cell,
        	[atomList], exclusions,
        	scale=0.06298,hkls=refList, error=error,  extinction=[0.0001054])
	
	z = 0
	xval = []
	y = []
	while (z < 1):
	
	    	#Set a range on the x value of the first atom in the model
    		m.atomListModel.atomModels[0].z.value = z
    		m.atomListModel.atomModels[0].z.range(0, 1)
    		problem = bumps.FitProblem(m)
		#    monitor = fitter.StepMonitor(problem, open("sxtalFitMonitor.txt","w"))
	
    	fitted = fitter.MPFit(problem)
    	x, dx = fitted.solve()
    	xval.append(x[0])
    	y.append(problem.chisq())
    	z += 0.005
	
	fig = plt.figure()
	mpl.pyplot.plot(xval, y)
	mpl.pyplot.xlabel("Pr z coordinate")
	mpl.pyplot.ylabel("X2 value")
	fig.savefig('/mnt/storage/prnio_chisq_vals.png')
	


#printChi2()
