import os,sys;sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
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
import bumps.lsqerror as lsqerr

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

    #Make a cell
    cell = Mod.makeCell(crystalCell, spaceGroup.xtalSystem)

    #Define a model
    m = S.Model([], [], backg, wavelength, spaceGroup, cell,
                [atomList], exclusions,
                scale=0.06298, error=[],  extinction=[0.0001054])

    #Set a range on the x value of the first atom in the model
    m.atomListModel.atomModels[0].z.value = 0.3
    m.atomListModel.atomModels[0].z.range(0,0.5)

    return m


def fit(model):

    #Create a problem from the model with bumps,
    #then fit and solve it
    problem = bumps.FitProblem(model)

    fitted = fitter.LevenbergMarquardtFit(problem)
    x, dx = fitted.solve()

    return x, dx, problem.chisq(), problem


def learn():

    #Q params
    epsilon = 1
    minEps = 0.01
    epsDecriment = 0.95

    alpha = .01
    gamma = .9

    maxEpisodes = 5000
    maxSteps = len(refList)
    rewards = []
    steps = []
    zvals = []
    chisqs = []

    qtable = np.zeros([len(refList)+1, len(refList)])    #qtable(state, action), first index of state is no data
#    qtable = readQTable()


    for episode in range(maxEpisodes):

        model = setInitParams()
        prevX2 = None

        remainingRefs = []

        for i in range(len(refList)):
            remainingRefs.append(i)

        visited = []
        observed = []
        totReward = 0
        stateIndex = 0

#        file = open("/mnt/storage/qtable-hkl-log-detailed" + str(episode) + ".txt", "a")
#        file.write("HKL Reward TotalReward ChiSq\n")
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

            #Need more data than parameters, have to wait to the second step to fit
            if step > 0:
                x, dx, chisq, prob = fit(model)

                reward -= 1
                if (prevX2 != None and chisq < prevX2):
                    reward = 1/chisq

                qtable[stateIndex, actionIndex] =  qtable[stateIndex, actionIndex] + \
                                                   alpha*(reward + gamma*(np.max(qtable[stateIndex,:])) - \
                                                   qtable[stateIndex, actionIndex])
                prevX2 = chisq

            state = action
            stateIndex = actionIndex+1  #shifted up one for states, so that the first index is no data
            totReward += reward

#            print(str(action.hkl[0])+ " " + str(action.hkl[1]) + " " + str(action.hkl[2]) + "\t" + str(reward) + "\t" + str(totReward) + "\t" + str(prevX2) + "\t" + str(model.atomListModel.atomModels[0].z.value)+ "\n")

            if (prevX2 != None and step > 50 and  chisq < 10):    #stop early if the fit is within certian bounds (i.e, "good enough")
                break

  #      file.close()

        #Decriment epsilon to exploit more as the model learns
        epsilon = epsilon*epsDecriment
        if (epsilon < minEps):
           epsilon = minEps

#        model.plot()

        #Write qtable to a file every ten episodes
        if ((episode % 15) == 0):
            rewards.append(totReward)
            chisqs.append(prevX2)
            zvals.append(model.atomListModel.atomModels[0].z.value)
            steps.append(episode)

        if((episode % 50) == 0):
            file = open("/mnt/storage/qtable-full-run4.txt", "w")
            pickle.dump(qtable, file)
            file.close()

            file = open("/mnt/storage/rewardsLog-qtable-full-run4.txt", "w")
            file.write("episode: " + str(episode))
            file.write(str(rewards[:]))
            file.close()

        if((episode % 500) == 0):

            plt.scatter(steps, rewards)
            plt.xlabel("Episodes")
            plt.ylabel("Reward")
            plt.savefig('/mnt/storage/rewards-qtable-full-training4-reward.png')
            plt.close()

            plt.scatter(steps, chisqs)
            plt.xlabel("Episodes")
            plt.ylabel("Final Chi Squared Value")
            plt.savefig('/mnt/storage/rewards-qtable-full-training4-chi.png')
            plt.close()

            plt.scatter(steps, zvals)
            plt.xlabel("Episodes")
            plt.ylabel("Z Value")
            plt.savefig('/mnt/storage/rewards-qtable-full-training4-z.png')
            plt.close()

def readQTable():

    file = open("/mnt/storage/qtable-full-run3.txt", "r")
    qtable = pickle.load(file)
    file.close()
    return qtable

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


#Graph the chi squared values at different values of the aprameter (Pr: z) and write it to a file
def printChi2():

	cell = Mod.makeCell(crystalCell, spaceGroup.xtalSystem)

	#Define a model
	m = S.Model(tt, sfs2, backg, wavelength, spaceGroup, cell,
        	[atomList], exclusions,
        	scale=0.06298,hkls=refList, error=error,  extinction=[0.0001054])

	z = 0
	xval = []
	y = []
	while (z < 0.5):

	    	#Set a range on the x value of the first atom in the model
    		m.atomListModel.atomModels[0].z.value = z
    		m.atomListModel.atomModels[0].z.range(0, 0.5)
    		problem = bumps.FitProblem(m)
		#    monitor = fitter.StepMonitor(problem, open("sxtalFitMonitor.txt","w"))

        	fitted = fitter.LevenbergMarquardtFit(problem)
    		x, dx = fitted.solve()
    		xval.append(x[0])
    		y.append(problem.chisq())
                print(x, problem.chisq())
    		z += 0.005

	fig = plt.figure()
	mpl.pyplot.plot(xval, y)
	mpl.pyplot.xlabel("Pr z coordinate")
	mpl.pyplot.ylabel("X2 value")
	fig.savefig('/mnt/storage/prnio_chisq_vals_optcfl_lm.png')

#printChi2()


