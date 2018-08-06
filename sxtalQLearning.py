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
import bumps.lsqerror as lsqerr
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
infoFile = os.path.join(DATAPATH,r"../prnio_optimized.cfl")

#Read data
spaceGroup, crystalCell, atomList = H.readInfo(infoFile)
# return wavelength, refList, sfs2, error, two-theta, and four-circle parameters
wavelength, refList, sfs2, error = S.readIntFile(observedFile, kind="int", cell=crystalCell)
tt = [H.twoTheta(H.calcS(crystalCell, ref.hkl), wavelength) for ref in refList]
backg = None
exclusions = []

qtable = []

#Set up initial model parameters
def setInitParams():

    #Make a cell
    cell = Mod.makeCell(crystalCell, spaceGroup.xtalSystem)

    #Define a model
    m = S.Model([], [], backg, wavelength, spaceGroup, cell,
                [atomList], exclusions,
                scale=0.06298, error=[],  extinction=[0.0001054])

    #Set a range on the x value of the first atom in the model
    #Replace this code with references to the parameters you would like to fit the model to
    m.atomListModel.atomModels[0].z.value = 0.3
    m.atomListModel.atomModels[0].z.range(0,0.5)

    return m

def fit(model):

    #Create a problem from the model with bumps,
    #then fit and solve it
    problem = bumps.FitProblem(model)
    fitted = fitter.LevenbergMarquardtFit(problem)
    x, dx = fitted.solve()

    #return:
    # x: parameter values
    # dx: uncertainty on parameters
    # chi squared value for model
    return x, dx, problem.chisq()

#Train the Q Learning Agent
def learn():

    #Epsilon annulement
    epsilon = 1
    minEps = 0.01
    epsDecriment = 0.95

    #Q Learning Parameters
    alpha = .01
    gamma = .9

    #Training Parameters
    maxEpisodes = 5000
    maxSteps = len(refList)
    rewards = []
    steps = []
    zvals = []
    chisqs = []

    qtable = np.zeros([len(refList)+1, len(refList)])    #qtable(state, action), first index of state is no data
    #qtable = readQTable()    --- use this line instead to read in a pre-trained table

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

        #Code to log hkl paths
#        file = open("/mnt/storage/agentPath_trained_" + str(episode) +".txt", 'w') 
#        file.write("HKL value\tReward\tTotal Reward\tChi Squared Value\n")

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

            chisq = 0
            #Need more data than parameters, have to wait to the second step to fit
            if step > 0:
                x, dx, chisq, prob = fit(model)

                reward -= 1
                if (prevX2 != None and chisq < prevX2):
                    reward = 1/chisq

                #Update the Q table
                qtable[stateIndex, actionIndex] =  qtable[stateIndex, actionIndex] + \
                                                   alpha*(reward + gamma*(np.max(qtable[stateIndex,:])) - \
                                                   qtable[stateIndex, actionIndex])
                prevX2 = chisq

                #print (str(action.hkl) + ":\t" +str(chisq) + "\t" + str(totReward)+ "\n")

            state = action
            stateIndex = actionIndex+1  #shifted up one for states, so that the first index is no data
            totReward += reward
#            hkl = str(action.hkl[0]) + " " + str(action.hkl[1]) + " " + str(action.hkl[2])
#            file.write(hkl + "\t" +str(reward) + "\t" + str(totReward)+ "\t" + str(chisq) + "\n")

            if (prevX2 != None and step > 50 and  chisq < 1):    #stop early if the fit is within certian bounds (i.e, "good enough")
                break

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

if __name__ == "__main__":
    # program run normally
    learn()

#Graph the chi squared values at different values of the aprameter (Pr: z) and write it to a file
def fitFullModel():

	cell = Mod.makeCell(crystalCell, spaceGroup.xtalSystem)

	#Define a model
	m = S.Model(tt, sfs2, backg, wavelength, spaceGroup, cell,
        	[atomList], exclusions,
        	scale=0.06298,hkls=refList, error=error,  extinction=[0.0001054])

        m.u.range(0,2)
        m.zero.pm(0.1)
        m.v.range(-2,0)
        m.w.range(0,2)
        m.eta.range(0,1)
        m.scale.range(0,10)
        m.base.pm(250)

        for atomModel in m.atomListModel.atomModels:
            atomModel.x.pm(0.1)
            atomModel.z.pm(0.1)
            if (atomModel.atom.multip == atomModel.sgmultip):
                # atom lies on a general position
                atomModel.x.pm(0.1)
                atomModel.y.pm(0.1)
                atomModel.z.pm(0.1)

	m.atomListModel.atomModels[0].z.value = 0.3
	m.atomListModel.atomModels[0].z.range(0,0.5) 

   	problem = bumps.FitProblem(m)
        fitted = fitter.SimplexFit(problem)
    	x, dx = fitted.solve(steps=50)
        problem.model_update()

        print(problem.summarize())
        print(x, dx, problem.chisq())


def plotSfs2():

	cell = Mod.makeCell(crystalCell, spaceGroup.xtalSystem)
	m = S.Model(tt, sfs2, backg, wavelength, spaceGroup, cell,
        	[atomList], exclusions,
        	scale=0.06298,hkls=refList, error=error,  extinction=[0.0001054])
	m.atomListModel.atomModels[0].z.range(0,0.5)

        x = sfs2
        y = m.theory()

        plt.scatter(x, y)
        plt.savefig('sfs2s.png')

def graphError():

	cell = Mod.makeCell(crystalCell, spaceGroup.xtalSystem)
	m = S.Model(tt, sfs2, backg, wavelength, spaceGroup, cell,
        	[atomList], exclusions,
        	scale=0.06298,hkls=refList, error=error,  extinction=[0.0001054])
	m.atomListModel.atomModels[0].z.range(0,0.5)

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

	fig = plt.figure()#
	mpl.pyplot.scatter(xval, yval)
	mpl.pyplot.xlabel("Pr z coordinate")
	mpl.pyplot.ylabel("X2 value")
	fig.savefig('/mnt/storage/prnio_chisq_vals_optcfl_lm.png')

