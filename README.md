# Reinforcement Learning with Pycrysfml

This is a set of Reinforcement Learning algorithms for use with pycrysfml. They can be used to train Reinforcement Learning agents using Q and Deep Q Learning

## Q Learning Setup

First set up and install pycrysfml. Follow the commands in the instructions file in pycrysfml/doc/Ubuntu_deps.txt

This code is designed to be run from:

     pycrysfml/hklgen/examples/sxtal/crystal-rl/sxtalQLearning.py
     pycrysfml/hklgen/examples/HOBK/simpleQAlg.py

simpleQAlg.py is a draft which was coverted to sxtalQLearning.py, it is not inteded to be functional.

The sxtalQLearning program is a simple Q Learning algorithm for use with the praseodymium nicolate data. Replace the prnio.cfl file in pycrysfml/hklgen/examples/sxtal/prnio.cfl with the prnio_optimized.cfl file
This file contains optimized atomic position values from fitting the data with FullProf. It will allow you to get better fits on the praseodymium nicolate crsytal model when only fitting a single parameter.
Adjust the files to write output to write to your desired file names and run the code with the command:

    python sxtalQLearning.py

The program will write out a log of the rewards earned by the agent over time. It also writes the Q table to a file, so that you can load the Q table and continue training from where you left off.

