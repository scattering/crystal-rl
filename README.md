# Reinforcement Learning with Pycrysfml

This is a set of Reinforcement Learning algorithms for use with pycrysfml. They can be used to train Reinforcement Learning agents using Q and Deep Q Learning

## Q Learning Setup

### Pycrysfml

First set up and install pycrysfml. Follow the commands in the instructions file in pycrysfml/doc/Ubuntu_deps.txt

Then, replace pycrysfml/hklgen/sxtal_model.py with the sxtal_model.py file in this repository. This updates the model class to accept empty lists for the observed data, so you can create a model without giving it all of the initial data.

### Code

This code is designed to be run from:

     pycrysfml/hklgen/examples/sxtal/crystal-rl/sxtalQLearning.py

The sxtalQLearning program is a simple Q Learning algorithm for use with the praseodymium nickolate data. Replace the prnio.cfl file in pycrysfml/hklgen/examples/sxtal/prnio.cfl with the prnio_optimized.cfl file
This file contains optimized atomic position values from fitting the data with FullProf. It will allow you to get better fits on the praseodymium nickolate crsytal model when only fitting a single parameter.
Adjust the files to write output to write to your desired file names and run the code with the command:

    python sxtalQLearning.py

The program will write out a log of the rewards earned by the agent over time. It also writes the Q table to a file, so that you can load the Q table and continue training from where you left off.

## Deep Q Learning Setup

### Tensorforce

Tensorforce relies on Tensorflow. Use a docker container with tensorforce to most easily set this up. You need to have nvidia-docker installed for this to work.
Run: 

    $  nvidia-docker run -it tensorflow/tensorflow:latest-gpu /bin/bash

This creates a docker container with tensorflow set up to use the gpu. Within the container, install TensorForce

    $ pip install tensorforce

For more instructions regarding tensorforce setup, see https://github.com/reinforceio/tensorforce

### Pycrysfml

Use above instructions in Q Learning section.

### Code

The training code will import TensorForce and set up an agent, model, and runner to train a model on the environment. You have to pass in an agent configuration.
A Deep Q Learning agent configuration is included here, got to https://github.com/reinforceio/tensorforce/tree/master/examples/configs for more examples. 

Place the files within pycrysfml, or define your filepath at the top of each file so they can find the data files

    pycrysfml/hklgen/examples/sxtal/crystal-rl/pycrysfmlTraining.py
    pycrysfml/hklgen/examples/sxtal/crystal-rl/pycrysfmlEnvironment.py

Run the training:

    python pycrysfmlTraining.py -a dqn_agent.json

Substitute your own agent after the -a tag to use a different one.
