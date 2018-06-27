from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os,sys;sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import os

import argparse
import time
import logging
import json
import gym

from tensorforce import TensorForceError
from tensorforce.execution import Runner
from tensorforce.agents import DQNAgent
from tensorforce.agents import PPOAgent
from tensorforce.core.preprocessors import Flatten

from pycrysfmlEnvironment import PycrysfmlEnvironment

from tensorforce.contrib.openai_gym import OpenAIGym


def main():

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(
        logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s"))
    logger.addHandler(console_handler)

    parser = argparse.ArgumentParser()

    parser.add_argument('-a', '--agent-config', help="Agent configuration file")
    parser.add_argument('-n', '--network-spec', default=None, help="Network specification file")

    args = parser.parse_args()


    logger.info("Start training")

    #From quickstart on docs
    # Network as list of layers
    #This is from mlp2_embedding_network.json
    network_spec = [
        {
            "type": "dense",
            "size":  32
        },
        {
            "type": "dense",
            "size": 32
        }
    ]

    DATAPATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    observedFile = os.path.join(DATAPATH,r"prnio.int")
    infoFile = os.path.join(DATAPATH,r"prnio.cfl")

    environment = PycrysfmlEnvironment(observedFile, infoFile)


    agent = DQNAgent(
            states=environment.states,
            actions=environment.actions,
            network=network_spec,
            #TODO add in other params
        )


#    pp_flat = Flatten()

    runner = Runner(
        agent=agent,
        environment=environment,
        repeat_actions=1
    )

    def episode_finished(r):
        if r.episode % 100 == 0:
            sps = r.timestep / (time.time() - r.start_time)
            logger.info("Finished episode {ep} after {ts} timesteps. Steps Per Second {sps}".format(ep=r.episode,
                                                                                                    ts=r.timestep,
                                                                                                    sps=sps))
            logger.info("Episode reward: {}".format(r.episode_rewards[-1]))
            logger.info("Episode timesteps: {}".format(r.episode_timestep))
            logger.info("Episode largest tile: {}".format(r.environment.largest_tile))
            logger.info("Average of last 500 rewards: {}".format(sum(r.episode_rewards[-500:]) / 500))
            logger.info("Average of last 100 rewards: {}".format(sum(r.episode_rewards[-100:]) / 100))
        return True

    runner.run(
        timesteps=6000000,
        episodes=1000,
        max_episode_timesteps=200,
        deterministic=True,
        episode_finished=episode_finished
    )

    terminal = False
    state = environment.reset()
 #   processedState = pp_flat.processed_shape(state)
    while not terminal:
        action = agent.act(processedState)
        state, terminal, reward = environment.execute(actions=action)
 #       processedState = pp_flat.process(state)
    environment.print_state()

    runner.close()




def quickStart():

   DATAPATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
   observedFile = os.path.join(DATAPATH,r"prnio.int")
   infoFile = os.path.join(DATAPATH,r"prnio.cfl")

   env = PycrysfmlEnvironment(observedFile, infoFile)


#   env = OpenAIGym('CartPole-v0', visualize=False)
 
   # Network as list of layers
   network_spec = [
       dict(type='dense', size=32, activation='tanh'),
       dict(type='dense', size=32, activation='tanh')
   ]

   agent = PPOAgent(
       states=env.states,
       actions=env.actions,
       network=network_spec,
       batch_size=4096,
       # BatchAgent
       keep_last_timestep=True,
       # PPOAgent
       step_optimizer=dict(
           type='adam',
           learning_rate=1e-3
       ),
       optimization_steps=10,
       # Model
       scope='ppo',
       discount=0.99,
       # DistributionModel
       distributions_spec=None,
       entropy_regularization=0.01,
       # PGModel
       baseline_mode=None,
       baseline=None,
       baseline_optimizer=None,
       gae_lambda=None,
       #  PGLRModel
       likelihood_ratio_clipping=0.2,
       summary_spec=None,
       distributed_spec=None
   )

   # Create the runner
   runner = Runner(agent=agent, environment=env)


   # Callback function printing episode statistics
   def episode_finished(r):
       print("Finished episode {ep} after {ts} timesteps (reward: {reward})".format(ep=r.episode, ts=r.episode_timestep,
                                                                                 reward=r.episode_rewards[-1]))
       return True


   # Start learning
   runner.run(episodes=3000, max_episode_timesteps=200, episode_finished=episode_finished)
   runner.close()

   # Print statistics
   print("Learning finished. Total episodes: {ep}. Average reward of last 100 episodes: {ar}.".format(
      ep=runner.episode,
       ar=np.mean(runner.episode_rewards[-100:]))
   )



if __name__ == '__main__':
    main()

