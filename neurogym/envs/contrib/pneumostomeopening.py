#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import math

import neurogym as ngym
from neurogym import spaces

# TODO: Move to collection
class Pneumostomeopening(ngym.BaseEnv):
    metadata = {
        'paper_link': 'https://jeb.biologists.org/content/199/3/683.long',
        'paper_name': '''Operant conditioning of aerial respiratory behaviour
        in Lymnaea stagnalis''',
        'tags': ['operant conditioning', 'learning', 'Lymnaea Stagnalis',
                 'associative learning', 'aerial respiratory behaviour',
                 'hypoxia', 'molluscan model', 'system', 'snail']
    }

    def __init__(self, dt=100, rewards=None, timing=None, sigma=1.0,
                 extra_input_param=None):
        super().__init__(dt=dt)

        self.t = 0

        ## action and observation spaces:
        ## action space
        ## breathing actions: [breathing pneumostome, breathing skin]
        ## swim actions: [up, down, left, right]
        ## make it 3D (forward/backward) or 2D is enough ?
        ## observation space
        ## Discretizing beaker into level of depth
        name = {
            'depth1': 0,
            'depth2': 1,
            'depth2': 2,
            'depth2': 3}
        self.observation_space = spaces.Discrete(4, name=name)
        name = {
            'breathingpneumostome': 0,
            'breathingskin': 1,
            'up': 2,
            'down': 3}
        self.action_space = spaces.Discrete(4, name=name)
        # self.action_space = spaces.MultiDiscrete([(0, 2), (0, 2)])

        ## TODO: can do breathingpneumostome only when at the surface ==depth 1

        ## task specific variables
        self.oxygen_level = 10

    def _get_new_oxygen_level(self, action):
        """
        Update oxygen level of the agent based on action. Follow an
        exponential decay.
        """
        decay_constant = 1

        if action == 0: #breathingpneumostome
            self.oxygen_level += 5
        elif action == 1: #breathingskin
            self.oxygen_level +=1
        else : #swimming actions
            self.oxygen_level = self.oxygen_level * math.exp(-decay_constant * self.t)
        self.oxygen_level = int(round(self.oxygen_level))
        print('oxygen_level: {}'.format(self.oxygen_level))
        return self.oxygen_level


    def phase(self, phase, **kwargs):
        """
        different phase of training.
        set_default_behavior trains the network so that breathing behavior corresponds
        to the default states of the animal where breathing through the peumostome
        is a common occurence.
        Training_session train the network for operant-conditioning of respiratory
        behaviour pattern. Simulating poking the pneumostome as a negative reward.
        Testing_session test the respiratory behaviour change after training_session
        has been run.

        """
        ## Rewards
        if phase == 'set_default_behavior':
            rewards = {
                'swim': +0.1,  # reward given for anything else
                'skin': +0.5,  # reward given when breathing through the skin
                'pneumostome': +1.  # reward given when breathing through the pneumostome
                #'miss': # to force some type of trial duration limit if implementing trials in this task ?
            }
        elif phase == 'training_session':
            rewards = {
                'swim': +0.1,
                'skin': +1.,
                'pneumostome': -1.
            }
        elif phase == 'testing_session':
            rewards = {
                'swim': 0.,
                'skin': 0.,
                'pneumostome': 0.
            }
        return rewards

    def reset(self):
        self.agent_pos = np.random.randint(7)
        self.done = False
        return self.agent_pos

    def step(self, action):
        new_trial = False

        # define the phase of the behavior
        self.rewards = self.phase('set_default_behavior')

        # update reward depending on action and phase of behavioral training
        # update agent position
        if action == 0: #breathingpneumostome
            self.reward = self.rewards['pneumostome']
            self.agent_pos = self.agent_pos
        elif action == 1: #breathingskin
            self.reward = self.rewards['skin']
            self.agent_pos = self.agent_pos
        else: #up #down
            self.reward = self.rewards['swim']
            if action == 2:
                self.agent_pos = self.agent_pos + 1
            if action == 3:
                self.agent_pos = self.agent_pos - 1

        self.agent_pos = np.clip(self.agent_pos, 0, len(self.observation_space.name))

        # updating oxygen level according to action taken
        self.oxygen_level = self._get_new_oxygen_level(action)

        # if oxygen level drop at 0 or below then end experiment
        if self.oxygen_level <= 0:
            self.done = True

        self.t = self.t + 1

        ## potential problems:
        ## - limit number of consecutive pneumostome opening using t
        ## - limit number of consecutive breathing event so that not only do that ?
                ## by using breathing even only when below threshold of O2 ?
                ## or use refractory period ?

        print([self.agent_pos], self.reward, self.done, {'new_trial': new_trial})
        return np.array([self.agent_pos]), self.reward, self.done, {'new_trial': new_trial}

    def render(self,mode='human'):
        pass

    def close(self):
        pass



if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from neurogym.tests import test_run, test_speed
    env = Pneumostomeopening()
    # test_run(env)
    # test_speed(env)
    from neurogym.utils.plotting import plot_env
    from neurogym.utils.test_plotting import test_plot
    # ngym.utils.plotting.run_env(env, num_steps=200)
    plot = plot_env(env, num_steps=100)
    plt.show(plot)
    # test_plot(env)

    # ob = data['ob']
    # ob.shape
    # actions = data['actions']
    #
    # ngym.utils.plotting.fig_(ob, actions, env=env)
    #
    # ob = np.array(ob)
    # actions = np.array(actions)
