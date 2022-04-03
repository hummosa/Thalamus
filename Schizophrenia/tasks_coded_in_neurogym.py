#tasks coded in neurogym
#%%
import numpy as np
rng = np.random.default_rng()
import matplotlib.pyplot as plt
import neurogym as ngym
import gym
from neurogym import TrialEnv
import sys; sys.path.insert(0, '.')
from utils import stats

class NoiseyMean(TrialEnv):
    def __init__(self, mean_noise= 0.1, mean_drift = 0, odd_balls_prob = 0.0, change_point_prob = 0.0, safe_trials = 5):
        super().__init__()
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(1,))#, name = ['outcome'])
        self.latent_space = gym.spaces.Box(low=0.0, high=1.0, shape=(1,))#, names = ['mean'])
        self.action_space = gym.spaces.Box(low=0.0, high=1.0, shape=(1,))
        
        #get initial random values
        self.outcome_now = self.observation_space.sample()
        self.mean_now = self.latent_space.sample()

        self.trial_len = 100
        self.far_definition = 0.3 *self.observation_space.__getattribute__('high')

        self.mean_drift = mean_drift 
        self.odd_balls_prob =odd_balls_prob 
        self.mean_noise= mean_noise
        self.change_point_prob = change_point_prob
        self.safe_trials = safe_trials

    def _new_trial(self):
        self.outcome_now = self.observation_space.sample()
        self.mean_now = self.latent_space.sample()
        outcomes, means = [], []
        oddballs = np.zeros(self.trial_len)
        changepoints = np.zeros(self.trial_len)
        s = self.safe_trials
        for i in range(self.trial_len):
            
            self.outcome_now = rng.normal(self.mean_now, self.mean_noise)
            while((self.outcome_now < self.observation_space.__getattribute__('low')) or (self.outcome_now > self.observation_space.__getattribute__('high'))):
                self.outcome_now = rng.normal(self.mean_now, self.mean_noise)
            self.mean_now = rng.normal(self.mean_now, self.mean_drift)
            self.outcome_now = np.clip(self.outcome_now, 0, 1)
            self.mean_now = np.clip(self.mean_now, 0, 1)
            
            if s == 0: # safety period over
                if rng.uniform() < self.odd_balls_prob:
                    oddballs[i] = 1.0
                    #ensure oddball is far enough from currently mean:
                    far_enough = False
                    while (not far_enough):
                        self.outcome_now = self.observation_space.sample()
                        far_enough = abs(self.outcome_now - self.mean_now) > self.far_definition
                    s = self.safe_trials
                
                if rng.uniform() < self.change_point_prob:
                    changepoints[i] = 1.0
                    far_enough = False
                    while(not far_enough):
                        mean_next = self.latent_space.sample()
                        far_enough = abs(mean_next - self.mean_now) > self.far_definition
                    self.mean_now = mean_next
                    s = self.safe_trials
            else:
                s = s-1
            outcomes.append(self.outcome_now)
            means.append(self.mean_now)
        
        ob = self.outcome_now  # observation previously computed
        # Sample observation for the next trial
        self.next_ob = np.random.uniform(0, 1, size=(1,))
        
        trial = dict()
        # Ground-truth is 1 if ob > 0, else 0
        trial['outcomes'] = np.stack(outcomes)
        trial['means'] = np.stack(means)
        trial['oddballs'] = oddballs
        trial['changepoints'] = changepoints
        trial['ground_truth'] = np.stack(outcomes)

        return trial
    
    def _step(self, action):
        ob = self.next_ob
        # If action equals to ground_truth, reward=1, otherwise 0
        reward = (action == self.trial['ground_truth']) * 1.0
        done = False
        info = {'new_trial': True}
        return ob, reward, done, info


class Shrew_task(TrialEnv):
    def __init__(self, dt=10, attend_to = 'either', context= 1, no_of_coherent_cues = None, timing=None):
        super().__init__(dt=dt)  # dt is passed to base task
        
        # Setting default task timing
        self.timing = {'cues': 200, 'delay': 50, 'stimulus': 30, 'decision': 20}
        self.trial_length = sum(self.timing.values())
        self.context = context
        self.no_of_coherent_cues = no_of_coherent_cues
        self.total_cues = 10
        # Update timing if provided externally
        if timing:
            self.timing.update(timing)
        
        self.attend_to = attend_to # restrict trials to only one modality or both  {audition, vision, or either}.
        # Here we use ngym.spaces, which allows setting name of each dimension
        self.observation_space = ngym.spaces.Box(
            low=0., high=1., shape=(6,), name={'stimulus': [0,1,2,3], 'cues': [4,5]})
        # self.cues_space = ngym.spaces.Box(
            # low=-1., high=1., shape=(2,), name={'cues': [0,1]})
        name = { 'choice': [0,1,2,3]}
        # name = { 'choice': [[0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1],]} #Turining this off because passing gt as index is the norm for neurogym.
        self.action_space = ngym.spaces.Discrete(1, name=name) # still self.action_space.shape returns shape () and that does not allow adding gt in R3

    def _new_trial(self):
        # Setting time periods for this trial
        periods = ['cues', 'delay', 'stimulus', 'decision']
        # Will add stimulus and decision periods sequentially using self.timing info
        self.add_period(periods)
        
        if self.attend_to == 'either':
            modality = np.choose(1, ['audition', 'vision'])
        else:
            modality = self.attend_to
            
        # Sample observation for the next trial
        cues = np.zeros(shape=(self.total_cues) ) 
        if self.no_of_coherent_cues is None:
            sampled_no_of_coherent_cues = rng.integers(self.total_cues//2, self.total_cues )
        else:
            sampled_no_of_coherent_cues = self.no_of_coherent_cues
        # print(sampled_no_of_coherent_cues)
        cues[:sampled_no_of_coherent_cues] = np.ones(sampled_no_of_coherent_cues)
        
        #shuffle cues
        idx = rng.permutation(range(len(cues)))
        cues = (cues[idx])
        if modality == 'audition':
            cues = np.vstack([cues, 1-cues, ]) # assign dominant cues to audition
        else:
            cues = np.vstack([1-cues, cues, ]) # assign dominant cues to vision

        cues_ones = np.ones(shape=(2, int(2*cues.shape[1]))) 
        cues_ones[:,::2] = cues # interperse cues with wideband noise.
        # cues = cues_ones.reshape((2,20,1)).repeat(10, axis = 2).reshape(2, -1) # some kungfoo to make each cue last for 10ms. Not relaly sure this is necessary
        cues = cues_ones.T
                
        stimulus = np.zeros(4)
        bin_a, bin_v = np.random.binomial(size=2, n=1, p= 0.5)
        stimulus[:2] = np.array([bin_a, 1-bin_a]) 
        stimulus[2:] = np.array([bin_v, 1-bin_v]) 
        
        # Add value 1 to stimulus period at fixation location
        self.add_ob(stimulus, period='stimulus', where='stimulus')
        # Add cues to cues period at cues location
        self.add_ob(cues, period='cues', where='cues')
        
        # Set ground_truth
        audition = 1.* (modality == 'audition') if (self.context == 1) else (1.*(modality=='vision')) # If context 2 flip the ground truth to the other context. 
        # groundtruth = np.concatenate([np.array(audition).reshape([1]), stimulus[:2].T if audition else stimulus[2:].T, ]) # choice is to report type of trial (audition (1) or not (0), and then the right choice of stimulus side  )
        groundtruth = np.concatenate([np.array(audition).reshape([1]), stimulus[:2].T if modality else stimulus[2:].T, ]) # choice is to report type of trial (audition (1) or not (0), and then the right choice of stimulus side  )
        # set up ground truth vector as [audition (1), left, right]
        gt_index = [ np.all(row == groundtruth) for row in [[0., 1., 0.], [0., 0., 1.], [1., 1., 0.], [1., 0., 1.]] ] #Turining this off because passing gt as index is the norm for neurogym.
        self.set_groundtruth(np.argwhere(gt_index).squeeze(), period='decision', where='choice')  
        # see note above. Self.action_space.shape still returns () which does not allow adding 3 dim for ground truth.

        trial = dict()
        trial['stimulus'] = stimulus
        trial['cues'] = cues
        trial['ground_truth'] = groundtruth
        
        return trial
    
    def _step(self, action):
        # self.ob_now and self.gt_now correspond to
        # current step observation and groundtruth

        # If action equals to ground_truth, reward=1, otherwise 0
        reward = (action == self.gt_now) * 1.0
        
        done = False
        # By default, the trial is not ended
        info = {'new_trial': False}
        return self.ob_now, reward, done, info

# env = Shrew_task(attend_to='either', no_of_coherent_cues=8)
# t = env.new_trial()
# ob_size = env.observation_space.shape[0]
# act_size = env.action_space.n
# print('ob_size :', ob_size, '   act_size: ', act_size)
# print(env.ob)
# print(env.gt)