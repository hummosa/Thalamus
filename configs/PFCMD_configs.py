
import os
import sys
from models import MODES
import gym

'''
Class names of configs are based on the class names of models:
    Base -> BaseConfig
    EWC  -> EWCConfig
'''

class BaseConfig(object):
    def __init__(self, args= []):
        # system
        self.device = 'cpu'
        self.ROOT_DIR = os.getcwd()


        self.env_kwargs = {'dt': 100}
        self.print_every_batches =  10
        
        import neurogym as ngym
        # self.tasks = ngym.get_collection('yang19')
        # This is the golden sequence, hand tuned curriculum to finish in the least no of trials
        self._tasks= [
                    'yang19.go-v0',
                    'yang19.rtgo-v0',
                    'yang19.dlygo-v0',
                    'yang19.dm1-v0',
                    'yang19.ctxdm1-v0',
                    'yang19.dms-v0',
                    'yang19.dmc-v0',
                    'yang19.dm2-v0',
                    'yang19.ctxdm2-v0',
                    'yang19.multidm-v0',
                    'yang19.rtanti-v0',
                    'yang19.anti-v0',
                    'yang19.dlyanti-v0',
                    'yang19.dnms-v0',
                    'yang19.dnmc-v0',
                    ] 
        # self._tasks += ['yang19.dlydm1-v0', 'yang19.dlydm2-v0', 'yang19.ctxdlydm1-v0', 'yang19.ctxdlydm2-v0', 'yang19.multidlydm-v0']
        self._tasks_id_name = [(i, self.tasks[i]) for i in range(len(self.tasks))]
        # Moving this line below as other things need defined first
        # self.tasks = self._tasks
        # This is yyyyya protected property to maintain the task_id no associated with each task based on this "standard" ordering
        self.human_task_names = ['{:<6}'.format(tn[7:-3]) for tn in self.tasks] #removes yang19 and -v0
        self.num_of_tasks = len(self.tasks)


        self.GoFamily = ['yang19.dlygo-v0', 'yang19.go-v0']
        self.AntiFamily = ['yang19.dlyanti-v0', 'yang19.anti-v0']
        self.DMFamily = ['yang19.dm1-v0', 'yang19.dm2-v0', 'yang19.ctxdm1-v0', 'yang19.ctxdm2-v0', 'yang19.multidm-v0']
        self.DMFamily += ['yang19.dlydm1-v0', 'yang19.dlydm2-v0', 'yang19.ctxdlydm1-v0', 'yang19.ctxdlydm2-v0', 'yang19.multidlydm-v0']
        self.MatchFamily = ['yang19.dms-v0', 'yang19.dmc-v0', 'yang19.dnms-v0', 'yang19.dnmc-v0']
        
        # MD
        self.md_dt = 0.001

        self.use_lstm = False

        self.env_kwargs = {'dt': 100}
        self.batch_size = 1

        #  training paradigm
        self.max_trials_per_task = 40000
        self.use_multiplicative_gates = True 
        self.use_additive_gates = False 
        self.train_to_criterion = True
        self.criterion = 0.98
        self.accuracy_momentum = 0.6    # how much of previous test accuracy to keep in the newest update.
        self.criterion_DMfam = 0.86
        
        self.same_rnn = True
        self.no_shuffled_trials = 40000
        self.paradigm_shuffle = False
        self.paradigm_sequential = not self.paradigm_shuffle

        # RNN model
        self.input_size = 33
        self.output_size = 17
        self.tau= 200
        self.lr = 1e-3

        #gates statis
        self.load_corr_gates = False
        self.MDeffect_mul = True # turns on or off the multiplicative MD effects
        self.MDeffect_add = False # turns on or off the additive MD effects
        self.train_gates = False
        self.gates_sparsity = 0.5
        self.gates_mean = 1.2
        self.gates_std = 0.05
        self.gates_gaussian_cut_off = -0.3
        self.MD2PFC_prob = 0.5
        # test & plot
        self.test_every_trials = 500
        self.test_num_trials = 1
        self.plot_every_trials = 4000
        self.args= args

################### New stuff from CL_neurogym    
        self.max_trials_per_task = 10000

        self.hidden_size = 600

        self.hidden_ctx_size = 400 # 450
        # self.sub_size = 200 # 150
        self.sub_active_size = 50 # this config is deprecated right now
        self.sub_active_prob = 0.50
        self.hidden_ctx_noise = 0.01
        # MD
        self.MDeffect = True
        self.md_active_size = 1
        self.md_dt = 0.001
        self.MDtoPFC_connect_prob = 1.00 # original 1.00

        self.tasks = self._tasks
        

        # save variables
        self.FILENAME = {
                        'config':    'config_PFCMD.npy',
                        'log':       f'log_PFCMD_{self.num_of_tasks}.npy',
                        'net':       'net_PFCMD.pt',
                        'plot_perf': f'performance_PFCMD_tasks{self.num_of_tasks}.png',
        }

    ############################################
        self.save_trained_model = True
        self.save_model = False
        self.load_saved_rnn1 = False 
        self.load_trained_cog_obs = False
        self.save_detailed = False

        self.use_rehearsal = False
        self.use_gates = False
        self.load_gates_corr = False
        self.use_cognitive_observer = False
        self.use_CaiNet = False

        self.train_cog_obs_only = False
        self.train_cog_obs_on_recent_trials = False
        self.use_md_optimizer = False  
        self.abandon_model = False
    ############################################   
    #  
    @property
    def tasks(self):
        return self._tasks
    @tasks.setter
    def tasks(self,tasks):
        self._tasks = tasks
        self.tasks_id_name = tasks # uses the property setter below to rearrange tasks_id_name to the new order but keep consistent the task ids
        self.num_of_tasks = len(tasks)
        # self.total_trials = int(self.max_trials_per_task * (self.num_of_tasks + 1)) # 70000
        self.sub_size = self.hidden_ctx_size//self.num_of_tasks # 150
        self.md_size = self.num_of_tasks # 3


    @property
    def tasks_id_name(self):
        return self._tasks_id_name
    @tasks_id_name.setter
    def tasks_id_name(self,tasks):
        new_task_ids = []
        for i in range(len(tasks)):
            new_task_ids.append( *[t[0] for t in self._tasks_id_name if t[1] == tasks[i]] )
        self._tasks_id_name = [(s,t) for s,t in zip(new_task_ids, tasks)]
        


    def set_strings(self, exp_name):
        self.exp_name = exp_name
        self.exp_signature = self.exp_name +f'_{self.args}_'+\
        f'{"same_rnn" if self.same_rnn else "separate"}_{"mul" if self.use_multiplicative_gates else "add"}'+\
        f'_{"tc" if self.train_to_criterion else "nc"}'

    def update(self, new_config):
        self.__dict__.update(new_config.__dict__)

    def __str__(self):
        return str(self.__dict__)

#################### various experiments :
class Gates_mul_config(BaseConfig):
    def __init__(self, args= []):
        super().__init__()
        self.same_rnn = True
        self.train_to_criterion = True
        self.use_rehearsal = True
        self.random_rehearsals = 0
        self.use_multiplicative_gates = True
        self.use_additive_gates = False

class Gates_add_config(Gates_mul_config):
    def __init__(self, args= []):
        super().__init__()
        self.use_multiplicative_gates = False
        self.use_additive_gates = True

class Shuffle_mul_config(Gates_mul_config):
    def __init__(self, args= []):
        super().__init__()
        self.train_to_plateau = False
        self.random_rehearsals = 300
        self.max_trials_per_task =     self.batch_size
        self.paradigm_shuffle = True
        self.paradigm_sequential = not     self.paradigm_shuffle
class Shuffle_add_config(Shuffle_mul_config):
    def __init__(self, args= []):
        super(Shuffle_add_config, self).__init__()
        self.use_multiplicative_gates = False
        self.use_additive_gates = True

class Gates_no_rehearsal_config(Gates_mul_config):
    def __init__(self, args= []):
        super(Gates_no_rehearsal_config, self).__init__()
        self.use_rehearsal = False

############################################



class PFCMDConfig(BaseConfig):
    def __init__(self):
        super(PFCMDConfig, self).__init__()
        import itertools
        unique_combinations = []
        permut = itertools.permutations(self.tasks, 2)
        for comb in permut:
            unique_combinations.append(comb)
        import numpy as np
        rng = np.random.default_rng(1)
        idx = rng.permutation(range(len(unique_combinations)))

        self.all_random_pairs = (np.array(unique_combinations)[idx]).tolist()
        
        # block training
        self.total_trials = 50000
        self.switch_points = [0, 20000, 40000]
        self.switch_taskid = [0, 1, 0] # this config is deprecated right now
        self.train_to_criterion = False
        self.criterion = 0.98
        self.accuracy_momentum = 0.6    # how much of previous test accuracy to keep in the newest update.
        self.criterion_DMfam = 0.86
        self.max_trials_per_task = 20000
        self.num_blocks = 3
        self.MDeffect_mul = True # turns on or off the multiplicative MD effects
        self.MDeffect_add = False # turns on or off the additive MD effects
        # RNN model
        self.input_size = 33
        self.hidden_size = 600
        self.output_size = 17
        self.lr = 1e-4
        self.tau = 100

        # test
        self.test_every_trials = 200
        self.test_num_trials = 30

        # plot
        self.plot_every_trials = 4000
        self.save_plots = True

        # PFC context
        self.hidden_ctx_size = 400
        self.sub_size = 200
        self.sub_active_size = 50 # this config is deprecated right now
        self.sub_active_prob = 0.25
        self.hidden_ctx_noise = 0.01
        # MD
        self.MDeffect = True
        self.md_size = 2
        self.md_active_size = 1
        self.md_dt = 0.001
        self.MD2PFC_prob = 0.5
        self.gates_mean = 1.
        self.gates_std = 0.1
        self.gates_sparsity = 0.5
        # self.EXPSIGNATURE += f'MDprob_{self.MD2PFC_prob}' 
        
        # save variables
        self.FILEPATH = './files/'
        self.EXPSIGNATURE = f'{self.task_seq[0][7:-3]}_{self.task_seq[1][7:-3]}_'
        self.FILENAME = {
                        'config':    'config_PFCMD.npy',
                        'log':       'log_PFCMD.npy',
                        'plot_perf': 'performance_PFCMD_task.png',
        }

class EWCConfig(BaseConfig):
    def __init__(self):
        super(EWCConfig, self).__init__()
        # EWC
        self.EWC = True
        self.EWC_weight = 1e6
        self.EWC_num_trials = 1500
        # save variables
        self.FILENAME = {
                        'config':    'config_EWC.npy',
                        'log':       'log_EWC.npy',
                        'plot_perf': 'performance_EWC_task.png',
        }

class SIConfig(BaseConfig):
    def __init__(self):
        super(SIConfig, self).__init__()
        # SI
        self.SI = True
        self.SI_c = 1e6
        self.c = self.SI_c
        self.SI_xi = 0.5
        self.xi = self.SI_xi

        # save variables
        self.FILENAME = {
                        'config':    'config_SI.npy',
                        'log':       'log_SI.npy',
                        'plot_perf': 'performance_SI_task.png',
        }

class SerialConfig(BaseConfig):
    def __init__(self, args= []):
        super(SerialConfig, self).__init__()
        self.use_multiplicative_gates = False
        self.use_additive_gates = False 
        
