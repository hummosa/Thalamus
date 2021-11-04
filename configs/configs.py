import os
import sys
from models import MODES

'''
Class names of configs are based on the class names of models:
    Base -> BaseConfig
    EWC  -> EWCConfig
'''

class BaseConfig(object):
    def __init__(self):
        # system
        self.device = 'cpu'
        # self.device = 'cuda:0'
        self.RNGSEED = 5
        self.ROOT_DIR = os.getcwd()
        
        # dataset
        # 1. Two tasks
        # self.task_seq = ['yang19.go-v0', 'yang19.rtgo-v0']
        # self.task_seq = ['yang19.dms-v0', 'yang19.dmc-v0']
        # self.task_seq = ['yang19.dnms-v0', 'yang19.dnmc-v0']
        self.task_seq = ['yang19.dlygo-v0', 'yang19.dnmc-v0']
        # self.task_seq = ['yang19.dlyanti-v0', 'yang19.dnms-v0']
        # self.task_seq = ['yang19.dlyanti-v0', 'yang19.dms-v0']
        # self.task_seq = ['yang19.rtgo-v0', 'yang19.ctxdm2-v0']
        # self.task_seq = ['yang19.dlygo-v0', 'yang19.dmc-v0']

        #sequences of interest:
        self.sequences=[['yang19.go-v0', 'yang19.rtgo-v0'],
        ['yang19.dms-v0', 'yang19.dmc-v0'],
        ['yang19.dnms-v0', 'yang19.dnmc-v0'],
        ['yang19.dlygo-v0', 'yang19.dnmc-v0'],
        ['yang19.dlyanti-v0', 'yang19.dnms-v0'],
        ['yang19.dlyanti-v0', 'yang19.dms-v0'],
        ['yang19.rtgo-v0', 'yang19.ctxdm2-v0'],
        ['yang19.dlygo-v0', 'yang19.dmc-v0'],]

        # self.tasks = ngym.get_collection('yang19')
        self.tasks= [
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
        self._tasks_id_name = [(i, self.tasks[i]) for i in range(len(self.tasks))]
        self.human_task_names = ['{:<6}'.format(tn[7:-3]) for tn in self.tasks] #removes yang19 and -v0
        
        import itertools
        unique_combinations = []
        permut = itertools.permutations(self.tasks, 2)
        for comb in permut:
            unique_combinations.append(comb)
        import numpy as np
        rng = np.random.default_rng(1)
        idx = rng.permutation(range(len(unique_combinations)))

        self.sequences = (np.array(unique_combinations)[idx]).tolist()
        
        self.GoFamily = ['yang19.dlygo-v0', 'yang19.go-v0']
        self.AntiFamily = ['yang19.dlyanti-v0', 'yang19.anti-v0']
        self.DMFamily = ['yang19.dm1-v0', 'yang19.dm2-v0', 'yang19.ctxdm1-v0', 'yang19.ctxdm2-v0', 'yang19.multidm-v0']
        self.MatchFamily = ['yang19.dms-v0', 'yang19.dmc-v0', 'yang19.dnms-v0', 'yang19.dnmc-v0']
        ### 2.1 two tasks
        # TaskA = self.GoFamily + self.AntiFamily
        # TaskB = self.MatchFamily + ['yang19.ctxdm1-v0', 'yang19.dm2-v0']
        # task_seqs = []
        # for a in TaskA:
        #     for b in TaskB:
        #         task_seqs.append((a, b))
        #         task_seqs.append((b, a))
        
        # task_seqs
        # self.sequences = task_seqs
        # 2. Three tasks
        # self.task_seq = ['yang19.dlygo-v0', 'yang19.dm1-v0', 'yang19.dnmc-v0']
        # self.task_seq = ['yang19.dlygo-v0', 'yang19.dm2-v0', 'yang19.dmc-v0']
        # self.task_seq = ['yang19.dlyanti-v0', 'yang19.dm1-v0', 'yang19.dnmc-v0']
        # 3. Four tasks
        # self.task_seq = ['yang19.dnms-v0', 'yang19.dnmc-v0', 'yang19.dlygo-v0', 'yang19.go-v0']
        # self.task_seq = ['yang19.dms-v0', 'yang19.dnms-v0', 'yang19.dlygo-v0', 'yang19.go-v0']
        # self.task_seq = ['yang19.dlygo-v0', 'yang19.go-v0', 'yang19.dmc-v0', 'yang19.dnmc-v0']
        # self.task_seq = ['yang19.dnms-v0', 'yang19.dnmc-v0', 'yang19.anti-v0', 'yang19.dlyanti-v0']
        self.env_kwargs = {'dt': 100}
        self.batch_size = 1

        # block training
        '''
        Customize the block training paradigm:
        1. Change self.task_seq, self.total_trials, self.switch_points.
            e.g.
            To train four tasks serially for 10000 trials each:
            self.task_seq=[task1, task2, task3, task4]
            self.total_trials=40000
            self.switch_points=[0, 10000, 20000, 30000]
           Change PFCMD configs.
        2. Change utils.get_task_id, utils.get_task_seqs
        3. Change the task_ids of CL_model.end_task() in the run_baselines.py & scaleup_baselines.py
        '''
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

        # save variables
        self.FILEPATH = './files/'
        self.EXPSIGNATURE = f'{self.task_seq[0][7:-3]}_{self.task_seq[1][7:-3]}_'
        self.FILENAME = {
                        'config':    'config_PFC.npy',
                        'log':       'log_PFC.npy',
                        'plot_perf': 'performance_PFC_task.png',
        }
        # continual learning mode
        self.mode = None
    def set_task_seq(self, task_seq):
        self.task_seq = task_seq
        self.num_task = len(self.task_seq)
    def set_tasks(self,tasks):
        self.tasks = tasks
        # self.tasks_id_name = [(i, self.tasks[i]) for i in range(len(self.tasks))]
        # self.human_task_names = ['{:<6}'.format(tn[7:-3]) for tn in self.tasks] #removes yang19 and -v0
    
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
        f'{"same_rnn" if self.same_rnn else "separate"}_{"gates" if self.use_gates else "nogates"}'+\
        f'_{"tc" if self.train_to_criterion else "nc"}'

    def update(self, new_config):
        self.__dict__.update(new_config.__dict__)

    def __str__(self):
        return str(self.__dict__)

class PFCMDConfig(BaseConfig):
    def __init__(self):
        super(PFCMDConfig, self).__init__()
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

        self.print_every_batches =  10
        self.device = 'cuda'
        
        import neurogym as ngym
        tasks = ngym.get_collection('yang19')

        self.tasks= [
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
        # This is a protected property to maintain the task_id no associated with each task based on this "standard" ordering
        self._tasks_id_name = [(i, self.tasks[i]) for i in range(len(self.tasks))]
        self.human_task_names = ['{:<6}'.format(tn[7:-3]) for tn in self.tasks] #removes yang19 and -v0
        
        # MD
        self.MDeffect = False
        self.md_size = len(self.tasks)
        self.md_active_size = 2
        self.md_dt = 0.001
        self.use_gates = False
        self.train_to_criterion = False
        self.same_rnn = True
        self.use_lstm = False

        self.task_seq = ['yang19.dnms-v0', 'yang19.dnmc-v0', 'yang19.dlygo-v0', 'yang19.go-v0']
        self.num_task = len(self.task_seq)
        self.env_kwargs = {'dt': 100}
        self.batch_size = 100

        # block training
        self.trials_per_task = 1000
        self.max_trials_per_task = 40000
        self.total_trials = int(self.num_task * self.trials_per_task)
        self.switch_points = list(range(0, self.total_trials, self.trials_per_task))
        self.switch_taskid = list(range(self.num_task) ) # this config is deprecated right now
        assert len(self.switch_points) == len(self.switch_taskid)

        # RNN model
        self.input_size = 33
        self.hidden_size = 356
        self.output_size = 17
        self.lr = 1e-3

        # test & plot
        self.test_every_trials = 500
        self.test_num_trials = 30
        self.plot_every_trials = 4000
        self.args= args


        # Add tasks gradually with rehearsal 1 2 1 2 3 1 2 3 4 ...
        task_sub_seqs = [[(i, self.task_seq[i]) for i in range(s)] for s in range(2, len(self.task_seq))] # interleave tasks and add one task at a time
        self.task_seq_with_rehearsal = []
        for sub_seq in task_sub_seqs: self.task_seq_with_rehearsal+=sub_seq
    
