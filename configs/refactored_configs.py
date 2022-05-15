
import os

'''
Class names of configs are based on the class names of models:
    Base -> BaseConfig
    EWC  -> EWCConfig
'''

class BaseConfig(object):
    def __init__(self, dataset= 'neurogym', args= []):
        # system
        self.device = 'cuda'
        self.ROOT_DIR = os.getcwd()

   #########################################################
        self.dataset= dataset
        self.get_dataset_config(dataset)


        self.GoFamily = ['yang19.dlygo-v0', 'yang19.go-v0']
        self.AntiFamily = ['yang19.dlyanti-v0', 'yang19.anti-v0']
        self.DMFamily = ['yang19.dm1-v0', 'yang19.dm2-v0', 'yang19.ctxdm1-v0', 'yang19.ctxdm2-v0', 'yang19.multidm-v0']
        self.DMFamily += ['yang19.dlydm1-v0', 'yang19.dlydm2-v0', 'yang19.ctxdlydm1-v0', 'yang19.ctxdlydm2-v0', 'yang19.multidlydm-v0']
        self.MatchFamily = ['yang19.dms-v0', 'yang19.dmc-v0', 'yang19.dnms-v0', 'yang19.dnmc-v0']
                
        self._tasks_id_name = [(i, self.tasks[i]) for i in range(len(self.tasks))]
        self.tasks = self._tasks
        # This is yyyyya protected property to maintain the task_id no associated with each task based on this "standard" ordering
        self.human_task_names = ['{:<6}'.format(tn[7:-3]) for tn in self.tasks] #removes yang19 and -v0
        
        # MD
        self.md_size = len(self.tasks)
        self.md_active_size = 2
        self.md_dt = 0.001

        self.print_every_tasks =  5
        self.print_every_batches =  100
        self.batch_size = 100

        #  training paradigm
        self.max_trials_per_task = int(400 * self.batch_size)
        self.use_multiplicative_gates = True 
        self.use_additive_gates = False 
        self.train_to_criterion = True
        self.use_rehearsal = False
        self.abort_rehearsal_if_accurate = False
        self.same_rnn = True
        self.no_shuffled_trials = 2700
        self.paradigm_shuffle = False
        self.paradigm_sequential = not self.paradigm_shuffle
        self.paradigm_alternate = False
        self.one_batch_optimization = False  # Use only one batch to infer task rule input. 
        self.abort_rehearsal_if_accurate
        self.random_rehearsals = 0
        self.use_latent_updates = True
        self.use_weight_updates = True
        self.max_no_of_latent_updates = 1000
        self.no_latent_updates = 0
        self.use_learning_rate_scheduler = False

        
        #gates statis
        self.train_gates = False
        self.gates_sparsity = 0.4
        self.gates_mean = 1.0
        self.gates_std = 0.0
        self.gates_gaussian_cut_off = -0.3
        self.MD2PFC_prob = 0.5
        # test & plot
        self.test_every_trials = 500
        self.test_num_trials = self.batch_size
        self.test_no_latent_updates = 400 
        self.plot_every_trials = 4000
        self.args= args
    
    ############################################
        self.save_trained_model = True
        self.save_model = False
        self.load_saved_rnn1 = False 
        self.load_trained_cog_obs = False
        self.save_detailed = False

    def get_dataset_config(self, dataset):
        if dataset == 'neurogym':
            self.env_kwargs = {'dt': 100}
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
            self.criterion_DMfam = 0.86
            self.accuracy_momentum = 0.6    # how much of previous test accuracy to keep in the newest update.
            self.criterion = 0.98
            self.converged_ttc_criterion = int(1+ len(self._tasks) /2)

            # RNN model
            self.model ='RNN'
            self.input_size = 33
            self.hidden_size = 356
            self.output_size = 17
            self.tau= 200
            self.lr = 1e-3

        elif dataset == 'split_mnist':
            self._tasks= [f'smnist.class{i}-v0' for i in range(5) ] 
            # RNN model
            self.model = 'MLP'
            self.input_size = 28*28
            self.hidden_size = 400
            self.output_size = 2
            self.lr = 1e-3

            self.criterion_DMfam = 0.86
            self.accuracy_momentum = 0.6    # how much of previous test accuracy to keep in the newest update.
            self.criterion = 0.94
            self.converged_ttc_criterion = int( 2)#len(self._tasks))

        elif dataset == 'rotated_mnist':
            self._tasks= [f'smnist.class{i:03d}-v0' for i in [0, 30, 60, 90, 150, 200, 250, 290, 320, 350] ] 

            # RNN model
            self.input_size = 28
            self.hidden_size = 356
            self.output_size = 5
            self.tau= 200
            self.lr = 1e-3
                    
            self.criterion_DMfam = 0.86
            self.accuracy_momentum = 0.6    # how much of previous test accuracy to keep in the newest update.
            self.criterion = 0.98
        else:
            print('dataset not found!')

    ############################################   
    #  
    @property
    def tasks(self):
        return self._tasks
    @tasks.setter
    def tasks(self,tasks):
        self._tasks = tasks
        self.tasks_id_name = tasks # uses the property setter below to rearrange tasks_id_name to the new order but keep consistent the task ids

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
    def __init__(self, dataset='neurogym', args= []):
        super().__init__(dataset=dataset)
        self.same_rnn = True
        self.train_to_criterion = True
        self.use_rehearsal = True
        self.random_rehearsals = 0
        self.use_multiplicative_gates = True
        self.use_additive_gates = False

class Gates_add_config(Gates_mul_config):
    def __init__(self, dataset='neurogym',  args= []):
        super().__init__(dataset=dataset)
        self.use_multiplicative_gates = False
        self.use_additive_gates = True

class Shuffle_mul_config(Gates_mul_config):
    def __init__(self, dataset='neurogym',  args= []):
        super().__init__(dataset=dataset)
        self.train_to_plateau = False
        self.random_rehearsals = 300
        self.max_trials_per_task =     self.batch_size
        self.paradigm_shuffle = True
        self.paradigm_sequential = not     self.paradigm_shuffle
class Shuffle_add_config(Shuffle_mul_config):
    def __init__(self, dataset='neurogym',  args= []):
        super().__init__(dataset=dataset)
        self.use_multiplicative_gates = False
        self.use_additive_gates = True

class Gates_no_rehearsal_config(Gates_mul_config):
    def __init__(self, dataset='neurogym',  args= []):
        super().__init__(dataset=dataset)
        self.use_rehearsal = False
class random_gates_only_config(Gates_mul_config):
    def __init__(self, dataset='neurogym',  args= []):
        super().__init__(dataset=dataset)
        self.use_rehearsal = False
        self.train_to_criterion = False
class Gates_rehearsal_no_train_to_criterion_config(Gates_mul_config):
    def __init__(self, dataset='neurogym',  args= []):
        super().__init__()
        self.use_rehearsal = True
        self.train_to_criterion = False

############################################
class SerialConfig(BaseConfig):
    def __init__(self, dataset='neurogym',  args= []):
        super().__init__(dataset=dataset)
        self.use_multiplicative_gates = False
        self.use_additive_gates = False 
        
class Schizophrenia_config(object):
    def __init__(self, exp_type=None, args= []):
        # system
        self.device = 'cuda'
        self.ROOT_DIR = os.getcwd()

        self.env_kwargs = {'dt': 10}
        self.print_every_batches =  10
        
        import neurogym as ngym
        # self.tasks = ngym.get_collection('yang19')
        # This is the golden sequence, hand tuned curriculum to finish in the least no of trials
        if exp_type is None:
            exp_type = 'noisy_mean'
        if exp_type == 'shrew_task':
            self._tasks= [
                    'shrew_task_cxt1', 'shrew_task_cxt2',# 'st_hierarchical', #'shrew_task_audition', 'shrew_task_vision', 'shrew_task_cxt1'
                    # 'st_hierarchical',# 'shrew_task_cxt1', 'shrew_task_cxt2',  #'shrew_task_audition', 'shrew_task_vision', 'shrew_task_cxt1'
                    ] 
        elif exp_type == 'noisy_mean':
            self._tasks= ['noisy_mean',  'oddball', 'changepoint','drifting_mean']
            # self._tasks= [ 'oddball4', 'oddball3','oddball2','oddball1',]

        # self._tasks += ['yang19.dlydm1-v0', 'yang19.dlydm2-v0', 'yang19.ctxdlydm1-v0', 'yang19.ctxdlydm2-v0', 'yang19.multidlydm-v0']
        self._tasks_id_name = [(i, self.tasks[i]) for i in range(len(self.tasks))]
        self.tasks = self._tasks
        # This is yyyyya protected property to maintain the task_id no associated with each task based on this "standard" ordering
        self.human_task_names = self.tasks 
        self.no_of_tasks = len(self.tasks)


        self.GoFamily = ['yang19.dlygo-v0', 'yang19.go-v0']
        self.AntiFamily = ['yang19.dlyanti-v0', 'yang19.anti-v0']
        self.DMFamily = ['yang19.dm1-v0', 'yang19.dm2-v0', 'yang19.ctxdm1-v0', 'yang19.ctxdm2-v0', 'yang19.multidm-v0']
        self.DMFamily += ['yang19.dlydm1-v0', 'yang19.dlydm2-v0', 'yang19.ctxdlydm1-v0', 'yang19.ctxdlydm2-v0', 'yang19.multidlydm-v0']
        self.MatchFamily = ['yang19.dms-v0', 'yang19.dmc-v0', 'yang19.dnms-v0', 'yang19.dnmc-v0']
        
        # MD
        self.MDeffect = False
        self.md_size = len(self.tasks)
        self.md_active_size = 2
        self.md_dt = 0.001

        self.use_lstm = False

        self.env_kwargs = {'dt': 100}
        self.batch_size = 100

        #  training paradigm
        self.max_trials_per_task = 80000
        self.use_multiplicative_gates = True 
        self.use_additive_gates = False 
        self.train_to_criterion = True
        self.criterion = 0.98
        self.accuracy_momentum = 0.6    # how much of previous test accuracy to keep in the newest update.
        self.criterion_DMfam = 0.86
        self.random_rehearsals = 0

        self.same_rnn = True
        self.no_shuffled_trials = 40000
        self.paradigm_shuffle = False
        self.paradigm_sequential = not self.paradigm_shuffle
        self.paradigm_alternate =True

        # RNN model
        if exp_type == 'shrew_task':
            self.input_size = 6
            self.hidden_size = 64
            self.output_size = 4
        elif exp_type == 'noisy_mean':
            self.input_size = 1
            self.hidden_size = 64
            self.output_size = 1
        self.tau= 200
        self.lr = 1e-3

        #gates statis
        self.train_gates = False
        self.gates_sparsity = 0.5
        self.gates_mean = 1.2
        self.gates_std = 0.05
        self.gates_gaussian_cut_off = -0.3
        self.MD2PFC_prob = 0.5
        # test & plot
        self.test_every_trials = 500
        self.test_num_trials = 30
        self.plot_every_trials = 4000
        self.args= args

    
    ############################################
        self.save_trained_model = False
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

    @property
    def tasks_id_name(self):
        return self._tasks_id_name
    @tasks_id_name.setter
    def tasks_id_name(self,tasks):
        new_task_ids = []
        for i in range(len(tasks)):
            new_task_ids.extend( [t[0] for t in self._tasks_id_name if t[1] == tasks[i]] )
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
        
