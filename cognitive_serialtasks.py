#!/usr/bin/env python
# coding: utf-8
# system
import os
import sys

# from torch._C import T
root = os.getcwd()
sys.path.append(root)
sys.path.append('..')
from pathlib import Path
import json
# tools
import time
import itertools
from collections import defaultdict
# computation
import math
import numpy as np
# rng = np.random.default_rng()
import random
import pandas as pd
import torch
import torch.nn as nn
from torch.nn import init
from torch.nn import functional as F
# tasks
import gym
import neurogym as ngym
from neurogym.wrappers import ScheduleEnvs
from neurogym.utils.scheduler import RandomSchedule
# models
from models.PFC_gated import RNN_MD
from configs.configs import PFCMDConfig, SerialConfig
from logger.logger import SerialLogger
from utils import stats, get_trials_batch, get_performance, accuracy_metric
# visualization
import matplotlib as mpl
mpl.rcParams['axes.spines.left'] = True
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['axes.spines.bottom'] = True
# mpl.rcParams['pdf.fonttype'] = 42
# mpl.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt
import seaborn as sns
import imageio
from tqdm import tqdm, trange


import argparse
my_parser = argparse.ArgumentParser(description='Train neurogym tasks sequentially')
my_parser.add_argument('exp_name',
                       default='cognitive_obs',
                       type=str, nargs='?',
                       help='Experiment name, also used to create the path to save results')
my_parser.add_argument('use_gates',
                       default=1, nargs='?',
                       type=int,
                       help='Use multiplicative gating or not')
my_parser.add_argument('same_rnn',
                       default=1, nargs='?',
                       type=int,
                       help='Train the same RNN for all task or create a separate RNN for each task')
my_parser.add_argument('train_to_criterion',
                       default=1, nargs='?',
                       type=int,
                       help='TODO')
my_parser.add_argument('--experiment_type',
                       default='cognitive_observer', nargs='?',
                       type=str,
                       help='Which experimental or setup to run: "pairs") task-pairs a b a "serial") Serial neurogym "interleave") Interleaved ')
my_parser.add_argument('--seed',
                       default=0, nargs='?',
                       type=int,
                       help='Seed')
my_parser.add_argument('--var1',
                       default=1, nargs='?',
                       type=int,
                       help='Seed')
my_parser.add_argument('--var2',
                       default=-0.3, nargs='?',
                       type=float,
                       help='the ratio of active neurons in gates ')
my_parser.add_argument('--var3',
                        default=100, nargs='?',
                        type=float,
                        help='tau')
my_parser.add_argument('--num_of_tasks',
                    default=5, nargs='?',
                    type=int,
                    help='number of tasks to train on')
my_parser.add_argument('--use_cog_obs',
                    default=0, nargs='?',
                    type=int,
                    help='Use a second RNN as cognitive observer or not')
my_parser.add_argument('--use_rehearsal',
                    default=1, nargs='?',
                    type=int,
                    help='Use rehearsal')


# Get args and set config
args = my_parser.parse_args()

exp_name = args.exp_name
os.makedirs('./files/'+exp_name, exist_ok=True)

if args.experiment_type == 'pairs':
    config = PFCMDConfig()
else:
    config = SerialConfig()

config.set_strings( exp_name)

if args.experiment_type == 'pairs':
    config.task_seq = config.sequences[args.seed]
    # config.human_task_names = ['{}'.format(tn[7:-3]) for tn in config.task_seq] #removes yang19 and -v0    {:<6}
    config.exp_signature = f'{config.human_task_names[0]}_{config.human_task_names[1]}_'
else:# args.experiment_type == 'serial':
    config.exp_signature = f'{args.seed:d}_{args.var2:1.1f}_{args.var3:1.1f}_'


# config.MDeffect_mul = True if bool(args.var3) else False
# config.MDeffect_add = not config.MDeffect_mul
config.tau= args.var3
config.MD2PFC_prob = 0.5
config.gates_gaussian_cut_off = args.var2

config.FILEPATH += exp_name +'/'
config.save_model = False
config.load_saved_rnn1 = True 
config.load_trained_cog_obs = False
config.save_detailed = True
config.use_external_inputs_mask = False
config.MDeffect = False
config.train_to_criterion = True

############################################
config.same_rnn = bool(args.same_rnn)
config.use_gates = bool(args.use_gates)
config.train_to_criterion = bool(args.train_to_criterion)
############################################
############################################
config.use_rehearsal = False
config.use_cognitive_observer = False
config.use_gates = False
config.load_gates_corr = False
############################################
config.same_rnn = True
config.use_gates = False
config.train_to_criterion = False
config.use_rehearsal = False
config.use_cognitive_observer = False
config.use_CaiNet = False
config.use_gates = False
config.load_gates_corr = False
config.save_gates_corr = False

config.train_cog_obs_only = False
config.train_cog_obs_on_recent_trials = False
config.use_md_optimizer = False  
config.abandon_model = False

if args.experiment_type == 'same_net': # this shows that sequential fails right away
    config.same_rnn = True
if args.experiment_type == 'train_to_criterion': # demo ttc and introduce metric ttc. Start testing for accelerateion
    config.same_rnn = True
    config.train_to_criterion = True
if args.experiment_type == 'rehearsal':  # introduce first time learning vs subsequent rehearsals.
    config.same_rnn = True
    config.train_to_criterion = True
    config.use_rehearsal = True
if args.experiment_type == 'random_gates': # Task aware algorithm. Improved CL but impaired FT
    config.same_rnn = True
    config.train_to_criterion = True
    config.use_rehearsal = True
    config.use_gates = True
if args.experiment_type == 'shuffle_with_gates': # Task aware algorithm. Improved CL but impaired FT
    config.same_rnn = True
    config.train_to_criterion = True
    config.train_to_plateau = False
    config.paradigm_shuffle = True
    config.paradigm_sequential = not config.paradigm_shuffle
    config.use_rehearsal = False
    config.use_gates = True
    config.random_rehearsals = 300
    config.max_trials_per_task = config.batch_size

if args.experiment_type == 'random_gates_no_rehearsal': # Task aware algorithm. Improved CL but impaired FT
    config.same_rnn = True
    config.train_to_criterion = True
    config.use_rehearsal = False
    config.use_gates = True
if args.experiment_type == 'record_correlations': # Correlated gates calculate and save corrleations btn tasks
    config.same_rnn = True
    config.train_to_criterion = True
    config.use_rehearsal = True
    config.use_gates = True
    config.load_gates_corr = True
    config.save_gates_corr = True
if args.experiment_type == 'correlated_gates': # Correlated gates improve the trade off
    config.same_rnn = True
    config.train_to_criterion = True
    config.use_rehearsal = True
    config.use_gates = True
    config.load_gates_corr = True
if args.experiment_type == 'CaiNet': # Chapter 3.
    config.same_rnn = True
    config.train_to_criterion = True
    config.use_rehearsal = True #False if config.load_saved_rnn1 else True
    config.use_gates = True
    config.load_gates_corr = True
    config.use_cognitive_observer = False
    config.use_CaiNet = True
    config.use_md_optimizer = True
    config.abandon_model = False  
if args.experiment_type == 'cognitive_observer': # Chapter 2.
    config.same_rnn = True
    config.train_to_criterion = True
    config.use_rehearsal = False if config.load_saved_rnn1 else True
    config.use_gates = True
    config.load_gates_corr = True
    config.use_cognitive_observer = True
############################################
config.exp_signature = config.exp_signature + f'_{"corr" if config.load_gates_corr else "nocorr"}_{"co" if config.use_cognitive_observer else "noc"}_{"reh" if config.use_rehearsal else "noreh"}_{"tc" if config.train_to_criterion else "nc"}_{"mul" if config.MDeffect_mul else "add"}_{"gates" if config.use_gates else "nog"}'


###--------------------------Training configs--------------------------###
rng = np.random.default_rng(int(args.seed))
if not args.seed == 0: # if given seed is not zero, shuffle the task_seq
    #Shuffle tasks
    if True:
        idx = rng.permutation(range(len(config.tasks)))
        config.set_tasks((np.array(config.tasks)[idx]).tolist())
        config.tasks_id_name = config.tasks # assigns the order to tasks_id_name but retains the standard task_id

# config.set_tasks((np.array(config.tasks)[:2]).tolist())

task_seq = [config.tasks_id_name[i] for i in range(args.num_of_tasks)]
# Add tasks gradually with rehearsal 1 2 1 2 3 1 2 3 4 ...
if config.use_rehearsal:
    task_sub_seqs = [[config.tasks_id_name[i] for i in range(s)] for s in range(2, args.num_of_tasks+1)] # interleave tasks and add one task at a time
    # for sub_seq in task_sub_seqs: 
        # random.shuffle(sub_seq)
        # task_seq+=sub_seq
    # task_seq+=list(reversed(sub_seq)) # One additional final rehearsal, but revearsed for best final score.
    # task_seq+=sub_seq # One additional final rehearsal, 
    # task_seq += [config.tasks_id_name[i] for i in range(14)]
else:
    task_seq = [config.tasks_id_name[i] for i in range(args.num_of_tasks)]
# Now adding many random rehearsals:
sub_seq = [config.tasks_id_name[i] for i in range(args.num_of_tasks)]
Random_rehearsals = 20 if not config.load_saved_rnn1 and (config.use_cognitive_observer or config.use_CaiNet )else 5
if args.experiment_type == 'shuffle_with_gates': 
    Random_rehearsals   = config.random_rehearsals 
    
for _ in range(Random_rehearsals):
    random.shuffle(sub_seq)
    task_seq+=sub_seq

if args.experiment_type == 'shuffle_with_gates': # add the last task from the unlearned pile:
    no_of_tasks_left = len(config.tasks_id_name)- args.num_of_tasks
    novel_task_id = args.num_of_tasks + rng.integers(no_of_tasks_left)
    task_seq+= config.tasks_id_name[novel_task_id]
# main loop

def create_model():
    # model
    if config.use_lstm:
        pass
    else:
        net = RNN_MD(config)
    net.to(config.device)
    return(net )

class Cognitive_Net(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Cognitive_Net, self).__init__()
        # self.bn = nn.BatchNorm1d(input_size)
        self.gru = nn.GRU(input_size, hidden_size, batch_first=False)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, inp):
        # inp = self.bn(inp)
        out, hidden = self.gru(inp)
        x = self.linear(out)
        return x, out

if config.use_cognitive_observer:
    cog_net = Cognitive_Net(input_size=10+config.hidden_size+config.output_size, hidden_size=1, output_size = config.md_size)
    cog_net.to(config.device)
                


def train(config, task_seq):

    if config.same_rnn:
        net = create_model()
    md_context_id = torch.zeros([1, config.md_size])

    # Replace the gates with the ones calculated offline from performance traces to measure tasks compatibility. 
    if config.use_gates:
        
        if config.load_gates_corr and (os.path.isfile(f'./data/perf_corr_mat_var1_{args.seed}.npy',)):#else pfc_gated will pick random overlapping gates. 
            print('------------------   loading correlations from ' +f'./data/perf_corr_mat_var1_0.npy')
            gates_tasks = np.load(f'./data/perf_corr_mat_var1_{args.seed}.npy', allow_pickle=True).item()
            gates_corr = gates_tasks['corr_mat'] 
            # assert(config.tasks== gates_tasks['tasks'])  # NOTE using the gates corr off policy here. 
            sampled_gates = np.random.multivariate_normal(np.zeros(net.rnn.gates.shape[0]), gates_corr, net.rnn.gates.shape[1]).T
            sampled_gates = torch.tensor(sampled_gates> config.gates_gaussian_cut_off, device=config.device).float()
            net.rnn.gates = sampled_gates.clone()
        else:
            #PFC_gated.py has already created random gates at init, so just not updating them.
            print('------------------   using random gates...')    

    # criterion & optimizer
    criterion = nn.MSELoss()
    #     print('training parameters:')
    training_params = list()
    for name, param in net.named_parameters():
        if not name.__contains__('md_context_id'):
            print(name)
            training_params.append(param)
        else:
            print('exluding: ', name)
    optimizer = torch.optim.Adam(training_params, lr=config.lr)

    md_optimizer = torch.optim.Adam([tp[1] for tp in net.named_parameters() if tp[0] == 'rnn.md_context_id'], lr=config.lr)
    if config.use_cognitive_observer:
        cog_training_params = list()
        print('cognitive network optimized parameters')
        for name, param in cog_net.named_parameters():
            print(name)
            cog_training_params.append(param)
            
        cog_optimizer = torch.optim.Adam(cog_training_params, lr=config.lr*100)


    training_log = SerialLogger(config=config)
    testing_log = SerialLogger(config=config)
    training_log.cog_obs_preds = []
    training_log.context_ids = []
    training_log.md_grads = []
    
    # Make all tasks, but reorder them from the tasks_id_name list of tuples
    envs = [None] * len(config.tasks_id_name)
    for task_id, task_name in config.tasks_id_name:
        envs[task_id] = gym.make(task_name, **config.env_kwargs)

    step_i = 0
    bar_tasks = tqdm(task_seq)
    for (task_id, task_name) in bar_tasks:
        
        #if the last task in shuffle training_ extend it to 400 trials


        env = envs[task_id]
        bar_tasks.set_description('i: ' + str(step_i))
        testing_log.switch_trialxxbatch.append(step_i)
        testing_log.switch_task_id.append(task_id)
        
        if not config.same_rnn:
            net= create_model()

        # training
        # train with no cog obs at all
        
        if config.MDeffect:
            net.rnn.md.learn = True
            net.rnn.md.sendinputs = True
        running_acc = 0
        training_bar = trange(config.max_trials_per_task//config.batch_size)
        for i in training_bar:
            # control training paradigm
            config.use_external_task_id = True 
            if config.use_cognitive_observer or config.use_CaiNet:
                trials_to_independence= 5500 if not config.load_saved_rnn1 else 10
                if step_i < trials_to_independence :
                    config.train_cog_obs_on_recent_trials = True
                else:  # False leads to using cog_obs predictions
                    config.use_external_task_id = False
                    config.train_cog_obs_only = True
                    config.train_cog_obs_on_recent_trials = True
                if (step_i == trials_to_independence): 
                    print(' Cognitive-observer-only training begins ----------------------------')

                if config.use_CaiNet:
                    config.train_cog_obs_only = False
                    config.train_cog_obs_on_recent_trials = False

                if config.save_model and (step_i ==trials_to_independence):
                    saved_model_path = './data/'+ config.exp_name+f'/saved_model_{config.exp_signature}.torch'
                    saved_gates_path = './data/'+ config.exp_name+f'/saved_gates_{config.exp_signature}.torch'

                    print(' Saving RNN1 to: ----', saved_model_path)
                    torch.save(net.state_dict(), saved_model_path)
                    torch.save(net.rnn.gates, saved_gates_path)
            if config.load_saved_rnn1 and (step_i==0):
                # saved_model_path = './data/cognitive_obs/saved_model_0_-0.3_100.0__corr_co_reh_tc_mul_gates.torch'
                if config.abandon_model:
                    # saved_model_path = './data/'+ config.exp_name+f'/saved_model_{config.exp_signature.replace("noreh", "reh")}.torch'
                    # saved_gates_path = './data/'+ config.exp_name+f'/saved_gates_{config.exp_signature.replace("noreh", "reh")}.torch'
                    saved_gates_path = './data/cognitive_obs/saved_gates_NO_ADDED_PARAM.torch'
                    saved_model_path = './data/cognitive_obs/saved_model_NO_ADDED_PARAM.torch'
                else:            
                    saved_gates_path = './data/cognitive_obs/saved_gates_with_context_id.torch'
                    saved_model_path = './data/cognitive_obs/saved_model_with_context_id.torch'
                    # saved_model_path = './data/'+ config.exp_name+f'/saved_model_{config.exp_signature.replace("noreh", "reh").replace("noc", "co")}.torch'
                    # saved_gates_path = './data/'+ config.exp_name+f'/saved_gates_{config.exp_signature.replace("noreh", "reh").replace("noc", "co")}.torch'
                try:
                    net.load_state_dict(torch.load(saved_model_path))
                    net.rnn.gates = torch.load(saved_gates_path)
                except:
                    raise Exception('No model file found!')
                

            if config.use_external_task_id:
                context_id = F.one_hot(torch.tensor([task_id]* config.batch_size), config.md_size).type(torch.float)

                # context_id = task_id if config.use_gates else 0 
            elif config.use_cognitive_observer and config.train_cog_obs_only:
                #########Gather cognitive inputs ###############
                horizon =50
                acts = np.stack(training_log.rnn_activity[-horizon:])  # (3127, 100, 356)
                outputs_horizon = np.stack(training_log.outputs[-horizon:])
                labels_horizon = np.stack(training_log.labels[-horizon:])
                task_ids = np.stack(training_log.task_ids[-horizon:])
                rl = np.argmax(labels_horizon, axis=-1)
                ro = np.argmax(outputs_horizon, axis=-1)
                accuracies = (rl==ro).astype(np.float32)

                # previous_acc = np.concatenate([accuracies[:1], accuracies ])[:-1] #shift by one place to make it run one step behind. 
                expanded_previous_acc = np.repeat(accuracies[..., np.newaxis], 10, axis=-1)   #Expanded merely to emphasize their signal over the numerous acts
                gathered_inputs = np.concatenate([acts, labels_horizon, expanded_previous_acc], axis=-1) #shape  7100 100 266

                task_ids_oh= F.one_hot(torch.from_numpy(task_ids).long(), 15)
                task_ids_repeated = task_ids[..., np.newaxis].repeat(100,1)
                
                training_inputs = gathered_inputs
                # training_outputs= task_ids_oh.reshape([task_ids_oh.shape[0], 1, task_ids_oh.shape[1]]).repeat([1,training_inputs.shape[1], 1])
                training_outputs= task_ids_repeated
                ins =  torch.tensor(training_inputs, device=config.device) # (input_length, 100, 266)
                outs = torch.tensor(training_outputs, device=config.device)
                #################################################
                cpred, _, = cog_net(ins)
                context_id  = F.softmax(cpred[-1], dim = 1) # will give 15 one_hot.
                context_id.retain_grad() # otherwise grad gets released from memory because is not leaf var.
                training_log.context_ids.append((step_i, context_id.detach().cpu().numpy()))
            elif config.use_CaiNet:
                # context_id = torch.zeros([config.batch_size, config.md_size])
                # context_id[:, torch.argmax(md_context_id, axis=1)] = 1. # Hard max
                if config.use_md_optimizer:
                
                    if config.abandon_model:
                    
                        context_id = torch.ones([1,config.md_size])/config.md_size    
                        context_id = context_id.repeat([config.batch_size, 1])
                    else:                   
                        context_id = net.rnn.md_context_id    
                        context_id = context_id.repeat([config.batch_size, 1])
                    
                else:
                    context_id = F.softmax(md_context_id.float(), dim=1 ) #alternatively softmax it. 
                    context_id = context_id.repeat([config.batch_size, 1])
                    context_id = context_id.to(config.device)
                    context_id.requires_grad_()
                    context_id.retain_grad()
                


            inputs, labels = get_trials_batch(envs=env, config = config, batch_size = config.batch_size)
    
            if config.use_lstm:
                outputs, rnn_activity = net(inputs)
            else:
                outputs, rnn_activity = net(inputs, sub_id=context_id)
            
                
            acc  = accuracy_metric(outputs.detach(), labels.detach())
            # print(f'shape of outputs: {outputs.shape},    and shape of rnn_activity: {rnn_activity.shape}')
            #Shape of outputs: torch.Size([20, 100, 17]),    and shape of rnn_activity: torch.Size ([20, 100, 256
            optimizer.zero_grad()
            if config.use_cognitive_observer: cog_optimizer.zero_grad()
            if config.use_md_optimizer:    md_optimizer.zero_grad()
            
            loss = criterion(outputs, labels)
            loss.backward()
            if step_i % config.print_every_batches == (config.print_every_batches - 1):
                fig, axes = plt.subplots(1,3)
                axes[0].matshow(context_id.detach().cpu().numpy()[:50])
                axes[0].set_title('context_id')
                axes[0].text(2, 52, f'Current Task_ID: {task_id}')
                if (config.use_CaiNet or config.train_cog_obs_only) and not config.use_external_task_id:
                    if context_id.grad is not None:
                        im =axes[1].matshow(context_id.grad.data.cpu().numpy()[:50])
                        axes[1].set_title('grad context_id')
                        cax = fig.add_axes([0.35, 0.3, 0.03, 0.3])
                        fig.colorbar(im, cax=cax, orientation='vertical')
                    # else: print('Context ID grad is None!')
                axes[2].bar(range(config.md_size), md_context_id.detach().cpu().numpy().squeeze())
                axes[2].set_title('md_context_id')
                
                plt.savefig(f'./files/to_animate/example_context_ids_{step_i:05}.jpg')
                plt.savefig(f'./files/to_animate/example_context_ids.jpg')
                plt.close('all')
            if config.use_cognitive_observer and not config.use_external_task_id:
                training_log.md_grads.append(context_id.grad.data.cpu().numpy())
            if config.use_CaiNet and not config.use_external_task_id:
                if config.use_md_optimizer:
                    if config.abandon_model:
                        optimizer.step()
                    
                    else:                   
                        md_optimizer.step()
                else:
                    caia_lr = 0.001
                    # grad_sum = context_id.grad.data.sum(0)
                    grad_sum = context_id.grad.data.sum(0)
                    md_context_id = md_context_id.to(config.device) - caia_lr* grad_sum# md_grads=(context_id.grad.data.cpu().numpy()) # shape [batch_size, 15]
                    # md_context_id = F.one_hot(torch.argmax(md_context_id, dim=1), md_context_id.shape[1])

            if config.use_external_task_id:
                optimizer.step()
            elif config.train_cog_obs_only:

                cog_optimizer.step()

            # from utils import show_input_output
            # show_input_output(inputs, labels, outputs)
            # plt.savefig('example_inpujt_label_output.jpg')
            # plt.close('all')
            # save loss

            training_log.write_basic(step_i, loss.item(), acc, task_id)
            training_log.gradients.append(np.array([torch.norm(p.grad).item() for p in net.parameters() if p.grad is not None]) )
            if config.save_detailed or config.use_cognitive_observer:
                training_log.write_detailed( rnn_activity= rnn_activity.detach().cpu().numpy().mean(0),
                inputs=   [] ,# inputs.detach().cpu().numpy(),
                outputs = outputs.detach().cpu().numpy()[-1, :, :],
                labels =   labels.detach().cpu().numpy()[-1, :, :],
                sampled_act = [], # rnn_activity.detach().cpu().numpy()[:,:, 1:356:36], # Sample about 10 neurons 
                # rnn_activity.shape             torch.Size([15, 100, 356])
                )
            ################################################
            #########Gather cognitive inputs ###############
            if config.use_cognitive_observer and config.train_cog_obs_on_recent_trials:
                horizon =50
                if (((step_i+1) % horizon) ==0):
                    acts = np.stack(training_log.rnn_activity[-horizon:])  # (3127, 100, 356)
                    outputs_horizon = np.stack(training_log.outputs[-horizon:])
                    labels_horizon = np.stack(training_log.labels[-horizon:])
                    task_ids = np.stack(training_log.task_ids[-horizon:])
                    rl = np.argmax(labels_horizon, axis=-1)
                    ro = np.argmax(outputs_horizon, axis=-1)
                    accuracies = (rl==ro).astype(np.float32)

                    # previous_acc = np.concatenate([accuracies[:1], accuracies ])[:-1] #shift by one place to make it run one step behind. 
                    expanded_previous_acc = np.repeat(accuracies[..., np.newaxis], 10, axis=-1)   #Expanded merely to emphasize their signal over the numerous acts
                    gathered_inputs = np.concatenate([acts, labels_horizon, expanded_previous_acc], axis=-1) #shape  7100 100 266

                    task_ids_oh= F.one_hot(torch.from_numpy(task_ids).long(), 15)
                    task_ids_repeated = task_ids[..., np.newaxis].repeat(100,1)
                    
                    training_inputs = gathered_inputs
                    # training_outputs= task_ids_oh.reshape([task_ids_oh.shape[0], 1, task_ids_oh.shape[1]]).repeat([1,training_inputs.shape[1], 1])
                    training_outputs= task_ids_repeated
                    ins =  torch.tensor(training_inputs, device=config.device) # (input_length, 100, 266)
                    outs = torch.tensor(training_outputs, device=config.device).long()
                    #################################################

                    cog_optimizer.zero_grad()
                    # cin = torch.cat([rnn1_means.detach(), labels_horizon[-1]],dim =-1)
                    # cin = cin.reshape([1, *cin.shape])
                    # cog_out = cog_net(cin)
                    cog_out, cog_acts = cog_net(ins) # Cog_out shape [horizon, batch, 15]
                    tids = torch.tensor([task_id]*100, device=config.device)
                    # cog_loss  = F.cross_entropy(input=cog_out.cpu().squeeze(), target=tids.type(torch.LongTensor))
                    #Train on all task_ids in horizon:
                    cog_loss = F.cross_entropy(input= cog_out.squeeze().permute([0,2,1]) , target = outs)
                    # Or just the current task:
                    # cog_loss = F.cross_entropy(input= cog_out.squeeze()[-1] , target = tids)
                    # if step_i < 4000: # start testing cog obs on useen data
                    cog_loss.backward()
                    cog_optimizer.step()

                    training_log.cog_obs_preds.append(cog_out.detach().cpu().numpy())

                    # if step_i > 100 and (step_i % (config.print_every_batches*50) == (config.print_every_batches*50 - 1)):
                        # _,_,cacc = test_model(cog_net, ins, task_ids, step_i)    
                    # training_bar.set_description('cog_ls, acc: {:0.3F}, {:0.2F} '.format(cog_loss.item(), acc)+ config.human_task_names[task_id])
            training_bar.set_description('ls, acc: {:0.3F}, {:0.2F} '.format(loss.item(), acc)+ config.human_task_names[task_id])

            # print statistics
            if step_i % config.print_every_batches == (config.print_every_batches - 1):
                ################################ test during training
                net.eval()
                if config.MDeffect:
                    net.rnn.md.learn = False
                # torch.set_grad_enabled(True)
                with torch.no_grad():
                    testing_log.stamps.append(step_i)
                    testing_context_ids = list(range(len(envs)))  # envs are ordered by task id sequentially now.
                    # testing_context_ids_oh = [F.one_hot(torch.tensor([task_id]* config.test_num_trials), config.md_size).type(torch.float) for task_id in testing_context_ids]

                    fix_perf, act_perf = get_performance(
                        net,
                        envs,
                        context_ids=testing_context_ids,
                        config = config,
                        batch_size = config.test_num_trials,
                        ) 
                    
                    testing_log.accuracies.append(act_perf)
                    testing_log.gradients.append(np.mean(np.stack(training_log.gradients[-config.print_every_batches:]),axis=0))
                # torch.set_grad_enabled(False)
                net.train()
                if config.MDeffect:
                    net.rnn.md.learn = True
                
                #### End testing

            if config.use_external_task_id:
                criterion_accuaracy = config.criterion if task_name not in config.DMFamily else config.criterion_DMfam
            elif config.train_cog_obs_only or config.use_CaiNet: # relax a little! Only optimizing context signal!
                criterion_accuaracy = config.criterion if task_name not in config.DMFamily else config.criterion_DMfam
                criterion_accuaracy -=0.08
            if ((running_acc > criterion_accuaracy) and config.train_to_criterion) or (i+1== config.max_trials_per_task//config.batch_size):
            # switch task if reached the max trials per task, and/or if train_to_criterion then when criterion reached
                # import pdb; pdb.set_trace()
                running_acc = 0.
                if args.experiment_type == 'pairs':
                    #move to next task
                    task_id = (task_id+1)%2 #just flip to the other task
                    print('switching to task: ', task_id, 'at trial: ', i)
                break # stop training current task if sufficient accuracy. Note placed here to allow at least one performance run before this is triggered.

            step_i+=1
            running_acc = 0.7 * running_acc + 0.3 * acc
        #no more than number of blocks specified
        if (args.experiment_type=='pairs') and(len(testing_log.switch_trialxxbatch) > config.num_blocks):
            break

        training_log.sample_input = inputs[:,0,:].detach().cpu().numpy().T
        training_log.sample_label = labels[:,0,:].detach().cpu().numpy().T
        training_log.sample_output = outputs[:,0,:].detach().cpu().numpy().T
    testing_log.total_batches = step_i

    if config.save_model:
        torch.save(net.state_dict(), './files/'+ config.exp_name+f'/saved_model_{config.exp_signature}')

    return(testing_log, training_log, )


testing_log, training_log = train(config, task_seq )
# ordered_corr_mat = corrmat # TODO needs reordered to match neurogym 19

if config.save_gates_corr:
    taa = []
    num_tasks = len(config.tasks)
    for logi in range(num_tasks):
        taa.append([test_acc[logi] for test_acc in testing_log.accuracies])

    tnp = np.stack(taa)
    corrmat = np.corrcoef(tnp)

    print('tnp shape: ', tnp.shape)
    print('ordered_corr_Mat shape: ', corrmat.shape)
    print('tasks: ', config.tasks)
    gates_corr = {'corr_mat': corrmat, 'tasks': config.tasks}
    if (os.path.isfile(f'./data/perf_corr_mat_var1_{args.seed}.npy',)):#else pfc_gated will pick random overlapping gates. 
        print('------------------   correlations file already exist! ' +f'./data/perf_corr_mat_var1_{args.seed}.npy')
    else:          
        np.save(f'./data/perf_corr_mat_var1_{args.seed}.npy', gates_corr, allow_pickle=True)


np.save('./files/'+ config.exp_name+f'/testing_log_{config.exp_signature}.npy', testing_log, allow_pickle=True)
np.save('./files/'+ config.exp_name+f'/training_log_{config.exp_signature}.npy', training_log, allow_pickle=True)
np.save('./files/'+ config.exp_name+f'/config_{config.exp_signature}.npy', config, allow_pickle=True)
print('testing logs saved to : '+ './files/'+ config.exp_name+f'/testing_log_{config.exp_signature}.npy')


## Plots

# plot the corr for comparison. Reorder them to standarized order.

# fig, ax = plt.subplots(1,1)
# standard_order = np.argsort(idx) # even accuracies are saved in standard order already, also envs
# ordered_corrmat = np.corrcoef(tnp)
# ax.matshow(ordered_corrmat)
# # numbered = [n for (i,n) in enumerate(np.array(config.human_task_names))] #[0,1,2,11,10,12,3,7,4,8,9,5,13,6,14]
# ax.set_xticks(range(num_tasks))
# _=ax.set_xticklabels(config.human_task_names, rotation = 90)
# ax.set_yticks(range(num_tasks))
# _=ax.set_yticklabels(config.human_task_names, rotation = 0)
# # now learn again, but this time the presence of the corr file will be detected and converted to proper gates

# In[9]:

import matplotlib
no_of_values = len(config.tasks)
norm = mpl.colors.Normalize(vmin=min([0,no_of_values]), vmax=max([0,no_of_values]))
cmap_obj = matplotlib.cm.get_cmap('Set1') # tab20b tab20
cmap = mpl.cm.ScalarMappable(norm=norm, cmap=cmap_obj)

log = testing_log
log.switch_trialxxbatch.append(log.stamps[-1])
num_tasks = len(config.tasks)
already_seen =[]
title_label = 'Training tasks sequentially ---> \n    ' + config.exp_name
max_x = log.stamps[-1]
fig, axes = plt.subplots(num_tasks+3,1, figsize=[9,7])
for logi in range(num_tasks):
        ax = axes[ logi ] # log i goes to the col direction -->
        ax.set_ylim([-0.1,1.1])
        ax.set_xlim([0, max_x])
#         ax.axis('off')
        ax.plot(log.stamps, [test_acc[logi] for test_acc in log.accuracies], linewidth=1)
        ax.plot(log.stamps, np.ones_like(log.stamps)*0.5, ':', color='grey', linewidth=1)
        ax.set_ylabel(config.human_task_names[logi], fontdict={'color': cmap.to_rgba(logi)})
        if (logi == num_tasks-1) and config.use_cognitive_observer and config.train_cog_obs_on_recent_trials: # the last subplot, put the preds from cog_obx
            cop = np.stack(training_log.cog_obs_preds).reshape([-1,100,15])
            cop_colors = np.argmax(cop, axis=-1).mean(-1)
            for ri in range(max_x-2):
                if ri< len(cop_colors):ax.axvspan(ri, ri+1, color =cmap.to_rgba(cop_colors[ri]) , alpha=0.2)
        else:            
            for ri in range(len(log.switch_trialxxbatch)-1):
                ax.axvspan(log.switch_trialxxbatch[ri], log.switch_trialxxbatch[ri+1], color =cmap.to_rgba(log.switch_task_id[ri]) , alpha=0.2)
for ti, id in enumerate(log.switch_task_id):
    if id not in already_seen:
        already_seen.append(id)
        task_name = config.human_task_names[id]
        axes[0].text(log.switch_trialxxbatch[ti], 1.3, task_name, color= cmap.to_rgba(id) )

gs = np.stack(log.gradients)

glabels = ['inp_w', 'inp_b', 'rnn_w', 'rnn_b', 'out_w', 'out_b']
ax = axes[num_tasks+0]
gi =0
ax.plot(log.stamps, gs[:,gi+1], label= glabels[gi+1])
ax.plot(log.stamps, gs[:,gi], label= glabels[gi])
ax.legend()
ax.set_xlim([0, max_x])
ax = axes[num_tasks+1]
gi =2
ax.plot(log.stamps, gs[:,gi+1], label= glabels[gi+1])
ax.plot(log.stamps, gs[:,gi], label= glabels[gi])
ax.legend()
ax.set_xlim([0, max_x])
ax = axes[num_tasks+2]
gi =4
ax.plot(log.stamps, gs[:,gi+1], label= glabels[gi+1])
ax.plot(log.stamps, gs[:,gi], label= glabels[gi])
ax.legend()
ax.set_xlim([0, max_x])
# ax.set_ylim([0, 0.25])

final_accuracy_average = np.mean(list(testing_log.accuracies[-1].values()))
plt.savefig('./files/'+ config.exp_name+f'/acc_summary_{config.exp_signature}_{testing_log.total_batches}_{final_accuracy_average:1.2f}.jpg', dpi=300)

