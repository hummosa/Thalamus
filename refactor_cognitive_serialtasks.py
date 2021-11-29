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
from models.PFC_gated import RNN_MD, Cognitive_Net
from configs.refactored_configs import *
from logger.logger import SerialLogger
from utils import stats, get_trials_batch, get_performance, accuracy_metric, sparse_with_mean
# visualization
import matplotlib as mpl
mpl.rcParams['axes.spines.left'] = True
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['axes.spines.bottom'] = True
import matplotlib.pyplot as plt
import seaborn as sns
import imageio
from tqdm import tqdm, trange

import argparse
my_parser = argparse.ArgumentParser(description='Train neurogym tasks sequentially')
my_parser.add_argument('exp_name',  default='temp', type=str, nargs='?', help='Experiment name, also used to create the path to save results')
my_parser.add_argument('--experiment_type', default='shuffle_mul', nargs='?', type=str, help='Which experimental or setup to run: "pairs") task-pairs a b a "serial") Serial neurogym "interleave") Interleaved ')
# my_parser.add_argument('--experiment_type', default='random_gates_mul', nargs='?', type=str, help='Which experimental or setup to run: "pairs") task-pairs a b a "serial") Serial neurogym "interleave") Interleaved ')
my_parser.add_argument('--seed', default=0, nargs='?', type=int,  help='Seed')
my_parser.add_argument('--var1',  default=1., nargs='?', type=float, help='gates mean ')
# my_parser.add_argument('--var2', default=-0.3, nargs='?', type=float, help='the ratio of active neurons in gates ')
my_parser.add_argument('--var3',  default=0.01, nargs='?', type=float, help='gates std')
my_parser.add_argument('--var4', default=0.5, nargs='?', type=float,  help='gates sparsity')
my_parser.add_argument('--num_of_tasks', default=5, nargs='?', type=int, help='number of tasks to train on')


# Get args and set config
args = my_parser.parse_args()

exp_name = args.exp_name
os.makedirs('./files/'+exp_name, exist_ok=True)
rng = np.random.default_rng(int(args.seed))

config = SerialConfig()

if args.experiment_type == 'same_net': # this shows that sequential fails right away
        config.same_rnn = True
if args.experiment_type == 'train_to_criterion': # demo ttc and introduce metric ttc. Start testing for accelerateion
        config.same_rnn = True
        config.train_to_criterion = True
if args.experiment_type == 'rehearsal':  # introduce first time learning vs subsequent rehearsals.
        config.same_rnn = True
        config.train_to_criterion = True
        config.use_rehearsal = True
if args.experiment_type == 'random_gates_mul': # Task aware algorithm. Improved CL but impaired FT
    config = Gates_mul_config()
if args.experiment_type == 'random_gates_add': # Task aware algorithm. Improved CL but impaired FT
    config = Gates_add_config()
if args.experiment_type == 'shuffle_mul': # Task aware algorithm. Improved CL but impaired FT
    config = Shuffle_mul_config()
if args.experiment_type == 'shuffle_add': # Task aware algorithm. Improved CL but impaired FT
    config = Shuffle_add_config()
if args.experiment_type == 'random_gates_no_rehearsal': # Task aware algorithm. Improved CL but impaired FT
    config = Gates_no_rehearsal_config()


############################################
config.set_strings( exp_name)
config.exp_signature = f"{args.var1:1.1f}_{args.var3:1.1f}_{args.var4:1.1f}_"
config.num_of_tasks = args.num_of_tasks
config.saved_model_sig = f'seed{args.seed}_paradigm_{"shuf" if config.paradigm_shuffle else "seq"}_{"mul" if config.use_multiplicative_gates else "add"}_tasks_{config.num_of_tasks}_'
config.exp_signature = config.saved_model_sig +  config.exp_signature #+ f'_{"corr" if config.load_gates_corr else "nocorr"}_{"co" if config.use_cognitive_observer else "noc"}_{"reh" if config.use_rehearsal else "noreh"}_{"tc" if config.train_to_criterion else "nc"}_{"mul" if config.use_multiplicative_gates else "add"}_{"gates" if config.use_gates else "nog"}'
config.saved_model_path = './data/'+ f'saved_model_{config.saved_model_sig}.torch'
config.saved_gates_path = './data/'+ f'saved_gates_{config.saved_model_sig}.torch'
config.gates_mean = args.var1
config.gates_std = args.var3
config.gates_sparsity = args.var4
config.train_gates = False

config.higher_order = True
if config.higher_order:
    config.random_rehearsals = 10
    config.save_model = False
    config.load_saved_rnn1 = True
###--------------------------Training configs--------------------------###
if not args.seed == 0: # if given seed is not zero, shuffle the task_seq
    #Shuffle tasks
    idx = rng.permutation(range(len(config.tasks)))
    new_order_tasks = ((np.array(config.tasks)[idx]).tolist())
    config.tasks = new_order_tasks # assigns the order to tasks_id_name but retains the standard task_id

task_seq = []
if config.paradigm_sequential:
    # Add tasks gradually with rehearsal 1 2 1 2 3 1 2 3 4 ...
    if config.use_rehearsal:
        task_sub_seqs = [[config.tasks_id_name[i] for i in range(s)] for s in range(2, args.num_of_tasks+1)] # interleave tasks and add one task at a time
        for sub_seq in task_sub_seqs: 
            task_seq+=sub_seq
        task_seq+=sub_seq # One additional final rehearsal, 
    else:
        task_seq = [config.tasks_id_name[i] for i in range(args.num_of_tasks)]

        
elif config.paradigm_shuffle:
    sub_seq = [config.tasks_id_name[i] for i in range(args.num_of_tasks)]
    task_seq = rng.choice(sub_seq, size = (config.no_shuffled_trials), replace=True)
    config.max_trials_per_task = config.batch_size  # set block size to 1
    config.print_every_batches = 100
    config.no_shuffled_trials = 12000

# add, at the end, the last 3 tasks from the unlearned pile:
task_seq_sequential = []
no_of_tasks_left = len(config.tasks_id_name)- args.num_of_tasks
if no_of_tasks_left > 0: novel_task_id = args.num_of_tasks + rng.integers(no_of_tasks_left)
# learn one novel task then rehearse previously learned + novel task
task_seq_sequential = [config.tasks_id_name[novel_task_id]] + sub_seq + [config.tasks_id_name[novel_task_id]] 

# if config.use_rehearsal:
#     to_no = min(args.num_of_tasks+4, len(config.tasks_id_name)+1)

#     task_sub_seqs = [[config.tasks_id_name[i] for i in range(args.num_of_tasks, s)] for s in range(args.num_of_tasks+2, to_no)] # interleave tasks and add one task at a time
#     for sub_seq in task_sub_seqs: 
#         # task_seq_sequential = np.concatenate([task_seq, np.stack(sub_seq)])
#         task_seq_sequential +=sub_seq
#     task_seq_sequential +=sub_seq # one last rehearsal
# else:
#     # task_seq = np.concatenate([task_seq, np.stack(config.tasks_id_name)[-no_of_tasks_left:]])
#     task_seq_sequential = (config.tasks_id_name)[-no_of_tasks_left:]

# Now adding many random rehearsals:
task_seq_random = []
for _ in range(config.random_rehearsals):
    sub_seq = [config.tasks_id_name[i] for i in range(args.num_of_tasks)]
    random.shuffle(sub_seq)
    task_seq_random+=sub_seq

# main loop
from train import train, optimize

net = RNN_MD(config)
net.to(config.device)
training_log = SerialLogger(config=config)
testing_log = SerialLogger(config=config)

if config.load_saved_rnn1:
    net.load_state_dict(torch.load(config.saved_model_path))
    # net.rnn.gates = torch.load(config.saved_gates_path)

else:
    config.criterion_shuffle_paradigm = 0.9
    testing_log, training_log, net = train(config, net, task_seq, testing_log, training_log , step_i = 0 )
    config.criterion_shuffle_paradigm = 10 # raise it too high so it is no longer stopping training. 
       
    if not config.higher_order: # run the novel task sequential test

        config.print_every_batches = 1
        config.train_to_criterion = True
        config.max_trials_per_task = 100000
        step_i = training_log.stamps[-1]+1 if training_log.stamps.__len__()>0 else 0
        training_log.start_testing_at , testing_log.start_testing_at = step_i, step_i
   
        testing_log, training_log, net = train(config, net, task_seq_sequential, testing_log, training_log, step_i = training_log.stamps[-1]+1 )
   
    else: # jump into random pre-training and save the net
        testing_log, training_log, net = train(config, net, task_seq_random, testing_log, training_log, step_i = training_log.stamps[-1]+1 )
        if config.save_model:
            print(' Saving RNN1 to: ----', config.saved_model_path)
            torch.save(net.state_dict(), config.saved_model_path)


if config.higher_order:
    cog_net = Cognitive_Net(input_size=10+config.hidden_size+config.output_size, hidden_size=1, output_size = config.md_size)
    cog_net.to(config.device)

    step_i = training_log.stamps[-1]+1 if training_log.stamps.__len__()>0 else 0
    training_log.start_optimizing_at , testing_log.start_optimizing_at = step_i, step_i
    
    if config.paradigm_shuffle:
        config.print_every_batches = 10
        config.train_to_criterion = True
        config.max_trials_per_task = 40000
    testing_log, training_log, net = optimize(config, net, cog_net, task_seq_random, testing_log, training_log, step_i = step_i )

np.save('./files/'+ config.exp_name+f'/testing_log_{config.exp_signature}.npy', testing_log, allow_pickle=True)
np.save('./files/'+ config.exp_name+f'/training_log_{config.exp_signature}.npy', training_log, allow_pickle=True)
np.save('./files/'+ config.exp_name+f'/config_{config.exp_signature}.npy', config, allow_pickle=True)
print('testing logs saved to : '+ './files/'+ config.exp_name+f'/testing_log_{config.exp_signature}.npy')


## Plots
import matplotlib
no_of_values = len(config.tasks)
norm = mpl.colors.Normalize(vmin=min([0,no_of_values]), vmax=max([0,no_of_values]))
cmap_obj = matplotlib.cm.get_cmap('Set1') # tab20b tab20
cmap = mpl.cm.ScalarMappable(norm=norm, cmap=cmap_obj)

log = testing_log
training_log.switch_trialxxbatch.append(training_log.stamps[-1])
num_tasks = len(config.tasks)
already_seen =[]
title_label = 'Training tasks sequentially ---> \n    ' + config.exp_name
max_x = training_log.stamps[-1]
fig, axes = plt.subplots(num_tasks+4,1, figsize=[9,7])
for logi in range(num_tasks):
        ax = axes[ logi ] # log i goes to the col direction -->
        ax.set_ylim([-0.1,1.1])
#         ax.axis('off')
        ax.plot(testing_log.stamps, [test_acc[logi] for test_acc in testing_log.accuracies], linewidth=1)
        ax.plot(testing_log.stamps, np.ones_like(testing_log.stamps)*0.5, ':', color='grey', linewidth=1)
        ax.set_ylabel(config.human_task_names[logi], fontdict={'color': cmap.to_rgba(logi)})
        ax.set_xlim([0, max_x])
        if (logi == num_tasks-1) and config.use_cognitive_observer and config.train_cog_obs_on_recent_trials: # the last subplot, put the preds from cog_obx
            cop = np.stack(training_log.cog_obs_preds).reshape([-1,100,15])
            cop_colors = np.argmax(cop, axis=-1).mean(-1)
            for ri in range(max_x-2):
                if ri< len(cop_colors):ax.axvspan(ri, ri+1, color =cmap.to_rgba(cop_colors[ri]) , alpha=0.2)
        else:            
            for ri in range(len(training_log.switch_trialxxbatch)-1):
                ax.axvspan(training_log.switch_trialxxbatch[ri], training_log.switch_trialxxbatch[ri+1], color =cmap.to_rgba(training_log.switch_task_id[ri]) , alpha=0.2)
for ti, id in enumerate(training_log.switch_task_id):
    if id not in already_seen:
        already_seen.append(id)
        task_name = config.human_task_names[id]
        axes[0].text(training_log.switch_trialxxbatch[ti], 1.3, task_name, color= cmap.to_rgba(id) )

gs = np.stack(training_log.gradients)

glabels =  ['inp_w', 'inp_b', 'rnn_w', 'rnn_b', 'out_w', 'out_b']
ax = axes[num_tasks+0]
gi =0
ax.plot(training_log.stamps, gs[:,gi+1], label= glabels[gi])
gi =2
ax.plot(training_log.stamps, gs[:,gi+1], label= glabels[gi])
gi = 4
ax.plot(training_log.stamps, gs[:,gi+1], label= glabels[gi])
ax.legend()
ax.set_xlim([0, max_x])
ax = axes[num_tasks+1]
ax.plot(testing_log.stamps,  [np.mean(list(la.values())) for la in testing_log.accuracies] )
ax.set_ylabel('avg acc')

ax.plot(testing_log.stamps, np.ones_like(testing_log.stamps)*0.9, ':', color='grey', linewidth=1)
ax.set_xlim([0, max_x])
ax.set_ylim([-0.1,1.1])

ax = axes[num_tasks+2]
ax.plot(training_log.switch_trialxxbatch[:-1], training_log.trials_to_crit)
ax.set_ylabel('ttc')
ax.set_xlim([0, max_x])

ax = axes[num_tasks+3]
if len(training_log.stamps) == len(training_log.frustrations):
    ax.plot(training_log.stamps, training_log.frustrations)
ax.set_ylabel('frust')
ax.set_xlim([0, max_x])


final_accuracy_average = np.mean(list(testing_log.accuracies[-1].values()))
plt.savefig('./files/'+ config.exp_name+f'/acc_summary_{config.exp_signature}_{training_log.stamps[-1]}_{final_accuracy_average:1.2f}.jpg', dpi=300)

