#!/usr/bin/env python
# coding: utf-8
use_PFCMD = False
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
import torch
import torch.nn as nn
from torch.nn import functional as F
# tasks
import gym
import neurogym as ngym
from neurogym.wrappers import ScheduleEnvs
from neurogym.utils.scheduler import RandomSchedule
# models
from models.PFC_gated import RNN_MD
# from models.GRUB import RNN_MD
from configs.refactored_configs import *
from models.PFC_gated import Cognitive_Net
from logger.logger import SerialLogger
from train import train, optimize
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
my_parser.add_argument('exp_name',  default='cluster', type=str, nargs='?', help='Experiment name, also used to create the path to save results')
# my_parser.add_argument('--experiment_type', default='shuffle_mul', nargs='?', type=str, help='Which experimental or setup to run: "pairs") task-pairs a b a "serial") Serial neurogym "interleave") Interleaved ')
# my_parser.add_argument('--experiment_type', default='noisy_mean', nargs='?', type=str, help='Which experimental or setup to run: "pairs") task-pairs a b a "serial") Serial neurogym "interleave") Interleaved ')
# my_parser.add_argument('--experiment_type', default='shrew_task', nargs='?', type=str, help='Which experimental or setup to run: "pairs") task-pairs a b a "serial") Serial neurogym "interleave") Interleaved ')
# to run shrew task: (1) set model to GRUB, (2) consider nll or mse main loss, (3) switch train.py to use net invoke command with gt.
my_parser.add_argument('--experiment_type', default='random_gates_add', nargs='?', type=str, help='Which experimental or setup to run: "pairs") task-pairs a b a "serial") Serial neurogym "interleave") Interleaved ')
# my_parser.add_argument('--experiment_type', default='random_gates_rehearsal_no_train_to_criterion', nargs='?', type=str, help='Which experimental or setup to run: "pairs") task-pairs a b a "serial") Serial neurogym "interleave") Interleaved ')
my_parser.add_argument('--seed', default=0, nargs='?', type=int,  help='Seed')
my_parser.add_argument('--var1',  default=0, nargs='?', type=float, help='gates mean ')
# my_parser.add_argument('--var2', default=-0.3, nargs='?', type=float, help='the ratio of active neurons in gates ')
my_parser.add_argument('--var3',  default=0.2, nargs='?', type=float, help='gates std')
my_parser.add_argument('--var4', default=0.4, nargs='?', type=float,  help='gates sparsity')
my_parser.add_argument('--num_of_tasks', default=14, nargs='?', type=int, help='number of tasks to train on')

# Get args and set config
args = my_parser.parse_args()

exp_name = args.exp_name
os.makedirs('./files/'+exp_name, exist_ok=True)
rng = np.random.default_rng(int(args.seed))

config = SerialConfig()

config.use_multiplicative_gates = False
config.use_additive_gates = False
if args.experiment_type == 'same_net': # this shows that sequential fails right away
        config.same_rnn = True
        config.train_to_criterion = False
        config.use_rehearsal = False
if args.experiment_type == 'train_to_criterion': # demo ttc and introduce metric ttc. Start testing for accelerateion
        config.same_rnn = True
        config.train_to_criterion = True
        config.use_rehearsal = False
if args.experiment_type == 'random_gates_no_rehearsal': 
    config = Gates_no_rehearsal_config()
if args.experiment_type == 'random_gates_only': 
    config = random_gates_only_config()
if args.experiment_type == 'random_gates_rehearsal_no_train_to_criterion': 
    config = Gates_rehearsal_no_train_to_criterion_config()

if args.experiment_type == 'rehearsal':  # introduce first time learning vs subsequent rehearsals.
        config.same_rnn = True
        config.train_to_criterion = True
        config.use_rehearsal = True
if args.experiment_type == 'random_gates_mul': 
    config = Gates_mul_config()
if args.experiment_type == 'random_gates_add': 
    config = Gates_add_config()
if args.experiment_type == 'random_gates_both': 
    config = Gates_add_config()
    config.use_multiplicative_gates =True  
if args.experiment_type == 'shuffle_mul': 
    config = Shuffle_mul_config()
if args.experiment_type == 'shuffle_add': 
    config = Shuffle_add_config()
if args.experiment_type == 'shrew_task' or args.experiment_type == 'noisy_mean': 
    config = Schizophrenia_config(args.experiment_type)
    args.num_of_tasks = len(config.tasks)
    config.use_additive_gates = True
    config.use_multiplicative_gates = False
    config.switches = 20
    config.max_trials_per_task = int(40 * config.batch_size)
    config.paradigm_alternate = True
    config.train_to_criterion = False
    config.abort_rehearsal_if_accurate = False
    config.one_batch_optimization = False
############################################
config.set_strings( exp_name)
config.exp_signature = f"{args.var1:1.1f}_{args.var3:1.1f}_{args.var4:1.1f}_"
config.num_of_tasks = args.num_of_tasks
config.saved_model_sig = f'seed{args.seed}_paradigm_{"shuf" if config.paradigm_shuffle else "seq"}_{"mul" if config.use_multiplicative_gates else "add"}_tasks_{config.num_of_tasks}_'
config.exp_signature = config.saved_model_sig +  config.exp_signature #+ f'_{"corr" if config.load_gates_corr else "nocorr"}_{"co" if config.use_cognitive_observer else "noc"}_{"reh" if config.use_rehearsal else "noreh"}_{"tc" if config.train_to_criterion else "nc"}_{"mul" if config.use_multiplicative_gates else "add"}_{"gates" if config.use_gates else "nog"}'
config.saved_model_path = './files/'+ config.exp_name+ f'/saved_model_{config.saved_model_sig}.torch'

config.gates_divider = 2.0
config.gates_offset = 0.10

config.train_gates = False
config.save_model = False
config.save_model = True

config.optimize_policy  = False
config.optimize_td      = False
config.optimize_bu      = True
config.cog_net_hidden_size = 100

config.higher_order = not config.save_model
if config.higher_order:
    config.train_to_criterion = True
    config.random_rehearsals = int(args.var1) if config.paradigm_sequential else 4000
    config.load_saved_rnn1 = not config.save_model
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
        # probabilistic rehearsal where old tasks are rehearsed exponentially less frequent.
        # task_sub_seqs = [[config.tasks_id_name[i] for i in range(s) if np.random.uniform() < np.exp(-0.1*((s-i)+.3))+config.rehearsal_base_prob] for s in range(2, args.num_of_tasks+1)] # interleave tasks and add one task at a time
        for sub_seq in task_sub_seqs: 
            task_seq+=sub_seq
        task_seq+=sub_seq # One additional final rehearsal, 
        # task_seq+=sub_seq # yet another additional final rehearsal, 
    else:
        task_seq = [config.tasks_id_name[i] for i in range(args.num_of_tasks)]
    if config.paradigm_alternate:
        task_seq = task_seq * config.switches
        
elif config.paradigm_shuffle:
    sub_seq = [config.tasks_id_name[i] for i in range(args.num_of_tasks)]
    task_seq = rng.choice(sub_seq, size = (config.no_shuffled_trials), replace=True)
    config.max_trials_per_task = config.batch_size  # set block size to 1
    config.print_every_batches = 100
    config.no_shuffled_trials = 12000

# add, at the end, one novel task from the unlearned pile:
task_seq_sequential = []
no_of_tasks_left = len(config.tasks_id_name)- args.num_of_tasks
if no_of_tasks_left > 0: 
    novel_task_id = args.num_of_tasks + rng.integers(no_of_tasks_left)
    sub_seq = [config.tasks_id_name[i] for i in range(args.num_of_tasks)]
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
task_seq_optimize = []
for _ in range(500):
    sub_seq = [config.tasks_id_name[i] for i in range(args.num_of_tasks)]
    if not config.paradigm_alternate: random.shuffle(sub_seq)
    task_seq_optimize+=sub_seq

# main loop
net = RNN_MD(config)
net.to(config.device)
training_log = SerialLogger(config=config)
testing_log = SerialLogger(config=config)

if config.load_saved_rnn1:
    net.load_state_dict(torch.load(config.saved_model_path))
    print('____ loading model from : ___ ', config.saved_model_path)
else: # if no pre-trained network proceed with the main training loop.
    config.criterion_shuffle_paradigm = 1.90 # accuracy crit for shuffled paradigm to acheive so it matches the sequential paradigm.
    testing_log, training_log, net = train(config, net, task_seq, testing_log, training_log , step_i = 0 )
    config.criterion_shuffle_paradigm = 10 # raise it too high so it is no longer stopping training. 
       
    if not config.higher_order: # run the novel task sequential test

        config.print_every_batches = 10
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
    # cog_net = Cognitive_Net(input_size=10+config.hidden_size+config.output_size, hidden_size=config.cog_net_hidden_size, output_size = config.md_size)
    cog_net = Cognitive_Net(input_size=10+config.hidden_size+config.md_size, hidden_size=config.cog_net_hidden_size, output_size = config.md_size)
    cog_net.to(config.device)

    step_i = training_log.stamps[-1]+1 if training_log.stamps.__len__()>0 else 0
    training_log.start_optimizing_at , testing_log.start_optimizing_at = step_i, step_i
    
    if config.paradigm_shuffle:
        config.print_every_batches = 10
        config.train_to_criterion = True
        config.max_trials_per_task = 80000
    testing_log, training_log, net = optimize(config, net, cog_net, task_seq_optimize, testing_log, training_log, step_i = step_i )

np.save('./files/'+ config.exp_name+f'/testing_log_{config.exp_signature}.npy', testing_log, allow_pickle=True)
np.save('./files/'+ config.exp_name+f'/training_log_{config.exp_signature}.npy', training_log, allow_pickle=True)
np.save('./files/'+ config.exp_name+f'/config_{config.exp_signature}.npy', config, allow_pickle=True)
if use_PFCMD:
    np.save('./files/'+ config.exp_name+f'/md_activities_{config.exp_signature}.npy', net.md_activities, allow_pickle=True)
print('testing logs saved to : '+ './files/'+ config.exp_name+f'/testing_log_{config.exp_signature}.npy')


## Plots
from analysis import visualization as viz

viz.plot_accuracies(config, training_log=training_log, testing_log=testing_log)

if config.higher_order and not config.optimize_policy:
    viz.plot_credit_assignment_inference(config, training_log=training_log, testing_log=testing_log)

# TODO Follow up on the gates_divider and its potential effects on TU BU or policy. 
# TODO Review 