#!/usr/bin/env python
# coding: utf-8
use_PFCMD = False
import os
import sys

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
# from models.PFC_gated import RNN_MD_GRU as RNN_MD
# from models.GRUB import RNN_MD
from configs.refactored_configs import *
from models.PFC_gated import Cognitive_Net
from logger.logger import SerialLogger
from train import train, optimize
from utils import *
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
my_parser.add_argument('exp_name',  default='cluster_2', type=str, nargs='?', help='Experiment name, also used to create the path to save results')
# my_parser.add_argument('exp_name',  default='cluster_convergence4/random_gates_mul', type=str, nargs='?', help='Experiment name, also used to create the path to save results')
# my_parser.add_argument('--experiment_type', default='shuffle_mul', nargs='?', type=str, help='Which experimental or setup to run: "pairs") task-pairs a b a "serial") Serial neurogym "interleave") Interleaved ')
# my_parser.add_argument('--experiment_type', default='noisy_mean', nargs='?', type=str, help='Which experimental or setup to run: "pairs") task-pairs a b a "serial") Serial neurogym "interleave") Interleaved ')
# my_parser.add_argument('--experiment_type', default='shrew_task', nargs='?', type=str, help='Which experimental or setup to run: "pairs") task-pairs a b a "serial") Serial neurogym "interleave") Interleaved ')
# to run shrew task: (1) set model to GRUB, (2) consider nll or mse main loss, (3) switch train.py to use net invoke command with gt.
# my_parser.add_argument('--experiment_type', default='same_net', nargs='?', type=str, help='Which experimental or setup to run: "pairs") task-pairs a b a "serial") Serial neurogym "interleave") Interleaved ')
my_parser.add_argument('--experiment_type', default='random_gates_mul', nargs='?', type=str, help='Which experimental or setup to run: "pairs") task-pairs a b a "serial") Serial neurogym "interleave") Interleaved ')
# my_parser.add_argument('--experiment_type', default='random_gates_rehearsal_no_train_to_criterion', nargs='?', type=str, help='Which experimental or setup to run: "pairs") task-pairs a b a "serial") Serial neurogym "interleave") Interleaved ')
my_parser.add_argument('--seed', default=4, nargs='?', type=int,  help='Seed')
my_parser.add_argument('--var1',  default=1000, nargs='?', type=float, help='no of loops optim task id')
# my_parser.add_argument('--var2', default=-0.3, nargs='?', type=float, help='the ratio of active neurons in gates ')
my_parser.add_argument('--var3',  default=1.0, nargs='?', type=float, help='actually use task_ids')
my_parser.add_argument('--var4', default=1.0, nargs='?', type=float,  help='gates sparsity')
my_parser.add_argument('--no_of_tasks', default=3, nargs='?', type=int, help='number of tasks to train on')

# Get args and set config
args = my_parser.parse_args()

exp_name = args.exp_name
os.makedirs('./files/'+exp_name, exist_ok=True)
rng = np.random.default_rng(int(args.seed))

### Dataset
dataset_name = 'neurogym' #'split_mnist'  'rotated_mnist' 'neurogym' 'hierarchical_reasoning' 'nassar'
config = SerialConfig(dataset_name) 

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
    config = Gates_no_rehearsal_config(dataset_name)
if args.experiment_type == 'random_gates_only': 
    config = random_gates_only_config(dataset_name)
if args.experiment_type == 'random_gates_rehearsal_no_train_to_criterion': 
    config = Gates_rehearsal_no_train_to_criterion_config(dataset_name)

if args.experiment_type == 'rehearsal':  # introduce first time learning vs subsequent rehearsals.
        config.same_rnn = True
        config.train_to_criterion = True
        config.use_rehearsal = True
if args.experiment_type == 'random_gates_mul': 
    # config = Gates_mul_config('split_mnist')
    config = Gates_mul_config(dataset_name)
if args.experiment_type == 'random_gates_add': 
    config = Gates_add_config(dataset_name)
if args.experiment_type == 'random_gates_both': 
    config = Gates_add_config(dataset_name)
    config.use_multiplicative_gates =True  
if args.experiment_type == 'shuffle_mul': 
    config = Shuffle_mul_config(dataset_name)
if args.experiment_type == 'shuffle_add': 
    config = Shuffle_add_config(dataset_name)
if args.experiment_type == 'shrew_task' or args.experiment_type == 'noisy_mean': 
    config = Schizophrenia_config(args.experiment_type)
    args.no_of_tasks = len(config.tasks)
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
config.no_of_tasks = args.no_of_tasks
config.saved_model_sig = f'seed{args.seed}_{"mul" if config.use_multiplicative_gates else "add"}_tasks_{config.no_of_tasks}_{dataset_name}'
config.exp_signature = config.saved_model_sig +  config.exp_signature #+ f'_{"corr" if config.load_gates_corr else "nocorr"}_{"co" if config.use_cognitive_observer else "noc"}_{"reh" if config.use_rehearsal else "noreh"}_{"tc" if config.train_to_criterion else "nc"}_{"mul" if config.use_multiplicative_gates else "add"}_{"gates" if config.use_gates else "nog"}'
config.saved_model_path = './files/'+ config.exp_name+ f'/saved_model_{config.saved_model_sig}.torch'

config.gates_divider = 1.0
config.gates_offset = 0.0
config.md_context_id_amplifier = float(args.var3)

config.train_gates = False
config.save_model = False
config.save_model = True

config.optimize_policy  = False
config.optimize_td      = False
config.optimize_bu      = True
config.cog_net_hidden_size = 100

config.no_latent_updates = int(args.var1)
config.actually_use_task_ids = False
config.lr_multiplier = float(args.var4)
config.bu_adam = True # SGD
config.use_weight_updates = True
config.detect_convergence = False
config.convergence_plan = 'save_and_present_novel'

config.lr_multiplier = 10 #100
config.train_to_criterion = False
config.max_trials_per_task = int(200 * config.batch_size)
config.use_rehearsal = False

ddir = './files/'+ config.exp_name + '/latent_updates'
import shutil
try:
    shutil.rmtree(ddir)
except:
    pass
os.makedirs(ddir, exist_ok=True)

config.test_latent_recall = False  # in the second block of training, repeatedly test recall of previous latents and ability to recover accuracy wiht latent updates as a function of weight updates on the next task.
config.md_loop_rehearsals = 20 # Train the sequence of tasks for this many times
config.train_novel_tasks = True  # then add a novel task, then the sequence again, and then the same novel task at the end. 
config.higher_order = False # not config.save_model
config.higher_cog_test_multiple = 2
config.training_loss = 'mse'



# add, rehearse the md_loop then at the end, one novel task from the unlearned pile:
novel_task_ids = get_novel_task_ids(args, rng, config) 

config.load_saved_rnn1 = not config.save_model


###--------------------------Training configs--------------------------###
if not args.seed == 0: # if given seed is not zero, shuffle the task_seq
    #Shuffle tasks
    if config.dataset == 'neurogym':
        new_order_tasks = get_tasks_order(args.seed) # make sure the first two are opposites and then shuffle the rest
    else:
        idx = rng.permutation(range(len(config.tasks)))
        new_order_tasks = ((np.array(config.tasks)[idx]).tolist())
    config.tasks = new_order_tasks # assigns the order to tasks_id_name but retains the standard task_id
# main loop
net = RNN_MD(config)
net.to(config.device)
training_log = SerialLogger(config=config)
testing_log = SerialLogger(config=config)

if config.load_saved_rnn1:
    net.load_state_dict(torch.load(config.saved_model_path))
    print('____ loading model from : ___ ', config.saved_model_path)
    # Train only LU to crit for testing purposes
    third_phase_multiple = 2
    # task_seq3 = [config.tasks_id_name[i] for i in range(args.no_of_tasks)]  * third_phase_multiple
    task_seq3 = [config.tasks_id_name[i] for i in novel_task_ids]  * third_phase_multiple
    config.train_to_criterion = True
    config.use_weight_updates = False
    config.detect_convergence = False
    config.max_trials_per_task = int(200*config.batch_size)
    cog_net = Cognitive_Net(input_size=10+config.hidden_size+config.md_size, hidden_size=config.cog_net_hidden_size, output_size = config.md_size)
    cog_net.to(config.device)
    testing_log, training_log, net = optimize(config, net,cog_net, task_seq3, testing_log, training_log , step_i = 0 )
    exp_name = args.exp_name + '_testing'
    config.exp_name = config.exp_name + '_testing'
    os.makedirs('./files/'+exp_name, exist_ok=True)
else: # if no pre-trained network proceed with the main training loop.
    ###############################################################################################################
    # Training Phases
    # Train with WU first
    first_phase_multiple = 2
    task_seq1 = [config.tasks_id_name[i] for i in range(args.no_of_tasks)] * first_phase_multiple
    config.train_to_criterion = False
    config.use_latent_updates = False
    testing_log, training_log, net = train(config, net, task_seq1 , testing_log, training_log , step_i = 0 )

    # Train with WU+LU but full blocks
    second_phase_multiple = 3
    config.use_latent_updates = True
    task_seq2 = [config.tasks_id_name[i] for i in range(args.no_of_tasks)]  * second_phase_multiple
    testing_log, training_log, net = train(config, net, task_seq2, testing_log, training_log , step_i = training_log.stamps[-1]+1 )

    # Train with WU+LU Train to Crit
    third_phase_multiple = 20
    task_seq3 = [config.tasks_id_name[i] for i in range(args.no_of_tasks)]  * third_phase_multiple
    config.train_to_criterion = False
    config.detect_convergence = True
    config.max_trials_per_task = int(100*config.batch_size)
    testing_log, training_log, net = train(config, net, task_seq3, testing_log, training_log , step_i = training_log.stamps[-1]+1 )

    if config.save_model:
        print(' Saving RNN1 to: ----', config.saved_model_path)
        torch.save(net.state_dict(), config.saved_model_path)

    ###############################################################################################################
    if config.train_novel_tasks: # run the novel task sequential test
        third_phase_multiple = 2
        # task_seq3 = [config.tasks_id_name[i] for i in range(args.no_of_tasks)]  * third_phase_multiple
        task_seq3 = [config.tasks_id_name[i] for i in novel_task_ids]  * third_phase_multiple
        # random.perm
        config.train_to_criterion = True
        config.use_weight_updates = False
        config.detect_convergence = False
        config.max_trials_per_task = int(200*config.batch_size)
        cog_net = Cognitive_Net(input_size=10+config.hidden_size+config.md_size, hidden_size=config.cog_net_hidden_size, output_size = config.md_size)
        cog_net.to(config.device)
        step_i = training_log.stamps[-1]+1 if training_log.stamps.__len__()>0 else 0
        training_log.start_testing_at , testing_log.start_testing_at = step_i, step_i
        testing_log, training_log, net = optimize(config, net,cog_net, task_seq3, testing_log, training_log , step_i = step_i )
       

if config.higher_order:
    config.train_to_criterion = True
    config.random_rehearsals = 5 if config.paradigm_sequential else 4000 # if needing to train the network from scratch use this number of rehearsals to prepare for the latent updates phase

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
print('testing logs saved to : '+ './files/'+ config.exp_name+f'/testing_log_{config.exp_signature}.npy')

## Plots
from analysis import visualization as viz

# viz.plot_accuracies(config, training_log=training_log, testing_log=testing_log)

# if config.higher_order and not config.optimize_policy:
viz.plot_long_term_cluster_discovery(config, training_log, testing_log)
try:
    viz.plot_credit_assignment_inference(config, training_log=training_log, testing_log=testing_log)
except:
    pass
