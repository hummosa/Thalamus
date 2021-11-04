#!/usr/bin/env python
import os
from subprocess import call
import subprocess as sp
from time import sleep
import sys    

#%% Create the parameters for experiments in a table (expVars list of lists)
# Var1 = range(500,1501, 500)
experiments = ['same_net', 'train_to_criterion', 'rehearsal', 'random_gates',  'correlated_gates', 'cognitive_observer', ]
experiments = [ 'rehearsal', 'random_gates',  'correlated_gates',]
experiments = [ 'record_correlations',  ]
experiments = [ 'cognitive_observer'  ]
experiments = [ 'rehearsal',]
# experiments = [ 'correlated_gates',]

exp_name = 'factorized'
Seeds = range(0,50)
Var2 = [-0.3] #MDprob, currently gaussian cuttoff #[0.0001, 0.001]#range(0,3, 1) #gates mean
Var3 = [100,] #0 1 add mul  #, 3.5, 4, 4.5, 5, 5.5, 6, 6.5]
Var4 = experiments #[.1, .3, .5, .9, 0.95, 1.]#[0.1, 0.5, 1.] #range(30,91, 20) #Gates density


expVars =  [[x1, x2, x3, x4] for x1 in Seeds for x2 in Var2 for x3 in Var3 for x4 in Var4 ]
# [expVars[i].insert(0, i) for i in range(len(expVars))] # INSERT exp ID # in the first col.
print ('Total no of experiments generated : ', len(expVars))
print(expVars)


#%% Load neuron
if sys.platform != 'win32':
    pass 
    # if slurm linux, can load any modules here.
    # sp.check_call("module load nrn/nrn-7.4", shell=True)  #Here it needs no splitting and shell=True
    

#%% Write the bash file to run an experiment

for par_set in expVars:
    seed, var2, var3, var4 = par_set
    experiment_type = var4
    exp_name = 'factorized/' + experiment_type
    
    sbatch_lines =["#!/bin/bash"
    ,  "#-------------------------------------------------------------------------------"
    , "#  SBATCH CONFIG"
    , "#-------------------------------------------------------------------------------"
    , "#SBATCH --nodes=1"
    , "#SBATCH -t 00:40:00"
    , "#SBATCH --gres=gpu:1"
    , "#SBATCH --constraint=high-capacity"
    , "#SBATCH --mem={}G".format(64 if experiment_type == 'cognitive_observer' else 10)
    # , '#SBATCH --output="showme.out"'
    , "#SBATCH --job-name=cl_{}".format(exp_name)
    , "#-------------------------------------------------------------------------------"
    , "## Run model "]
    
    # args = ['-c "x%d=%g" ' % (i, par_set[i]) for i in range(len(par_set))]
    #                                   use_gates same_rnn train_to_critoeron
    command_line = 'python cognitive_serialtasks.py  {} 1 1 1 --seed={} --var2={} --var3={} --experiment_type={}'.format(exp_name, seed, var2, var3, experiment_type) 
    win_command_line = command_line + ' --os=windows'
    
    fsh = open('serial_bash_generated.sh', 'w')
    fsh.write("\n".join(sbatch_lines))
    fsh.writelines(["\n", command_line])
    fsh.close()

    # Run the simulation
    if sys.platform == 'win32':
        sp.check_call(win_command_line.split(' '))
    else: #assuming Linux
        sp.check_call('sbatch serial_bash_generated.sh'.split(' '))
    
    sleep(1)    # Inserts a 1 second pause
    
    