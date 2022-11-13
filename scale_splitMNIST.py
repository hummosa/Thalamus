#!/usr/bin/env python
import os, re
from subprocess import call
import subprocess as sp
from time import sleep
import sys    

check_every = 100 # seconds to recheck if jobs are done
job_limit = 100 # jobs allowed submitted at once.

#%% Create the parameters for experiments in a table (expVars list of lists)
experiments = ['few_shot_testing']
experiments = ['iclr_testing']

no_of_tasks_to_run = [5] 
# exp_sig = 'cluster_split_mnist_sparsity_rehearsal_5tasks_testing_dataset'
# exp_sig = 'cluster_neurogym_tasks10_to_convergence2'
exp_sig = 'cluster_iclr_Wz_tests2'

Seeds = list(range(3,12))#[6, 7, 8, 10,  14, ]#range(11,15)
Var1 = [0,1] # train or not LU optim SGD, ADam 
Var2 = no_of_tasks_to_run # used to pass no of exp  #[-0.3] #MDprob, currently gaussian cuttoff #[0.0001, 0.001]#range(0,3, 1) #gates mean
Var3 = [0., 0.5,]# gates sparsity LU LR multiplier ,100] # [(x/10) for x in range(1,5, 1)] #gates_std  #, 3.5, 4, 4.5, 5, 5.5, 6, 6.5]
Var4 = [1., 1.5, 2.] # gates std few_)shots [(x/10) for x in range(0,6, 4)] #gates_sparsity  #, 3.5, 4, 4.5, 5, 5.5, 6, 6.5]

expVars =  [[seed, experiment, x1, x2, x3, x4] for seed in Seeds for experiment in experiments for x1 in Var1 for x2 in Var2 for x3 in Var3 for x4 in Var4 ]
# [expVars[i].insert(0, i) for i in range(len(expVars))] # INSERT exp ID # in the first col.
print ('Total no of experiments generated : ', len(expVars))
# psqnt(expVars)

#%% Load neuron
if sys.platform != 'win32':
    pass 
    # if slurm linux, can load any modules here.
    # sp.check_call("module load nrn/nrn-7.4", shell=True)  #Here it needs no splitting and shell=True
    

#%% Write the bash file to run an experiment

for jobi, par_set in enumerate(expVars):
    seed, experiment, var1, var2, var3, var4 = par_set
    experiment_type = experiment
    exp_name = exp_sig +'/' + experiment_type
    
    ## SKIPS:
    # if jobi < 3: continue
    # if seed==0: continue
    
    sbatch_lines =["#!/bin/bash"
    ,  "#-------------------------------------------------------------------------------"
    , "#  SBATCH CONFIG"
    , "#-------------------------------------------------------------------------------"
    , "#SBATCH --nodes=1"
    , "#SBATCH -t 01:40:00"
    , "#SBATCH --gres=gpu:1"
    # , "#SBATCH --constraint=high-capacity"
    # , "#SBATCH -p halassa"
    , "#SBATCH --mem={}G".format(64 if experiment_type == 'cognitive_observer' else 10)
    , '#SBATCH --output=./slurm/%j.out'
    , "#SBATCH --job-name={}_{}".format(jobi,exp_name)
    , "#-------------------------------------------------------------------------------"
    , "## Run model "]
    
    # args = ['-c "x%d=%g" ' % (i, par_set[i]) for i in range(len(par_set))]
    #                                   use_gates same_rnn train_to_critoeron
    command_line = 'python run.py  {} --seed={} --var1={} --var3={} --var4={} --experiment_type={} --no_of_tasks={}'\
                                                    .format(exp_name, seed,  var1, var3, var4, experiment_type, var2) 
    win_command_line = command_line + ' --os=windows'
    
    fsh = open('serial_bash_generated.sh', 'w')
    fsh.write("\n".join(sbatch_lines))
    fsh.writelines(["\n", command_line])
    fsh.close()

    # Run the simulation
    if sys.platform == 'win32':
        sp.check_call(win_command_line.split(' '))
    else: #assuming Linux
        cp_process = sp.run('sbatch serial_bash_generated.sh'.split(' '), capture_output=True)
        # sp.check_call('sbatch serial_bash_generated.sh'.split(' '))
    cp_stdout = cp_process.stdout.decode()
    # print(cp_stdout)
    job_id = cp_stdout[-9:-1]
    print(f'submitted: {jobi} out of {len(expVars)} job_id : {job_id}')
    ## Get squee and count how many jobs are active for user ahummos
    result = sp.run('squeue -u ahummos'.split(' '), stdout=sp.PIPE)
    res = len(re.findall('(?= ahummos)', str(result.stdout)))
    while(res > job_limit):
        result = sp.run('squeue -u ahummos'.split(' '), stdout=sp.PIPE)
        res = len(re.findall('(?= ahummos)', str(result.stdout)))
        sleep(check_every)    # Inserts a 100 second pause
    
# cp_process = subprocess.run(['sbatch',
#                             get_jobfile(python_cmd, job_n,
#                                         dep_ids=pre_job_ids,
#                                         email=send_email,
#                                         hours=config.hours)],
#                         capture_output=True, check=True)
# cp_stdout = cp_process.stdout.decode()
# print(cp_stdout)
# job_id = cp_stdout[-9:-1]
# job_ids[group_num].append(job_id)