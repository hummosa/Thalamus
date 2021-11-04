import numpy as np
import torch
import torch.nn as nn
from torch.nn import init
from torch.nn import functional as F
import matplotlib as mpl
mpl.rcParams['axes.spines.left'] = True
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['axes.spines.bottom'] = True
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt
import seaborn as sns
import imageio
from pygifsicle import optimize

# RNN activities
def plot_rnn_activity(rnn_activity, confing):
    font = {'family':'Times New Roman','weight':'normal', 'size':20}
    plt.figure()
    plt.plot(rnn_activity[-1, 0, :].cpu().detach().numpy())
    plt.title('PFC activities', fontdict=font)
    plt.show()

# MD related variables
def plot_MD_variables(net, config):
    font = {'family':'Times New Roman','weight':'normal', 'size':20}
    # Presynaptic traces
    plt.figure(figsize=(12, 9))
    plt.subplot(2, 2, 1)
    plt.plot(net.rnn.md.md_preTraces[-1, :])
    plt.axhline(y=net.rnn.md.md_preTrace_thresholds[-1], color='r', linestyle='-')
    plt.title('Pretrace', fontdict=font)
    # Binary presynaptic traces
    sub_size = config.sub_size
    plt.subplot(2, 2, 2)
    plt.plot(net.rnn.md.md_preTraces_binary[-1, :])
    plt.axhline(y=net.rnn.md.md_preTrace_binary_thresholds[-1], color='r', linestyle='-')
    plt.title( 'Pretrace_binary\n' +
                f'L: {sum(net.rnn.md.md_preTraces_binary[-1, :sub_size]) / sub_size}; ' +
                f'R: {sum(net.rnn.md.md_preTraces_binary[-1, sub_size:]) / sub_size}; ' +
                f'ALL: {sum(net.rnn.md.md_preTraces_binary[-1, :]) / len(net.rnn.md.md_preTraces_binary[-1, :])}',
                fontdict=font)
    # MD activities
    plt.subplot(2, 2, 3)
    plt.plot(net.rnn.md.md_output_t[-1, :])
    plt.title('MD activities', fontdict=font)
    # Heatmap wPFC2MD
    ax = plt.subplot(2, 2, 4)
    ax = sns.heatmap(net.rnn.md.wPFC2MD, cmap='Reds')
    ax.set_xticks([0, config.hidden_ctx_size-1])
    ax.set_xticklabels([1, config.hidden_ctx_size], rotation=0)
    ax.set_yticklabels([i for i in range(config.md_size)], rotation=0)
    ax.set_xlabel('PFC neuron index', fontdict=font)
    ax.set_ylabel('MD neuron index', fontdict=font)
    ax.set_title('wPFC2MD', fontdict=font)
    cbar = ax.collections[0].colorbar
    cbar.set_label('connection weight', fontdict=font)
    ## Heatmap wMD2PFC
    # ax = plt.subplot(2, 2, 4)
    # ax = sns.heatmap(net.rnn.md.wMD2PFC, cmap='Blues_r')
    # ax.set_xticklabels([i for i in range(config.md_size)], rotation=0)
    # ax.set_yticks([0, config.hidden_size-1])
    # ax.set_yticklabels([1, config.hidden_size], rotation=0)
    # ax.set_xlabel('MD neuron index', fontdict=font)
    # ax.set_ylabel('PFC neuron index', fontdict=font)
    # ax.set_title('wMD2PFC', fontdict=font)
    # cbar = ax.collections[0].colorbar
    # cbar.set_label('connection weight', fontdict=font)
    ## Heatmap wMD2PFCMult
    # font = {'family':'Times New Roman','weight':'normal', 'size':20}
    # ax = plt.subplot(2, 3, 6)
    # ax = sns.heatmap(net.rnn.md.wMD2PFCMult, cmap='Reds')
    # ax.set_xticklabels([i for i in range(config.md_size)], rotation=0)
    # ax.set_yticks([0, config.hidden_size-1])
    # ax.set_yticklabels([1, config.hidden_size], rotation=0)
    # ax.set_xlabel('MD neuron index', fontdict=font)
    # ax.set_ylabel('PFC neuron index', fontdict=font)
    # ax.set_title('wMD2PFCMult', fontdict=font)
    # cbar = ax.collections[0].colorbar
    # cbar.set_label('connection weight', fontdict=font)
    plt.tight_layout()
    plt.show()

# loss curve
def plot_loss(config, log):
    font = {'family':'Times New Roman','weight':'normal', 'size':25}
    plt.figure()
    plt.plot(np.array(log.losses))
    plt.xlabel('Trials', fontdict=font)
    plt.ylabel('Training MSE loss', fontdict=font)
    plt.tight_layout()
    if config.save_plots:
        import os
        os.makedirs('./animation/', exist_ok = True)
        plt.savefig('./animation/'+'CEloss.png')
    else:   
        plt.show()

# performance curve (fixation performance and action performance)
def plot_fullperf(config, log):
    label_font = {'family':'Times New Roman','weight':'normal', 'size':20}
    title_font = {'family':'Times New Roman','weight':'normal', 'size':25}
    legend_font = {'family':'Times New Roman','weight':'normal', 'size':12}
    for env_id in range(config.num_task):
        plt.figure()
        plt.plot(log.stamps, log.fix_perfs[env_id], label='fix')
        plt.plot(log.stamps, log.act_perfs[env_id], label='act')
        plt.fill_between(x=[config.switch_points[0], config.switch_points[1]], y1=0.0, y2=1.01, facecolor='red', alpha=0.05)
        plt.fill_between(x=[config.switch_points[1], config.switch_points[2]], y1=0.0, y2=1.01, facecolor='green', alpha=0.05)
        plt.fill_between(x=[config.switch_points[2], config.total_trials    ], y1=0.0, y2=1.01, facecolor='red', alpha=0.05)
        plt.legend(bbox_to_anchor = (1.15, 0.7), prop=legend_font)
        plt.xlabel('Trials', fontdict=label_font)
        plt.ylabel('Performance', fontdict=label_font)
        plt.title('Task{:d}: '.format(env_id+1)+config.task_seq[env_id], fontdict=title_font)
        # plt.xticks(ticks=[i*500 - 1 for i in range(7)], labels=[i*500 for i in range(7)])
        plt.xlim([0.0, None])
        plt.ylim([0.0, 1.01])
        plt.yticks([0.1*i for i in range(11)])
        plt.tight_layout()
        if config.save_plots:
            plt.savefig(config.FILEPATH +config.EXPSIGNATURE + '_'+ f'taskseq_{env_id}_task' + 'full_perf.png' )
            plt.close()
        else:
            plt.show()

# performance curve
def plot_perf(config, log, task_seq_id=None):
    label_font = {'family':'Times New Roman','weight':'normal', 'size':10}
    title_font = {'family':'Times New Roman','weight':'normal', 'size':12}
    legend_font = {'family':'Times New Roman','weight':'normal', 'size':10}
    for env_id in range(config.num_task):
        plt.figure(figsize=[4,3])
        plt.plot(log.stamps, log.act_perfs[env_id], color='red', label='$ MD+ $')
        shade_points = config.switch_points +[config.total_trials]
        for i in range(0, len(shade_points)-2,2):
            # print(f'i: {i} len(shded_points {len(shade_points)} ')
            plt.fill_between(x=[shade_points[i+0], shade_points[i+1]], y1=0.0, y2=1.01, facecolor='red', alpha=0.1)
            plt.fill_between(x=[shade_points[i+1], shade_points[i+2]], y1=0.0, y2=1.01, facecolor='blue', alpha=0.1)
        plt.legend(bbox_to_anchor = (1.25, 0.7), prop=legend_font)
        plt.xlabel('Trials', fontdict=label_font)
        plt.ylabel('Performance', fontdict=label_font)
        plt.title('Task{:d}: '.format(env_id+1)+config.task_seq[env_id], fontdict=title_font)
        # plt.xticks(ticks=[i*500 - 1 for i in range(7)], labels=[i*500 for i in range(7)])
        plt.xlim([0.0, None])
        plt.ylim([0.0, 1.01])
        plt.yticks([0.1*i for i in range(11)])
        plt.tight_layout()
        if config.save_plots:
            plt.savefig(config.FILEPATH +config.EXPSIGNATURE + '_'+ f'{task_seq_id}_taskseq_{env_id}_task' + config.FILENAME['plot_perf'] )
            plt.close()
        else:
            plt.show()
# performance curve
def plot_both_perf(config, log, net, color1= 'tab:blue', color2=  'tab:red', label1='$ Task 1 $', label2='Task 2'):
    label_font = {'family':'Times New Roman','weight':'normal', 'size':10}
    title_font = {'family':'Times New Roman','weight':'normal', 'size':12}
    legend_font = {'family':'Times New Roman','weight':'normal', 'size':10}

    plt.figure(figsize=[4,3])
    
    plt.plot(log.stamps, log.act_perfs[0], color=color1, label=label1)
    plt.plot(log.stamps, log.act_perfs[1], color=color2, label=label2)
    # overlay MD mean activities :
    shade_points = config.switch_points +[config.total_trials]
    for i in range(0, len(shade_points)-2,2):
        # print(f'i: {i} len(shded_points {len(shade_points)} ')
        plt.axvspan(shade_points[i+0], shade_points[i+1],  alpha=0.1, color=color1)
        plt.axvspan(shade_points[i+1], shade_points[i+2],  alpha=0.1, color=color2)
    # plt.legend(bbox_to_anchor = (1.25, 0.7), prop=legend_font)
    plt.legend()
    plt.xlabel('Trials', fontdict=label_font)
    plt.ylabel('Performance', fontdict=label_font)
    plt.title('Tasks: {} {}'.format(config.task_seq[0][7:-3], config.task_seq[1][7:-3]), fontdict=title_font)
        # plt.xticks(ticks=[i*500 - 1 for i in range(7)], labels=[i*500 for i in range(7)])
    plt.xlim([0.0, None])
    plt.ylim([0.0, 1.01])
    plt.yticks([0.1*i for i in range(11)])
    plt.tight_layout()
    if config.save_plots:
        plt.savefig(config.FILEPATH +config.EXPSIGNATURE + '_'+ config.FILENAME['plot_perf'] )
        plt.close()
    else:
        plt.show()
        
def plot_md_activity(net, log, config):
    fig, axes = plt.subplots(2,1)    
    ax = axes[0]
    env_id = 0
    ax.plot(log.stamps, log.md_activity[env_id])
    ax = axes[1]
    env_id = 1
    ax.plot(log.stamps, log.md_activity[env_id])
    if config.save_plots:
        plt.savefig(config.FILEPATH +config.EXPSIGNATURE + '_MD_activtiies_'+ config.FILENAME['plot_perf'] )
        plt.close()
    else:
        plt.show()


def plot_accuracy_matrix(logs, config):
    num_tasks = len(config['tasks'])
    title_label = 'Training tasks sequentially ---> \n    ' + label 
    plt.close('all')
    max_x = config['trials_per_task']//config['batch_size']
    fig, axes = plt.subplots(num_tasks,num_tasks, figsize=[9,7])
    for logi in range(num_tasks):
        for li in range(num_tasks):
            ax = axes[ li, logi ] # log i goes to the col direction -->
            ax.set_ylim([-0.1,1.1])
            ax.set_xlim([0, max_x])
    #         ax.axis('off')
            log = testing_logs[logi]
            ax.plot(log['stamps'], [test[li] for test in log['accuracy']], linewidth=2)
            ax.plot(log['stamps'], np.ones_like(log['stamps'])*0.5, ':', color='grey', linewidth=0.5)
            if li == 0: ax.set_title(config['human_task_names'][logi])
            if logi == 0: ax.set_ylabel(config['human_task_names'][li])
            ax.set_yticklabels([]) 
            ax.set_xticklabels([])
            if logi== li:
                ax.axvspan(*ax.get_xlim(), facecolor='grey', alpha=0.2)
            if li == num_tasks-1 and logi in [num_tasks//2 - 4, num_tasks//2, num_tasks//2 + 4] :
                ax.set_xlabel('batch #')
    axes[num_tasks-1, num_tasks//2-2].text(-8., -2.5, title_label, fontsize=12)     
    # exp_parameters = f'Exp parameters: {config["exp_name"]}\nRNN: {"same" if config["same_rnn"] else "separate"}\n\
    #       mul_gate: {"True" if config["use_gates"] else "False"}\
    #           {exp_signature}'
    # axes[num_tasks-1, 0].text(-7., -2.2, exp_parameters, fontsize=7)
    # plt.show(fig)
    plt.savefig('./files/'+label+'_all_logs.png', dpi=300)
    
