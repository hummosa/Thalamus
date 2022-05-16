import numpy as np
import torch
import torch.nn as nn
from torch.nn import init
from torch.nn import functional as F
import matplotlib as mpl
import matplotlib
import sys, os
root = os.getcwd()
sys.path.append(root)
sys.path.append('..')
from utils import convert_train_to_test_idx
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

def plot_accuracies( config, training_log, testing_log):
    
    
    no_of_values = len(config.tasks)
    norm = mpl.colors.Normalize(vmin=min([0,no_of_values]), vmax=max([0,no_of_values]))
    cmap_obj = mpl.cm.get_cmap('Set1') # tab20b tab20
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

    lens = [len(tg) for tg in training_log.gradients]
    m = min(lens)
    training_log.gradients = [tg[:m] for tg in training_log.gradients]
    try:
        gs = np.stack(training_log.gradients)
    except:
        pass

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
    identifiers = 9 # f'{training_log.stamps[-1]}_{final_accuracy_average:1.2f}'
    plt.savefig('./files/'+ config.exp_name+f'/acc_summary_{config.exp_signature}_{identifiers}.jpg', dpi=300)
def plot_thalamus_accuracies( config, training_log, testing_log):
        
    no_of_values = len(config.tasks)+1
    
    norm = mpl.colors.Normalize(vmin=min([0,no_of_values]), vmax=max([0,no_of_values]))
    cmap_obj = matplotlib.cm.get_cmap('Set1') # tab20b
    cmap = mpl.cm.ScalarMappable(norm=norm, cmap=cmap_obj)

    log = testing_log
    # training_log.switch_trialxxbatch.append(training_log.stamps[-1])
    switches = training_log.switch_trialxxbatch + [training_log.stamps[-1]]
    num_tasks = len(config.tasks)
    already_seen =[]
    title_label = 'Training tasks sequentially ---> \n    ' + config.exp_name
    if hasattr(training_log, 'start_optimizing_at'):
            max_x = training_log.start_optimizing_at #training_log.switch_trialxxbatch[num_tasks] #* config.print_every_batches
    else:
            max_x = training_log.start_testing_at #training_log.switch_trialxxbatch[num_tasks] #* config.print_every_batches
            # max_x = training_log.stamps[-1]
    fig, axes = plt.subplots(num_tasks,1, figsize=[15/2.53,6.5/2.53])
    for i, (tid, tn) in enumerate(config.tasks_id_name[:num_tasks]):
            # print(f'currently plot i {i} tid: {tid}  and tn {tn} ')
            ax = axes[ i ] # log i goes to the col direction -->
            ax.set_ylim([-0.1,1.1])
            ax.set_xlim([0, max_x])
    #         ax.axis('off')
            ax.plot(log.stamps, [test_acc[tid] for test_acc in log.accuracies], linewidth=1.5)
            ax.plot(log.stamps, np.ones_like(log.stamps)*0.5, ':', color='grey', linewidth=1)
            ax.set_ylabel(tn[7:-3], fontdict={'color': cmap.to_rgba(tid)})
            for ri in range(len(training_log.switch_trialxxbatch)):
                    ax.axvspan(switches[ri], switches[ri+1], color =cmap.to_rgba(training_log.switch_task_id[ri]) , alpha=0.2)
            xtl = ax.get_xticklabels()
            if not (i +1== num_tasks):        ax.set_xticklabels([])
    for ti, id in enumerate(training_log.switch_task_id):
        if id not in already_seen and (training_log.switch_trialxxbatch[ti] < max_x):
            if len(already_seen) >= num_tasks: break # do not go beyond how many tasks are to be displayed
            already_seen.append(id)
            task_name = config.human_task_names[id]
            # print(id)
            task_name = config.human_task_names[id]
            axes[0].text(training_log.switch_trialxxbatch[ti], 1.3, task_name, color= cmap.to_rgba(id) )
    # axes[0].text(400, 1.7, 'Tasks being trained -->', color= 'black', fontsize=6 ,)
    # axes[4].text(-0.1, 0.1, 'Accuracy on other tasks via latent updates', color= 'black', fontsize=6 ,rotation=90, transform=axes[4].transAxes)
    axes[-1].set_xlabel('Trials (x100)')
    final_accuracy_average = np.mean(list(testing_log.accuracies[-1].values()))
    print('final avg acc', final_accuracy_average)
        
    final_accuracy_average = np.mean(list(testing_log.accuracies[-1].values()))
    identifiers = f'{training_log.stamps[-1]}_{final_accuracy_average:1.2f}'
    plt.savefig('./files/'+ config.exp_name+f'/acc_summary_{config.exp_signature}_{identifiers}.jpg', dpi=300)


def plot_long_term_cluster_discovery( config, training_log, testing_log):
    # if len(training_log.bu_context_ids) > 0: context_ids =  training_log.bu_context_ids
    if len(training_log.td_context_ids) > 0: context_ids =  training_log.td_context_ids
    elif len(training_log.md_context_ids) > 0: context_ids =  training_log.md_context_ids
    else: 
        policy_context_id = np.ones([1,config.md_size])/config.md_size
        context_ids = [policy_context_id.repeat(config.batch_size, 0)] * training_log.stamps[-1]


    x0, x1 = 0, training_log.stamps[-1]
    no_of_values = len(config.tasks)
    norm = mpl.colors.Normalize(vmin=min([0,no_of_values]), vmax=max([0,no_of_values]))
    cmap_obj = mpl.cm.get_cmap('Set1') # tab20b tab20
    cmap = mpl.cm.ScalarMappable(norm=norm, cmap=cmap_obj)

    switches=  training_log.switch_trialxxbatch[:] 
    # switches=  training_log.switch_trialxxbatch[1:] # earlier on I must have added zero as a switch trial 

    fig, axes = plt.subplots(3,1, figsize=[20/2.53,5], sharex = False)

    ax = axes[0]
    # ax.set_position(mpl.transforms.Bbox([[0.125, 0.715], [.747, 0.880]]))
    ax.plot(np.array(training_log.stamps)[x0:x1], np.array(training_log.accuracies)[x0:x1], linewidth=1)
    for ri in range(len(switches)-1):
        ax.axvspan(training_log.switch_trialxxbatch[ri], training_log.switch_trialxxbatch[ri]+1, color =cmap.to_rgba(training_log.switch_task_id[ri]) , alpha=0.5)
        id = training_log.switch_task_id[ri]
        task_name = config.human_task_names[id]
        ax.text(training_log.switch_trialxxbatch[ri], 1.0 + np.random.uniform(-0.1, 0.25), task_name, color= cmap.to_rgba(id) , fontsize=7)
    ax.set_ylabel('current task accuracy')
    ax.set_xlim([x0, x1])
    # print('axis 0 position: ',ax.get_position())
    
    ax = axes[1] # context ids
    md = np.stack([m[0] for m in context_ids])
    # print(fig.get_axes())
    im = sns.heatmap(md.T, cmap='Reds', ax = ax)#, vmax=md.max()/3)
    # print(fig.get_axes())
    # print('colorbar pos:', fig.get_axes()[-1].get_position())
    fig.get_axes()[-1].set_position(mpl.transforms.Bbox([[0.9037,0.39],[.90785, 0.600]]))
    # ax.get_shared_x_axes().join(ax, axes[0])
    ax.set_xticks(axes[0].get_xticks()[:-1])
    ax.set_xticklabels(axes[0].get_xticklabels()[:-1], rotation=0)
    # ax.set_xlabel('Batches (100 trials)')
    ax.set_ylabel('Latent z vector')
    ax.set_position(mpl.transforms.Bbox([[0.125,0.39],[.902, 0.613]]))
    # ax.set_position(mpl.transforms.Bbox([[0.125,0.52],[.99, 0.683]]))
    # print(ax.get_position())

    ax = axes[2] # mean_bu
    # ax.plot(np.array(training_log.stamps)[x0:x1], np.array(training_log.accuracies)[x0:x1])
    # ax.set_xlim([x0, x1])
    # ax.set_ylabel('current task accuracy')
    # ax.set_xlabel('Batches (100 trials)')
    # # print(ax.get_position())
    # # =0.125, y0=0.32195652173913036, x1=0.9, y1=0.4860869565217391
    # ax.set_position(mpl.transforms.Bbox([[0.125,0.32],[.9, 0.45]]))
    # ax.plot(training_log.trials_to_crit, label = 'trials to crit')
    ax.plot(training_log.switch_trialxxbatch, training_log.trials_to_crit, label = 'Weight updates', color='tab:blue', linewidth=1)
    ax.plot(training_log.switch_trialxxbatch,training_log.trials_to_crit, 'o', markersize=4, color='tab:blue')
    filter=10
    filtered_mean = np.convolve(np.array(training_log.trials_to_crit), np.ones(filter)/filter, 'same')
    try:
        ax.plot(training_log.switch_trialxxbatch,filtered_mean, label=f'Weight updates avg', color='tab:orange',)
        mpl.rcParams['axes.spines.right'] = True
        ax2 =  ax.twinx()
        ax2.set_ylabel('Latent updates', color= 'tab:red') 
        ax2.plot(training_log.switch_trialxxbatch,np.clip(np.array(training_log.latents_to_crit),0, a_max=1000), 'x',markersize=4,color='tab:red', label = 'Latent updates')
        ax2.plot([la[0]for la in training_log.lu_stamps_acc_improve], np.stack([la[1]for la in training_log.lu_stamps_acc_improve])*1000, '.', color='tab:green')
        ax2.tick_params(axis='y', color='tab:red', labelcolor='tab:red')
        mpl.rcParams['axes.spines.right'] = False
    except:
        pass
    ax.set_ylabel('Weight updates to criterion')
    ax.set_xlabel('Trials')
    ax.set_xlim([x0, x1])
    ax.set_ylim(0, 100)#filtered_mean.max()*1.5)
    if hasattr(training_log, 'converged_detected_at'):
        ax.axvline(training_log.converged_detected_at, color='tab:green', alpha=0.5, linewidth=2, label='Converged')
    ax.legend()
    # print(ax.get_position())
    ax.set_position(mpl.transforms.Bbox([[0.125,0.125],[.90, 0.33]]))
    identifiers = 9
    plt.savefig('./files/'+ config.exp_name+f'/BU_Long_cluster_discovery_{config.exp_signature}_{identifiers}.jpg', dpi=200)

def plot_cluster_discovery( config, bubuffer, training_log, testing_log, bu_accs, latent_losses):
    if len(training_log.bu_context_ids) > 0: context_ids =  training_log.bu_context_ids
    elif len(training_log.td_context_ids) > 0: context_ids =  training_log.td_context_ids
    else: 
        # len(training_log.md_context_ids) > 0: 
        policy_context_id = np.ones([1,config.md_size])/config.md_size
        context_ids = [policy_context_id.repeat(config.batch_size, 0)] * training_log.stamps[-1]

    mean_bu = np.stack(bubuffer).squeeze()
    ex_bu = np.stack([bu[0] for bu in bubuffer])
    
    x0, x1 = 0, training_log.stamps[-1]
    no_of_values = len(config.tasks)
    norm = mpl.colors.Normalize(vmin=min([0,no_of_values]), vmax=max([0,no_of_values]))
    cmap_obj = mpl.cm.get_cmap('Set1') # tab20b tab20
    cmap = mpl.cm.ScalarMappable(norm=norm, cmap=cmap_obj)

    switches=  training_log.switch_trialxxbatch[:] 
    # switches=  training_log.switch_trialxxbatch[1:] # earlier on I must have added zero as a switch trial 

    fig, axes = plt.subplots(4,1, figsize=[12,6], sharex = False)

    ax = axes[0]
    ax.set_position(mpl.transforms.Bbox([[0.125, 0.715], [.747, 0.880]]))
    ax.plot(np.array(training_log.stamps)[x0:x1], np.array(training_log.accuracies)[x0:x1])
    for ri in range(len(switches)-1):
        ax.axvspan(training_log.switch_trialxxbatch[ri], training_log.switch_trialxxbatch[ri]+1, color =cmap.to_rgba(training_log.switch_task_id[ri]) , alpha=0.5)
        id = training_log.switch_task_id[ri]
        task_name = config.human_task_names[id]
        ax.text(training_log.switch_trialxxbatch[ri], 1.0 + np.random.uniform(-0.1, 0.25), task_name, color= cmap.to_rgba(id) , fontsize=10)
    ax.set_xlim([x0, x1])
    ax.set_ylabel('current task accuracy')
    
    ax = axes[1] # context ids
    md = np.stack([m[0] for m in training_log.md_context_ids])
    im = sns.heatmap(md.T, cmap='Reds', ax = ax)#, vmax=md.max()/3)
    ax.get_shared_x_axes().join(ax, axes[0])
    ax.set_xticks(axes[0].get_xticks())
    
    ax = axes[2] # mean_bu
    # im = sns.heatmap(mean_bu.T, cmap='Reds', ax = ax, vmax=mean_bu.max()/3)
    try:
        bu_all = np.concatenate((mean_bu[0], mean_bu[-1], mean_bu[-1]-mean_bu[0]))
        ax.bar(range(len(bu_all)), bu_all)
        ax.set_title('last md')
        # ax.set_xlim([x0, x1])
        ax.set_ylim([-1, 4])
        ax.axvspan(14,15, alpha=0.4)
        ax.axvspan(29,30, alpha=0.4)
    except:
        pass
    ax = axes[3]
    # ax.plot(bu_accs)
    ax.plot(np.stack(latent_losses))
    identifiers = 9
    ddir = './files/'+ config.exp_name + '/latent_updates'
    plt.savefig(ddir+f'/BU_cluster_discovery_{config.exp_signature}_i{training_log.stamps[-1]}_{identifiers}.jpg', dpi=200)

def plot_credit_assignment_inference( config, training_log, testing_log):
    if len(training_log.bu_context_ids) > 0: context_ids =  training_log.bu_context_ids
    elif len(training_log.td_context_ids) > 0: context_ids =  training_log.td_context_ids
    else: 
        # len(training_log.md_context_ids) > 0: 
        policy_context_id = np.ones([1,config.md_size])/config.md_size
        context_ids = [policy_context_id.repeat(config.batch_size, 0)] * training_log.stamps[-1]

    
    mg = np.stack(training_log.md_grads)
    mg = mg.mean(1) #(7957, 15) # average across batch

    x0, x1 = 0, training_log.stamps[-1]
    no_of_values = len(config.tasks)
    norm = mpl.colors.Normalize(vmin=min([0,no_of_values]), vmax=max([0,no_of_values]))
    cmap_obj = mpl.cm.get_cmap('Set1') # tab20b tab20
    cmap = mpl.cm.ScalarMappable(norm=norm, cmap=cmap_obj)

    switches=  training_log.switch_trialxxbatch[:] 
    # switches=  training_log.switch_trialxxbatch[1:] # earlier on I must have added zero as a switch trial 

    fig, axes = plt.subplots(4,1, figsize=[12,6], sharex = True)

    ax = axes[0]
    print(ax.get_position())
    ax.set_position(mpl.transforms.Bbox([[0.125, 0.715], [.747, 0.880]]))

    ax.plot(np.array(training_log.stamps)[x0:x1], np.array(training_log.accuracies)[x0:x1])
    for ri in range(len(switches)-1):
        ax.axvspan(training_log.switch_trialxxbatch[ri], training_log.switch_trialxxbatch[ri]+1, color =cmap.to_rgba(training_log.switch_task_id[ri]) , alpha=0.5)
        id = training_log.switch_task_id[ri]
        task_name = config.human_task_names[id]
        ax.text(training_log.switch_trialxxbatch[ri], 1.0 + np.random.uniform(-0.1, 0.25), task_name, color= cmap.to_rgba(id) , fontsize=10)
    ax.set_ylabel('current task accuracy')
    
    # ax.set_xlim([x0, x1])
    # for ri in range(len(training_log.switch_trialxxbatch)-1):
    #     ax.axvspan(training_log.switch_trialxxbatch[ri], training_log.switch_trialxxbatch[ri+1], color =cmap.to_rgba(training_log.switch_task_id[ri]) , alpha=0.2)
    ax = axes[1]
    try:
        t = 0
        d = training_log.stamps[-1]
        average_acc =[]
        taa = []
        for logi in range(config.no_of_tasks):
            taa.append([test_acc[logi] for test_acc in testing_log.accuracies])
        average_acc.append(np.stack(taa))

        testing_t = convert_train_to_test_idx(training_log, testing_log, t)
        testing_e =convert_train_to_test_idx(training_log, testing_log, t+d)
        ax.plot(testing_log.stamps[testing_t:testing_e], average_acc[0].mean(0)[testing_t:testing_e])
        ax.axvspan(0, d, color='tab:blue', alpha=0.2)
        ax.set_ylim([0,1])
        ax.set_ylabel('all tasks with Rule accuracy')
        # ax.set_title('With task rule input provided')
        # print(ax.get_position())
    except:
        pass
    ax.set_position(mpl.transforms.Bbox([[0.125, 0.519], [.747, 0.683]]))    

    ax = axes[2]
    tdci = np.stack(context_ids)
    mtd = tdci.mean(1) # (7729, 15)
    im = sns.heatmap(mtd.T, ax = ax)
    ax.set_xlim([x0, x1])
    ax.set_ylabel('md activity')
    # ax.colorbar()
    # plt.colorbar(im) #, ax=ax.ravel().tolist())


    ax.set_xlim([x0, x1])

    # ax.set_ylim([0,1])
    for ri in range(len(switches)-1):
        # print(ri)
        ax.scatter(switches[ri], (training_log.switch_task_id[ri]+0.5), color =cmap.to_rgba(training_log.switch_task_id[ri]) ,  linewidth=4, )#alpha=0.2)
        ax.axvspan(training_log.switch_trialxxbatch[ri], training_log.switch_trialxxbatch[ri]+1, color =cmap.to_rgba(training_log.switch_task_id[ri]) , alpha=0.9)

    ax = axes[3]
    ax = sns.heatmap(mg.T, cmap='Reds', ax = ax, vmax=mg.max()/3)
    # ax.set_xticks(list(range(0, x1, 200)))
    # # ax.set_xticklabels([str(i) for i in list(range(0, x1, 200))])
    # for index, label in enumerate(bar_plot.get_xticklabels()):
    #     if index % 2 == 0:
    #     label.set_visible(True)
    #     else:
    #     label.set_visible(False)
    xticks = list(range(0, x1, int(ax.get_xlim()[1]//4)))
    ax.set_xticks(xticks)
    ax.set_xticklabels([str(i) for i in xticks])
    ax.set_xlim([x0, x1])
    ax.set_ylabel('MD grads')
    ax.set_xlabel('Trials (100)')
    # plt.colorbar(ax)

    # sample_rate = 1
    # hm = sns.heatmap(mg.T, cmap='Reds', ax = ax)
    # # ax.set_yticklabels([str(i) for i in labels])
    # ax.set_ylabel('MD neuron idx', fontsize=8)
    # ax.set_xticks(list(range(0, x1, 200)))
    # ax.set_xticklabels([str(i) for i in list(range(0, x1, 200))])
    # # _=ax.set_xlabel('Trial (1000)', fontsize=8)
    # ax.set_title('MD grads')

    # try:
    #     plt.savefig('./files/'+ config.exp_name+f'/CAI_summary_{config.exp_signature}_bottom_up_optimizing.jpg', dpi=200)
    # except:
    #     # plt.savefig('./../files/'+ config.exp_name+f'/CAI_summary_{config.exp_signature}_bottom_up_optimizing.jpg', dpi=200)
    #     plt.savefig('./../files/'+ config.exp_name+f'/CAI_summary_{config.exp_signature}_bu_optimizing_many_batches.jpg', dpi=300)
        # plt.savefig('./../files/'+ config.exp_name+f'/CAI_summary_{config.exp_signature}_policy_optimizing.jpg', dpi=200)
    identifiers = 9
    plt.savefig('./files/'+ config.exp_name+f'/CAI_summary_{config.exp_signature}_{identifiers}.jpg', dpi=200)


def get_activity(config, net, env, num_trial=1000):
    """Get activity of equal-length trials"""
    trial_list = list()
    activity_list = list()
    for i in range(num_trial):
        env.new_trial()
        ob = env.ob
        ob = ob[:, np.newaxis, :]  # Add batch axis
        inputs = torch.from_numpy(ob).type(torch.float).to(config.device)
        
        context_id = F.one_hot(torch.tensor([env.i_env]* 1), config.md_size).type(torch.float)

        action_pred, activity = net(inputs, sub_id= context_id)
        activity = activity.detach().cpu().numpy()
        trial_list.append(env.trial)
        activity_list.append(activity)

    activity = np.concatenate(activity_list, axis=1)
    return activity, trial_list

def plot_task_variance(config, net):
    from neurogym.wrappers.block import MultiEnvs
    import gym
    import neurogym as ngym

    # Environment
    timing = {'fixation': ('constant', 500)}
    kwargs = {'dt': 100, 'timing': timing}
    seq_len = 100
    tasks = tasks = config.tasks # ngym.get_collection('yang19')
    envs = [gym.make(task, **kwargs) for task in tasks]
    env = MultiEnvs(envs, env_input=True) # adds env_ID as input, also compiles envs into env which needs env_i
    
    # Compute and Plot task variance
    task_variance_list = list()
    for i in range(len(tasks)):
        env.set_i(i)
        activity, trial_list = get_activity(net, env, num_trial=500)  #activity shape (10, 500, 356)   (time_steps, batch, no_neurons)
        # Compute task variance
        task_variance = np.var(activity, axis=1).mean(axis=0)
        task_variance_list.append(task_variance)
    task_variance = np.array(task_variance_list)  # (n_task, n_units)
    thres = 1e-6
    task_variance = task_variance[:, task_variance.sum(axis=0)>thres]

    norm_task_variance = task_variance / np.max(task_variance, axis=0)
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.metrics import silhouette_score
    X = norm_task_variance.T
    silhouette_scores = list()
    n_clusters = np.arange(2, 20)
    for n in n_clusters:
        cluster_model = AgglomerativeClustering(n_clusters=n)
        labels = cluster_model.fit_predict(X)
        silhouette_scores.append(silhouette_score(X, labels))
    plt.figure()
    plt.plot(n_clusters, silhouette_scores, 'o-')
    plt.xlabel('Number of clusters')
    plt.ylabel('Silhouette score')

    n_cluster = n_clusters[np.argmax(silhouette_scores)]
    cluster_model = AgglomerativeClustering(n_clusters=n_cluster)
    labels = cluster_model.fit_predict(X)

    # Sort clusters by its task preference (important for consistency across nets)
    label_prefs = [np.argmax(norm_task_variance[:, labels==l].sum(axis=1)) for l in set(labels)]

    ind_label_sort = np.argsort(label_prefs)
    label_prefs = np.array(label_prefs)[ind_label_sort]
    # Relabel
    labels2 = np.zeros_like(labels)
    for i, ind in enumerate(ind_label_sort):
        labels2[labels==ind] = i
    labels = labels2

    # Sort neurons by labels
    ind_sort = np.argsort(labels)
    labels = labels[ind_sort]
    norm_task_variance = norm_task_variance[:, ind_sort]


    # Plot Normalized Variance
    figsize = (3.5,2.5)
    rect = [0.25, 0.2, 0.6, 0.7]
    rect_color = [0.25, 0.15, 0.6, 0.05]
    rect_cb = [0.87, 0.2, 0.03, 0.7]
    tick_names = [task[len('yang19.'):-len('-v0')] for task in tasks]
    fs = 6
    labelpad = 13

    vmin, vmax = 0, 1
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes(rect)
    im = ax.imshow(norm_task_variance, cmap='magma',
                aspect='auto', interpolation='nearest', vmin=vmin, vmax=vmax)

    plt.yticks(range(len(tick_names)), tick_names,
            rotation=0, va='center', fontsize=fs)
    plt.xticks([])
    plt.title('Units', fontsize=7, y=0.97)
    plt.xlabel('Clusters', fontsize=7, labelpad=labelpad)
    ax.tick_params('both', length=0)
    for loc in ['bottom','top','left','right']:
        ax.spines[loc].set_visible(False)
    ax = fig.add_axes(rect_cb)
    cb = plt.colorbar(im, cax=ax, ticks=[vmin,vmax])
    cb.outline.set_linewidth(0.5)
    clabel = 'Normalized Task Variance'

    cb.set_label(clabel, fontsize=7, labelpad=0)
    plt.tick_params(axis='both', which='major', labelsize=7)


    # Plot color bars indicating clustering
    cmap = mpl.cm.get_cmap('tab10')
    ax = fig.add_axes(rect_color)
    for il, l in enumerate(np.unique(labels)):
        color = cmap(il % 10)
        ind_l = np.where(labels==l)[0][[0, -1]]+np.array([0,1])
        ax.plot(ind_l, [0,0], linewidth=4, solid_capstyle='butt',
                color=color)
        ax.text(np.mean(ind_l), -0.5, str(il+1), fontsize=6,
                ha='center', va='top', color=color)
    ax.set_xlim([0, len(labels)])
    ax.set_ylim([-1, 1])
    ax.axis('off')
    plt.savefig('./files/'+ config.exp_name+f'/task_var_analysis_{config.exp_signature}.jpg', dpi=300)

