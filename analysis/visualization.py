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


def plot_credit_assignment_inference( config, training_log, testing_log):
    if (config.optimize_bu): context_ids = training_log.bu_context_ids
    if (config.optimize_td): context_ids =  training_log.td_context_ids

    
    mg = np.stack(training_log.md_grads)
    mg = mg.mean(1) #(7957, 15)
    mg_repeated = np.repeat(mg, repeats =10, axis=1)
    mg_repeated.shape # xxx, 45

    x0, x1 = 0, min(training_log.stamps[-1], 1000)
    fig, axes = plt.subplots(3,1, figsize=[12,6])
    ax = axes[0]
    ax.matshow(mg_repeated[x0:x1].T)
    ax.set_xlim([x0, x1])
    # plt.colorbar(ax)

    ax = axes[1]
    tdci = np.stack(context_ids)
    mtd = tdci.mean(1) # (7729, 15)
    ax.matshow(np.repeat(mtd[x0:x1], repeats=10, axis=1).T)
    ax.set_xlim([x0, x1])

    ax = axes[1]
    no_of_values = len(config.tasks)-5
    norm = mpl.colors.Normalize(vmin=min([0,no_of_values]), vmax=max([0,no_of_values]))
    cmap_obj = mpl.cm.get_cmap('Set1') # tab20b tab20
    cmap = mpl.cm.ScalarMappable(norm=norm, cmap=cmap_obj)

    switches=  training_log.switch_trialxxbatch 
    ax.set_xlim([x0, x1])

    # ax.set_ylim([0,1])
    for ri in range(len(switches)):
        ax.scatter(switches[ri], (training_log.switch_task_id[ri]+0.5)*10, color =cmap.to_rgba(training_log.switch_task_id[ri]) ,  linewidth=4, )#alpha=0.2)
        ax.axvspan(training_log.switch_trialxxbatch[ri], training_log.switch_trialxxbatch[ri]+1, color =cmap.to_rgba(training_log.switch_task_id[ri]) , alpha=0.9)

    ax = axes[2]
    ax.matshow(np.clip(mg_repeated[x0:x1].T, max= mg_repeated.max()/2))
    ax.plot(np.array(training_log.stamps)[x0:x1], np.array(training_log.accuracies)[x0:x1])
    for ri in range(len(switches)):
        ax.axvspan(training_log.switch_trialxxbatch[ri], training_log.switch_trialxxbatch[ri]+1, color =cmap.to_rgba(training_log.switch_task_id[ri]) , alpha=0.5)
    ax.set_xlim([x0, x1])
    # for ri in range(len(training_log.switch_trialxxbatch)-1):
    #     ax.axvspan(training_log.switch_trialxxbatch[ri], training_log.switch_trialxxbatch[ri+1], color =cmap.to_rgba(training_log.switch_task_id[ri]) , alpha=0.2)

    plt.savefig('./files/'+ config.exp_name+f'/CAI_summary_{config.exp_signature}_{training_log.stamps[-1]}_{3.3:1.2f}.jpg', dpi=300)


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

