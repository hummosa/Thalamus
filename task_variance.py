"""Analyze."""

import os

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch
import gym
import neurogym as ngym
from neurogym.wrappers.block import MultiEnvs

from models import RNNNet, get_performance


# Environment
timing = {'fixation': ('constant', 500)}
kwargs = {'dt': 100, 'timing': timing}
seq_len = 100
tasks = ngym.get_collection('yang19')
envs = [gym.make(task, **kwargs) for task in tasks]
env = MultiEnvs(envs, env_input=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
net = RNNNet(input_size=53, hidden_size=256, output_size=17,
             dt=env.dt).to(device)
fname = os.path.join('files', 'model.pt')
net.load_state_dict(torch.load(fname, map_location=torch.device(device)))


def get_activity(net, env, num_trial=1000):
    """Get activity of equal-length trials"""
    trial_list = list()
    activity_list = list()
    for i in range(num_trial):
        env.new_trial()
        ob = env.ob
        ob = ob[:, np.newaxis, :]  # Add batch axis
        inputs = torch.from_numpy(ob).type(torch.float).to(device)

        action_pred, activity = net(inputs)
        activity = activity.detach().numpy()
        trial_list.append(env.trial)
        activity_list.append(activity)

    activity = np.concatenate(activity_list, axis=1)
    return activity, trial_list

# Get performance
for i in range(20):
    env.set_i(i)
    perf = get_performance(net, env, num_trial=200)
    print('Average performance {:0.2f} for task {:s}'.format(perf, tasks[i]))

# Compute and Plot task variance
task_variance_list = list()
for i in range(20):
    env.set_i(i)
    activity, trial_list = get_activity(net, env, num_trial=500)
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
cmap = matplotlib.cm.get_cmap('tab10')
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