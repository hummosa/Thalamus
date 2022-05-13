from asyncore import loop
import itertools
import numpy as np
import random
import torch
from torch.nn import init
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader

from torch.utils.data import DataLoader, Dataset
from torchvision.transforms.functional import rotate

import gym
import neurogym as ngym
from Schizophrenia.tasks_coded_in_neurogym import *
# from analysis.visualization import plot_cluster_discovery, plot_long_term_cluster_discovery

def get_novel_task_ids(args, rng, config):
    no_of_tasks_left = len(config.tasks_id_name)- args.no_of_tasks
    if no_of_tasks_left > 0: 
        novel_task_id = args.no_of_tasks + rng.integers(no_of_tasks_left) # sample one task randomly. Not used at the moment.
        novel_task_ids = list(range(args.no_of_tasks, len (config.tasks_id_name) ))
    return (novel_task_ids)


def test_in_training(net, dataset, config, log, trial_idx):
    '''
    Compute performances for every task in the given task sequence(config.task_seq).
    '''
    # turn on test mode
    net.eval()
    if hasattr(config, 'MDeffect'):
        if config.MDeffect:
            net.rnn.md.learn = False
    # testing
    with torch.no_grad():
        log.stamps.append(trial_idx+1)
        #   fixation & action performance
        print('Performance')
        for env_id in range(config.num_task):
            fix_perf, act_perf, md_activity = get_full_performance(net=net, dataset=dataset, task_id=env_id, config=config)
            log.fix_perfs[env_id].append(fix_perf)
            log.act_perfs[env_id].append(act_perf)
            log.md_activity[env_id].append(md_activity)
            print('  fix performance, task {:d}, cycle {:d}: {:0.2f}'.format(env_id+1, trial_idx+1, fix_perf))
            print('  act performance, task {:d}, cycle {:d}: {:0.2f}'.format(env_id+1, trial_idx+1, act_perf))
    # back to train mode
    net.train()
    if hasattr(config, 'MDeffect'):
        if config.MDeffect:
            net.rnn.md.learn = True

# Parse argunents passed to python file.:
def get_args_from_parser(my_parser):
    my_parser.add_argument('exp_name',
                       default='pairs',
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
                        default=0, nargs='?',
                        type=int,
                        help='TODO')
    my_parser.add_argument('--var1',
                        default=3, nargs='?',
                        type=int,
                        help='Generic var to be used in various places, Currently, the variance of the fixed multiplicative MD to RNN weights')
    my_parser.add_argument('--var2',
                        default=0.5, nargs='?',
                        type=float,
                        help='Generic var to be used in various places, Currently, the sparsity of the gates, fraction set to zero')
    my_parser.add_argument('--var3',
                        default=1, nargs='?',
                        type=int,
                        help='0 for additive MD and 1 for multiplicative')
    my_parser.add_argument('--var4',
                        default=0.1, nargs='?',
                        type=float,
                        help='std of the gates. mean assumed 1.')
    my_parser.add_argument('--no_of_tasks',
                        default=30, nargs='?',
                        type=int,
                        help='number of tasks to train on')
    args = my_parser.parse_args()

    return (args)


def stats(var, var_name=None):
    if type(var) == type([]): # if a list
        var = np.array(var)
    elif type(var) == type(np.array([])):
        pass #if already a numpy array, just keep going.
    else: #assume torch tensor
        # pass
        var = var.detach().cpu().numpy()
    if var_name:
        print(var_name, ':')   
    out = ('Mean, {:2.5f}, var {:2.5f}, min {:2.3f}, max {:2.3f}, norm {}'.format(var.mean(), var.var(), var.min(), var.max(),np.linalg.norm(var) ))
    print(out)
    return (out)

# save variables
def save_variables(config, log, task_seq_id):
    np.save(config.FILEPATH + f'{task_seq_id}_' + config.FILENAME['config'], config)
    np.save(config.FILEPATH + f'{task_seq_id}_' + config.FILENAME['log'], log)
    # log = np.load('./files/'+'log.npy', allow_pickle=True).item()
    # config = np.load('./files/'+'config.npy', allow_pickle=True).item()

import math

def sparse_with_mean(tensor, sparsity, mean= 1.,  std=0.01):
    r"""Coppied from PyTorch source code. Fills the 2D input `Tensor` as a sparse matrix, where the
    non-zero elements will be drawn from the normal distribution
    :math:`\mathcal{N}(0, 0.01)`, as described in `Deep learning via
    Hessian-free optimization` - Martens, J. (2010).

    Args:
        tensor: an n-dimensional `torch.Tensor`
        sparsity: The fraction of elements in each column to be set to zero
        std: the standard deviation of the normal distribution used to generate
            the non-zero values

    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.sparse_(w, sparsity=0.1)
    """
    if tensor.ndimension() != 2:
        raise ValueError("Only tensors with 2 dimensions are supported")

    rows, cols = tensor.shape
    num_zeros = int(math.ceil(sparsity * rows))

    with torch.no_grad():
        tensor.normal_(mean, std)
        for col_idx in range(cols):
            row_indices = torch.randperm(rows)
            zero_indices = row_indices[:num_zeros]
            tensor[zero_indices, col_idx] = 0
    return tensor


from collections import defaultdict
def get_performance(net, envs, context_ids, config, batch_size=100):
    if type(envs) is not type([]):
        envs = [envs]

    fixation_accuracies = defaultdict()
    action_accuracies = defaultdict()
    for (context_id, env) in (zip(context_ids, envs)):
        # import pdb; pdb.set_trace()
        inputs, labels,_ = get_trials_batch(env, batch_size, config, test_batch=True)
        context_id_oh = F.one_hot(torch.tensor([context_id]* batch_size), config.md_size).type(torch.float)        
        action_pred, _ = net(inputs, sub_id=(context_id_oh/config.gates_divider)+config.gates_offset) # shape [500, 10, 17]
        ap = torch.argmax(action_pred, -1) # shape ap [500, 10]

        gt = torch.argmax(labels, -1)

        fix_lens = torch.sum(gt==0, 0)
        act_lens = gt.shape[0] - fix_lens 

        fixation_accuracy = ((gt==0)==(ap==0)).sum() / np.prod(gt.shape)## get fixation performance. overlap between when gt is to fixate and when model is fixating
           ## then divide by number of time steps.
        fixation_accuracies[context_id] = fixation_accuracy.detach().cpu().numpy()

        action_accuracy = accuracy_metric(action_pred, labels)
        action_accuracies[context_id] = action_accuracy
#         import pdb; pdb.set_trace()
        
    return((fixation_accuracies, action_accuracies))

# In[5]:

def accuracy_metric(outputs, labels):
    ap = torch.argmax(outputs, -1) # shape ap [500, 10]
    gt = torch.argmax(labels, -1)
    if labels.shape[-1] > 1: # checking it is not the Nassar tasks
        action_accuracy = (gt[-1,:] == ap[ -1,:]).sum() / gt.shape[1] # take action as the argmax of the last time step
    else: # IF NASSAR tasks
        action_accuracy = ((abs(outputs - labels ) < 0.05).float()).mean()
    return(action_accuracy.detach().cpu().numpy())

def plot_Nassar_task(env, config, context_id, task_name, training_log, net):
    # ap = torch.argmax(outputs, -1) # shape ap [500, 10]
    # gt = torch.argmax(labels, -1)
    input, output, trials = get_trials_batch(env, 100, config,return_dist_mean_for_Nassar_tasks=True)
    # distMeans = []
    # [distMeans.append(trial['means']) for trial in trials]
    pred, _ = net(input, sub_id=context_id)
    plt.close('all')
    fig, ax = plt.subplots(1,1)
    # ax = axes.flatten()[0]
    color1 = 'tab:blue'
    color2 = 'tab:red'
    ax.plot(pred.detach().cpu().numpy()[:,-1, : ], '.', label='RNN preds', color=color2 ,markersize = 5, alpha=1)
    ax.plot(output.cpu().numpy()[:,-1, :], '.', label='Ground truth',  color=color1,markersize = 4, alpha=0.7)
    # ax.plot(distMeans[-1], ':', label='Dist mean', color=color1)
    # ax.plot(rnn.zs_block[-1,:,:]  , label='Z', linewidth=0.5)
    ax.set_xlabel('Trials')
    ax.set_ylabel('Rewarded position')
    ax.set_ylim([-0.1, 1.1])
    ax.legend()
    # ax.set_title('Oddball condition' if exp_type == 'Oddball' else 'Change-point condition')
    plt.savefig('./files/'+ config.exp_name+f'/Example_perf_{config.exp_signature}_{training_log.stamps[-1]}_{task_name}.jpg', dpi=200)


    # save outputs, means, inputs. also CHange points and oddballs. 
    np.save('./files/'+ config.exp_name+f'/data_{config.exp_signature}_{training_log.stamps[-1]}_{task_name}.data', trials, allow_pickle=True)
    np.save('./files/'+ config.exp_name+f'/preds_{config.exp_signature}_{training_log.stamps[-1]}_{task_name}.data', pred.detach().cpu().numpy(), allow_pickle=True)
    # calculate how far from actual mean..
    # calculate the response to oddballs vs changepoints. 


def get_trials_batch(envs, batch_size, config, return_dist_mean_for_Nassar_tasks=False, test_batch=False):
    # check if only one env or several and ensure it is a list either way.
    if type(envs) is not type([]):
        envs = [envs]
    additional_data = None
    if config.dataset in ['split_mnist', 'rotated_mnist']:
        env = envs[0]
        batch = next(env)
        inputs, labels = batch
        inputs.squeeze_()
        zeros = torch.zeros([inputs.shape[0], inputs.shape[1], config.output_size]) # batch, time sequence, one-hot for 10 classes.
        zeros[:, -1, :] = F.one_hot(labels,config.output_size)
        labels = zeros

    else:
        # fetch and batch data
        obs, gts, dts = [], [], []
        for bi in range(batch_size):
            env = envs[np.random.randint(0, len(envs))] # randomly choose one env to sample from, if more than one env is given
            trial= env.new_trial()
            ob, gt = env.ob, env.gt # gt shape: (15,)  ob.shape: (15, 33)
            assert not np.any(np.isnan(ob))
            obs.append(ob), gts.append(gt)
            if return_dist_mean_for_Nassar_tasks:
                dts.append(trial)
        # Make trials of equal time length:
        obs_lens = [len(o) for o in obs]
        max_len = np.max(obs_lens)
        for o in range(len(obs)):
            while len(obs[o]) < max_len:
                obs[o]= np.insert(obs[o], 0, obs[o][0], axis=0)
    #             import pdb; pdb.set_trace()
        gts_lens = [len(o) for o in gts]
        max_len = np.max(gts_lens)
        for o in range(len(gts)):
            while len(gts[o]) < max_len:
                gts[o]= np.insert(gts[o], 0, gts[o][0], axis=0)
        obs = np.stack(obs) # shape (batch_size, 32, 33)
        gts = np.stack(gts) # shape (batch_size, 32)
        # numpy -> torch
        inputs = torch.from_numpy(obs).type(torch.float)
        labels = torch.from_numpy(gts).type(torch.float)
        # if shrew:
        if hasattr(env, 'update_context') and not test_batch:
            env.update_context()
            # print(f'remaining: \t {env.trials_remaining}\t cxt: {env.current_context} ')
            # print(env.switches)
        # index -> one-hot vector
        if labels.shape[-1] > 1:
            labels = (F.one_hot(labels.type(torch.long), num_classes=config.output_size)).float()  # Had to make it into integers for one_hot then turn it back to float.
        else: #If Nassar task
            labels = labels # Keeping as floats for Nassar tasks. No one-hot encoding. 
    if return_dist_mean_for_Nassar_tasks:
        return (inputs.permute([1,0,2]).to(config.device), labels.permute([1,0,2]).to(config.device), dts) # using time first [time, batch, input]
    else:
        return (inputs.permute([1,0,2]).to(config.device), labels.permute([1,0,2]).to(config.device), additional_data) # using time first [time, batch, input]


# In[16]:
import matplotlib.pyplot as plt
def show_input_output(inputs, labels, outputs=None, axes=None, ex_id=0):
    input = inputs.detach().cpu().numpy()[:,ex_id,:]
    label = labels.detach().cpu().numpy()[:,ex_id,:]
    output= outputs.detach().cpu().numpy()[:,ex_id,:]
    
    if axes is None:
        fig, axes = plt.subplots(3)
                
    no_output = True if output is None else False
    
    axes[0].imshow(input)
    axes[1].imshow(label)
    if output is not None: axes[2].imshow(output)
    
    axes[0].set_xlabel('Time steps')
#     ax.set_ylabel('fr')
#     ax.set_yticks([1, 17, 33, 34, 49])



def test_model(model, test_inputs, test_outputs, step_i=0 ):
    '''test_outputs are given as task_ids as integers '''
    model.eval()
    model_acts= []
    model_preds= []
    test_input_length = test_inputs.shape[0]
    with torch.no_grad():
        eins= torch.tensor(test_inputs)
        gpreds, gacts = model(eins) # torch.Size([50, 100, 1])  gacts:  torch.Size([50, 100, 256])
        #     gacts, gpreds = model(eins) # torch.Size([50, 100, 1])  gacts:  torch.Size([50, 100, 256])
        model_acts =gacts.detach().cpu().numpy()
        model_preds= gpreds.detach().cpu().numpy()
        # print('gru_preds shape : ', model_preds.shape)
        # print('taskIDs : ', task_ids[b_example:b_example+test_input_length])
        preds = np.argmax(model_preds, axis=-1)

        # preds # [60, 100] for the whole seq, and 100 batchs. not each seq step of 60 compared to its
        acc = 0
        for s in range(test_input_length):
            acc += (preds[s]== test_outputs[s]).sum()/ len(preds[s])
        acc = acc/test_input_length 
        # print('accuracy: ', acc)
        # acc = accuracy_measure(input= gpreds.squeeze().permute([0,2,1]), target=torch.Tensor(task_ids_repeated[b_example:b_example+test_input_length],))

    model.train()
    # model_preds = np.stack(model_preds)
    # model_acts = np.stack(model_acts)
    if False: #plot
        fig, axes = plt.subplots(1,2, figsize=[8,12])
        ax = axes[0]
        ax.matshow(model_preds.reshape([-1, 100, 15]).mean(1))
        # ax.plot(range(15), [29.5]*15, linewidth=(3))
        # ax.text(4, 31, 'Testing data', {'color': 'white'})
        ax.set_ylabel('Trials')
        ax.set_xlabel('Task ID')

        ax = axes[1]
        to_oh = F.one_hot(torch.from_numpy(test_outputs).long(),config.md_size).numpy()
        ax.matshow(to_oh)
        # ax.plot(range(15), [29.5]*15, linewidth=(3))
        # ax.text(4, 31, 'Testing data', {'color': 'white'})
        ax.set_ylabel('Trials')
        ax.set_xlabel('Task ID')

        plt.savefig(f'./files/cog_observer_sample_preds{step_i}.jpg')
        plt.close('')
    return (model_preds, model_acts, acc)

def latent_recall_test(config, net, testing_log, training_log, bu_optimizer, bu_running_acc, criterion_accuaracy, envs, test_task_id, test_task_context=None):
    env = envs[int(test_task_id)]
    inputs, labels = get_trials_batch(envs=env, config = config, batch_size = config.batch_size)
    
    ### Swap out the context the network has
    context_id_before_loop = net.rnn.md_context_id.detach().clone().cpu().numpy()
    net.rnn.md_context_id.data = torch.from_numpy(test_task_context).to(config.device)

    bubuffer, bu_accs = [], []
    for _ in range(config.no_latent_updates):
        bubuffer.append(net.rnn.md_context_id.detach().clone().cpu().numpy())
        bu_acc,_ = md_error_loop(config, net, training_log, criterion, bu_optimizer, inputs, labels, accuracy_metric)
        bu_accs.append(bu_acc)
        bu_running_acc = 0.7 * bu_running_acc + 0.3 * bu_acc
        if bu_running_acc > (criterion_accuaracy-0.1): #stop optim if reached criter
            break
    # put back the context id for the ongoing training task
    context_id_after_loop = net.rnn.md_context_id.detach().clone().cpu().numpy()
    net.rnn.md_context_id.data = torch.from_numpy(context_id_before_loop).to(config.device) # torch.cuda.FloatTensor()
    # net.rnn.md_context_id.data = torch.from_numpy(context_id_before_loop).to(config.device) # torch.cuda.FloatTensor()

    print(f'Deviation from previous context {test_task_context} by {(context_id_after_loop-test_task_context)}')
    # if len(bubuffer) > 0:
        # plot_cluster_discovery(config, bubuffer, training_log, testing_log, bu_accs)
    return bu_running_acc, context_id_after_loop

def test_in_training(config, net, testing_log, training_log, step_i, envs):
    # torch.set_grad_enabled(False)
    net.eval()
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
    try:
        gradients_past = min(step_i-training_log.start_optimizing_at, config.print_every_batches) # to avoid np.stack gradients from training and optimization. They might be of different lengths
        testing_log.gradients.append(np.mean(np.stack(training_log.gradients[-gradients_past:]),axis=0))
    except:
        pass
                # torch.set_grad_enabled(True)
    net.train()


class MyRotationTransform:
    """Rotate by one of the given angles."""

    def __init__(self, angle= 45):
        self.angle = angle

    def __call__(self, x):
        return rotate(x, self.angle)

class Rotated_mnist(Dataset):
    """Rotated mnist dataset."""
    def __init__(self, mnist_dataset, rotation_angle):
        """
        Args:
                mnist datasets or subdatasets
                angle: angle to apply rotation.
        """
        self.dataset = mnist_dataset
        self.transform = MyRotationTransform(rotation_angle)
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, idx):
        img , label = self.dataset[idx]
        timg = self.transform(img.reshape([1,1,28,28]) )
        return (timg, label)
            
def cycle(iterable):
    while True:
        for x in iterable:
            yield x


def build_env(config, envs):
    if config.dataset =='nassar':
        for task_id, task_name in config.tasks_id_name:
            if task_name in ['noisy_mean', 'drifting_mean', 'oddball', 'changepoint', 'oddball1', 'oddball2', 'oddball3','oddball4',]:
                params = {
                    'noisy_mean':       [0.05, 0, 0 , 0], 
                    'drifting_mean':    [0.05, 0.05, 0 , 0],
                    'oddball':          [0.05,  0.05, 0.1, 0],
                    'changepoint':      [0.05, 0.0, 0.0 , 0.1],
                    'oddball1':          [0.05,  0.05, 0.1, 0],
                    'oddball2':          [0.05,  0.05, 0.2, 0],
                    'oddball3':          [0.05,  0.05, 0.3, 0],
                    'oddball4':          [0.05,  0.05, 0.4, 0],
                }
                param= params[task_name]
                envs[task_id] =  NoiseyMean(mean_noise= param[0], mean_drift = param[1], odd_balls_prob = param[2], change_point_prob = param[3], safe_trials = 5)
    if config.dataset =='hierarchical_reasoning':
        for task_id, task_name in config.tasks_id_name:
        
            if task_name == 'shrew_task_audition':
                envs[task_id] = Shrew_task(dt =10, attend_to='audition')
            if task_name == 'shrew_task_vision':
                envs[task_id] = Shrew_task(dt =10, attend_to='vision')
                
            if task_name == 'shrew_task_cxt1':
                envs[task_id] = Shrew_task(dt =10, attend_to='either', no_of_coherent_cues=None)
            if task_name == 'shrew_task_cxt2':
                envs[task_id] = Shrew_task(dt =10, attend_to='either', context=2, no_of_coherent_cues=None)
            if task_name == 'st_hierarchical':
                envs[task_id] = Shrew_task_hierarchical()

    if config.dataset == 'neurogym':
        for task_id, task_name in config.tasks_id_name:
    
            from neurogym.envs.collections.yang19 import go, rtgo, dlygo, anti, rtanti, dlyanti, dm1, dm2, ctxdm1, ctxdm2, multidm, dlydm1, dlydm2,ctxdlydm1, ctxdlydm2, multidlydm, dms, dnms, dmc, dnmc
            neurogym_tasks_dict= {
                        'yang19.go-v0': go() ,
                        'yang19.rtgo-v0': rtgo(),
                        'yang19.dlygo-v0': dlygo(),
                        'yang19.dm1-v0': dm1(),
                        'yang19.ctxdm1-v0': ctxdm1(),
                        'yang19.dms-v0': dms(),
                        'yang19.dmc-v0': dmc(),
                        'yang19.dm2-v0': dm2(),
                        'yang19.ctxdm2-v0': ctxdm2(),
                        'yang19.multidm-v0': multidm(),
                        'yang19.rtanti-v0': rtanti(),
                        'yang19.anti-v0': anti(),
                        'yang19.dlyanti-v0': dlyanti(),
                        'yang19.dnms-v0': dnms(),
                        'yang19.dnmc-v0': dnmc(),
                        } 
            # envs[task_id] = gym.make(task_name, **config.env_kwargs)
            envs[task_id] = neurogym_tasks_dict[task_name]
    if config.dataset == 'split_mnist':
        from continual_learning.data import get_multitask_experiment
        # scenario = ['task', 'class']
        print("\nPreparing the data...")
        (train_datasets, test_datasets), data_config, classes_per_task = get_multitask_experiment(
        name='splitMNIST', scenario='domain', tasks=5, 
        verbose=True, exception=True , )
        for task_id, task_name in config.tasks_id_name:
            envs[task_id]= iter(cycle(DataLoader(dataset=train_datasets[task_id], batch_size=config.batch_size, shuffle=True, drop_last=True)))
            
    if config.dataset == 'rotated_mnist':
        from continual_learning.data import get_multitask_experiment
        # scenario = ['task', 'class']
        print("\nPreparing the data...")
        (train_datasets, test_datasets), data_config, classes_per_task = get_multitask_experiment(
        name='splitMNIST', scenario='domain', tasks=2,          verbose=True, exception=True , )
        for task_id, task_name in config.tasks_id_name:
            # use the same one dataset with two digits to classify, but specifiy different rotations
            rotated_dataset = Rotated_mnist(train_datasets[0], rotation_angle= int(task_name[-6:-3]))
            envs[task_id]= iter(cycle(DataLoader(dataset=rotated_dataset, batch_size=config.batch_size, shuffle=True, drop_last=True)))

def get_tasks_order(seed):
    tasks= [    'yang19.go-v0',
                'yang19.rtgo-v0',
                'yang19.dlygo-v0',
                'yang19.dm1-v0',
                'yang19.ctxdm1-v0',
                'yang19.dms-v0',
                'yang19.dmc-v0',
                'yang19.dm2-v0',
                'yang19.ctxdm2-v0',
                'yang19.multidm-v0',
                'yang19.rtanti-v0',
                'yang19.anti-v0',
                'yang19.dlyanti-v0',
                'yang19.dnms-v0',
                'yang19.dnmc-v0',
                ] 
    # choose opposite combos
    GoFamily = ['yang19.dlygo-v0', 'yang19.go-v0', 'yang19.rtgo-v0']
    AntiFamily = ['yang19.dlyanti-v0', 'yang19.anti-v0', 'yang19.rtanti-v0']
    DM1family = [ 'yang19.dms-v0', 'yang19.dmc-v0','yang19.ctxdm1-v0',]
    # 'yang19.dm1-v0', 'yang19.ctxdm1-v0', 'yang19.dm2-v0', 'yang19.ctxdm2-v0','yang19.multidm-v0'
    DM2family = [  'yang19.dnms-v0', 'yang19.dnmc-v0', 'yang19.ctxdm2-v0',]
    ### 2.1 two tasks
    TaskA = GoFamily +  DM1family
    TaskB = AntiFamily + DM2family
    task_seqs = []
    for a in range(len(GoFamily)):
        task_seqs.append([GoFamily[a], AntiFamily[a]])
        task_seqs.append([AntiFamily[a], GoFamily[a] ])
        task_seqs.append([DM1family[a], DM2family[a]])
        task_seqs.append([DM2family[a], DM1family[a]])

    import random
    random.seed(seed)
    task_seq = task_seqs[seed%len(task_seqs)] # choose one conflictual pairs
    [random.shuffle(tasks) for _ in range(seed)] # then shuffle uniquely for each seed even if they have the same pair.
    tasks = task_seq + [task for task in tasks if task not in task_seq] ## add all the other tasks shuffled
    return(tasks)



def get_logs_and_files(data_folder, exp_name, file_sig='testing_log', search_strs=[]):
    import os
    lfiles = os.listdir(data_folder+f'{exp_name}/')
    lfiles = [fn for fn in lfiles if fn.__contains__(file_sig)]
    for sstr in search_strs:
        lfiles = [fn for fn in lfiles if fn.__contains__(sstr)]

    logs = []
    for fi, fn in enumerate(lfiles):
        try:
            logs.append(np.load(data_folder+f'{exp_name}/' + fn, allow_pickle=True).item())
        except:
            print('some problem with ', fn)
    return(logs, lfiles)

def convert_train_to_test_idx(training_log, testing_log, training_idx):
    test_idx = training_log.stamps[training_idx]
    diff_arra = np.abs(np.array(testing_log.stamps) - test_idx)
    test_t_idx = np.argmin(diff_arra)
    return(test_t_idx)