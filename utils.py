import itertools
import numpy as np
import random
import torch
from torch.nn import init
from torch.nn import functional as F
import gym
import neurogym as ngym

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def get_task_seqs():
    '''
    Generate task sequences
    '''
    ## 1. all pairs
    # num_tasks = 2
    # tasks = ['yang19.dms-v0',
    #          'yang19.dnms-v0',
    #          'yang19.dmc-v0',
    #          'yang19.dnmc-v0',
    #          'yang19.dm1-v0',
    #          'yang19.dm2-v0',
    #          'yang19.ctxdm1-v0',
    #          'yang19.ctxdm2-v0',
    #          'yang19.multidm-v0',
    #          'yang19.dlygo-v0',
    #          'yang19.dlyanti-v0',
    #          'yang19.go-v0',
    #          'yang19.anti-v0',
    #          'yang19.rtgo-v0',
    #          'yang19.rtanti-v0']
    # task_seqs = list(itertools.permutations(tasks, num_tasks))
    # task_seqs = [val for val in task_seqs for i in range(num_tasks)]
    ## 2. pairs from different task families
    GoFamily = ['yang19.dlygo-v0', 'yang19.go-v0']
    AntiFamily = ['yang19.dlyanti-v0', 'yang19.anti-v0']
    DMFamily = ['yang19.dm1-v0', 'yang19.dm2-v0', 'yang19.ctxdm1-v0', 'yang19.ctxdm2-v0', 'yang19.multidm-v0']
    MatchFamily = ['yang19.dms-v0', 'yang19.dmc-v0', 'yang19.dnms-v0', 'yang19.dnmc-v0']
    ### 2.1 two tasks
    TaskA = GoFamily + AntiFamily
    TaskB = MatchFamily + ['yang19.ctxdm1-v0', 'yang19.dm2-v0']
    task_seqs = []
    for a in TaskA:
        for b in TaskB:
            task_seqs.append((a, b))
            task_seqs.append((b, a))
    ### 2.2 three tasks
    # TaskA = GoFamily + AntiFamily
    # TaskB = ['yang19.dm2-v0', 'yang19.ctxdm1-v0', 'yang19.multidm-v0']
    # TaskC = ['yang19.dms-v0', 'yang19.dnms-v0', 'yang19.dnmc-v0']
    # task_seqs = []
    # for a in TaskA:
    #     for b in TaskB:
    #         for c in TaskC:
    #             task_seqs.append((a, b, c))
    ### 2.3 four tasks
    # task_seqs = []
    # for a in itertools.combinations(GoFamily, 2):
    #     for b in itertools.combinations(MatchFamily, 2):
    #         task_seqs.append(list(a) + list(b))
    #         task_seqs.append(list(b) + list(a))
    # for a in itertools.combinations(AntiFamily, 2):
    #     for b in itertools.combinations(DMFamily, 2):
    #         task_seqs.append(list(a) + list(b))
    #         task_seqs.append(list(b) + list(a))
    return task_seqs

# training
def get_task_id(config, trial_idx, prev_task_id):
    # 1. Two tasks
    # Sequential training between blocks
    if trial_idx >= config.switch_points[0] and trial_idx < config.switch_points[1]:
        task_id = 0
    elif trial_idx >= config.switch_points[1] and trial_idx < config.switch_points[2]:
        task_id = 1
    elif trial_idx >= config.switch_points[2]:
        task_id = 0
    # 2. Three tasks
    # Sequential training between blocks
    # if trial_idx >= config.switch_points[0] and trial_idx < config.switch_points[1]:
    #     task_id = 0
    # elif trial_idx >= config.switch_points[1] and trial_idx < config.switch_points[2]:
    #     task_id = 1
    # elif trial_idx >= config.switch_points[2] and trial_idx < config.switch_points[3]:
    #     task_id = 2
    # elif trial_idx >= config.switch_points[3]:
    #     task_id = 0
    # 3. Four tasks
    # # Sequential training between blocks
    # if trial_idx == config.switch_points[0]:
    #     task_id = 0
    # elif trial_idx == config.switch_points[1]:
    #     task_id = 2
    # elif trial_idx == config.switch_points[2]:
    #     task_id = 0
    # # Interleaved training within blocks
    # if trial_idx >= config.switch_points[0] and trial_idx < config.switch_points[1]:
    #     if prev_task_id == 0:
    #         task_id = 1
    #     elif prev_task_id == 1:
    #         task_id = 0
    # elif trial_idx >= config.switch_points[1] and trial_idx < config.switch_points[2]:
    #     if prev_task_id == 2:
    #         task_id = 3
    #     elif prev_task_id == 3:
    #         task_id = 2
    # elif trial_idx >= config.switch_points[2]:
    #     if prev_task_id == 0:
    #         task_id = 1
    #     elif prev_task_id == 1:
    #         task_id = 0

    return task_id

def get_optimizer(net, config):
    print('training parameters:')
    training_params = list()
    named_training_params = dict()
    for name, param in net.named_parameters():
        # if 'rnn.h2h' not in name: # reservoir
        # if True: # learnable RNN
        if ('rnn.input2PFCctx' not in name):
            print(name)
            training_params.append(param)
            named_training_params[name] = param
    optimizer = torch.optim.Adam(training_params, lr=config.lr)
    return optimizer, training_params, named_training_params

def forward_backward(net, opt, crit, inputs, labels, task_id):
    '''
    forward + backward + optimize
    '''
    opt.zero_grad()
    outputs, rnn_activity, _ = net(inputs, task_id=task_id)
    loss = crit(outputs, labels)
    loss.backward()
    opt.step()
    return loss, rnn_activity

# testing

def get_full_performance(net, dataset, task_id, config):
    num_trial = config.test_num_trials
    fix_perf = 0.
    act_perf = 0.
    md_activity = []
    num_no_act_trial = 0
    for i in range(num_trial):
        ob, gt = dataset.new_trial(task_id)
        inputs = torch.from_numpy(ob).type(torch.float).to(config.device)
        inputs = inputs[:, np.newaxis, :]
        action_pred, _, _md_activity = net(inputs, task_id=task_id)
        action_pred = action_pred.detach().cpu().numpy()
        action_pred = np.argmax(action_pred, axis=-1)

        md_activity.append(_md_activity)
        fix_len = sum(gt == 0)
        act_len = len(gt) - fix_len
        assert all(gt[:fix_len] == 0)
        fix_perf += sum(action_pred[:fix_len, 0] == 0)/fix_len
        if act_len != 0:
            assert all(gt[fix_len:] == gt[-1])
            # act_perf += sum(action_pred[fix_len:, 0] == gt[-1])/act_len
            act_perf += (action_pred[-1, 0] == gt[-1])
        else: # no action in this trial
            num_no_act_trial += 1
    md_activity_mean_for_env = np.stack(md_activity).mean(0)
    fix_perf /= num_trial
    act_perf /= num_trial - num_no_act_trial
    return fix_perf, act_perf, md_activity_mean_for_env

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
    my_parser.add_argument('--num_of_tasks',
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
        inputs, labels = get_trials_batch(env, batch_size, config)
        if config.use_lstm:
            action_pred, _ = net(inputs, update_md= False) # shape [500, 10, 17]
        else:
            context_id_oh = F.one_hot(torch.tensor([context_id]* batch_size), config.md_size).type(torch.float)        
            action_pred, _ = net(inputs, sub_id=context_id_oh) # shape [500, 10, 17]
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


def get_trials_batch(envs, batch_size, config, return_dist_mean_for_Nassar_tasks=False):
    # check if only one env or several and ensure it is a list either way.
    if type(envs) is not type([]):
        envs = [envs]
        
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
    inputs = torch.from_numpy(obs).type(torch.float).to(config.device)
    labels = torch.from_numpy(gts).type(torch.float).to(config.device)

    # index -> one-hot vector
    if labels.shape[-1] > 1:
        labels = (F.one_hot(labels.type(torch.long), num_classes=config.output_size)).float()  # Had to make it into integers for one_hot then turn it back to float.
    else: #If Nassar task
        labels = labels # Keeping as floats for Nassar tasks. No one-hot encoding. 
    if return_dist_mean_for_Nassar_tasks:
        return (inputs.permute([1,0,2]), labels.permute([1,0,2]), dts) # using time first [time, batch, input]
    else:
        return (inputs.permute([1,0,2]), labels.permute([1,0,2])) # using time first [time, batch, input]


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