import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from logger.logger import SerialLogger
from tqdm import tqdm, trange
import gym
import neurogym as ngym
import matplotlib.pyplot as plt
from utils import stats, get_trials_batch, get_performance, accuracy_metric
from Schizophrenia.tasks_coded_in_neurogym import NoiseyMean, Shrew_task
import utils

def criterion(output, labels, use_loss='nll'):    # criterion & optimizer
    if use_loss =='mse':
        crit = nn.MSELoss()
        loss = crit(output, labels)
    elif use_loss =='nll':
        crit = F.nll_loss
        
        loss = crit(torch.log_softmax(output[-1,...], dim = -1, dtype=torch.float), torch.argmax(labels[-1,...], dim=-1))
    return loss


def train(config, net, task_seq, testing_log, training_log, step_i  = 0):
    print('training parameters:')
    training_params = list()
    for name, param in net.named_parameters():
        if (not name.__contains__('md_context_id')) and (not ('gates' in name) or config.train_gates):
        # if (not ('gates' in name) or config.train_gates):
            print(name)
            training_params.append(param)
        else:
            print('exluding: ', name)
    optimizer = torch.optim.Adam(training_params, lr=config.lr)

    # Make all tasks, but reorder them from the tasks_id_name list of tuples
    envs = [None] * len(config.tasks_id_name)
    for task_id, task_name in config.tasks_id_name:
        build_env(config, envs, task_id, task_name)

    task_i = 0
    bar_tasks = tqdm(task_seq)
    for (task_id, task_name) in bar_tasks:
        #if the last task in shuffle training_ extend it to 400 trials

        task_id = int(task_id)
        env = envs[int(task_id)]
        bar_tasks.set_description('i: ' + str(step_i))
        training_log.switch_trialxxbatch.append(step_i)
        training_log.switch_task_id.append(task_id)
        training_log.trials_to_crit.append(0) #add a zero and increment it in the training loop.
        
        running_frustration = 0
        running_acc = 0
        training_bar = trange(config.max_trials_per_task//config.batch_size)
        for i in training_bar:
            
            context_id = F.one_hot(torch.tensor([task_id]* config.batch_size), config.md_size).type(torch.float)
            inputs, labels = get_trials_batch(envs=env, config = config, batch_size = config.batch_size)
            inputs.refine_names('timestep', 'batch', 'input_dim')
            labels.refine_names('timestep', 'batch', 'output_dim')
            outputs, rnn_activity = net(inputs, sub_id=(context_id/config.gates_divider)+config.gates_offset)
            # outputs, rnn_activity = net(inputs, sub_id=(context_id/config.gates_divider)+config.gates_offset, gt=labels)
            acc  = accuracy_metric(outputs.detach(), labels.detach())
            # print(f'shape of outputs: {outputs.shape},    and shape of rnn_activity: {rnn_activity.shape}')
            #Shape of outputs: torch.Size([20, 100, 17]),    and shape of rnn_activity: torch.Size ([20, 100, 256
            optimizer.zero_grad()
            
            loss = criterion(outputs, labels, use_loss='mse')
            loss.backward()
           
            optimizer.step()
            # from utils import show_input_output
            # show_input_output(inputs, labels, outputs)
            # plt.savefig('example_inpujt_label_output.jpg')
            # plt.close('all')
            # save loss

            if type(rnn_activity) is tuple:
                rnn_activity, md = rnn_activity
                training_log.md_context_ids.append(md.detach().cpu().numpy())
                training_log.md_grads.append(md.grad.cpu().numpy())
            training_log.write_basic(step_i, loss.item(), acc, task_id)
            training_log.gradients.append(np.array([torch.norm(p.grad).item() for p in net.parameters() if p.grad is not None]) )
            fidx = min(i, 100)
            frustration = 1-np.sum(np.diff(np.stack(training_log.accuracies[-fidx:])))
            frustration_alpha = 0.05
            running_frustration = frustration_alpha* frustration + (1-frustration_alpha)* running_frustration
            training_log.frustrations.append(running_frustration)
            if config.save_detailed:
                training_log.write_detailed( rnn_activity= rnn_activity.detach().cpu().numpy().mean(0),
                inputs=   [] ,# inputs.detach().cpu().numpy(),
                outputs = outputs.detach().cpu().numpy()[-1, :, :],
                labels =   labels.detach().cpu().numpy()[-1, :, :],
                sampled_act = [], # rnn_activity.detach().cpu().numpy()[:,:, 1:356:36], # Sample about 10 neurons 
                # rnn_activity.shape             torch.Size([15, 100, 356])
                )
            training_bar.set_description('ls, acc: {:0.3F}, {:0.2F} '.format(loss.item(), acc)+ config.human_task_names[task_id])

            # print statistics
            if step_i % config.print_every_batches == (config.print_every_batches - 1):
                ################################ test during training
                net.eval()
                # torch.set_grad_enabled(False)
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
                testing_log.gradients.append(np.mean(np.stack(training_log.gradients[-config.print_every_batches:]),axis=0))
                            
                # torch.set_grad_enabled(True)
                net.train()
 
                #### End testing
            step_i+=1
            training_log.trials_to_crit[-1] += 1 # increment the total trials spent on this current task
            criterion_accuaracy = config.criterion if task_name not in config.DMFamily else config.criterion_DMfam
            if ((running_acc > criterion_accuaracy) and config.train_to_criterion) or (i+1== config.max_trials_per_task//config.batch_size):
                running_acc = 0.
                utils.plot_Nassar_task(envs[task_id], config, context_id=context_id, task_name=task_name, training_log=training_log, net=net )
                break # stop training current task if sufficient accuracy. Note placed here to allow at least one performance run before this is triggered.
            running_acc = 0.7 * running_acc + 0.3 * acc


        if config.paradigm_shuffle and config.train_to_criterion and step_i > config.print_every_batches+10:
            unique_tasks = np.unique(training_log.switch_task_id)
            current_average_accuracy = np.mean([testing_log.accuracies[-1][ut] for ut in unique_tasks]) 
            if current_average_accuracy > config.criterion_shuffle_paradigm:
                break
        #no more than number of blocks specified
        task_i +=1
        training_log.sample_input = inputs[:,0,:].detach().cpu().numpy().T
        training_log.sample_label = labels[:,0,:].detach().cpu().numpy().T
        training_log.sample_output = outputs[:,0,:].detach().cpu().numpy().T
    testing_log.total_batches, training_log.total_batches = step_i, step_i

    return(testing_log, training_log, net)

def build_env(config, envs, task_id, task_name):
    if task_name in ['noisy_mean', 'drifting_mean', 'oddball', 'changepoint']:
        params = {
                'noisy_mean':       [0.05, 0, 0 , 0], 
                'drifting_mean':    [0.05, 0.05, 0 , 0],
                'oddball':          [0.05,  0.05, 0.1, 0],
                'changepoint':      [0.05, 0.0, 0.0 , 0.1]
            }
        param= params[task_name]
        envs[task_id] =  NoiseyMean(mean_noise= param[0], mean_drift = param[1], odd_balls_prob = param[2], change_point_prob = param[3], safe_trials = 5)
    elif task_name in ['shrew_task_audition', 'shrew_task_vision', 'shrew_task_either',  'shrew_task_either2']:
        if task_name == 'shrew_task_audition':
            envs[task_id] = Shrew_task(dt =10, attend_to='audition')
        if task_name == 'shrew_task_vision':
            envs[task_id] = Shrew_task(dt =10, attend_to='vision')
               
        if task_name == 'shrew_task_either':
            envs[task_id] = Shrew_task(dt =10, attend_to='either', no_of_coherent_cues=9)
        if task_name == 'shrew_task_either2':
            envs[task_id] = Shrew_task(dt =10, attend_to='either', context=2, no_of_coherent_cues=9)

    else: # assume a neurogym yang19 task:
        envs[task_id] = gym.make(task_name, **config.env_kwargs)

def optimize(config, net, cog_net, task_seq, testing_log,  training_log,step_i  = 0):

    md_context_id = torch.zeros([1, config.md_size])

    # criterion & optimizer
    criterion = nn.MSELoss()
    print('Policy training parameters:')
    training_params = list()
    for name, param in net.named_parameters():
        if (not name.__contains__('md_context_id')) and (not ('gates' in name) or config.train_gates):
            print(name)
            training_params.append(param)
        else:
            print('exluding: ', name)
    policy_optimizer = torch.optim.Adam(training_params, lr=config.lr)

    bu_optimizer = torch.optim.Adam([tp[1] for tp in net.named_parameters() if tp[0] == 'rnn.md_context_id'], 
    lr=config.lr*100)
    
    td_training_params = list()
    print('cognitive network optimized parameters')
    for name, param in cog_net.named_parameters():
        print(name)
        td_training_params.append(param)
        
    td_optimizer = torch.optim.Adam(td_training_params, lr=config.lr*(100 if cog_net.gru.hidden_size ==1 else 1))

    # Make all tasks, but reorder them from the tasks_id_name list of tuples
    envs = [None] * len(config.tasks_id_name)
    for task_id, task_name in config.tasks_id_name:
        build_env(config, envs, task_id, task_name)


    # initialize buffer
    config.horizon =50
    buffer_acts = []
    buffer_task_ids = []
    buffer_labels = []
    buffer_accuracies = []

    task_i = 0
    bar_tasks = tqdm(task_seq)
    for (task_id, task_name) in bar_tasks:

        env = envs[task_id]
        bar_tasks.set_description('i: ' + str(step_i))
        training_log.switch_trialxxbatch.append(step_i)
        training_log.switch_task_id.append(task_id)
        training_log.trials_to_crit.append(0) #add a zero and increment it in the training loop.
        
        running_frustration = 0
        running_acc = 0
        training_bar = trange(config.max_trials_per_task//config.batch_size)
        for i in training_bar:

            if config.optimize_td:
                #########Gather cognitive inputs ###############
                if len(buffer_acts) > config.horizon-1:
                    # expanded_previous_acc = np.repeat(np.array(buffer_accuracies)[..., np.newaxis], 10, axis=-1)   #Expanded merely to emphasize their signal over the numerous acts
                    expanded_previous_acc = np.array(buffer_accuracies).reshape([-1, 1, 1]).repeat(config.batch_size, 1).repeat(10, 2) # expand batch dim, but alos repeat to emphsaize them
                    gathered_inputs = np.concatenate([buffer_acts, buffer_labels, expanded_previous_acc], axis=-1) #shape  7100 100 266
                    # task_ids_repeated = buffer_task_ids[..., np.newaxis].repeat(100,1)
                    
                    training_inputs = gathered_inputs
                    # training_outputs= task_ids_repeated
                    ins =  torch.tensor(training_inputs, device=config.device) # (input_length, 100, 266)
                    # outs = torch.tensor(training_outputs, device=config.device)
                    #################################################
                    cpred, _, = cog_net(ins)
                    td_context_id  = F.gumbel_softmax(cpred[-1], dim = 1) # will give 15 one_hot.
                else:
                    td_context_id = torch.ones([config.batch_size,config.md_size])/config.md_size    
                    # td_context_id = td_context_id.repeat([config.batch_size, 1])
                training_log.td_context_ids.append(td_context_id.detach().cpu().numpy())

            if config.optimize_bu:
                # context_id = torch.zeros([config.batch_size, config.md_size])
                # context_id[:, torch.argmax(md_context_id, axis=1)] = 1. # Hard max
                
                bu_context_id = net.rnn.md_context_id    
                bu_context_id = F.softmax(bu_context_id.float(), dim=1 ) 
                bu_context_id = bu_context_id.repeat([config.batch_size, 1])
                bu_context_id = bu_context_id.to(config.device)
                bu_context_id.requires_grad_()
                training_log.bu_context_ids.append(bu_context_id.detach().cpu().numpy())
                
            if config.optimize_policy:
                # create non-informative uniform context_ID
                policy_context_id = torch.ones([1,config.md_size])/config.md_size    
                policy_context_id = policy_context_id.repeat([config.batch_size, 1])

            if config.optimize_td or config.optimize_bu:
                if i == 0: # only get one batch of trials at the outset and keep reiterating on them. 
                    inputs, labels = get_trials_batch(envs=env, config = config, batch_size = config.batch_size)
            else:
                inputs, labels = get_trials_batch(envs=env, config = config, batch_size = config.batch_size)
            
            # combine context signals.
            if config.optimize_policy: context_id = policy_context_id 
            if (config.optimize_bu): context_id = bu_context_id
            if (config.optimize_td): context_id =  td_context_id 
            context_id.requires_grad_().retain_grad() #shape batch_size x Md_size
            # print(context_id[0,:])
            # context_id = config.optimize_bu * bu_context_id +\
            #     config.optimize_td * td_context_id + config.optimize_policy * policy_context_id 
            
            outputs, rnn_activity = net(inputs, sub_id=context_id)
            acc  = accuracy_metric(outputs.detach(), labels.detach())
            
            buffer_acts.append(rnn_activity.detach().cpu().numpy().mean(0))
            buffer_labels.append(labels.detach().cpu().numpy()[-1, :, :])
            buffer_accuracies.append(acc)
            buffer_task_ids.append(task_id)
            if len(buffer_acts) > config.horizon: buffer_task_ids.pop(0); buffer_accuracies.pop(0); buffer_labels.pop(0); buffer_acts.pop(0) 
            # print(f'shape of outputs: {outputs.shape},    and shape of rnn_activity: {rnn_activity.shape}')
            #Shape of outputs: torch.Size([20, 100, 17]),    and shape of rnn_activity: torch.Size ([20, 100, 256
            if config.optimize_bu: bu_optimizer.zero_grad()
            if config.optimize_td: td_optimizer.zero_grad()
            if config.optimize_policy: policy_optimizer.zero_grad()
            
            loss = criterion(outputs, labels)
            loss.backward()
            # if step_i % config.print_every_batches == (config.print_every_batches - 1):
                # plot_context_id(config, td_context_id, bu_context_id, task_id)
            if not context_id.grad is None: # if not using gates at all, grad will be none
                training_log.md_grads.append(context_id.grad.data.cpu().numpy())
            else: 
                assert (not (config.use_multiplicative_gates or config.use_additive_gates) ), 'context_ID grad is None'
                

            if False: # handGD
                    caia_lr = 0.001
                    # grad_sum = context_id.grad.data.sum(0)
                    grad_sum = context_id.grad.data.sum(0)
                    md_context_id = md_context_id.to(config.device) - caia_lr* grad_sum# md_grads=(context_id.grad.data.cpu().numpy()) # shape [batch_size, 15]
                    # md_context_id = F.one_hot(torch.argmax(md_context_id, dim=1), md_context_id.shape[1])
            if config.optimize_bu:      bu_optimizer.step()
            if config.optimize_td:      td_optimizer.step()
            if config.optimize_policy:  policy_optimizer.step()

            # from utils import show_input_output
            # show_input_output(inputs, labels, outputs)
            # plt.savefig('example_inpujt_label_output.jpg')
            # plt.close('all')
            # save loss

            training_log.write_basic(step_i, loss.item(), acc, task_id)
            training_log.gradients.append(np.array([torch.norm(p.grad).item() for p in net.parameters() if p.grad is not None]) )
            if config.save_detailed or config.use_cognitive_observer:
                training_log.write_detailed( rnn_activity= rnn_activity.detach().cpu().numpy().mean(0),
                inputs=   [] ,# inputs.detach().cpu().numpy(),
                outputs = outputs.detach().cpu().numpy()[-1, :, :],
                labels =   labels.detach().cpu().numpy()[-1, :, :],
                sampled_act = [], # rnn_activity.detach().cpu().numpy()[:,:, 1:356:36], # Sample about 10 neurons 
                # rnn_activity.shape             torch.Size([15, 100, 356])
                )
            training_bar.set_description('ls, acc: {:0.3F}, {:0.2F} '.format(loss.item(), acc)+ config.human_task_names[task_id])
            # print statistics
            if step_i % config.print_every_batches == (config.print_every_batches - 1):
                ################################ test during training
                net.eval()
                # torch.set_grad_enabled(True)
                with torch.no_grad():
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
                    testing_log.gradients.append(np.mean(np.stack(training_log.gradients[-config.print_every_batches:]),axis=0))
                # torch.set_grad_enabled(False)
                net.train()
 
                #### End testing
            training_log.trials_to_crit[-1] += 1 # increment the total trials spent on this current task
            # relax a little! Only optimizing context signal!
            criterion_accuaracy = config.criterion if task_name not in config.DMFamily else config.criterion_DMfam
            criterion_accuaracy -=0.1
            if ((running_acc > criterion_accuaracy) and config.train_to_criterion) or (i+1== config.max_trials_per_task//config.batch_size):
            # switch task if reached the max trials per task, and/or if train_to_criterion then when criterion reached
                running_acc = 0.
                utils.plot_Nassar_task(envs[task_id], config, context_id=context_id, task_name=task_name, training_log=training_log, net=net )
                break # stop training current task if sufficient accuracy. Note placed here to allow at least one performance run before this is triggered.
            step_i+=1
            running_acc = 0.7 * running_acc + 0.3 * acc
        task_i +=1

    training_log.optimizing_total_batches, testing_log.optimizing_total_batches = step_i, step_i

    return(testing_log, training_log, net)





def plot_context_id(config, bu_context_id, td_context_id, task_id, step_i = 0):
    fig, axes = plt.subplots(1,3)
    axes[0].matshow(td_context_id.detach().cpu().numpy()[:50])
    axes[0].set_title('context_id')
    axes[0].text(2, 52, f'Current Task_ID: {task_id}')
    if (config.use_CaiNet or config.train_cog_obs_only) and not config.use_external_task_id:
        if td_context_id.grad is not None:
            im =axes[1].matshow(td_context_id.grad.data.cpu().numpy()[:50])
            axes[1].set_title('grad context_id')
            cax = fig.add_axes([0.35, 0.3, 0.03, 0.3])
            fig.colorbar(im, cax=cax, orientation='vertical')
        # else: print('Context ID grad is None!')
    axes[2].bar(range(config.md_size), bu_context_id.detach().cpu().numpy().squeeze())
    axes[2].set_title('md_context_id')
    
    plt.savefig(f'./files/to_animate/example_context_ids_{step_i:05}.jpg')
    plt.savefig(f'./files/to_animate/example_context_ids.jpg')
    plt.close('all')


def pre_train_td(config, cog_net, training_log):
    config.horizon =50
    cog_optimizer = 'missing optimizer'
    acts = np.stack(training_log.rnn_activity[-config.horizon:])  # (3127, 100, 356)
    outputs_horizon = np.stack(training_log.outputs[-config.horizon:])
    labels_horizon = np.stack(training_log.labels[-config.horizon:])
    task_ids = np.stack(training_log.task_ids[-config.horizon:])
    rl = np.argmax(labels_horizon, axis=-1)
    ro = np.argmax(outputs_horizon, axis=-1)
    accuracies = (rl==ro).astype(np.float32)

    # previous_acc = np.concatenate([accuracies[:1], accuracies ])[:-1] #shift by one place to make it run one step behind. 
    expanded_previous_acc = np.repeat(accuracies[..., np.newaxis], 10, axis=-1)   #Expanded merely to emphasize their signal over the numerous acts
    gathered_inputs = np.concatenate([acts, labels_horizon, expanded_previous_acc], axis=-1) #shape  7100 100 266

    task_ids_oh= F.one_hot(torch.from_numpy(task_ids).long(), config.md_size)
    task_ids_repeated = task_ids[..., np.newaxis].repeat(100,1)
    
    training_inputs = gathered_inputs
    # training_outputs= task_ids_oh.reshape([task_ids_oh.shape[0], 1, task_ids_oh.shape[1]]).repeat([1,training_inputs.shape[1], 1])
    training_outputs= task_ids_repeated
    ins =  torch.tensor(training_inputs, device=config.device) # (input_length, 100, 266)
    outs = torch.tensor(training_outputs, device=config.device).long()
    #################################################
    current_task_id = 'where to get from'
    cog_optimizer.zero_grad()
    # cin = torch.cat([rnn1_means.detach(), labels_horizon[-1]],dim =-1)
    # cin = cin.reshape([1, *cin.shape])
    # cog_out = cog_net(cin)
    cog_out, cog_acts = cog_net(ins) # Cog_out shape [horizon, batch, 15]
    tids = torch.tensor([current_task_id]*100, device=config.device)
    # cog_loss  = F.cross_entropy(input=cog_out.cpu().squeeze(), target=tids.type(torch.LongTensor))
    #Train on all task_ids in horizon:
    cog_loss = F.cross_entropy(input= cog_out.squeeze().permute([0,2,1]) , target = outs)
    # Or just the current task:
    # cog_loss = F.cross_entropy(input= cog_out.squeeze()[-1] , target = tids)
    # if step_i < 4000: # start testing cog obs on useen data
    cog_loss.backward()
    cog_optimizer.step()

    training_log.cog_obs_preds.append(cog_out.detach().cpu().numpy())

    # if step_i > 100 and (step_i % (config.print_every_batches*50) == (config.print_every_batches*50 - 1)):
        # _,_,cacc = test_model(cog_net, ins, task_ids, step_i)    
    # training_bar.set_description('cog_ls, acc: {:0.3F}, {:0.2F} '.format(cog_loss.item(), acc)+ config.human_task_names[task_id])
