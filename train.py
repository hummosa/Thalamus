from cmath import nan
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from analysis.visualization import plot_cluster_discovery, plot_long_term_cluster_discovery
from logger.logger import SerialLogger
from tqdm import tqdm, trange
import gym
import neurogym as ngym
import matplotlib.pyplot as plt
from utils import *
from Schizophrenia.tasks_coded_in_neurogym import *
import utils

def criterion(output, labels, use_loss='nll'):    # criterion & optimizer
    if use_loss =='mse':
        crit = nn.MSELoss()
        loss = crit(output, labels)
    elif use_loss =='nll':
        crit = F.nll_loss
        if output.ndim > 2:
            loss = crit(torch.log_softmax(output[-1,...], dim = -1, dtype=torch.float), torch.argmax(labels[-1,...], dim=-1))
        else:
            loss = crit(torch.log_softmax(output, dim = -1, dtype=torch.float), torch.argmax(labels, dim=-1))

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
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[1, 10,20, 30, 40, 50], gamma=0.8)
    
    if not str(net.rnn) == 'GRU(33, 356)':
        if config.bu_adam:
            bu_optimizer = torch.optim.Adam([tp[1] for tp in net.named_parameters() if tp[0] == 'rnn.md_context_id'], 
                lr=config.lr*config.lr_multiplier, weight_decay=(config.lr/10)*config.weight_decay_multiplier)
                # adding l2 reg (weeight decay) with an order of magnitude lower than lr.
        else:
            bu_optimizer = torch.optim.SGD([tp[1] for tp in net.named_parameters() if tp[0] == 'rnn.md_context_id'],  lr=config.lr*config.lr_multiplier, momentum=0.95)
        
    # create non-informative uniform context_ID
    context_id = torch.ones([1,config.md_size])/config.md_size    
    context_id = context_id.repeat([config.batch_size, 1])

    # Make all tasks, but reorder them from the tasks_id_name list of tuples
    envs = [None] * len(config.tasks_id_name)
    build_env(config, envs)
    running_acc = 0 # just to init
    bu_running_acc = 0
    converged = False # flag used to detect when the model has converged
    tasks_after_converged = int(config.no_of_tasks*2)
    recall_test_context_id, test_task_id = None, None # init these to use for testing latent recall
    training_log.recall_bu_accuracy, training_log.recall_latent_correlation = [], []
    
    task_i = 0
    bar_tasks = tqdm(task_seq)
    for (task_id, task_name) in bar_tasks:
        # optimizer = torch.optim.SGD(training_params, lr=config.lr*10, momentum=0.99)
    
        if (converged):  # after converged. Run tasks one more juist for a demo and them move on
            tasks_after_converged-=1
            if tasks_after_converged == 0:
                break
        #if the last task in shuffle training_ extend it to 400 trials
        task_id = int(task_id)
        env = envs[int(task_id)]
        bar_tasks.set_description('i: ' + str(step_i))
        training_log.switch_trialxxbatch.append(step_i)
        training_log.switch_task_id.append(task_id)
        training_log.trials_to_crit.append(0) #add a zero and increment it in the training loop.
        training_log.latents_to_crit.append(0) #add a zero and increment it in the training loop.
        criterion_accuaracy = config.criterion if task_name not in config.DMFamily else config.criterion_DMfam
        ## Adjust learning rate of BU optim:
        # if len(training_log.trials_to_crit) > 5:
        #     for param_group in bu_optimizer.param_groups:
        #         param_group['lr'] = config.lr * (1+max(0., 50 - np.mean(training_log.trials_to_crit[-10:])))
        if config.detect_convergence and (len(training_log.trials_to_crit)> (config.no_of_tasks*2)):
            recent_average_ttc = np.mean( np.stack(training_log.trials_to_crit[-config.no_of_tasks*2:]))
            if recent_average_ttc < config.converged_ttc_criterion:
                converged = True
                config.detect_convergence = False
                # config.train_to_criterion = True
                tn = (bar_tasks.total - bar_tasks.n)
                run_more = min(tn, config.no_of_tasks * 2)
                bar_tasks.update(tn-run_more) # run no_of_tasks  more tasks and stop.
                config.use_weight_updates = False
                if not hasattr(training_log, 'converged_detected_at'):
                    training_log.converged_detected_at = step_i
                
        lu_attempts = 5
        running_frustration = 0
        training_bar = trange(config.max_trials_per_task//config.batch_size)
        for i in training_bar:
            if config.abort_rehearsal_if_accurate and config.paradigm_sequential:
                if len(testing_log.accuracies)>0: 
                    if (i==0) and (testing_log.accuracies[-1][task_id]) > (criterion_accuaracy-0.05): # criterion with some leniency. Only check at the begning of this current task.
                        print(f'task name: {task_name}, \t task ID: {task_id}\t skipped at accuracy: {testing_log.accuracies[-1][task_id]}')
                        break # stop training this task and jump to the next.
            inputs, labels, _ = get_trials_batch(envs=env, config = config, batch_size = config.batch_size)
            if inputs.shape[0] < config.batch_size:
                inputs, labels, _ = get_trials_batch(envs=env, config = config, batch_size = config.batch_size)
            # inputs.refine_names('timestep', 'batch', 'input_dim')
            # labels.refine_names('timestep', 'batch', 'output_dim')
            if config.actually_use_task_ids and not hasattr(training_log, 'start_testing_at'): #use ids only in the first exposure to the tasks.
                context_id = F.one_hot(torch.tensor([task_id]* config.batch_size), config.md_size).type(torch.float)
            ########################################################################
            outputs, rnn_activity = net(inputs, sub_id=(context_id/config.gates_divider)+config.gates_offset)
            ########################################################################
            # outputs, rnn_activity = net(inputs, sub_id=(context_id/config.gates_divider)+config.gates_offset, gt=labels)
            acc  = accuracy_metric(outputs.detach(), labels.detach(), config)
            
            # if acc is < running_acc by 0.2. run optim and get a new context_id
            # if config.use_latent_updates: bu_running_acc, context_id_after_lu, total_latent_updates = latent_updates(1, config, net, testing_log, training_log, bu_optimizer, bu_running_acc, criterion_accuaracy, envs, inputs, labels)
            training_log.md_context_ids.append(context_id.detach().cpu().numpy())
            if ((running_acc-acc) > 0.2 ) and config.use_latent_updates: # assume some novel something happened or (converged and (bu_running_acc<(criterion_accuaracy-.1)))
                max_latent_updates = int(min(15 * len(training_log.trials_to_crit)-10, config.max_no_of_latent_updates))
                # max_latent_updates = config.max_no_of_latent_updates
                bu_running_acc, context_id_after_lu, total_latent_updates = latent_updates(max_latent_updates, config, net, testing_log, training_log, bu_optimizer, bu_running_acc, criterion_accuaracy, envs, inputs, labels)
                training_log.latents_to_crit[-1] += total_latent_updates # add the total number of latent updates
                context_id = net.latent_activation_function(torch.from_numpy(context_id_after_lu ), dim=1).to(config.device)
                training_log.lu_stamps_acc_improve.append( (step_i, bu_running_acc- acc))
                print(f'============== Latent update improvement: {training_log.lu_stamps_acc_improve[-1] } LU_acc: {bu_running_acc}   ACC {acc}')
                ###############
            if converged and (bu_running_acc < criterion_accuaracy -0.1): #If converged, then tasks should be solved. If they are not, try to resample the latent for a number of attempts.
                while( lu_attempts and (bu_running_acc < criterion_accuaracy -0.1)):
                    net.rnn.md_context_id.data = torch.rand_like(net.rnn.md_context_id.data) # resample randomly a new md embedding and try again.
                    bu_running_acc, context_id_after_lu, total_latent_updates = latent_updates(config.max_no_of_latent_updates, config, net, testing_log, training_log, bu_optimizer, bu_running_acc, criterion_accuaracy, envs, inputs, labels)
                    print(f'resampling latent due to failure at trial stamp: {training_log.stamps[-1]}')
                    print(f'attempts left: {lu_attempts}, bu_rnning_acc: {bu_running_acc}')
                    lu_attempts-=1
                    ################
                    training_log.latents_to_crit[-1] += total_latent_updates # add the total number of latent updates
                    context_id = net.latent_activation_function(torch.from_numpy(context_id_after_lu ), dim=1).to(config.device)
                    training_log.lu_stamps_acc_improve.append( (step_i, bu_running_acc- acc))
                    print(f'============  Latent update improvement: {training_log.lu_stamps_acc_improve[-1] }')
                

            if (not (recall_test_context_id is None)) and config.test_latent_recall:
                test_bu_running_acc, test_context_id_after_lu = latent_recall_test(config, net, testing_log, training_log, bu_optimizer, bu_running_acc, criterion_accuaracy, envs, 
                test_task_id=test_task_id, test_task_context=recall_test_context_id)
                latent_correlation = np.correlate(recall_test_context_id.squeeze(), test_context_id_after_lu.squeeze())
            else:
                latent_correlation, test_bu_running_acc = nan, nan
            training_log.recall_latent_correlation.append(latent_correlation)
            training_log.recall_bu_accuracy.append(test_bu_running_acc)


                # plt.close('all')
                # plot_long_term_cluster_discovery(config, training_log, testing_log)
            # training_log.bu_context_ids.append(context_id.detach().cpu().numpy())
            # print(f'shape of outputs: {outputs.shape},    and shape of rnn_activity: {rnn_activity.shape}')
            #Shape of outputs: torch.Size([20, 100, 17]),    and shape of rnn_activity: torch.Size ([20, 100, 256
            optimizer.zero_grad()
            
            loss = criterion(outputs, labels, use_loss=config.training_loss)
            loss.backward()
           
            if config.use_weight_updates and not (bu_running_acc > (criterion_accuaracy-0.1)): # do not learn if optimzing rule input already close enough to solving task
                optimizer.step()
            # from utils import show_input_output
            # show_input_output(inputs, labels, outputs)
            # plt.savefig('example_inpujt_label_output.jpg')
            # plt.close('all')
            # save loss

            if type(rnn_activity) is tuple:
                rnn_activity, md = rnn_activity
                # training_log.md_context_ids.append(md.detach().cpu().numpy())
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

            if step_i % config.print_every_batches == (config.print_every_batches - 1):
                plot_long_term_cluster_discovery(config, training_log, testing_log)
                # test_in_training(config, net, testing_log, training_log, step_i, envs)
                latent_test_in_training(config,
                    net, testing_log, training_log, bu_optimizer,
                    bu_running_acc, criterion_accuaracy, envs, inputs, labels, step_i)
            
            training_log.lrs.append(optimizer.param_groups[0]["lr"])
            step_i+=1
            
            running_acc = 0.7 * running_acc + 0.3 * acc
            if ((running_acc > criterion_accuaracy) ) or (bu_running_acc > (criterion_accuaracy-0.1)) or (i+1== config.max_trials_per_task//config.batch_size):
            # switch task if reached the max trials per task, and/or if train_to_criterion then when criterion reached
                # running_acc = 0.
                if training_log.trials_to_crit[-1] == 0: # if no trial to crit recorded previously
                    training_log.trials_to_crit[-1] = i # log the total trials spent on this current task
                # utils.plot_Nassar_task(envs[task_id], config, context_id=context_id, task_name=task_name, training_log=training_log, net=net )
                if (config.train_to_criterion) or (i+1== config.max_trials_per_task//config.batch_size):
                    # test_in_training(config, net, testing_log, training_log, step_i, envs)
                    bu_running_acc = 0
                    if not str(net.rnn) == 'GRU(33, 356)':
                        recall_test_context_id, test_task_id = net.rnn.md_context_id.clone().detach().cpu().numpy(), task_id
                    break # stop training current task if sufficient accuracy. Note placed here to allow at least one performance run before this is triggered.


        if config.accuracy_convergence and step_i > config.print_every_batches+10:
            # unique_tasks = np.unique(training_log.switch_task_id)
            current_average_accuracy = np.mean([testing_log.accuracies[-1][ut] for ut in range(len(config.tasks))]) 
            if current_average_accuracy > config.average_accuracy_criterion:
                break
        #no more than number of blocks specified
        task_i +=1
        if len(training_log.trials_to_crit) % config.no_of_tasks == (config.no_of_tasks - 1) and config.use_learning_rate_scheduler:
            scheduler.step() # lower learning rate every no of tasks learned.
            pass
        # training_log.sample_input = inputs[:,0,:].detach().cpu().numpy().T
        # training_log.sample_label = labels[:,0,:].detach().cpu().numpy().T
        # training_log.sample_output = outputs[:,0,:].detach().cpu().numpy().T
    testing_log.total_batches, training_log.total_batches = step_i, step_i

    return(testing_log, training_log, net)

def latent_updates(max_latent_updates, config, net, testing_log, training_log, bu_optimizer, bu_running_acc, criterion_accuaracy, envs, inputs, labels):
    
    context_id_before_loop = net.rnn.md_context_id.detach().clone()
    bubuffer, bu_accs, latent_losses = [], [], []
    total_latent_updates = 0
    latents_to_criterion = 0
    for total_latent_updates in range(max_latent_updates):
        bubuffer.append(net.rnn.md_context_id.detach().clone().cpu().numpy())
        bu_acc,latent_loss = md_error_loop(config, net, training_log, criterion, bu_optimizer, inputs, labels, accuracy_metric)
        bu_accs.append(bu_acc); latent_losses.append(latent_loss)
        bu_running_acc = 0.7 * bu_running_acc + 0.3 * bu_acc
        if bu_running_acc > (criterion_accuaracy-0.1) and (not (latents_to_criterion ==0)): #stop optim if reached criter
            print(f'LU solved task at trial stamp: {training_log.stamps[-1]}',)
            latents_to_criterion = total_latent_updates
            # break
        # if (total_latent_updates == int(config.no_latent_updates//2)): # if LUs are not getting anywhere, assume stuck, resample randomly a new md embedding and try again.
        #     net.rnn.md_context_id.data = torch.rand_like(net.rnn.md_context_id.data) 
        #     print(f'resampling latent due to failure at trial stamp: {training_log.stamps[-1]}')
    if (latents_to_criterion ==0): # if the solved question up there was not triggered, record the total usaed updates 
        latents_to_criterion = total_latent_updates
    context_id_after_loop = net.rnn.md_context_id.detach().clone().cpu().numpy()
    cb = context_id_before_loop.detach().cpu().numpy()
    print(f'MD updates: {total_latent_updates}: {cb} by {(context_id_after_loop-cb)}')

    # if len(bubuffer) > 0:
        # plot_cluster_discovery(config, bubuffer, training_log, testing_log, bu_accs, latent_losses)
    return bu_running_acc, context_id_after_loop, latents_to_criterion


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

    if config.bu_adam:
        bu_optimizer = torch.optim.Adam([tp[1] for tp in net.named_parameters() if tp[0] == 'rnn.md_context_id'], 
            lr=config.lr*config.lr_multiplier, weight_decay=(config.lr/10)*config.weight_decay_multiplier)
            # adding l2 reg (weeight decay) with an order of magnitude lower than lr.
    else:
        bu_optimizer = torch.optim.SGD([tp[1] for tp in net.named_parameters() if tp[0] == 'rnn.md_context_id'],  lr=config.lr*config.lr_multiplier, momentum=0.95)
    
    # bu_optimizer = torch.optim.Adam([tp[1] for tp in net.named_parameters() if tp[0] == 'rnn.md_context_id'], 
    # lr=config.lr*50,
    # weight_decay=config.lr*0.1)
    # bu_optimizer = torch.optim.SGD([tp[1] for tp in net.named_parameters() if tp[0] == 'rnn.md_context_id'],  lr=config.lr*30000)

    
    td_training_params = list()
    print('cognitive network optimized parameters')
    for name, param in cog_net.named_parameters():
        print(name)
        td_training_params.append(param)
        
    td_optimizer = torch.optim.Adam(td_training_params, lr=config.lr*(100 if cog_net.gru.hidden_size ==1 else 1))

    # Make all tasks, but reorder them from the tasks_id_name list of tuples
    envs = [None] * len(config.tasks_id_name)
    build_env(config, envs)


    # initialize buffer
    config.horizon =150
    buffer_acts = []
    buffer_grads = []
    buffer_grads_targets = []
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
        training_log.latents_to_crit.append(0) #add a zero and increment it in the training loop.
        
        cognitive_loss = None
        running_frustration = 0
        running_acc = 0
        training_bar = trange(config.max_trials_per_task//config.batch_size)
        config.one_batch_success = False
        for i in training_bar:

            if config.one_batch_optimization and not config.one_batch_success:
                if i == 0: # only get one batch of trials at the outset and keep reiterating on them. 
                    inputs, labels, _ = get_trials_batch(envs=env, config = config, batch_size = config.batch_size)
            else:
                inputs, labels, _ = get_trials_batch(envs=env, config = config, batch_size = config.batch_size)

            if config.optimize_bu:
                # context_id = torch.zeros([config.batch_size, config.md_size])
                # context_id[:, torch.argmax(md_context_id, axis=1)] = 1. # Hard max
                
                bu_context_id = net.latent_activation_function(net.rnn.md_context_id, dim=1)
                bu_context_id = bu_context_id.repeat([config.batch_size, 1])
                bu_context_id = bu_context_id.to(config.device)
                bu_context_id.requires_grad_()
                if config.md_loop_rehearsals > 0:
                    training_log.md_context_ids.append(bu_context_id.detach().cpu().numpy())
                else:
                    training_log.bu_context_ids.append(bu_context_id.detach().cpu().numpy())
                
            config.no_latent_updates = 0
            if config.no_latent_updates:
                context_id_before_loop = net.rnn.md_context_id.detach()
                buffer_grads.append(context_id_before_loop.detach().cpu().numpy())
                for _ in range(config.no_latent_updates):
                    md_error_loop(config, net, training_log, criterion, bu_optimizer, inputs, labels, accuracy_metric)
                context_id_after_loop = net.rnn.md_context_id.detach()
                buffer_grads_targets.append(context_id_after_loop.detach().cpu().numpy())
            
            if config.optimize_td:
                #########Gather cognitive inputs ###############
                if len(buffer_acts) > config.horizon-1:
                    # expanded_previous_acc = np.repeat(np.array(buffer_accuracies)[..., np.newaxis], 10, axis=-1)   #Expanded merely to emphasize their signal over the numerous acts
                    expanded_previous_acc = np.array(buffer_accuracies).reshape([-1, 1, 1]).repeat(config.batch_size, axis=1).repeat(10, axis=2) # expand batch dim, but also repeat to emphsaize them
                    # gathered_inputs = np.concatenate([buffer_acts, buffer_labels, expanded_previous_acc], axis=-1) #shape  7100 100 266
                    buffer_grads_tensor= np.stack(buffer_grads).repeat(config.batch_size, axis=1)
                    gathered_inputs = np.concatenate([buffer_acts, buffer_grads_tensor[1:], expanded_previous_acc], axis=-1) #shape  7100 100 266
                    # task_ids_repeated = buffer_task_ids[..., np.newaxis].repeat(100,1)
                    
                    training_inputs = gathered_inputs
                    # training_outputs= task_ids_repeated
                    ins =  torch.tensor(training_inputs, device=config.device) # (input_length, 100, 266)
                    # outs = torch.tensor(training_outputs, device=config.device)
                    #################################################
                    cpred, _, = cog_net(ins)
                    buffer_grads_targets_tensor= np.stack(buffer_grads_targets[1:]).repeat(config.batch_size, axis=1)
                    cognitive_loss = F.mse_loss(cpred, torch.from_numpy(buffer_grads_targets_tensor).to(config.device))
                    # td_context_id  = F.gumbel_softmax(cpred[-1], dim = 1) # will give 15 one_hot.
                    td_context_id  = net.latent_activation_function(cpred[-1], dim = 1)  
                else:
                    td_context_id = torch.ones([config.batch_size,config.md_size])/config.md_size  
                    cognitive_loss = None  
                    # td_context_id = td_context_id.repeat([config.batch_size, 1])
                training_log.td_context_ids.append(td_context_id.detach().cpu().numpy())

            if config.optimize_policy:
                # create non-informative uniform context_ID
                policy_context_id = torch.ones([1,config.md_size])/config.md_size    
                policy_context_id = policy_context_id.repeat([config.batch_size, 1])

            # combine context signals.
            if config.optimize_policy: context_id = policy_context_id 
            if (config.optimize_bu): context_id = bu_context_id
            if (config.optimize_td): context_id =  td_context_id 
            context_id.requires_grad_().retain_grad() #shape batch_size x Md_size
            # print(context_id[0,:])
            # context_id = config.optimize_bu * bu_context_id +\
            #     config.optimize_td * td_context_id + config.optimize_policy * policy_context_id 
            
            outputs, rnn_activity = net(inputs, sub_id=context_id)
            acc  = accuracy_metric(outputs.detach(), labels.detach(), config)
            
            buffer_acts.append(rnn_activity.detach().cpu().numpy().mean(0))
            if config.model =='RNN':
                buffer_labels.append(labels.detach().cpu().numpy()[-1, :, :])
            else:
                buffer_labels.append(labels.detach().cpu().numpy())
            buffer_accuracies.append(acc)
            buffer_task_ids.append(task_id)
            if len(buffer_acts) > config.horizon: buffer_task_ids.pop(0); buffer_accuracies.pop(0); buffer_labels.pop(0); buffer_acts.pop(0)
            if (len(buffer_acts) > config.horizon) and config.optimize_td: buffer_grads_targets.pop(0);buffer_grads.pop(0)
            # print(f'shape of outputs: {outputs.shape},    and shape of rnn_activity: {rnn_activity.shape}')
            #Shape of outputs: torch.Size([20, 100, 17]),    and shape of rnn_activity: torch.Size ([20, 100, 256
            if config.optimize_bu: bu_optimizer.zero_grad()
            if config.optimize_td: td_optimizer.zero_grad()
            if config.optimize_policy: policy_optimizer.zero_grad()
            
            if cognitive_loss:
                loss = criterion(outputs, labels) + cognitive_loss
            else:
                loss = criterion(outputs, labels)
            loss.backward()
            # if step_i % config.print_every_batches == (config.print_every_batches - 1):
                # plot_context_id(config, td_context_id, bu_context_id, task_id)
            if not context_id.grad is None: # if not using gates at all, grad will be none
                training_log.md_grads.append(context_id.grad.data.cpu().numpy())
            else: 
                assert (not (config.use_multiplicative_gates or config.use_additive_gates) ), 'context_ID grad is None'

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
            # if step_i % config.print_every_batches == (config.print_every_batches - 1):
                # test_in_training(config, net, testing_log, training_log, step_i, envs)
            step_i+=1
            # relax a little! Only optimizing context signal!
            criterion_accuaracy = config.criterion if task_name not in config.DMFamily else config.criterion_DMfam
            criterion_accuaracy -=0.1
            
            if ((running_acc > criterion_accuaracy) ) or (i+1== config.max_trials_per_task//config.batch_size):
            # switch task if reached the max trials per task, and/or if train_to_criterion then when criterion reached
                running_acc = 0.
                if training_log.latents_to_crit[-1] == 0: # if no trial to crit recorded previously
                    training_log.latents_to_crit[-1] = i # log the total trials spent on this current task
                # utils.plot_Nassar_task(envs[task_id], config, context_id=context_id, task_name=task_name, training_log=training_log, net=net )
                if (config.train_to_criterion) or (i+1== config.max_trials_per_task//config.batch_size):
                    break # stop training current task if sufficient accuracy. Note placed here to allow at least one performance run before this is triggered.
            running_acc = 0.7 * running_acc + 0.3 * acc
        task_i +=1


    training_log.optimizing_total_batches, testing_log.optimizing_total_batches = step_i, step_i

    return(testing_log, training_log, net)

def md_error_loop(config, net, training_log, criterion, bu_optimizer, inputs, labels, accuracy_metric):
    
    bu_context_id = net.rnn.md_context_id    
    bu_context_id = net.latent_activation_function(bu_context_id.float(), dim=1 ) # /config.gates_divider 
    bu_context_id = bu_context_id.repeat([config.batch_size, 1])
    bu_context_id = bu_context_id.to(config.device)
    bu_context_id.requires_grad_()
    # training_log.bu_context_ids.append(bu_context_id.detach().cpu().numpy())

    context_id = bu_context_id
    context_id.requires_grad_().retain_grad() #shape batch_size x Md_size
                        
    outputs, rnn_activity = net(inputs, sub_id=context_id)
    acc  = accuracy_metric(outputs.detach(), labels.detach(), config)
    bu_optimizer.zero_grad()
    loss = criterion(outputs, labels, config.training_loss) 
    loss.backward()
    bu_optimizer.step()
    return (acc, loss.detach().cpu().numpy())

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

def latent_test_in_training(config,
         net, testing_log, training_log, bu_optimizer,
         bu_running_acc, criterion_accuaracy, envs, inputs, labels, step_i):
    # torch.set_grad_enabled(False)
    # net.eval()
    testing_log.stamps.append(step_i)
    testing_context_ids = list(range(len(envs)))  # envs are ordered by task id sequentially now.
                # testing_context_ids_oh = [F.one_hot(torch.tensor([task_id]* config.test_num_trials), config.md_size).type(torch.float) for task_id in testing_context_ids]

    action_accuracies = defaultdict()
    testing_latents_to_crits = defaultdict()
        ### Swap out the context the network has
    context_id_before_testing = net.rnn.md_context_id.detach().clone().cpu().numpy()
    testing_context_ids = list(range(len(envs)))  # envs are ordered by task id sequentially now.
    for (context_id, env) in (zip(testing_context_ids, envs)):

        test_task_context = torch.rand_like(net.rnn.md_context_id.data) # resample randomly
        net.rnn.md_context_id.data = test_task_context.to(config.device)
        inputs, labels,_ = get_trials_batch(env, config.test_num_trials, config, test_batch=True)
        max_latent_updates = config.test_no_latent_updates
        bu_running_acc, context_id_after_lu, total_latent_updates = latent_updates(max_latent_updates,config,
         net, testing_log, training_log, bu_optimizer,
         bu_running_acc, criterion_accuaracy, envs, inputs, labels)
        context_id_finalized = net.latent_activation_function(torch.from_numpy(context_id_after_lu ), dim=1).to(config.device)
        
        action_pred, _ = net(inputs, sub_id=(context_id_finalized)) # shape [500, 10, 17]
        
        ap = torch.argmax(action_pred, -1) # shape ap [500, 10]
        gt = torch.argmax(labels, -1)

        action_accuracy = accuracy_metric(action_pred, labels, config)
        action_accuracies[context_id] = action_accuracy
        testing_latents_to_crits[context_id] = total_latent_updates

    # put model md back to where we found it.
    net.rnn.md_context_id.data = torch.from_numpy(context_id_before_testing).to(config.device) 
    testing_log.accuracies.append(action_accuracies)
    try:
        gradients_past = min(step_i-training_log.start_optimizing_at, config.print_every_batches) # to avoid np.stack gradients from training and optimization. They might be of different lengths
        testing_log.gradients.append(np.mean(np.stack(training_log.gradients[-gradients_past:]),axis=0))
    except:
        pass
    # net.train()


def get_latent_performance(config,net, testing_log, training_log, bu_optimizer, bu_running_acc, criterion_accuaracy, envs, context_ids, batch_size=100):
    action_accuracies = defaultdict()
        ### Swap out the context the network has
    context_id_before_testing = net.rnn.md_context_id.detach().clone().cpu().numpy()

    for (context_id, env) in (zip(context_ids, envs)):

        test_task_context = torch.rand_like(net.rnn.md_context_id.data) # resample randomly
        net.rnn.md_context_id.data = torch.from_numpy(test_task_context).to(config.device)
        inputs, labels,_ = get_trials_batch(env, batch_size, config, test_batch=True)
        config.no_latent_updates= 400
        bu_running_acc, context_id_after_lu, total_latent_updates = latent_updates(config,
         net, testing_log, training_log, bu_optimizer,
         bu_running_acc, criterion_accuaracy, envs, inputs, labels)
        context_id_finalized = net.latent_activation_function(torch.from_numpy(context_id_after_lu ), dim=1).to(config.device)
        
        action_pred, _ = net(inputs, sub_id=(context_id_finalized)) # shape [500, 10, 17]
        
        ap = torch.argmax(action_pred, -1) # shape ap [500, 10]
        gt = torch.argmax(labels, -1)

        action_accuracy = accuracy_metric(action_pred, labels)
        action_accuracies[context_id] = action_accuracy
        
    return( action_accuracies)
