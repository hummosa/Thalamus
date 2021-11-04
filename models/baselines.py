from argparse import Namespace
import numpy as np
import torch
import torchvision
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch import autograd

# Continual learning model API
class ContinualModel(nn.Module):
    """
    Continual learning model.
    """
    NAME = None
    COMPATIBILITY = []

    def __init__(self, backbone: nn.Module, loss: nn.Module,
                       config: Namespace, transform: torchvision.transforms,
                       opt, device,
                       parameters, named_parameters) -> None:
        super(ContinualModel, self).__init__()

        self.net = backbone
        self.loss = loss
        self.config = config
        self.transform = transform
        self.opt = opt
        self.device = device
        self.net.to(self.device)
        self.parameters = parameters
        self.named_parameters = named_parameters

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes a forward pass.
        :param x: batch of inputs
        :param task_label: some models require the task label
        :return: the result of the computation
        """
        return self.net(x)

    def observe(self, inputs: torch.Tensor, labels: torch.Tensor,
                not_aug_inputs: torch.Tensor) -> float:
        """
        Compute a training step over a given batch of examples.
        :param inputs: batch of examples
        :param labels: ground-truth labels
        :param kwargs: some methods could require additional parameters
        :return: the value of the loss function
        """
        pass

    def end_task(self, dataset=None, task_ids=None, config=None):
        """
        At the end of training a task, register parameters.
        :param dataset: dataset
        :param task_ids: indices of learned tasks
        :param config: configs
        """
        pass

    def get_params(self) -> torch.Tensor:
        """
        Returns all the parameters concatenated in a single tensor.
        :return: parameters tensor
        """
        params = []
        for pp in list(self.parameters):
            params.append(pp.view(-1))
        return torch.cat(params)

    def get_grads(self) -> torch.Tensor:
        """
        Returns all the gradients concatenated in a single tensor.
        :return: gradients tensor (input_size * 100 + 100 + 100 * 100 + 100 +
                                   + 100 * output_size + output_size)
        """
        grads = []
        for pp in list(self.parameters):
            grads.append(pp.grad.view(-1))
        return torch.cat(grads)

# Base Model
class Base(ContinualModel):
    NAME = 'Base'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il']

    def __init__(self, backbone, loss, config, transform, opt, device, parameters, named_parameters):
        super(Base, self).__init__(backbone, loss, config, transform, opt, device, parameters, named_parameters)

    def observe(self, inputs, labels, not_aug_inputs, task_id=None):
        self.opt.zero_grad()
        outputs, rnn_activity = self.net(inputs)
        loss = self.loss(outputs, labels)
        loss.backward()
        nn.utils.clip_grad.clip_grad_value_(self.parameters, 1)
        self.opt.step()
        return loss, rnn_activity

# Elastic Weight Consolidation
class EWC(ContinualModel):
    NAME = 'Elastic Weight Consolidation'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il']

    def __init__(self, backbone, loss, config, transform, opt, device, parameters, named_parameters):
        super(EWC, self).__init__(backbone, loss, config, transform, opt, device, parameters, named_parameters)

    def _update_mean_params(self):
        for param_name, param in self.named_parameters.items():
            _buff_param_name = param_name.replace('.', '__')
            self.net.register_buffer(_buff_param_name+'_estimated_mean', param.data.clone())
    
    # CE loss
    def _update_fisher_params(self, current_ds, task_ids, num_batch):
        log_liklihoods = []
        for task_id in task_ids:
            for i in range(num_batch):
                # fetch data
                ob, gt = current_ds.new_trial(task_id=task_id)
                inputs = torch.from_numpy(ob).type(torch.float).to(self.device)
                labels = torch.from_numpy(gt).type(torch.long).to(self.device)
                inputs = inputs[:, np.newaxis, :]
                outputs, _ = self.net(inputs)
                # compute log_liklihoods
                outputs = F.log_softmax(outputs, dim=-1) # the last dim
                log_liklihoods.append(torch.flatten(outputs[:, :, labels]))
        log_likelihood = torch.cat(log_liklihoods).mean()
        grad_log_liklihood = autograd.grad(log_likelihood, self.parameters)
        _buff_param_names = [param[0].replace('.', '__') for param in self.named_parameters.items()]
        for _buff_param_name, param in zip(_buff_param_names, grad_log_liklihood):
            self.net.register_buffer(_buff_param_name+'_estimated_fisher', param.data.clone() ** 2)

    # MSE loss
    # def _update_fisher_params(self, current_ds, task_id, num_batch):
    #     log_liklihoods = []
    #     for i in range(num_batch):
    #         # fetch data
    #         inputs, labels = current_ds(task_id=task_id)
    #         outputs, _ = self.net(inputs)
    #         # compute log_liklihoods
    #         outputs = F.mse_loss(outputs, labels)
    #         log_liklihoods.append(outputs)
    #     log_likelihood = torch.mean(torch.stack(log_liklihoods), dim=0)
    #     grad_log_liklihood = autograd.grad(log_likelihood, self.parameters)
    #     _buff_param_names = [param[0].replace('.', '__') for param in self.named_parameters.items()]
    #     for _buff_param_name, param in zip(_buff_param_names, grad_log_liklihood):
    #         self.net.register_buffer(_buff_param_name+'_estimated_fisher', param.data.clone() ** 2)

    def penalty(self, weight):
        try:
            losses = []
            for param_name, param in self.named_parameters.items():
                _buff_param_name = param_name.replace('.', '__')
                estimated_mean = getattr(self.net, '{}_estimated_mean'.format(_buff_param_name))
                estimated_fisher = getattr(self.net, '{}_estimated_fisher'.format(_buff_param_name))
                losses.append((estimated_fisher * (param - estimated_mean) ** 2).sum())
            return (weight / 2) * sum(losses)
        except AttributeError:
            return 0
    
    def end_task(self, dataset, task_ids, config):
        num_batches = config.EWC_num_trials
        self._update_fisher_params(dataset, task_ids, num_batches)
        self._update_mean_params()

    def observe(self, inputs, labels, not_aug_inputs, task_id=None):
        self.opt.zero_grad()
        output, rnn_activity = self.net(inputs)
        loss = self.loss(output, labels) + self.penalty(self.config.EWC_weight)
        loss.backward()
        self.opt.step()
        return loss, rnn_activity


# Synaptic Intelligence
class SI(ContinualModel):
    NAME = 'Synaptic Intelligence'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il']

    def __init__(self, backbone, loss, config, transform, opt, device, parameters, named_parameters):
        super(SI, self).__init__(backbone, loss, config, transform, opt, device, parameters, named_parameters)

        self.checkpoint = self.get_params().data.clone().to(self.device)
        self.big_omega = None
        self.small_omega = 0

    def penalty(self):
        if self.big_omega is None:
            return torch.tensor(0.0).to(self.device)
        else:
            penalty = (self.big_omega * ((self.get_params() - self.checkpoint) ** 2)).sum()
            return penalty

    def end_task(self, dataset=None, task_ids=None, config=None):
        # big omega calculation step
        if self.big_omega is None:
            self.big_omega = torch.zeros_like(self.get_params()).to(self.device)

        self.big_omega += self.small_omega / ((self.get_params().data - self.checkpoint) ** 2 + self.config.xi)

        # store parameters checkpoint and reset small_omega
        self.checkpoint = self.get_params().data.clone().to(self.device)
        self.small_omega = 0

    def observe(self, inputs, labels, not_aug_inputs, task_id=None):
        self.opt.zero_grad()
        outputs, rnn_activity = self.net(inputs)
        penalty = self.penalty()
        loss = self.loss(outputs, labels) + self.config.c * penalty
        loss.backward()
        nn.utils.clip_grad.clip_grad_value_(self.parameters, 1)
        self.opt.step()

        self.small_omega += self.config.lr * self.get_grads().data ** 2

        return loss, rnn_activity