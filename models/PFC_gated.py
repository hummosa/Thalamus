''' Defines a gated PFC model that takes an index to retreive the relevant multiplicative gate mask for the task '''

import random
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import matplotlib.pyplot as plt
from utils import sparse_with_mean, stats

class CTRNN_MD(nn.Module):
    """Continuous-time RNN that can take MD inputs.
    Args:
        input_size: Number of input neurons
        hidden_size: Number of hidden neurons
        sub_size: Number of subpopulation neurons
    Inputs:
        input: (seq_len, batch, input_size), network input
        hidden: (batch, hidden_size), initial hidden activity
    """

    def __init__(self, config, dt=100, **kwargs):
        super().__init__()
        self.input_size =  config.input_size
        self.hidden_size = config.hidden_size
        self.output_size = config.output_size
        self.md_size = config.md_size
        self.use_multiplicative_gates = config.use_multiplicative_gates
        self.use_additive_gates = config.use_additive_gates
        self.divide_gating_to_input_and_recurrence = config.divide_gating_to_input_and_recurrence
        self.device = config.device
        self.g = .5 # Conductance for recurrents. Consider higher values for realistic richer RNN dynamics, at least initially.

        self.tau = config.tau
        # if dt is None:
            # alpha = 1
        # else:
        alpha = dt / self.tau
        self.alpha = alpha
        self.oneminusalpha = 1 - alpha

        if self.use_multiplicative_gates:
            # self.gates = torch.normal(config.md_mean'], 1., size=(config.md_size'], config.hidden_size'], ),
            #   dtype=torch.float) #.type(torch.LongTensor) device=self.device,
            # control density /

            # simple binary gates:
            # self.init_mul_gates = np.random.uniform(0, 1, size=(config.md_size, config.hidden_size, )) 
            # self.init_mul_gates = (self.init_mul_gates < config.MD2PFC_prob).astype(float)
            # self.init_mul_gates = torch.from_numpy(self.init_mul_gates).to(config.device).float() #* torch.abs(self.gates) 
            # self.gates = self.init_mul_gates
            
            # Gaussian with sparsity gates
            self.init_mul_gates = torch.empty((config.md_size, config.hidden_size, )) 
            sparse_with_mean(self.init_mul_gates, config.gates_sparsity, config.gates_mean, config.gates_std)
            self.init_mul_gates = torch.nn.functional.relu(self.init_mul_gates) #* config.gates_divider + config.gates_offset
            self.register_parameter(name='mul_gates', param=torch.nn.Parameter(self.init_mul_gates))
            # import pdb; pdb.set_trace()
        if self.use_additive_gates:
            # self.gates = torch.normal(config.md_mean'], 1., size=(config.md_size'], config.hidden_size'], ),
            #   dtype=torch.float) #.type(torch.LongTensor) device=self.device,
            # control density /
            self.init_add_gates = torch.zeros(size=(config.md_size, config.hidden_size, )) 
            self.init_add_gates = torch.nn.init.xavier_uniform_(self.init_add_gates, gain=nn.init.calculate_gain('relu')).to(config.device).float()
            # self.gates = self.add_gates
            self.register_parameter(name='add_gates', param=torch.nn.Parameter(self.init_add_gates))

            # self.init_mul_gates = (self.init_mul_gates < config.MD2PFC_prob).astype(float)
            # self.init_mul_gates = torch.from_numpy(self.init_mul_gates).to(config.device).float() #* torch.abs(self.gates) 
          
            # torch.uniform(config.md_mean'], 1., size=(config.md_size'], config.hidden_size'], ),
            #  device=self.device, dtype=torch.float)
                # *config.G/np.sqrt(config.Nsub*2)
            # Substract mean from each row.
            # self.gates -= np.mean(self.gates, axis=1)[:, np.newaxis]
        self.register_parameter(name='md_context_id', param=torch.nn.Parameter(torch.ones([1,config.md_size])/config.md_size ))
        self.md_context_id.retain_grad()
        # sensory input layer
        self.input2h = nn.Linear(self.input_size, self.hidden_size)

        # hidden layer
        self.h2h = nn.Linear(self.hidden_size, self.hidden_size)
        self.reset_parameters()


    def reset_parameters(self):
        # identity*0.5
        nn.init.eye_(self.h2h.weight)
        self.h2h.weight.data *= self.g

    def init_hidden(self, input):
        batch_size = input.shape[1]
        # as zeros
        hidden = torch.zeros(batch_size, self.hidden_size)
        # as uniform noise
        # hidden = 1/self.hidden_size*torch.rand(batch_size, self.hidden_size)
        return hidden.to(input.device)

    def recurrence(self, input, sub_id, hidden):
        """Recurrence helper."""
        ext_input = self.input2h(input)
        rec_input = self.h2h(hidden)

        if self.divide_gating_to_input_and_recurrence:
            neurons_per_md = sub_id.shape[1]//2
            mask1, mask2 = torch.zeros_like(sub_id), torch.zeros_like(sub_id)
            mask1[:, :neurons_per_md] =  torch.ones_like(sub_id)[:, :neurons_per_md]
            mask2[:, neurons_per_md:] =  torch.ones_like(sub_id)[:, neurons_per_md:]
            if self.use_multiplicative_gates:
                batch_sub_encoding = sub_id # to prevent from having to output 1 value from softmax # already onehot encoded. Skipping this step. 
                gates = torch.matmul((mask1 * batch_sub_encoding).to(self.device), self.mul_gates)
                rec_input = torch.multiply( gates, rec_input)
                # ext_input = torch.multiply( gates, ext_input)
            if self.use_additive_gates:
                batch_sub_encoding = sub_id 
                gates = torch.matmul((mask2 * batch_sub_encoding).to(self.device), self.add_gates)
                # rec_input = torch.add( gates, rec_input)
                ext_input = torch.add( -torch.abs(gates), ext_input)
        else:  # only gate RNN  
            if self.use_multiplicative_gates:
                batch_sub_encoding = sub_id # to prevent from having to output 1 value from softmax # already onehot encoded. Skipping this step. 
                gates = torch.matmul(batch_sub_encoding.to(self.device), self.mul_gates)
                rec_input = torch.multiply( gates, rec_input)
            if self.use_additive_gates:
                batch_sub_encoding = sub_id 
                gates = torch.matmul(batch_sub_encoding.to(self.device), self.add_gates)
                rec_input = torch.add( gates, rec_input)
            
        pre_activation = ext_input + rec_input
        
        h_new = torch.relu(hidden * self.oneminusalpha + pre_activation * self.alpha)

        return h_new

    def forward(self, input, sub_id, hidden=None):
        """Propogate input through the network."""
        
        num_tsteps = input.size(0)

        # init network activities
        if hidden is None:
            hidden = self.init_hidden(input)
        
        # initialize variables for saving network activities
        output = []
        for i in range(num_tsteps):
            hidden = self.recurrence(input[i], sub_id, hidden)
                
            # save PFC activities
            output.append(hidden)
        
        output = torch.stack(output, dim=0)
        return output, hidden


class MLP_MD(nn.Module):
    """ An MLP that can take MD inputs.
    Args:
        input_size: Number of input neurons
        hidden_sizes: Number of hidden neurons
        sub_size: Number of subpopulation neurons
    Inputs:
        input: (seq_len, batch, input_size), network input
        hidden: (batch, hidden_size), initial hidden activity
    """

    def __init__(self, config, dt=100, **kwargs):
        super().__init__()
        self.input_size =  config.input_size
        self.hidden_size = config.hidden_size
        self.output_size = config.output_size
        self.md_size = config.md_size
        self.use_multiplicative_gates = config.use_multiplicative_gates
        self.use_additive_gates = config.use_additive_gates

        self.device = config.device
        self.g = .5 # Conductance for recurrents. Consider higher values for realistic richer RNN dynamics, at least initially.

        if self.use_multiplicative_gates:
            # Gaussian with sparsity gates
            self.init_mul_gates = torch.empty((config.md_size, config.hidden_size, )) 
            sparse_with_mean(self.init_mul_gates, config.gates_sparsity, config.gates_mean, config.gates_std)
            self.init_mul_gates = torch.nn.functional.relu(self.init_mul_gates) #* config.gates_divider + config.gates_offset
            self.register_parameter(name='mul_gates', param=torch.nn.Parameter(self.init_mul_gates))
        if self.use_additive_gates:
            self.init_add_gates = torch.zeros(size=(config.md_size, config.hidden_size, )) 
            self.init_add_gates = torch.nn.init.xavier_uniform_(self.init_add_gates, gain=nn.init.calculate_gain('relu')).to(config.device).float()
            self.register_parameter(name='add_gates', param=torch.nn.Parameter(self.init_add_gates))
        self.register_parameter(name='md_context_id', param=torch.nn.Parameter(torch.ones([1,config.md_size])/config.md_size ))
        self.md_context_id.retain_grad()
        # sensory input layer
        
        self.input2h = nn.Linear(self.input_size, self.hidden_size)
        self.h2h1 = nn.Linear(self.hidden_size, self.hidden_size)
        # self.h2h2 = nn.Linear(self.hidden_size, self.hidden_size)

    def forward(self, input, sub_id, hidden=None):
        """Propogate input through the network."""
        ## inputs are shaped [28, batch_size, 28] due to using sequence first for the rnn. 
        local_batch_size,h, w = input.shape

        x = input.reshape([local_batch_size, h*w ]) ## Flatten images
        x = F.relu(self.input2h(x)) 
        # x = F.relu(self.h2h2(x))

        if self.use_multiplicative_gates:
            batch_sub_encoding = sub_id # to prevent from having to output 1 value from softmax # already onehot encoded. Skipping this step. 
            gates = torch.matmul(batch_sub_encoding.to(self.device), self.mul_gates)
            x_gated = torch.multiply( gates, x)
        if self.use_additive_gates:
            batch_sub_encoding = sub_id 
            gates = torch.matmul(batch_sub_encoding.to(self.device), self.add_gates)
            x_gated = torch.add( gates, x)

        x = F.relu(self.h2h1(x_gated))

        if self.use_multiplicative_gates:
            batch_sub_encoding = sub_id # to prevent from having to output 1 value from softmax # already onehot encoded. Skipping this step. 
            gates = torch.matmul(batch_sub_encoding.to(self.device), self.mul_gates)
            x_gated = torch.multiply( gates, x)
        if self.use_additive_gates:
            batch_sub_encoding = sub_id 
            gates = torch.matmul(batch_sub_encoding.to(self.device), self.add_gates)
            x_gated = torch.add( gates, x)

        x = F.relu(x_gated)
        return (x, 0) # the additional 0 just for compatibility with the return from RNN
                

class RNN_MD(nn.Module):
    """Recurrent network model.
    Args:
        input_size: int, input size
        hidden_size: int, hidden size
        sub_size: int, subpopulation size
        output_size: int, output size
        rnn: str, type of RNN, lstm, rnn, ctrnn, or eirnn
    """

    def __init__(self, config):
        super().__init__()
        if config.model == 'RNN':
            self.rnn = CTRNN_MD(config)    
        elif config.model =='MLP':
            self.rnn = MLP_MD(config)
        elif config.model =='MLP+RNN':
            self.rnn = MLP_RNN_MD(config)
        self.drop_layer = nn.Dropout(p=0.05)
        self.fc = nn.Linear(config.hidden_size, config.output_size)
        # self.latent_activation_function = self.normalized_activation
        self.latent_activation_function = F.softmax

    def forward(self, x, sub_id):
        rnn_activity, _ = self.rnn(x, sub_id)
        rnn_activity = self.drop_layer(rnn_activity)
        out = self.fc(rnn_activity)
        return out, rnn_activity

    def normalized_activation(self, logits, dim =1):
        logits -=torch.min(logits)
        normalized = logits / torch.norm(logits, p=1)
        return(normalized)


class RNN_MD_GRU(nn.Module):
    """GRU compariosn
    """
    def __init__(self, config):
        super().__init__()
        
        self.rnn = nn.GRU(config.input_size, config.hidden_size, batch_first=False)
        self.drop_layer = nn.Dropout(p=0.05)
        self.fc = nn.Linear(config.hidden_size, config.output_size)

    def forward(self, x, sub_id):
        rnn_activity, _ = self.rnn(x)
        rnn_activity = self.drop_layer(rnn_activity)
        out = self.fc(rnn_activity)
        return out, rnn_activity

class Cognitive_Net(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Cognitive_Net, self).__init__()
        # self.bn = nn.BatchNorm1d(input_size)
        self.gru = nn.GRU(input_size, hidden_size, batch_first=False)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, inp):
        # inp = self.bn(inp)
        out, hidden = self.gru(inp)
        x = self.linear(out)
        return x, out