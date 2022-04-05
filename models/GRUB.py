''' Defines a gated PFC model that takes an index to retreive the relevant multiplicative gate mask for the task '''

from pickletools import optimize
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
        self.bias = True

        self.md_size = config.md_size
        self.use_multiplicative_gates = config.use_multiplicative_gates
        self.use_additive_gates = config.use_additive_gates

        self.device = config.device
        self.g = .5 # Conductance for recurrents. Consider higher values for realistic richer RNN dynamics, at least initially.
        self.tau = config.tau
        # if dt is None:
            # alpha = 1
        # else:
        alpha = dt / self.tau
        self.alpha = alpha
        self.oneminusalpha = 1 - alpha

        self.md = None # initial set to None, later initizlized andn maintained after input comes
        
        self.x2h = nn.Linear(self.input_size, 2 * self.hidden_size, bias=self.bias)
        self.h2h = nn.Linear(self.hidden_size, 2 * self.hidden_size, bias=self.bias)

        self.h2md = nn.Linear(self.hidden_size, self.md_size, bias=self.bias)
        self.x2md = nn.Linear(self.input_size, self.md_size, bias=self.bias)
        # self.x2md.get_parameter('weight').data *=1.5
        # self.md2h = nn.Linear(self.md_size, self.hidden_size, bias=self.bias)
        self.drop_layer = nn.Dropout(p=0.05)
        self.h2r = nn.Linear(config.hidden_size, self.output_size)
        self.h2r.weight.data +=.06

        if self.use_multiplicative_gates:
            # Gaussian with sparsity gates
            self.init_mul_gates = torch.empty((config.md_size, config.hidden_size, )) 
            sparse_with_mean(self.init_mul_gates, config.gates_sparsity, config.gates_mean, config.gates_std)
            self.init_mul_gates = torch.nn.functional.relu(self.init_mul_gates)
            self.register_parameter(name='mul_gates', param=torch.nn.Parameter(self.init_mul_gates))
        if self.use_additive_gates:
            self.init_add_gates = torch.nn.init.xavier_uniform_(self.init_add_gates, gain=nn.init.calculate_gain('relu')).to(config.device).float()
            self.register_parameter(name='add_gates', param=torch.nn.Parameter(self.init_add_gates))

        self.register_parameter(name='md_context_id', param=torch.nn.Parameter(torch.ones([1,config.md_size])))
        self.md_context_id.retain_grad()
        # sensory input layer
        self.reset_parameters()
        # loss monitoring
        self.mean_loss = 100
        self.loss_momentum = 0.9

        #md optimizer
        # self.optimize_md = torch.optim.Adam(self.md)
        
    def reset_parameters(self):
        # identity*0.5
        nn.init.eye_(self.h2h.weight)
        self.h2h.weight.data *= self.g

    def init_hidden(self, input):
        batch_size = input.shape[1]
        # as zeros
        hidden = torch.zeros(batch_size, self.hidden_size)
        md = torch.zeros(batch_size, self.md_size)
        # as uniform noise
        # hidden = 1/self.hidden_size*torch.rand(batch_size, self.hidden_size)
        return (hidden.to(input.device), md.to(input.device))

    def recurrence(self, input, hidden, md, update_md = True):
        """Recurrence helper."""
        # ext_input = self.input2h(input)
        # rec_input = self.h2h(hidden)

        #calc the two gru gates tand the recurrent input.
        gate_x = self.x2h(input).squeeze()
        gate_h = self.h2h(hidden).squeeze()

        hidden_to_keep, ext_input = gate_h.chunk(2, 1)
        input_to_keep, rec_input = gate_x.chunk(2, 1)
        keepgate = torch.sigmoid(input_to_keep + hidden_to_keep)

        if update_md: 
            md_keep_ratio = 0.9
            md_new = torch.relu(self.h2md(hidden) + self.x2md(input))
            md = md_keep_ratio * md + (1-md_keep_ratio) * md_new
            md = F.gumbel_softmax(md)

        if self.use_multiplicative_gates:
            gates = torch.matmul(md, self.mul_gates)
            rec_input = torch.multiply( gates, rec_input)
        if self.use_additive_gates:
            batch_sub_encoding = md
            gates = torch.matmul(batch_sub_encoding, self.add_gates)
            rec_input = torch.add( gates, rec_input)
        
        pre_activation = ext_input + rec_input
        new_hidden = torch.sigmoid(pre_activation)
        
        # h_new = torch.relu(hidden * self.oneminusalpha + pre_activation * self.alpha)
        h_new = keepgate * hidden + (1 - keepgate) * new_hidden
        h_new = self.drop_layer(h_new)
        out = torch.nn.functional.relu(self.h2r(h_new))
        
        return (out, h_new, md)

    def forward(self, input, task_id, gt = None, hidden=None, md=None, update_md= True):
        """Propogate input through the network."""
        
        num_tsteps = input.size(0)

        # init network activities
        if (hidden is None) :
            hidden, _ = self.init_hidden(input)
        if (md is None) :
            if (self.md is None) :
                _, md = self.init_hidden(input)
                md.requires_grad = True
            else:
                md = self.md.detach().clone().reshape([1,-1]).repeat([input.shape[1], 1] ) # expand into a batch size
        
        # initialize variables for saving network activities
        output = []
        rnn_activity = []
        for i in range(num_tsteps):
            out,hidden, md_main_path = self.recurrence(input[i], hidden, md, update_md=True)
                
            # save PFC activities
            rnn_activity.append(hidden)
            output.append(out)
        md_main_path.retain_grad()
        
        output = torch.stack(output, dim=0)
        # rnn_activity = torch.stack(rnn_activity, dim=0)
        if gt is not None:
            model_loss = self.loss_func(output, gt, use_loss='nll')
            self.mean_loss = self.loss_momentum * self.mean_loss + (1-self.loss_momentum) * model_loss.sum()

            if (model_loss - self.mean_loss) > 0.5 * (self.mean_loss):
                md = md_main_path.detach() # remove connecion to the previous graph.
                # deliberate until loss is down or 200 trials.
                deliberate_i = 0
                while ((model_loss - self.mean_loss) > 1.0 * (self.mean_loss) and (deliberate_i < 30)):
                    # model_loss.backward() # run through to just clear the grad graph already constructed.
                    self.h2h.zero_grad(); self.h2md.zero_grad(); self.mul_gates.grad = None
                    md = md.detach() # remove connecion to the previous graph.
                    md.requires_grad = True
                    outputs = []
                    for i in range(num_tsteps):
                        output_temp, hidden, md = self.recurrence(input[i], hidden, md, update_md=False)
                        outputs.append(output_temp)
                    outputs = torch.stack(outputs, dim=0)
                    # model_loss = self.loss_func(outputs, gt)
                    acc_loss = nn.functional.nll_loss(torch.log_softmax(outputs[-1,...], dim = -1, dtype=torch.float), torch.argmax(gt[-1,...], dim=-1))
                    # acc_loss.backward()
                    md_grad = torch.autograd.grad(outputs= acc_loss, inputs=md, 
                            only_inputs=True, create_graph=False, retain_graph=False)[0]
                    # stats(md_grad, 'md_grad')
                    print(f'md: {md.mean(0).detach().cpu().numpy()}, \t md_grads: {md_grad.mean(0).detach().cpu().numpy()}  \t loss: {acc_loss}\t outputs {outputs[-1,0,:].detach().cpu().numpy()}')
                    md = (md - 1e1* md_grad.detach().mean(0)) 
                    self.mean_loss = self.loss_momentum * self.mean_loss + (1-self.loss_momentum) * acc_loss

                    deliberate_i +=1

            output, hidden, md_main_path = self.forward(input, task_id, md = md.detach(), update_md = False) # rerun the normal forward to build the graph again for the normal loss learning. 

        if update_md:
            self.md= md.mean(0).detach() # reduce to a single estimate across batck. detach to cut off gradient graph in between trials. 

            # b = torch.randn(1, 1, 128, 256, dtype=torch.float64)
            # sys.getsizeof(b)
            # sys.getsizeof(b.storage())

        return (output, hidden, md_main_path)
    
    def loss_func(self, output, labels, use_loss='nll'):    # criterion & optimizer
        if use_loss =='mse':
            crit = nn.MSELoss()
            loss = crit(output, labels)
        elif use_loss =='nll':
            crit = F.nll_loss
            loss = crit(torch.log_softmax(output[-1,...], dim = -1, dtype=torch.float), torch.argmax(labels[-1,...], dim=-1))
        return loss

class RNN_MD(nn.Module):
    """Recurrent network model. A GRU with an MD-like bottleneck gating.
    Args:
        input_size: int, input size
        hidden_size: int, hidden size
        sub_size: int, subpopulation size
        output_size: int, output size
        rnn: str, type of RNN, lstm, rnn, ctrnn, or eirnn
    """

    def __init__(self, config):
        super().__init__()
        self.rnn = CTRNN_MD(config)

    def forward(self, x, sub_id, gt=None):
        out, rnn_activity,md = self.rnn(x, sub_id, gt)
        return (out, (rnn_activity, md))

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