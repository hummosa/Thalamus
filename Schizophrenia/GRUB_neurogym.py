#%%
import numpy as np
rng = np.random.default_rng()
import matplotlib.pyplot as plt
import neurogym as ngym
import gym
from neurogym import TrialEnv
import sys; sys.path.insert(0, '.')
from utils import stats

class MyTrialEnv(TrialEnv):
    def __init__(self, mean_noise= 0.1, mean_drift = 0, odd_balls_prob = 0.0, change_point_prob = 0.0, safe_trials = 5):
        super().__init__()
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(1,))#, name = ['outcome'])
        self.latent_space = gym.spaces.Box(low=0.0, high=1.0, shape=(1,))#, names = ['mean'])
        self.action_space = gym.spaces.Box(low=0.0, high=1.0, shape=(1,))
        
        #get initial random values
        self.outcome_now = self.observation_space.sample()
        self.mean_now = self.latent_space.sample()

        self.trial_len = 100
        self.far_definition = 0.3 *self.observation_space.__getattribute__('high')

        self.mean_drift = mean_drift 
        self.odd_balls_prob =odd_balls_prob 
        self.mean_noise= mean_noise
        self.change_point_prob = change_point_prob
        self.safe_trials = safe_trials

    def _new_trial(self):
        self.outcome_now = self.observation_space.sample()
        self.mean_now = self.latent_space.sample()
        outcomes, means = [], []
        oddballs = np.zeros(self.trial_len)
        changepoints = np.zeros(self.trial_len)
        s = self.safe_trials
        for i in range(self.trial_len):
            
            self.outcome_now = rng.normal(self.mean_now, self.mean_noise)
            while((self.outcome_now < self.observation_space.__getattribute__('low')) or (self.outcome_now > self.observation_space.__getattribute__('high'))):
                self.outcome_now = rng.normal(self.mean_now, self.mean_noise)
            self.mean_now = rng.normal(self.mean_now, self.mean_drift)
            self.outcome_now = np.clip(self.outcome_now, 0, 1)
            self.mean_now = np.clip(self.mean_now, 0, 1)
            
            if s == 0: # safety period over
                if rng.uniform() < self.odd_balls_prob:
                    oddballs[i] = 1.0
                    #ensure oddball is far enough from currently mean:
                    far_enough = False
                    while (not far_enough):
                        self.outcome_now = self.observation_space.sample()
                        far_enough = abs(self.outcome_now - self.mean_now) > self.far_definition
                    s = self.safe_trials
                
                if rng.uniform() < self.change_point_prob:
                    changepoints[i] = 1.0
                    far_enough = False
                    while(not far_enough):
                        mean_next = self.latent_space.sample()
                        far_enough = abs(mean_next - self.mean_now) > self.far_definition
                    self.mean_now = mean_next
                    s = self.safe_trials
            else:
                s = s-1
            outcomes.append(self.outcome_now)
            means.append(self.mean_now)
        
        ob = self.outcome_now  # observation previously computed
        # Sample observation for the next trial
        self.next_ob = np.random.uniform(0, 1, size=(1,))
        
        trial = dict()
        # Ground-truth is 1 if ob > 0, else 0
        trial['outcomes'] = np.stack(outcomes)
        trial['means'] = np.stack(means)
        trial['oddballs'] = oddballs
        trial['changepoints'] = changepoints
        trial['ground_truth'] = np.stack(outcomes)

        return trial
    
    def _step(self, action):
        ob = self.next_ob
        # If action equals to ground_truth, reward=1, otherwise 0
        reward = (action == self.trial['ground_truth']) * 1.0
        done = False
        info = {'new_trial': True}
        return ob, reward, done, info




# %%
import torch
import torch.nn as nn

batch_size = 200
training_steps = 200
device = 'cuda:0'
# device = 'cpu:0'
learning_rate = 0.01
hidden_size = 32
seq_len = 100
input_dim = 1

import torch
import torch.nn as nn
import math


class GRUCell(nn.Module):
    def __init__(self, input_size, hidden_size, bias=True):
        super(GRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.x2h = nn.Linear(input_size, 3 * hidden_size, bias=bias)
        self.h2h = nn.Linear(hidden_size, 3 * hidden_size, bias=bias)
        # self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, x, hidden):
        x = x.view(-1, x.size(1))

        gate_x = self.x2h(x)
        gate_h = self.h2h(hidden)

        gate_x = gate_x.squeeze()
        gate_h = gate_h.squeeze()

        i_r, i_i, i_n = gate_x.chunk(3, 1)
        h_r, h_i, h_n = gate_h.chunk(3, 1)

        resetgate = torch.sigmoid(i_r + h_r)
        inputgate = torch.sigmoid(i_i + h_i)
        newgate = torch.tanh(i_n + (resetgate * h_n))

        hy = newgate + inputgate * (hidden - newgate)

        return hy
        
class GRUBCell(nn.Module):
    def __init__(self, input_size, hidden_size, bias=True):
        super(GRUBCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.z_size = 2

        # self.register_parameter(name='z', param=torch.nn.Parameter(self.init_mul_gates))
        # self.z =torch.nn.Parameter(self.init_mul_gates)
        self.z =torch.ones(self.z_size).to(device)

        self.x2h = nn.Linear(input_size, 2 * hidden_size, bias=bias)
        self.h2h = nn.Linear(hidden_size, 2 * hidden_size, bias=bias)

        self.h2z = nn.Linear(hidden_size, self.z_size, bias=bias)
        self.x2z = nn.Linear(input_size, self.z_size, bias=bias)
        # self.x2z.get_parameter('weight').data *=1.5
        self.z2r = nn.Linear(self.z_size, hidden_size, bias=bias)
         
    def forward(self, x, hidden):
        # x = x.view(-1, x.size(1))

        gate_x = self.x2h(x).squeeze()
        gate_h = self.h2h(hidden).squeeze()

        i_i, i_n = gate_x.chunk(2, 1)
        h_i, h_n = gate_h.chunk(2, 1)

        z = torch.relu(self.h2z(hidden) + self.x2z(x))
        assert z.shape[0] == x.shape[0]
        z_update_rate = 0.5
        self.z = (1. - z_update_rate) * self.z + z_update_rate * z #z.detach() # just take a copyfd
        self.z = torch.softmax(self.z, dim=1)
        # self.z = torch.nn.functional.gumbel_softmax(self.z, dim=1)
        # self.z.requires_grad_(True)
        resetgate = self.z2r(self.z)
        resetgate = torch.sigmoid(resetgate)
        # resetgate = self.z2r(z)
        
        inputgate = torch.sigmoid(i_i + h_i)
        new_hidden = torch.tanh(i_n + (resetgate * h_n))

        # hy = newgate + inputgate * (hidden - newgate)
        # hy = newgate + inputgate *hidden - inputgate * newgate
        hy = inputgate * hidden + ( 1 - inputgate) * new_hidden 

        return (hy, self.z)

class GRUModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim,  bias=True):
        super(GRUModel, self).__init__()
        self.hidden_dim = hidden_dim
        # self.rnn = GRUCell(input_dim, hidden_dim)
        self.rnn = GRUBCell(input_dim, hidden_dim)

        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.linear = nn.Linear(hidden_dim, output_dim)
        self.linearloss = nn.Linear(hidden_dim, output_dim)
        self.zs_block = []

    def forward(self, x, z=None):
        hn = torch.zeros(x.size(0), self.hidden_dim).to(x.device)
        outs = []
        zs = []
        zgrads = []
        if z is None: # infer z through CAI
            for seq in range(x.size(1)):
                hn, zn = self.rnn(x[:, seq, :], hn)
                hn = self.relu1(hn)
                out = self.linear(hn)
                out = self.relu2(out)
                outloss = self.linearloss(hn)
                outloss = torch.relu(outloss)
                # how far is the current prediction from the input that induced it?
                # using current input as an estimate os current belief about mean, and measuring surprise of current input
                gt_mse = nn.functional.mse_loss(out.detach(), x[:, seq, :].detach(), reduction='none')
                # stats(gt_mse, 'gt_mse')
                # optm 
                lossloss = nn.functional.mse_loss(outloss , gt_mse)
                z_grad = torch.autograd.grad(outputs= lossloss, inputs=self.rnn.z, 
                    only_inputs=True, create_graph=True, retain_graph=True)[0]
                # stats(z_grad, 'z_grad')
                self.rnn.z = (self.rnn.z - 1e5 * z_grad.detach()) 
                zs.append(self.rnn.z.detach())
                # zs.append(z_grad.detach())
                zgrads.append(z_grad.detach())
                outs.append(out)
                optz.zero_grad()
            
        else: # if z sequence is given
            for seq in range(x.size(1)):
                self.rnn.z = z[seq] 
                hn, _ = self.rnn(x[:, seq, :], hn)
                hn = self.relu1(hn)
                out = self.linear(hn)
                out = self.relu2(out)
                outs.append(out)
                # zs.append(self.rnn.z.detach())
            
            # print('does z require grad: ', self.rnn.z.requires_grad)
	    # lossloss.backward(retain_graph=True)
            # optz.param_groups[0]['lr'] = 0.1
            # optz.step()
            # self.rnn.z2r.get_parameter('weight').detach_()
            # self.rnn.z2r.get_parameter('bias').detach_()


        outx = torch.stack(outs, 1)
        if len(zs)>0:     
            zs = torch.stack(zs)
            self.zs_block = zs.detach().cpu().numpy()
        return (outx, zs)

def get_batch(exp_type='Oddball', batch_size=batch_size):
    batches, distMeans = [], []
    params = {
        'noisy_mean':       [0.05, 0, 0 , 0], 
        'drifting_mean':    [0.05, 0.05, 0 , 0],
        'oddball':          [0.05,  0.05, 0.1, 0],
        'changepoint':      [0.05, 0.0, 0.0 , 0.1]
    }
    param= params[exp_type]
    env = MyTrialEnv(mean_noise= param[0], mean_drift = param[1], odd_balls_prob = param[2], change_point_prob = param[3], safe_trials = 5)

    for i in range(batch_size):
        trial = env.new_trial()
        distMean, outcome = trial['means'], trial['outcomes']
        
        batches.append(outcome)
        distMeans.append(distMean)
    batches = torch.from_numpy(np.stack(batches))
    batches.refine_names('batch', 'timesteps', 'output_size')
    batches.align_to('batch', ...) # batch first
    # tbatches = torch.nn.functional.one_hot(torch.from_numpy(batches.astype(int)), screenWidth)
    # return(np.stack(distMeans), tbatches.squeeze().float().to(device))
    return(np.stack(distMeans), batches.float().to(device))

def test_model(exp_type):
    # torch.set_grad_enabled(False)
    #with torch.no_grad():
    distMeans, batch = get_batch(exp_type)
    input = batch[:, :-2, :]
    output = batch[:, -1, :]

    pred, _ = rnn(input.float())
    acc = (pred[:, -1,]) - (output)
    acc = torch.abs(acc) < .1 # consider anthing within 4 steps away as accurate.
    acc = torch.mean(acc.float())
    plt.close('all')
    fig, ax = plt.subplots(1,1)
    # ax = axes.flatten()[0]
    color1 = 'tab:blue'
    color2 = 'tab:red'
    ax.plot(pred.detach().cpu().numpy()[-1,  ], '.', label='RNN preds', color=color2 ,markersize = 5, alpha=1)
    ax.plot(batch.cpu().numpy()[-1, :-1], '.', label='Ground truth',  color=color1,markersize = 4, alpha=0.7)
    ax.plot(distMeans[-1], ':', label='Dist mean', color=color1)
    if hasattr(rnn, 'zs_block'):
        ax.plot(rnn.zs_block[-1,:,:]  , label='Z', linewidth=0.5)
    ax.set_xlabel('Trials')
    ax.set_ylabel('Rewarded position')
    ax.legend()
    ax.set_ylim([-0.1,1.1])
    ax.set_title('Oddball condition' if exp_type == 'Oddball' else 'Change-point condition')
    plt.savefig(f'./Schizophrenia/files/current_results_{exp_type}.jpg', dpi=200)
    plt.show()
    
    # torch.set_grad_enabled(True)
    return(acc.detach().cpu().numpy())

class RNN_out(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        # self.bn = nn.BatchNorm1d(input_size)
        # self.gru = nn.RNN(input_size, hidden_size, batch_first=True, nonlinearity='relu')
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, inp):
        # inp = self.bn(inp)
        out, hidden = self.gru(inp)
        x = self.linear(out)
        return x, out

# rnn = RNN_out(input_size = input_dim, hidden_size=hidden_size, output_size=input_dim).to(device)
rnn = GRUModel( input_dim, hidden_size, input_dim).to(device)

training_params = list()
for name, param in rnn.named_parameters():
    # if (not name.__contains__('rnn.z')) :
        print(name)
        training_params.append(param)
    # else:
        # print('exluding: ', name)
opt = torch.optim.Adam(training_params, lr= learning_rate)

print('___---___ z opt params')
training_params = list()
for name, param in rnn.named_parameters():
    # if (name.__contains__('z')) :
    if (name=='rnn.z') :
        print(name)
        training_params.append(param)
    else:
        print('exluding: ', name)
optz = torch.optim.Adam(rnn.parameters(), lr= learning_rate)

for train_i in range(training_steps):
    exp_type = 'drifting_mean'# if train_i < training_steps*0.5 else 'drifting_mean'
    exp_type = 'noisy_mean' #if train_i < training_steps*0.5 else 'changepoint'
    exp_type = 'changepoint' #if train_i < training_steps*0.5 else 'oddball'
    exp_type = 'oddball' #if train_i < training_steps*0.5 else 'changepoint'
    _,batch = get_batch(exp_type)
    input = batch[:, :-1, :]
    output = batch[:, 1:, :]
    # with torch.autograd.set_detect_anomaly(True):
    pred, zs = rnn(input.float()) # torch.Size([100, 199, 40])
    opt.zero_grad()
    pred, _ = rnn(input.float(), zs) # torch.Size([100, 199, 40])
    loss = nn.functional.mse_loss(pred,output)
    loss.backward()
    opt.step()

    if train_i % 10 == 0:
        acc = test_model(exp_type) #('changepoint')
        print(f'step: {train_i}  acc: {acc:0.2f}')



# %%
