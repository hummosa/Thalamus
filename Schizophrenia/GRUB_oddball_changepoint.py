# %% [markdown]
# Reproducing the Schizophrenia Helicopter experiments. 
# 

# %%
import numpy as np
rng = np.random.default_rng()
import matplotlib.pyplot as plt

## Refactor
def get_oddball_trials(numOutcomes=200, sigma = 20, Haz=.125, safe=2, screenWidth=300, drift = 7.5):
    '''# numOutcomes     #how long should the block of trials be?
    # sigma = 20         #standard deviation of the generative dist...
    # Haz=.125           #probability of a change-point on any given trial
    # safe=4;            #except that we set hazard rate equal to zero for "safe" trials after a change-point
    # screenWidth=300
    # drift = 7.5 # '''
    #generate outcomes
    mean=round(rng.uniform()*screenWidth)

    outcome=np.ones((numOutcomes, 1)); #this will be an array of outcomes
    distMean=np.ones((numOutcomes, 1));#this will be an array of distribution mean
    cp=np.zeros((numOutcomes, 1));     #this will be an array of binary change-point variable
    s=safe

    for i in range(1, numOutcomes ):
        if i >1:
            trialDrift=rng.normal(0,drift)

            if  (distMean[i-1]+trialDrift)>0 and (distMean[i-1]+trialDrift)<screenWidth:
                mean= distMean[i-1]+trialDrift
            else:   #If the drift would push you off screen, drift in other direction. 
                mean= distMean[i-1]-trialDrift
        
        if rng.uniform()<Haz and s==0: # Jumpy anywhere.
            outcome[i]=rng.uniform()*screenWidth
            s=safe
        else: #Drift around mean.
            outcome[i]=np.round(rng.normal(mean, sigma))
            counter = 0
            while (outcome[i]>screenWidth) or (outcome[i]<0): # while outcome value not acceptable keep resampling 
                outcome[i]=np.round(rng.normal(mean, sigma))
                counter+=1
                if counter > 5:
                    pass
                    # print('mean: ', mean, ' i: ', i,' outcome[i]: ', outcome[i])
                if counter > 200:
                    print('failed to recover from out of bounds! Breaking')
                    break
            s=max([s-1, 0]) # decrement no of safe trials left.
            
        distMean[i]=mean
    return(distMean,outcome)

distMean, outcome = get_oddball_trials()
plt.plot(distMean, '--k')
plt.plot(outcome, '.r')




# %%
# Shift and update condition
def get_Change_point_trials(numOutcomes=200, sigma = 20, Haz=.125, safe=8, screenWidth=300):
    '''# numOutcomes     #how long should the block of trials be?
    # sigma = 20         #standard deviation of the generative dist...
    # Haz=.125           #probability of a change-point on any given trial
    # safe=4;            #except that we set hazard rate equal to zero for "safe" trials after a change-point
    # screenWidth=300
    # drift = 7.5 # NO drift in this condition'''
    #generate outcomes
    mean=round(rng.uniform()*screenWidth)

    outcome=np.ones((numOutcomes, 1)); #this will be an array of outcomes
    distMean=np.ones((numOutcomes, 1));#this will be an array of distribution mean
    cp=np.zeros((numOutcomes, 1));     #this will be an array of binary change-point variable
    s=safe

    for i in range(1, numOutcomes ):
        if rng.uniform()<Haz and s==0: # Jumpy anywhere.
            mean= np.round(rng.uniform()*screenWidth)
            cp[i]=1
            s= safe
        else:
            s=max([s-1, 0])
        outcome[i]=np.round(rng.normal(mean, sigma))
        while (outcome[i]>screenWidth) or (outcome[i]<0): # while outcome value not acceptable keep resampling 
            outcome[i]=np.round(rng.normal(mean, sigma))
        s=max([s-1, 0]) # decrement no of safe trials left.
        distMean[i]=mean
    return(distMean,outcome)

distMean, outcome = get_Change_point_trials(sigma=5, screenWidth=40)
# distMean, outcome = get_trials(numOutcomes= seq_len, sigma=3, screenWidth=screenWidth-1)
plt.plot(distMean, '--k')
plt.plot(outcome, '.r')



# %%
import torch
import torch.nn as nn

batch_size = 100
training_steps = 150
device = 'cuda:0'
# device = 'cpu:0'
hidden_size = 128
seq_len = 200
screenWidth = 40
input_dim = 1
# exp_type = 'Oddball' 
# exp_type = 'Change-point'

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

        self.init_mul_gates = torch.empty((self.z_size )) 
        # sparse_with_mean(self.init_mul_gates, config.gates_sparsity, config.gates_mean, config.gates_std)
        self.init_mul_gates = torch.nn.functional.relu(self.init_mul_gates)
        self.register_parameter(name='z', param=torch.nn.Parameter(self.init_mul_gates))

        self.x2h = nn.Linear(input_size, 3 * hidden_size, bias=bias)
        self.h2h = nn.Linear(hidden_size, 3 * hidden_size, bias=bias)
        self.z2r = nn.Linear(self.z_size, hidden_size, bias=bias)

    def forward(self, x, hidden):
        x = x.view(-1, x.size(1))

        gate_x = self.x2h(x).squeeze()
        gate_h = self.h2h(hidden).squeeze()

        i_r, i_i, i_n = gate_x.chunk(3, 1)
        h_r, h_i, h_n = gate_h.chunk(3, 1)

        resetgate = torch.sigmoid(i_r + h_r)
        # TAKE ONLY 2 gating vars, discard the rest of resetgate values.
        if self.z_size > 0:
            self.z.data = resetgate[:, :self.z_size].data
            resetgate = self.z2r(self.z)
        inputgate = torch.sigmoid(i_i + h_i)
        newgate = torch.tanh(i_n + (resetgate * h_n))

        hy = newgate + inputgate * (hidden - newgate)

        return (hy, self.z)

class GRUModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, layer_dim=1,  bias=True):
        super(GRUModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        # self.rnn = GRUCell(input_dim, hidden_dim)
        self.rnn = GRUBCell(input_dim, hidden_dim)
        # self.fc = nn.Linear(hidden_dim, output_dim)
        self.zs_block = []

    def forward(self, x):
        hn = torch.zeros(x.size(0), self.hidden_dim).to(x.device)
        outs = []
        zs = []

        for seq in range(x.size(1)):
            hn, zn = self.rnn(x[:, seq, :], hn)
            outs.append(hn)
            zs.append(zn)

        outx = torch.stack(outs, 1)
        self.zs_block = torch.stack(zs, 1)
        return (outx, outx[:,-1,:])

class RNN_out(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        # self.bn = nn.BatchNorm1d(input_size)
        # self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.gru = GRUModel(input_size, hidden_size, output_size)
        self.relu = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, inp):
        # inp = self.bn(inp)
        out, _ = self.gru(inp)
        x = self.linear(self.relu(out))
        x = self.relu2(x)
        return x*100, out

def get_batch(exp_type='Oddball', batch_size=batch_size):
    batches, distMeans = [], []
    for i in range(batch_size):
        if exp_type == 'Oddball':
            distMean, outcome = get_oddball_trials(seq_len, sigma=3, screenWidth=screenWidth-1, drift=2)
        else:
            distMean, outcome = get_Change_point_trials(numOutcomes= seq_len, sigma=3, screenWidth=screenWidth-1)

        batches.append(outcome)
        distMeans.append(distMean)
    batches = torch.from_numpy(np.stack(batches))
    # tbatches = torch.nn.functional.one_hot(torch.from_numpy(batches.astype(int)), screenWidth)
    # return(np.stack(distMeans), tbatches.squeeze().float().to(device))
    return(np.stack(distMeans), batches.float().to(device))

def test_model(exp_type):
    # torch.set_grad_enabled(False)
    with torch.no_grad():
        distMeans, batch = get_batch(exp_type)
        input = batch[:, :-1, :]
        output = batch[:, -1, :]

        pred, _ = rnn(input.float())
        acc = (10*pred[:, -1,]) - (output)
        acc = torch.abs(acc) < 4 # consider anthing within 4 steps away as accurate.
        acc = torch.mean(acc.float())
    plt.close('all')
    fig, ax = plt.subplots(1,1)
    # ax = axes.flatten()[0]
    color1 = 'tab:blue'
    color2 = 'tab:red'
    ax.plot(pred.detach().cpu().numpy()[-1,  ], '.', label='RNN preds', color=color2 ,markersize = 5, alpha=1)
    ax.plot(batch.cpu().numpy()[-1, :-1], '.', label='Ground truth',  color=color1,markersize = 4, alpha=0.7)
    ax.plot(distMeans[-1], ':', label='Dist mean', color=color1)
    ax.plot(rnn.gru.zs_block[-1,:,:].detach().cpu().numpy() *3 , label='Z', linewidth=0.5)
    ax.set_xlabel('Trials')
    ax.set_ylabel('Rewarded position')
    ax.legend()
    ax.set_title('Oddball condition' if exp_type == 'Oddball' else 'Change-point condition')
    plt.savefig(f'./current_results_{exp_type}.jpg', dpi=200)
    plt.show()
    
    # torch.set_grad_enabled(True)
    return(acc.detach().cpu().numpy())

rnn = RNN_out(input_size = input_dim, hidden_size=hidden_size, output_size=input_dim).to(device)
opt = torch.optim.Adam(rnn.parameters(), lr= 0.001)
optz = torch.optim.Adam(rnn.parameters(), lr= 0.001)

for train_i in range(training_steps):
    exp_type = 'Change-point' if train_i < training_steps*0.5 else 'Oddball'
    _,batch = get_batch(exp_type)
    input = batch[:, :-1, :]
    output = batch[:, 1:, :]

    pred, _ = rnn(input.float()) # torch.Size([100, 199, 40])
    loss = nn.functional.mse_loss(pred,output)
    opt.zero_grad()
    loss.backward()
    opt.step()
    with torch.no_grad:
        world = nn.functional.mse_loss(pred,output, reduction='none') # shape (batch size, time-steps, 1)

    if train_i % 10 == 0:
        acc = test_model(exp_type)
        print(f'step: {train_i}  acc: {acc:0.2f}')


