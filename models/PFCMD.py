import random
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from utils import stats, sparse_with_mean

# MD for neurogym tasks
class MD_GYM():
    def __init__(self, config, positiveRates=True, dt=0.001):
        self.hidden_size = config.hidden_size
        self.hidden_ctx_size = config.hidden_ctx_size
        self.md_size = config.md_size
        self.positiveRates = positiveRates
        self.num_active = config.md_active_size # num_active: num MD active per context
        self.learn = True # update MD weights or not
        self.sendinputs = True # send inputs to RNN or not
        self.config = config

        self.tau = 0.02
        self.tau_times = 4
        self.dt = dt
        self.tau_pretrace = 1000 # unit, time steps
        self.tau_posttrace = 1000 # unit, time steps
        self.Hebb_learning_rate = 1e-4
        Gbase = 0.75  # determines also the cross-task recurrence

        # initialize weights
        self.init_weights()
        # initialize activities
        self.init_traces()

        # Choose G based on the type of activation function
        #  unclipped activation requires lower G than clipped activation,
        #  which in turn requires lower G than shifted tanh activation.
        if self.positiveRates:
            self.G = Gbase
            self.tauMD = self.tau * self.tau_times  ##self.tau
        else:
            self.G = Gbase
            self.MDthreshold = 0.4
            self.tauMD = self.tau * 10 * self.tau_times
        self.init_activity()

    def init_traces(self):
        self.MDpreTrace = np.zeros(shape=(self.hidden_ctx_size))
        self.MDpreTrace_filtered = np.zeros(shape=(self.hidden_ctx_size))
        self.MDpreTrace_binary = np.zeros(shape=(self.hidden_ctx_size))
        self.MDpostTrace = np.zeros(shape=(self.md_size))
        self.MDpreTrace_threshold = 0

    def init_weights(self):
        # PFC context -> MD weights
        self.wPFC2MD = np.random.normal(0,
                                        1 / np.sqrt(self.md_size * self.hidden_ctx_size),
                                        size=(self.md_size, self.hidden_ctx_size))
        # MD -> PFC weights
        # self.wMD2PFC = np.random.normal(0,
        #                                 1 / np.sqrt(self.md_size * self.hidden_size),
        #                                 size=(self.hidden_size, self.md_size))
        self.wMD2PFC = np.zeros(shape=(self.hidden_size, self.md_size))
        for i in range(self.wMD2PFC.shape[0]):
            j = np.floor(np.random.rand()*self.md_size).astype(int)
            self.wMD2PFC[i, j] = -5
        
        ### Relaxing the separation.
        # stats(self.gates)
        # Mean, -2.45417, var 6.24790, min -5.000, max 0.000, norm 121.34661099511597
        # stats(self.wMD2PFC)
        # Mean, -2.50000, var 6.25000, min -5.000, max 0.000, norm 122.47448713915891

            
            # Binary gates formulation:

        


        # self.wMD2PFCMult = get_corr_gates(self.mul_gates, self.config)
        config = self.config
        import os

        if config.load_corr_gates and (os.path.isfile(f'./data/perf_corr_mat_var1_0.npy',)):#else pfc_gated will pick random overlapping gates. 
            print('------------------   loading correlations from ' +f'./data/perf_corr_mat_var1_0.npy')
            gates_tasks = np.load(f'./data/perf_corr_mat_var1_0.npy', allow_pickle=True).item()
            gates_corr = gates_tasks['corr_mat'] 
            task_order = gates_tasks['tasks'] 
            task_ids = np.argwhere(np.array(task_order) == config.task_seq[0]), np.argwhere(np.array(task_order) == config.task_seq[1])
            task_ids = task_ids[0].squeeze(), task_ids[1].squeeze()
            print('task_ids: ', task_ids)
            gates_cov = np.array([[ 1., gates_corr[task_ids].squeeze(),], [gates_corr[task_ids].squeeze(), 1.]])
            sampled_gates = np.random.multivariate_normal(np.zeros(self.md_size), gates_cov, self.hidden_size)
            self.wMD2PFC = sampled_gates.copy()
            sampled_gates = torch.tensor(sampled_gates> config.gates_gaussian_cut_off, device=config.device).float()
            self.wMD2PFCMult = sampled_gates.clone()

            # NOTE As a control for the correlated gates exp I'm comparing to the additive effects
            self.gates = np.random.uniform(size=(self.hidden_size, self.md_size))
            self.gates = (self.gates < self.config.MD2PFC_prob).astype(float) 
            self.wMD2PFC = (self.gates -1) *5 #shift it to match Zhongxuan's -5 inhibition.
            # self.wMD2PFCMult = self.gates

        else:
            #Sparse uniform formulation
            self.mul_gates = torch.empty(size=(self.hidden_size, self.md_size))
            self.add_gates = torch.empty(size=(self.hidden_size, self.md_size))
            # torch.nn.init.sparse_(self.gates, self.config.gates_sparsity, std=self.config.gates_std)
            
            self.wMD2PFCMult = sparse_with_mean(self.mul_gates, self.config.gates_sparsity, mean=1., std=self.config.gates_std)
            self.wMD2PFC = sparse_with_mean(self.add_gates, self.config.gates_sparsity, mean=0, std=self.config.gates_std)
            # torch.nn.init.sparse_(tensor, sparsity, std=0.01)
            # N(0, std=0.01)
            # sparsity – The fraction of elements in each column to be set to zero
            # std – the standard deviation of the normal distribution used to generate the non-zero values

        stats(self.wMD2PFCMult, 'wMD2PFC mul: ')
        stats(self.wMD2PFC, 'wMD2PFC add:')

        # self.wMD2PFCMult = np.random.normal(0,
                                            # 1 / np.sqrt(self.md_size * self.hidden_size),
                                            # size=(self.hidden_size, self.md_size))
        # Hebbian learning mask
        self.wPFC2MDdelta_mask = np.ones(shape=(self.md_size, self.hidden_ctx_size))
        self.MD_mask = np.array([])
        self.PFC_mask = np.array([])

    def init_activity(self):
        self.MDinp = np.zeros(shape=self.md_size)
        
    def __call__(self, input, *args, **kwargs):
        """Run the network one step

        For now, consider this network receiving input from PFC,
        input stands for activity of PFC neurons
        output stands for output current to MD neurons

        Args:
            input: array (n_input,)
            

        Returns:
            output: array (n_output,)
        """
        # compute MD outputs
        #  MD decays 10x slower than PFC neurons,
        #  so as to somewhat integrate PFC input over time
        if self.positiveRates:
            self.MDinp += self.dt / self.tauMD * (-self.MDinp + np.dot(self.wPFC2MD, input))
        else:
            # shift PFC rates, so that mean is non-zero to turn MD on
            self.MDinp += self.dt / self.tauMD * (-self.MDinp + np.dot(self.wPFC2MD, (input + 0.5)))      
        MDout = self.winner_take_all(self.MDinp)

        # update
        if self.learn:
            # update PFC-MD weights
            self.update_weights(input, MDout)

        return MDout

    def update_trace(self, rout, MDout):
        self.MDpreTrace += 1. / self.tau_pretrace * (-self.MDpreTrace + rout)
        self.MDpostTrace += 1. / self.tau_posttrace * (-self.MDpostTrace + MDout)
        MDoutTrace = self.winner_take_all(self.MDpostTrace)

        return MDoutTrace

    def update_weights(self, rout, MDout):
        """Update weights with plasticity rules.

        Args:
            rout: input to MD
            MDout: activity of MD
        """

        # compute MD outputs
        # 1. non-filtered rout
        MDoutTrace = self.update_trace(rout, MDout)
        # 2. filtered rout
        # kernel = np.ones(shape=(1,))/1
        # rout_filtered = np.convolve(rout, kernel, mode='same')
        # MDoutTrace = self.update_trace(rout_filtered, MDout)

        # compute binary pretraces
        # mean
        # self.MDpreTrace_threshold = np.mean(self.MDpreTrace)
        # self.MDpreTrace_binary = (self.MDpreTrace>self.MDpreTrace_threshold).astype(float)
        # 1. mean of small part
        pretrace_part = int(0.8*len(self.MDpreTrace))
        self.MDpreTrace_threshold = np.mean(np.sort(self.MDpreTrace)[0:pretrace_part])
        self.MDpreTrace_binary = (self.MDpreTrace>self.MDpreTrace_threshold).astype(float)
        # 2. mean of big part
        # pretrace_part = int(0.8*len(self.MDpreTrace))
        # self.MDpreTrace_threshold = np.mean(np.sort(self.MDpreTrace)[-pretrace_part:])
        # self.MDpreTrace_binary = (self.MDpreTrace>self.MDpreTrace_threshold).astype(float)
        # 3. median
        # pretrace_part = int(0.8*len(self.MDpreTrace))
        # self.MDpreTrace_threshold = np.median(np.sort(self.MDpreTrace)[-pretrace_part:])
        # self.MDpreTrace_binary = (self.MDpreTrace>self.MDpreTrace_threshold).astype(float)
        # 4. filtered pretraces
        # kernel = np.ones(shape=(5,))/5
        # self.MDpreTrace_filtered = np.convolve(self.MDpreTrace, kernel, mode='same')
        # self.MDpreTrace_threshold = np.mean(self.MDpreTrace_filtered)
        # self.MDpreTrace_binary = (self.MDpreTrace_filtered>self.MDpreTrace_threshold).astype(float)
        
        # compute thresholds
        # self.MDpreTrace_binary_threshold = np.mean(self.MDpreTrace_binary)
        self.MDpreTrace_binary_threshold = 0.5
        MDoutTrace_threshold = 0.5
        
        # update and clip the PFC context -> MD weights
        wPFC2MDdelta = 0.5 * self.Hebb_learning_rate * np.outer(MDoutTrace - MDoutTrace_threshold, self.MDpreTrace_binary - self.MDpreTrace_binary_threshold)
        self.wPFC2MDdelta_mask = 5e-2 * np.exp(np.log(1/5e-2) * np.outer(MDoutTrace, self.MDpreTrace_binary))
        wPFC2MDdelta = wPFC2MDdelta * self.wPFC2MDdelta_mask
        self.wPFC2MD = np.clip(self.wPFC2MD + wPFC2MDdelta, 0., 1.)

    def winner_take_all(self, MDinp):
        '''Winner take all on the MD
        '''

        # Thresholding
        # MDout = np.zeros(self.md_size)
        # MDinp_sorted = np.sort(MDinp)

        # MDthreshold = np.median(MDinp_sorted[-int(self.num_active) * 2:])
        # # MDthreshold = np.mean(MDinp_sorted[-int(self.num_active) * 2:])
        # # MDthreshold  = np.mean(MDinp)
        # index_pos = np.where(MDinp >= MDthreshold)
        # index_neg = np.where(MDinp < MDthreshold)
        # MDout[index_pos] = 1
        # MDout[index_neg] = 0

        # An equivalent thresholding
        MDout = np.zeros(self.md_size)
        MDout[np.argsort(MDinp)[-int(self.num_active):]] = 1

        return MDout

    def update_mask(self, prev):
        MD_mask_new = np.where(prev == 1.0)[0]
        PFC_mask_new = np.where(np.mean(self.wPFC2MD[MD_mask_new, :], axis=0) > 0.5)[0] # the threshold here is dependent on <self.wMD2PFC = np.clip(self.wMD2PFC + wPFC2MDdelta.T, -1, 0.)>
        self.MD_mask = np.concatenate((self.MD_mask, MD_mask_new)).astype(int)
        self.PFC_mask = np.concatenate((self.PFC_mask, PFC_mask_new)).astype(int)
        if len(self.MD_mask) > 0:
            self.wPFC2MDdelta_mask[self.MD_mask, :] = 0
        if len(self.PFC_mask) > 0:
            self.wPFC2MDdelta_mask[:, self.PFC_mask] = 0


# CTRNN with MD layer
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

    def __init__(self, config, dt=None, **kwargs):
        super().__init__()

        self.input_size = config.input_size
        self.hidden_size = config.hidden_size
        self.hidden_ctx_size = config.hidden_ctx_size
        self.sub_size = config.sub_size
        self.sub_active_size = config.sub_active_size
        self.output_size = config.output_size
        self.MDeffect = config.MDeffect
        self.md_size = config.md_size
        self.md_active_size = config.md_active_size
        self.md_dt = config.md_dt
        self.config = config

        self.tau = 100
        if dt is None:
            alpha = 1
        else:
            alpha = dt / self.tau
        self.alpha = alpha
        self.oneminusalpha = 1 - alpha

        # sensory input -> PFC layer
        self.input2h = nn.Linear(self.input_size, self.hidden_size)

        # PFC layer
        self.h2h = nn.Linear(self.hidden_size, self.hidden_size)
        
        # MD related layers
        if self.MDeffect:
            # sensory input -> PFC context layer
            self.input2PFCctx = nn.Linear(self.input_size, self.hidden_ctx_size, bias=False)
            # MD layer
            self.md = MD_GYM(config, positiveRates=True)
            self.md.md_output = np.zeros(self.md_size)
            index = np.random.permutation(self.md_size)
            self.md.md_output[index[:self.md_active_size]] = 1 # randomly set part of md_output to 1
            self.md.md_output_t = np.array([])
        
        self.reset_parameters()
        
        # report block switching
        if self.MDeffect:
            self.prev_actMD = np.zeros(shape=(self.md_size)) # actiavted MD neurons in the previous <odd number> trials

    def reset_parameters(self):
        ### input weights initialization
        if self.MDeffect:
            # all uniform noise
            k = (1./self.hidden_size)**0.5
            self.input2PFCctx.weight.data = k*torch.rand(self.input2PFCctx.weight.data.size()) # noise ~ U(leftlim=0, rightlim=k)

        ### recurrent weights initialization
        # identity*0.5
        nn.init.eye_(self.h2h.weight)
        self.h2h.weight.data *= 0.5

        # identity*other value
        # nn.init.eye_(self.h2h.weight)
        # self.h2h.weight.data *= 0.2

        # block identity + positive uniform noise
        # weights = []
        # for i in range(self.num_task):
        #     k = 1e-1*(1./self.hidden_size)**0.5
        #     weights.append(torch.eye(self.sub_size)*0.5 + k*torch.rand(self.sub_size, self.sub_size)) # noise ~ U(leftlim=0, rightlim=k)
        # self.h2h.weight.data = torch.block_diag(*weights)

        # block identity + uniform noise
        # weights = []
        # for i in range(self.num_task):
        #     k = (1./self.hidden_size)**0.5
        #     weights.append(torch.eye(self.sub_size) + 2*k*torch.rand(self.sub_size, self.sub_size) - k) # noise ~ U(leftlim=-k, rightlim=k)
        # self.h2h.weight.data = torch.block_diag(*weights)

        # block identity + normal noise
        # weights = []
        # for i in range(self.num_task):
        #     k = (1./self.hidden_size)**0.5
        #     weights.append(torch.eye(self.sub_size) + k*torch.randn(self.sub_size, self.sub_size)) # noise ~ N(mean=0, std=1/hidden_size)
        # self.h2h.weight.data = torch.block_diag(*weights)

        # block positive uniform noise
        # weights = []
        # for i in range(self.num_task):
        #     k = 1e-1*(1./self.hidden_size)**0.5
        #     weights.append(k*torch.rand(self.sub_size, self.sub_size)) # noise ~ U(leftlim=0, rightlim=k)
        # self.h2h.weight.data = torch.block_diag(*weights)

        # random orthogonal noise
        # nn.init.orthogonal_(self.h2h.weight, gain=0.5)

        # all uniform noise
        # k = (1./self.hidden_size)**0.5
        # self.h2h.weight.data += 2*k*torch.rand(self.h2h.weight.data.size()) - k # noise ~ U(leftlim=-k, rightlim=k)

        # all normal noise
        # k = (1./self.hidden_size)**0.5
        # self.h2h.weight.data += k*torch.randn(self.h2h.weight.data.size()) # noise ~ N(mean=0, std=1/hidden_size)

        # the same as pytorch built-in RNN module, used in reservoir
        # k = (1./self.hidden_size)**0.5
        # nn.init.uniform_(self.h2h.weight, a=-k, b=k)
        # nn.init.uniform_(self.h2h.bias, a=-k, b=k)

        # default initialization
        # pass

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

        

        # external inputs & activities of PFC neurons containing context info
        if self.MDeffect:
            ext_input_ctx = self.input2PFCctx(input)
            # PFC-context neurons get disjoint inputs
            # 1. The context information is not deterministic
            # 2. The PFC-context layer is noisy
            ext_input_mask = torch.zeros_like(ext_input_ctx)
            mask_idx = torch.where(torch.rand(self.sub_size) < self.config.sub_active_prob)[0].tolist()
            for batch_idx in range(ext_input_mask.shape[0]):
                ext_input_mask[batch_idx, sub_id*self.sub_size:(sub_id+1)*self.sub_size][mask_idx] = 1
            PFC_ctx_input = torch.relu(ext_input_ctx.mul(ext_input_mask) + (self.config.hidden_ctx_noise)*torch.randn(ext_input_ctx.size()))

        # md inputs
        if self.MDeffect:
            assert hidden.shape[0] == 1, 'batch size should be 1'
            assert rec_input.shape[0] == 1, 'batch size should be 1'

            # original MD inputs
            # self.md.md_output = self.md(hidden.cpu().detach().numpy()[0, :])
            # self.md.MD2PFCMult = np.dot(self.md.wMD2PFCMult, self.md.md_output)
            # rec_inp = rec_input.cpu().detach().numpy()[0, :]
            # md2pfc_weights = (self.md.MD2PFCMult/self.md.md_size)
            # md2pfc = md2pfc_weights * rec_inp
            # md2pfc += np.dot((self.md.wMD2PFC/self.md.md_size), self.md.md_output)
            # md2pfc = torch.from_numpy(md2pfc).view_as(hidden).to(input.device)

            # only MD additive inputs
            # self.md.md_output = self.md(hidden.cpu().detach().numpy()[0, :])
            self.md.md_output = self.md(PFC_ctx_input.cpu().detach().numpy()[0, :])
            # md2pfc = np.dot((self.md.wMD2PFC/self.md.md_size), np.logical_not(self.md.md_output).astype(float))
            md2pfc = np.dot((self.md.wMD2PFC/self.md.md_size), (self.md.md_output).astype(float))
            md2pfc_mul = np.dot((self.md.wMD2PFCMult), (self.md.md_output).astype(float))
            md2pfc = torch.from_numpy(md2pfc).view_as(hidden).to(input.device)
            md2pfc_mul = torch.from_numpy(md2pfc_mul).view_as(hidden).to(input.device)

            # ideal MD inputs analysis
            # self.md.md_output = self.md(hidden.cpu().detach().numpy()[0, :])
            # rec_inp = rec_input.cpu().detach().numpy()[0, :]
            # #  ideal multiplicative inputs
            # md2pfc_weights = np.zeros(shape=(self.hidden_size))
            # md2pfc_weights[sub_id*self.sub_size:(sub_id+1)*self.sub_size] = 0.5
            # md2pfcMult = md2pfc_weights * rec_inp
            # #  ideal additive inputs
            # md2pfcAdd = np.ones(shape=(self.hidden_size))*(-0.5)
            # md2pfcAdd[sub_id*self.sub_size:(sub_id+1)*self.sub_size] = 0
            # #  ideal inputs
            # md2pfc = md2pfcAdd
            # md2pfc = torch.from_numpy(md2pfc).view_as(hidden).to(input.device)
            # stats(pre_activation)
            if self.md.sendinputs:
                if self.config.MDeffect_mul:
                    rec_input *= md2pfc_mul
                if self.config.MDeffect_add:
                    rec_input += md2pfc
            pre_activation = ext_input + rec_input

        h_new = torch.relu(hidden * self.oneminusalpha + pre_activation * self.alpha)
        
        # shutdown analysis
        # shutdown_mask = torch.zeros_like(h_new)
        # shutdown_mask[:, sub_id*self.sub_size:(sub_id+1)*self.sub_size] = 1
        # h_new = h_new.mul(shutdown_mask)

        return h_new

    def forward(self, input, sub_id, hidden=None):
        """Propogate input through the network."""
        
        num_tsteps = input.size(0)

        # init network activities
        if hidden is None:
            hidden = self.init_hidden(input)
        if self.MDeffect:
            self.md.init_activity()

        # initialize variables for saving network activities
        output = []
        if self.MDeffect:
            self.md.md_preTraces = np.zeros(shape=(num_tsteps, self.hidden_ctx_size))
            self.md.md_preTraces_binary = np.zeros(shape=(num_tsteps, self.hidden_ctx_size))
            self.md.md_preTrace_thresholds = np.zeros(shape=(num_tsteps, 1))
            self.md.md_preTrace_binary_thresholds = np.zeros(shape=(num_tsteps, 1))
            self.md.md_output_t *= 0

        for i in range(num_tsteps):
            hidden = self.recurrence(input[i], sub_id, hidden)
            
            # save PFC activities
            output.append(hidden)
            # save MD activities
            if self.MDeffect:
                self.md.md_preTraces[i, :] = self.md.MDpreTrace
                self.md.md_preTraces_binary[i, :] = self.md.MDpreTrace_binary
                self.md.md_preTrace_thresholds[i, :] = self.md.MDpreTrace_threshold
                self.md.md_preTrace_binary_thresholds[i, :] = self.md.MDpreTrace_binary_threshold
                if i==0:
                    self.md.md_output_t = self.md.md_output.reshape((1, self.md.md_output.shape[0]))
                else:
                    self.md.md_output_t = np.concatenate((self.md.md_output_t, self.md.md_output.reshape((1, self.md.md_output.shape[0]))),axis=0)

        # report block switching during training
        if self.MDeffect:
            if self.md.learn:
                # time constant
                alpha = 0.01
                # previous activated MD neurons
                prev_actMD_sorted = np.sort(self.prev_actMD)
                prev = (self.prev_actMD > np.median(prev_actMD_sorted[-int(self.md.num_active)*2:])).astype(float)
                # update self.prev_actMD
                new_actMD = np.mean(self.md.md_output_t, axis=0)
                self.prev_actMD[:] = (1-alpha)*self.prev_actMD + alpha*new_actMD
                # current activated MD neurons
                curr_actMD_sorted = np.sort(self.prev_actMD)
                curr = (self.prev_actMD > np.median(curr_actMD_sorted[-int(self.md.num_active)*2:])).astype(float)
                # compare prev and curr
                flag = sum(np.logical_xor(prev, curr).astype(float))
                if (flag >= 2*self.md.num_active) and (sum(self.prev_actMD) > 0.4*self.md.num_active):
                    # when task switching correctly, flag = 2*num_active
                    # At the beginning of training, MD activities are not stable. So we use sum(self.prev_actMD) to determine whether this is the beginning of training.
                    print('Switching!')
                    print(prev, curr, self.prev_actMD, sep='\n')
                    # change self.prev_actMD to penalize many switching
                    self.prev_actMD[:] = curr


        output = torch.stack(output, dim=0)
        return output, hidden, np.mean(self.md.md_output_t, axis=0)

class RNN_MD(nn.Module):
    """Recurrent network model.
    Args:
        input_size: int, input size
        hidden_size: int, hidden size
        sub_size: int, subpopulation size
        output_size: int, output size
        rnn: str, type of RNN, lstm, rnn, ctrnn, or eirnn
    """

    def __init__(self, config, **kwargs):
        super().__init__()

        self.rnn = CTRNN_MD(config, **kwargs)
        self.drop_layer = nn.Dropout(p=0.0)
        self.fc = nn.Linear(config.hidden_size, config.output_size)
        self.md_activities = []

    def forward(self, x, task_id):
        rnn_activity, _, md_mean_activity = self.rnn(x, sub_id=task_id)
        rnn_activity = self.drop_layer(rnn_activity)
        out = self.fc(rnn_activity)
        return out, rnn_activity, md_mean_activity