class BaseLogger(object):
    def __init__(self, config):
        self.losses = []
        self.stamps = []
        self.fix_perfs = []
        self.act_perfs = []
        self.md_activity = []
        for _ in range(config.num_task):
            self.fix_perfs.append([])
            self.act_perfs.append([])
            self.md_activity.append([])
        self.PFCouts_all = []

class PFCMDLogger(BaseLogger):
    def __init__(self, config):
        super(PFCMDLogger, self).__init__(config)
        self.MDouts_all = []
        self.MDpreTraces_all = []
        self.MDpreTraces_binary_all = []
        self.MDpreTrace_threshold_all = []
        self.MDpreTrace_binary_threshold_all = []
        self.wPFC2MD_list = []
        self.wMD2PFC_list = []


class SerialLogger(object):
    def __init__(self, config):
        self.losses = []
        self.frustrations = []
        self.stamps = []
        self.accuracies = []
        self.fix_perfs = []
        self.act_perfs = []
        self.switch_trialxxbatch = []
        self.switch_task_id = []
        self.trials_to_crit =[]
        self.latents_to_crit = []
        self.lu_stamps_acc_improve = []
        self.lrs = []
        self.gradients = []
        self.cog_obs_preds = []
        self.bu_context_ids = []
        self.td_context_ids = []
        self.md_context_ids = []
        self.md_grads = []
        self.task_ids = []

        for _ in range(len(config.tasks)):
            self.fix_perfs.append([])
            self.act_perfs.append([])
        self.PFCouts_all = []
        if config.save_detailed:
            self.rnn_activity = []
            self.inputs = []
            self.outputs = []
            self.labels = []
            self.sampled_act = []

    def write_basic(self, stamp, loss, acc, task_id):
        self.stamps.append(stamp)
        self.losses.append(loss)
        self.accuracies.append(acc)
        self.task_ids.append(task_id)
    def write_detailed(self, rnn_activity, inputs, outputs, labels, sampled_act, task_id):#  gradients,):
        self.rnn_activity.append(rnn_activity)
        self.inputs.append(inputs)
        self.outputs.append(outputs)
        self.labels.append(labels)
        self.sampled_act.append(sampled_act)
        # self.gradients.append(gradients)