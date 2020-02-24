import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from rl_main.utils import AddBiases, util_init
from rl_main.main_constants import device

"""
Modify standard PyTorch distributions so they are compatible with this code.
"""

#
# Standardize distribution interfaces
#

# Categorical
FixedCategorical = torch.distributions.Categorical

old_sample = FixedCategorical.sample
FixedCategorical.sample = lambda self: old_sample(self).unsqueeze(-1)
FixedCategorical.old_sample = old_sample

log_prob_cat = FixedCategorical.log_prob
FixedCategorical.log_probs = lambda self, actions: log_prob_cat(self, actions.squeeze(-1)).view(actions.size(0), -1).sum(-1).unsqueeze(-1)
FixedCategorical.log_prob_cat = log_prob_cat

FixedCategorical.mode = lambda self: self.probs.argmax(dim=-1, keepdim=True)

# Normal
FixedNormal = torch.distributions.Normal

log_prob_normal = FixedNormal.log_prob
FixedNormal.log_probs = lambda self, actions: log_prob_normal(self, actions).sum(-1, keepdim=True)

normal_entropy = FixedNormal.entropy
FixedNormal.entropy = lambda self: normal_entropy(self).sum(-1)

FixedNormal.mode = lambda self: self.mean


class DistCategorical(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(DistCategorical, self).__init__()

        init_ = lambda m: util_init(
            m,
            nn.init.orthogonal_,
            lambda x: nn.init.constant_(x, 0),
            gain=0.01
        )

        self.linear = init_(nn.Linear(num_inputs, num_outputs))

    def forward(self, x):
        x = F.leaky_relu(self.linear(x))
        return FixedCategorical(logits=x)


class DistDiagGaussian(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(DistDiagGaussian, self).__init__()

        init_ = lambda m: util_init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0))

        self.linear = init_(nn.Linear(num_inputs, num_outputs))
        self.logstd = AddBiases(torch.zeros(num_outputs).to(device))

    def forward(self, x):
        action_mean = torch.tanh(self.linear(x))

        #  An ugly hack for my KFAC implementation.
        zeros = torch.zeros(action_mean.size())
        if x.is_cuda:
            zeros = zeros.cuda()

        action_logstd = self.logstd(zeros)
        return FixedNormal(loc=action_mean, scale=action_logstd.exp())


if __name__ == "__main__":
    logits = torch.tensor(data=[0.3, 0.7])

    softmax = logits.softmax(dim=0)
    print("############# FixedCategorical #############")
    print("Softmax: {0} with regard to logits: {1}\n".format(softmax, logits))

    fc = FixedCategorical(logits=logits) # FixedCategorical := torch.distributions.Categorical

    for i in range(10):
        sample = fc.sample()
        log_prob = fc.log_probs(sample)
        mode = fc.mode()
        print("sample: {0} (size: {1}), log_prob: {2} (size: {3}) :: {4:7.4}, {5:7.4}, {6:7.4}, mode: {7}".format(
            sample,
            sample.size(),
            log_prob,
            log_prob.size(),
            math.log(logits[sample.item()], math.e),
            math.log(softmax[sample.item()], math.e), #softmax[1] --> 0.5987 --> log_e 0.5987 = -0.5130
            math.exp(log_prob.item()),
            mode
        ))

    print()

    for i in range(10):
        old_sample = fc.old_sample()
        log_prob_cat = fc.log_prob_cat(old_sample)
        print("old_sample: {0} (size:{1}), log_prob_cat: {2} (size:{3})".format(
            old_sample,
            old_sample.size(),
            log_prob_cat,
            log_prob_cat.size()
        ))

    print()
    print("############# FixedNormal #############")

    fn = FixedNormal(loc=0.0, scale=1.0) # FixedNormal := torch.distributions.Normal (loc --> mean, scale --> std)

    for i in range(10):
        sample = fn.sample()
        log_prob = fn.log_probs(sample)
        mode = fn.mode()
        print("sample: {0}, log_prob: {1} ({2:7.4}), mode: {3}".format(
            sample,
            log_prob,
            math.exp(log_prob.item()),
            mode,
            end=", "
        ))