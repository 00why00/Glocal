import torch
import torch.nn as nn


class AutomaticWeightedLoss(nn.Module):
    def __init__(self, args, num=2):
        super(AutomaticWeightedLoss, self).__init__()
        self.args = args
        params = torch.ones(num, requires_grad=True)
        self.params = torch.nn.Parameter(params)

    def forward(self, *x):
        loss_sum = 0
        # if self.args.weighted_combine or self.args.task_specific_param:
        #     for i, loss in enumerate(x):
        #         loss_sum += self.params[i] * loss
        # else:
        for i, loss in enumerate(x):
            loss_sum += loss
        return loss_sum
