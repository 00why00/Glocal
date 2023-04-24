import torch.nn as nn
import torch


class OpenLoss(nn.Module):
    def __init__(self, args, lam):
        super(OpenLoss, self).__init__()
        self.ce = nn.CrossEntropyLoss()
        self.args = args
        self.lam = lam

    def forward(self, x, y):
        args = self.args
        known_x = x[:args.closed_way * args.query, :]
        known_y = y[:args.closed_way * args.query]
        unknown_x = x[args.open_way * args.query:, :]

        loss_known1 = self.ce(known_x, known_y)

        one_hot_labels = torch.zeros(args.closed_way * args.query,
                                     args.closed_way).cuda().scatter_(1, known_y.view(-1, 1),1)
        dim1 = torch.nonzero(one_hot_labels)
        known_x_gt = known_x[dim1[:, 0], dim1[:, 1]]
        loss_known2 = 2 - known_x_gt
        loss_known2 = torch.clamp(loss_known2, 0)
        loss_known2 = loss_known2.mean()

        loss_unknown2 = unknown_x + 2
        loss_unknown2 = torch.clamp(loss_unknown2, 0)
        loss_unknown2 = loss_unknown2.mean()

        total_loss = self.lam * (loss_known2 + loss_unknown2) + loss_known1
        return total_loss
