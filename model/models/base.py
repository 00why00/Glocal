import numpy as np
import torch
import torch.nn as nn
import abc

from model.modules.attention import MultiHeadAttention


class Base(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        # encoder
        if args.backbone_class == 'ConvNet':
            from model.networks.convnet import ConvNet
            self.hidden_dims = [64, 64, 64, 64]
            self.encoder = ConvNet(flag_pool=False)
        elif args.backbone_class == 'Res12':
            self.hidden_dims = [64, 160, 320, 640]
            from model.networks.res12 import ResNet
            self.encoder = ResNet(avg_pool=False, dropblock_size=2 if args.dataset in ['CIFAR-FS', 'FC100'] else 5)
        elif args.backbone_class == 'Res18':
            self.hidden_dims = [64, 128, 256, 512]
            from model.networks.res18 import resnet18
            self.encoder = resnet18(avg_pool=False)
        elif args.backbone_class == 'WRN':
            self.hidden_dims = [16, 160, 320, 640]
            from model.networks.WRN28 import Wide_ResNet
            self.encoder = Wide_ResNet(28, 10, 0.5)
        else:
            raise ValueError('')

        # classifier
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        if args.attention:
            self.slf_attn = MultiHeadAttention(1, self.hidden_dims[-1], self.hidden_dims[-1], self.hidden_dims[-1],
                                               dropout=0.5)

        self.feat_dim = 0
        self.emb_dim = 0
        self.num_batch = 0
        self.num_proto = 0

        self.kquery = None
        self.uquery = None
        self.bproto = None
        self.proto = None

    def split_instances(self):
        args = self.args
        if self.training:
            return (torch.Tensor(np.arange(args.way * args.shot)).long().view(1, args.shot, args.way),
                    torch.Tensor(np.arange(args.way * args.shot,
                                           args.way * (args.shot + args.query))).long().view(1, args.query, args.way))
        else:
            return (
                torch.Tensor(np.arange(args.eval_way * args.eval_shot)).long().view(1, args.eval_shot, args.eval_way),
                torch.Tensor(np.arange(args.eval_way * args.eval_shot,
                                       args.eval_way * (args.eval_shot + args.eval_query))).long().view(1,
                                                                                                        args.eval_query,
                                                                                                        args.eval_way))

    @abc.abstractmethod
    def forward(self, x):
        pass

    @abc.abstractmethod
    def _forward(self, *args, **kwargs):
        pass

    def classify(self, x, is_emb=False):
        """
        classify in mini batch way
        :param x: batch data
        :param is_emb: use fc or not
        :return: logits
        """
        out = self.encoder(x)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        if not is_emb:
            out = self.fc(out)
        return out

    def embedding_process(self, instance_embs, support_idx, query_idx):
        closed_way = self.args.closed_way if self.training else self.args.closed_eval_way

        support = instance_embs[support_idx.contiguous().view(-1)].contiguous().view(*(support_idx.shape + (-1,)))
        support = support[:, :, :closed_way, ...].contiguous()
        query = instance_embs[query_idx.contiguous().view(-1)].contiguous().view(*(query_idx.shape + (-1,)))

        # get mean of the support
        proto = support.mean(dim=1)  # N task x NK x d
        self.bproto = proto

        if self.args.attention:
            proto = self.slf_attn(proto, proto, proto)

        kquery = query[:, :, :closed_way].contiguous()
        uquery = query[:, :, closed_way:].contiguous()

        self.proto = proto
        self.kquery = kquery
        self.uquery = uquery

        return proto, kquery, uquery

    def get_logits(self, proto, kquery, uquery):
        kquery = kquery.view(-1, self.emb_dim).unsqueeze(1)  # (N batch * Nq * Nw, 1, d)
        uquery = uquery.view(-1, self.emb_dim).unsqueeze(1)
        kproto = proto.unsqueeze(1).expand(self.num_batch, kquery.shape[0], self.num_proto,
                                           self.emb_dim).contiguous()
        kproto = kproto.view(self.num_batch * kquery.shape[0], self.num_proto, self.emb_dim)
        uproto = proto.unsqueeze(1).expand(self.num_batch, uquery.shape[0], self.num_proto,
                                           self.emb_dim).contiguous()
        uproto = uproto.view(self.num_batch * uquery.shape[0], self.num_proto, self.emb_dim)

        klogits = - torch.sum((kproto - kquery) ** 2, 2) / self.args.temperature
        ulogits = - torch.sum((uproto - uquery) ** 2, 2) / self.args.temperature

        return klogits, ulogits
