import torch
import torch.nn as nn

from model.loss.energy_loss import EnergyLoss
from model.models.base import Base


class GEL(Base):
    def __init__(self, args):
        super(GEL, self).__init__(args)
        self.args = args

        if args.backbone_class == "Res18" and args.dataset not in ['CIFAR-FS', 'FC100']:
            self.downsample = nn.Sequential(
                nn.Conv2d(self.hidden_dims[-1], self.hidden_dims[-1], kernel_size=1, stride=2, bias=False),
                nn.BatchNorm2d(self.hidden_dims[-1]),
            )

        if args.pixel_conv:
            self.conv = nn.Sequential(nn.Conv2d(self.hidden_dims[-1],
                                                self.hidden_dims[-1] // 2, kernel_size=1, stride=1, padding=0),
                                      nn.BatchNorm2d(self.hidden_dims[-1] // 2),
                                      nn.PReLU())

            for m in self.conv.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

        self.EnergyLoss = EnergyLoss(args)

    def forward(self, x):
        # feature extraction
        instance_feats = self.encoder(x)
        if self.args.backbone_class == "Res18" and self.args.dataset not in ['CIFAR-FS', 'FC100']:
            instance_feats = self.downsample(instance_feats)
        instance_embs = self.classify(x, is_emb=True)
        if self.args.pixel_conv:
            instance_feats = self.conv(instance_feats)
        # split support query set for few-shot data
        support_idx, query_idx = self.split_instances()
        emb, feat = self._forward(instance_feats, instance_embs, support_idx, query_idx)
        return emb, feat

    def _forward(self, instance_feats, instance_embs, support_idx, query_idx):
        self.emb_dim = instance_embs.size(-1)
        self.feat_dim = instance_feats.shape[1:]

        proto, kquery, uquery = self.embedding_process(instance_embs, support_idx, query_idx)
        f_proto, f_kquery, f_uquery = self.feature_process(instance_feats, support_idx, query_idx)

        self.num_batch = proto.shape[0]
        self.num_proto = proto.shape[1]

        return (self.get_logits(proto, kquery, uquery),
                self.get_feat_logits(f_proto, f_kquery, f_uquery, distance=self.args.distance))

    def feature_process(self, instance_feats, support_idx, query_idx):
        closed_way = self.args.closed_way if self.training else self.args.closed_eval_way

        support = instance_feats[support_idx.contiguous().view(-1)].contiguous().view(
            *(support_idx.shape + self.feat_dim))
        support = support[:, :, :closed_way, ...].contiguous()
        query = instance_feats[query_idx.contiguous().view(-1)].contiguous().view(*(query_idx.shape + self.feat_dim))

        # get mean of the support
        proto = support.mean(dim=1)  # N task x NK x d

        kquery = query[:, :, :closed_way].contiguous()
        uquery = query[:, :, closed_way:].contiguous()

        return proto, kquery, uquery

    def get_feat_logits(self, proto, kquery, uquery, distance="euclidean"):
        if distance == "euclidean":
            kquery = kquery.view(-1, *self.feat_dim).unsqueeze(1)  # (N batch * Nq * Nw, 1, d)
            uquery = uquery.view(-1, *self.feat_dim).unsqueeze(1)
            kproto = proto.unsqueeze(1).expand(self.num_batch, kquery.shape[0], self.num_proto,
                                               *self.feat_dim).contiguous()
            kproto = kproto.view(self.num_batch * kquery.shape[0], self.num_proto, *self.feat_dim)
            uproto = proto.unsqueeze(1).expand(self.num_batch, uquery.shape[0], self.num_proto,
                                               *self.feat_dim).contiguous()
            uproto = uproto.view(self.num_batch * uquery.shape[0], self.num_proto, *self.feat_dim)
            klogits = - torch.sum((kproto - kquery) ** 2, dim=[2, 3, 4]) / self.args.temperature / 10
            ulogits = - torch.sum((uproto - uquery) ** 2, dim=[2, 3, 4]) / self.args.temperature / 10
        elif distance == "pixel_sim":
            kquery = kquery.view(-1, self.feat_dim[0], self.feat_dim[1] * self.feat_dim[2]
                                 ).unsqueeze(1).permute(0, 1, 3, 2).unsqueeze(-1)
            uquery = uquery.view(-1, self.feat_dim[0], self.feat_dim[1] * self.feat_dim[2]
                                 ).unsqueeze(1).permute(0, 1, 3, 2).unsqueeze(-1)
            proto = proto.squeeze().view(-1, self.feat_dim[0], self.feat_dim[1] * self.feat_dim[2]
                                         ).unsqueeze(0).unsqueeze(2)

            klogits = torch.nn.CosineSimilarity(dim=3)(kquery, proto)
            ulogits = torch.nn.CosineSimilarity(dim=3)(uquery, proto)
            if self.args.top_method == 'query':
                klogits = klogits.topk(self.args.top_k, dim=3).values.sum(dim=[2, 3]) / self.args.top_k
                ulogits = ulogits.topk(self.args.top_k, dim=3).values.sum(dim=[2, 3]) / self.args.top_k
            elif self.args.top_method == 'proto':
                klogits = klogits.topk(self.args.top_k, dim=2).values.sum(dim=[2, 3]) / self.args.top_k
                ulogits = ulogits.topk(self.args.top_k, dim=2).values.sum(dim=[2, 3]) / self.args.top_k
            else:
                klogits = klogits.topk(self.args.top_k, dim=2).values / self.args.top_k
                ulogits = ulogits.topk(self.args.top_k, dim=2).values / self.args.top_k
                klogits = klogits.topk(self.args.top_k, dim=3).values.sum(dim=[2, 3]) / self.args.top_k
                ulogits = ulogits.topk(self.args.top_k, dim=3).values.sum(dim=[2, 3]) / self.args.top_k
        else:
            raise NotImplementedError

        return klogits, ulogits
