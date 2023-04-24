import torch.nn  as nn
from model.models.base import Base
from model.modules.attention import MultiHeadAttention


class SnaTCHerF(Base):
    def __init__(self, args):
        super(SnaTCHerF, self).__init__(args)
        self.args = args
        self.hdim = self.hidden_dims[-1]
        self.slf_attn = MultiHeadAttention(1, self.hdim, self.hdim, self.hdim, dropout=0.5)

    def forward(self, x):
        # feature extraction
        x = x.squeeze(0)
        instance_embs = self.classify(x, is_emb=True)
        # split support query set for few-shot data
        support_idx, query_idx = self.split_instances()

        output = self._forward(instance_embs, support_idx, query_idx)
        return output

    def _forward(self, instance_embs, support_idx, query_idx):
        self.probe_instance_embs = instance_embs
        self.probe_support_idx = support_idx
        self.probe_query_idx = query_idx

        return instance_embs
