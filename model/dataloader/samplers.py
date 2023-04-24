import numpy as np
import torch


class FewShotSampler:
    def __init__(self, label, n_batch, n_cls, n_per):
        self.n_batch = n_batch
        self.n_cls = n_cls
        self.n_per = n_per

        label = np.array(label)
        self.m_ind = []
        for i in range(max(label) + 1):
            ind = np.argwhere(label == i).reshape(-1)
            ind = torch.from_numpy(ind)
            self.m_ind.append(ind)

    def __len__(self):
        return self.n_batch

    def __iter__(self):
        for i_batch in range(self.n_batch):
            batch = []
            classes = torch.randperm(len(self.m_ind))[:self.n_cls]
            for c in classes:
                label_id = self.m_ind[c]
                pos = torch.randperm(len(label_id))[:self.n_per]
                batch.append(label_id[pos])
            batch = torch.stack(batch).t().reshape(-1)
            yield batch


class FewShotOpenSetSampler(FewShotSampler):
    def __init__(self, label, n_batch, n_cls, n_per, random_open=False):
        super(FewShotOpenSetSampler, self).__init__(label, n_batch, n_cls, n_per)
        self.random_open = random_open

    def __iter__(self):
        if not self.random_open:
            for i_batch in range(self.n_batch):
                batch = []
                classes = torch.randperm(len(self.m_ind))[:self.n_cls]
                for c in classes:
                    label_id = self.m_ind[c]
                    pos = torch.randperm(len(label_id))[:self.n_per]
                    batch.append(label_id[pos])
                batch = torch.stack(batch).t().reshape(-1)
                yield batch
        else:
            for i_batch in range(self.n_batch):

                # random sample 5 closed classes
                random_classes = torch.randperm(len(self.m_ind))
                closed_classes = random_classes[:self.n_cls // 2]
                open_classes = random_classes[self.n_cls // 2:]
                open_ind = torch.tensor([], dtype=torch.int64)
                for i in open_classes:
                    open_ind = torch.cat([open_ind, self.m_ind[i]], dim=0)

                batch = []
                open_samples = open_ind[torch.randperm(len(open_ind))]

                # closed
                for c in closed_classes:
                    label_id = self.m_ind[c]
                    pos = torch.randperm(len(label_id))[:self.n_per]
                    batch.append(label_id[pos])

                # open
                for i in range(len(closed_classes)):
                    batch.append(open_samples[i * self.n_per: (i + 1) * self.n_per])

                batch = torch.stack(batch).t().reshape(-1)
                yield batch
