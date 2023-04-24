import numpy as np
import torch
import torch.nn.functional as f
from tqdm import tqdm

from model.models.snatcher_f import SnaTCHerF
from model.trainer.base import Trainer
from model.trainer.helpers import get_dataloader, prepare_model
from model.utils.train_utils import count_acc, calc_auroc


class FSORTrainerSnaTCherF(Trainer):
    def __init__(self, args):
        super(FSORTrainerSnaTCherF, self).__init__(args)
        self.train_loader, self.val_loader, self.test_loader = get_dataloader(args)
        self.model = SnaTCHerF(args)
        self.model = prepare_model(args, self.model)
        self.emb_dim = self.model.hdim

    def prepare_label(self):
        args = self.args

        # prepare one-hot label
        label = torch.arange(args.way, dtype=torch.int16).repeat(args.query)
        label_aux = torch.arange(args.way, dtype=torch.int8).repeat(args.shot + args.query)

        label = label.type(torch.LongTensor)
        label_aux = label_aux.type(torch.LongTensor)

        if torch.cuda.is_available():
            label = label.cuda()
            label_aux = label_aux.cuda()

        return label, label_aux

    def train(self):
        pass

    def evaluate(self, data_loader):
        pass

    def evaluate_test(self):
        # restore model args
        args = self.args
        self.model.eval()
        test_steps = 600

        record = np.zeros((test_steps, 2))  # loss and acc
        auroc_record = np.zeros((test_steps, 10))

        way = args.closed_way
        label = torch.arange(way).repeat(15).cuda()

        for i, batch in tqdm(enumerate(self.test_loader, 1)):
            if i > test_steps:
                break

            if torch.cuda.is_available():
                data = batch[0].cuda()
            else:
                data = batch[0]

            with torch.no_grad():
                _ = self.model(data)
                instance_embs = self.model.probe_instance_embs
                support_idx = self.model.probe_support_idx
                query_idx = self.model.probe_query_idx

                support = instance_embs[support_idx.flatten()].view(*(support_idx.shape + (-1,)))
                query = instance_embs[query_idx.flatten()].view(*(query_idx.shape + (-1,)))
                emb_dim = support.shape[-1]

                support = support[:, :, :way].contiguous()
                # get mean of the support
                bproto = support.mean(dim=1)  # Ntask x NK x d
                proto = bproto

                kquery = query[:, :, :way].contiguous()
                uquery = query[:, :, way:].contiguous()

                # get mean of the support
                proto = self.model.slf_attn(proto, proto, proto)
                proto = proto[0]

            klogits = -(kquery.reshape(-1, 1, emb_dim) - proto).pow(2).sum(2) / 64.0
            ulogits = -(uquery.reshape(-1, 1, emb_dim) - proto).pow(2).sum(2) / 64.0

            loss = f.cross_entropy(klogits, label)
            acc = count_acc(klogits, label)

            """ Probability """
            known_prob = f.softmax(klogits, 1).max(1)[0]
            unknown_prob = f.softmax(ulogits, 1).max(1)[0]

            known_scores = known_prob.cpu().detach().numpy()
            unknown_scores = unknown_prob.cpu().detach().numpy()
            known_scores = 1 - known_scores
            unknown_scores = 1 - unknown_scores

            auroc = calc_auroc(known_scores, unknown_scores)

            """ Distance """
            kdist = -(klogits.max(1)[0])
            udist = -(ulogits.max(1)[0])
            kdist = kdist.cpu().detach().numpy()
            udist = udist.cpu().detach().numpy()
            dist_auroc = calc_auroc(kdist, udist)

            """ Snatcher """
            with torch.no_grad():
                snatch_known = []
                for j in range(75):
                    pproto = bproto.clone().detach()
                    c = klogits.argmax(1)[j]
                    pproto[0][c] = kquery.reshape(-1, emb_dim)[j]
                    pproto = self.model.slf_attn(pproto, pproto, pproto)[0]
                    pdiff = (pproto - proto).pow(2).sum(-1).sum() / 64.0
                    snatch_known.append(pdiff)

                snatch_unknown = []
                for j in range(ulogits.shape[0]):
                    pproto = bproto.clone().detach()
                    c = ulogits.argmax(1)[j]
                    pproto[0][c] = uquery.reshape(-1, emb_dim)[j]
                    pproto = self.model.slf_attn(pproto, pproto, pproto)[0]
                    pdiff = (pproto - proto).pow(2).sum(-1).sum() / 64.0
                    snatch_unknown.append(pdiff)

                pkdiff = torch.stack(snatch_known)
                pudiff = torch.stack(snatch_unknown)
                pkdiff = pkdiff.cpu().detach().numpy()
                pudiff = pudiff.cpu().detach().numpy()

                snatch_auroc = calc_auroc(pkdiff, pudiff)

            record[i - 1, 0] = loss.item()
            record[i - 1, 1] = acc
            auroc_record[i - 1, 0] = auroc[0]
            auroc_record[i - 1, 1] = snatch_auroc[0]
            auroc_record[i - 1, 2] = dist_auroc[0]

            if i % 100 == 0:
                vdata = record[:, 1]
                vdata = 1.0 * np.array(vdata)
                vdata = vdata[:i]
                va = np.mean(vdata)
                std = np.std(vdata)
                vap = 1.96 * (std / np.sqrt(i))

                audata = auroc_record[:, 0]
                audata = np.array(audata, np.float32)
                audata = audata[:i]
                aua = np.mean(audata)
                austd = np.std(audata)
                auap = 1.96 * (austd / np.sqrt(i))

                sdata = auroc_record[:, 1]
                sdata = np.array(sdata, np.float32)
                sdata = sdata[:i]
                sa = np.mean(sdata)
                sstd = np.std(sdata)
                sap = 1.96 * (sstd / np.sqrt(i))

                ddata = auroc_record[:, 2]
                ddata = np.array(ddata, np.float32)[:i]
                da = np.mean(ddata)
                dstd = np.std(ddata)
                dap = 1.96 * (dstd / np.sqrt(i))

                print("acc: {:.4f} + {:.4f} Prob: {:.4f} + {:.4f}"
                      " Dist: {:.4f} + {:.4f} SnaTCHer: {:.4f} + {:.4f}".format(va, vap, aua, auap, da, dap, sa, sap))
