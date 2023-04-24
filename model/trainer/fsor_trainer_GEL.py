import os.path as osp
import time

import numpy as np
import torch
import torch.nn.functional as f
import wandb
from tqdm import tqdm

from model.trainer.base import Trainer
from model.models.glocal_energy import GEL
from model.trainer.helpers import get_dataloader, prepare_model, prepare_optimizer
from model.utils.train_utils import Averager, ListAverager, count_acc, compute_confidence_interval, calc_auroc


class FSORTrainerGEL(Trainer):
    def __init__(self, args):
        self.grab_gpu = torch.zeros([1024, 1024, 1024], dtype=torch.double).cuda()
        super().__init__(args)
        self.train_loader, self.val_loader, self.test_loader = get_dataloader(args)
        del self.grab_gpu
        self.model = GEL(args)
        wandb.watch(self.model)
        self.model = prepare_model(args, self.model)
        self.optimizer, self.lr_scheduler = prepare_optimizer(self.model, args)
        self.EnergyLoss = self.model.EnergyLoss
        self.training = True

    def prepare_label(self):
        args = self.args
        closed_way = args.closed_way if self.training else args.closed_eval_way

        # prepare one-hot label
        label = torch.arange(closed_way, dtype=torch.int16).repeat(args.query)
        label = label.type(torch.LongTensor)

        if torch.cuda.is_available():
            label = label.cuda()

        return label

    def train(self):
        args = self.args
        self.model.train()
        if self.args.fix_BN:
            self.model.encoder.eval()

        # start FSL training
        for epoch in range(1, args.max_epoch + 1):
            self.training = True
            print("epoch: {}".format(epoch))
            self.train_epoch += 1
            self.model.train()

            acc_ave, energy_auroc_ave = Averager(), ListAverager(num=4)
            logits_auroc_ave, combine_auroc_ave = ListAverager(num=4), ListAverager(num=4)
            l_few_ave, l_open_ave = Averager(), Averager()
            l_energy_ave, l_all_ave = Averager(), Averager()

            start_tm = time.time()
            for i, batch in enumerate(self.train_loader):
                data, gt_label = self.loader_process(batch)
                label = self.prepare_label()

                self.train_step += 1
                data_tm = time.time()
                self.data_time.add(data_tm - start_tm)

                result = self.post_process(data, label)
                l_few, l_open, l_energy, total_loss, acc, energy_auroc, logits_auroc, combine_auroc = result

                forward_tm = time.time()
                self.forward_time.add(forward_tm - data_tm)

                self.optimizer.zero_grad()
                total_loss.backward()
                backward_tm = time.time()
                self.backward_time.add(backward_tm - forward_tm)

                self.optimizer.step()

                optimizer_tm = time.time()
                self.optimizer_time.add(optimizer_tm - backward_tm)

                # refresh start_tm
                start_tm = time.time()
                self.try_logging(l_few_ave.add(l_few.item()),
                                 l_open_ave.add(l_open.item()),
                                 l_energy_ave.add(l_energy.item()),
                                 l_all_ave.add(total_loss.item()),
                                 acc_ave.add(acc),
                                 energy_auroc_ave.add(energy_auroc),
                                 logits_auroc_ave.add(logits_auroc),
                                 combine_auroc_ave.add(combine_auroc))

            self.lr_scheduler.step()
            self.try_evaluate(epoch)

            print('ETA:{}/{}'.format(
                self.timer.measure(),
                self.timer.measure(self.train_epoch / args.max_epoch))
            )

        torch.save(self.train_log, osp.join(args.save_path, 'train_log'))
        self.save_model('epoch-last')

    def evaluate_base(self, data_loader):
        self.training = False
        # restore model args
        args = self.args
        # evaluation mode
        self.model.eval()

        record = np.zeros((args.num_eval_episodes, 2))  # loss and acc
        auroc_record = np.zeros((args.num_eval_episodes, 3))
        fpr95_record = np.zeros((args.num_eval_episodes, 3))
        auc_pr_record = np.zeros((args.num_eval_episodes, 3))
        f1_score_record = np.zeros((args.num_eval_episodes, 3))

        label = self.prepare_label()
        with torch.no_grad():
            for i, batch in enumerate(tqdm(data_loader), 1):
                data, _ = self.loader_process(batch)

                _, _, _, loss, acc, energy_auroc, logits_auroc, combine_auroc = self.post_process(data, label)

                record[i - 1, 0] = loss.item()
                record[i - 1, 1] = acc

                auroc_record[i - 1, 0] = energy_auroc[0]
                auroc_record[i - 1, 1] = logits_auroc[0]
                auroc_record[i - 1, 2] = combine_auroc[0]

                fpr95_record[i - 1, 0] = energy_auroc[1]
                fpr95_record[i - 1, 1] = logits_auroc[1]
                fpr95_record[i - 1, 2] = combine_auroc[1]

                auc_pr_record[i - 1, 0] = energy_auroc[2]
                auc_pr_record[i - 1, 1] = logits_auroc[2]
                auc_pr_record[i - 1, 2] = combine_auroc[2]

                f1_score_record[i - 1, 0] = energy_auroc[3]
                f1_score_record[i - 1, 1] = logits_auroc[3]
                f1_score_record[i - 1, 2] = combine_auroc[3]

        result_list = [compute_confidence_interval(record[:, 0])[0]]
        result_list.extend(compute_confidence_interval(record[:, 1]))
        result_list.extend(compute_confidence_interval(auroc_record[:, 0]))
        result_list.extend(compute_confidence_interval(fpr95_record[:, 0]))
        result_list.extend(compute_confidence_interval(auc_pr_record[:, 0]))
        result_list.extend(compute_confidence_interval(f1_score_record[:, 0]))
        result_list.extend(compute_confidence_interval(auroc_record[:, 1]))
        result_list.extend(compute_confidence_interval(fpr95_record[:, 1]))
        result_list.extend(compute_confidence_interval(auc_pr_record[:, 1]))
        result_list.extend(compute_confidence_interval(f1_score_record[:, 1]))
        result_list.extend(compute_confidence_interval(auroc_record[:, 2]))
        result_list.extend(compute_confidence_interval(fpr95_record[:, 2]))
        result_list.extend(compute_confidence_interval(auc_pr_record[:, 2]))
        result_list.extend(compute_confidence_interval(f1_score_record[:, 2]))

        return result_list

    def val_log_string(self):
        return ('best acc epoch {}, best auroc epoch {}, best val acc={:.4f} + {:.4f}, best val auroc={:.4f} + {:.4f}, '
                'total loss={:.4f}, acc={:.4f} + {:.4f}, '
                'energy auroc={:.4f}+{:.4f}, energy fpr95={:.4f} + {:.4f}, '
                'energy auc-pr={:.4f} + {:.4f}, energy f1 score={:.4f} + {:.4f}, '
                'logits auroc={:.4f}+{:.4f}, logits fpr95={:.4f} + {:.4f}, '
                'logits auc-pr={:.4f} + {:.4f}, logits f1 score={:.4f} + {:.4f}, '
                'combine auroc={:.4f}+{:.4f}, combine fpr95={:.4f} + {:.4f}, '
                'combine auc-pr={:.4f} + {:.4f}, combine f1 score={:.4f} + {:.4f}'.format(*self.val_log.values()))

    @staticmethod
    def test_log_string(result_list):
        return ("acc: {:.4f} + {:.4f} "
                "Energy: auroc={:.4f}+{:.4f}, fpr95={:.4f}+{:.4f}, auc-pr={:.4f}+{:.4f}, f1 score={:.4f}+{:.4f} "
                "Logits: auroc={:.4f}+{:.4f}, fpr95={:.4f}+{:.4f}, auc-pr={:.4f}+{:.4f}, f1 score={:.4f}+{:.4f} "
                "Combine: auroc={:.4f}+{:.4f}, fpr95={:.4f}+{:.4f}, auc-pr={:.4f}+{:.4f}, f1 score={:.4f}+{:.4f}"
                " ".format(*result_list[1:]))

    def evaluate(self, data_loader):
        print(self.val_log_string())
        result_list = self.evaluate_base(data_loader)
        return result_list

    def evaluate_test(self, path):
        if self.args.test:
            model_path = self.args.test_model_path
            save_path = self.args.cross_save_path
        else:
            model_path = save_path = self.args.save_path
        weights = torch.load(osp.join(model_path, path))
        model_weights = weights['params']
        self.model.load_state_dict(model_weights, strict=False)

        result_list = self.evaluate_base(self.test_loader)

        for idx, key in enumerate(self.test_log.keys()):
            self.test_log[key] = result_list[idx]

        print(self.test_log_string(result_list))
        print(self.val_log_string())

        # save the best performance in a txt file
        with open(osp.join(save_path, '{}+{}.txt'.format(self.test_log['test_acc'],
                                                         self.test_log['test_energy_auroc'])), 'w') as file:
            file.write(self.val_log_string())
            file.write(self.test_log_string(result_list))

        # wandb log
        wandb.log(self.test_log)

    def auroc_process(self, bce_score, klogits, ulogits):
        args = self.args
        closed_way = args.closed_way if self.training else args.closed_eval_way

        bce_known_prob = bce_score[:closed_way * args.query]
        bce_unknown_prob = bce_score[closed_way * args.query:]

        """ Energy """
        known_scores = bce_known_prob.cpu().detach().numpy()
        unknown_scores = bce_unknown_prob.cpu().detach().numpy()
        energy_auroc = calc_auroc(known_scores, unknown_scores)

        if not args.SnaTCHer:
            """ logits """
            known_dist = -(klogits.max(1)[0])
            unknown_dist = -(ulogits.max(1)[0])
            known_dist_score = known_dist.cpu().detach().numpy()
            unknown_dist_score = unknown_dist.cpu().detach().numpy()
            logits_auroc = calc_auroc(known_dist_score, unknown_dist_score)
        else:
            """ SnaTCHer """
            with torch.no_grad():
                emb_dim = 640 if args.backbone_class == 'Res12' else 512
                kquery = self.model.kquery
                uquery = self.model.uquery
                bproto = self.model.bproto
                proto = self.model.proto

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

                logits_auroc = calc_auroc(pkdiff, pudiff)

        """ combine """
        if not args.SnaTCHer:
            known_dist = -(klogits.max(1)[0])
            unknown_dist = -(ulogits.max(1)[0])
            known_dist_score = known_dist.cpu().detach().numpy() + known_scores
            unknown_dist_score = unknown_dist.cpu().detach().numpy() + unknown_scores
            combine_auroc = calc_auroc(known_dist_score, unknown_dist_score)
        else:
            combine_known = pkdiff + known_scores
            combine_unknown = pudiff + unknown_scores
            combine_auroc = calc_auroc(combine_known, combine_unknown)

        return energy_auroc, logits_auroc, combine_auroc

    def post_process(self, data, label):
        args = self.args

        emb, feat = self.model(data)
        e_klogits, e_ulogits = emb
        f_klogits, f_ulogits = feat

        # energy loss
        l_energy, energy_score = self.EnergyLoss(f_klogits, f_ulogits, e_klogits, e_ulogits)

        # few shot loss
        l_few = f.cross_entropy(e_klogits, label)

        # open few loss
        if args.open_loss:
            l_open = f.softmax(e_ulogits, dim=1) * f.log_softmax(e_ulogits, dim=1)
            l_open = l_open.sum(dim=1).mean() * args.open_loss_scale
        else:
            l_open = torch.Tensor([0]).cuda()

        acc = count_acc(e_klogits, label)
        energy_auroc, logits_auroc, combine_auroc = self.auroc_process(energy_score, e_klogits, e_ulogits)
        total_loss = l_few + l_open + l_energy

        return l_few, l_open, l_energy, total_loss, acc, energy_auroc, logits_auroc, combine_auroc

    @staticmethod
    def loader_process(batch):
        if torch.cuda.is_available():
            data = batch[0].cuda()
            gt_label = batch[1]
        else:
            data, gt_label = batch[0], batch[1]
        #
        # for index, path in enumerate(gt_label):
        #     import cv2
        #     img = cv2.imread(path)
        #     cv2.imwrite("./vis/{}.jpg".format(index), img)

        return data, gt_label
