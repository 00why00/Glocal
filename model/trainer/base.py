import abc
import os
import os.path as osp

import torch
import wandb

from model.utils.train_utils import Averager, Timer, ensure_path


class Trainer(object, metaclass=abc.ABCMeta):
    def __init__(self, args):
        self.args = args
        if not args.test:
            ensure_path(
                self.args.save_path,
                scripts_to_save=['model/models', 'model/networks', __file__],
                debug=args.debug
            )

        if args.cross is not None:
            os.makedirs(args.cross_save_path, exist_ok=True)

        self.model = None
        self.optimizer = None
        self.train_loader, self.val_loader, self.test_loader = None, None, None

        self.train_step = 0
        self.train_epoch = 0
        self.max_steps = args.episodes_per_epoch * args.max_epoch
        self.data_time, self.forward_time = Averager(), Averager()
        self.backward_time, self.optimizer_time = Averager(), Averager()
        self.timer = Timer()

        # train statistics
        self.train_log = {
            'few_shot_loss': 0.0,
            'open_set_loss': 0.0,
            'energy_loss': 0.0,
            'total_loss': 0.0,
            'acc': 0.0,
            'energy_auroc': 0.0,
            'energy_auroc_interval': 0.0,
            'energy_fpr95': 0.0,
            'energy_fpr95_interval': 0.0,
            'energy_auc_pr': 0.0,
            'energy_auc_pr_interval': 0.0,
            'energy_f1_score': 0.0,
            'energy_f1_score_interval': 0.0,
            'logits_auroc': 0.0,
            'logits_auroc_interval': 0.0,
            'logits_fpr95': 0.0,
            'logits_fpr95_interval': 0.0,
            'logits_auc_pr': 0.0,
            'logits_auc_pr_interval': 0.0,
            'logits_f1_score': 0.0,
            'logits_f1_score_interval': 0.0,
            'combine_auroc': 0.0,
            'combine_auroc_interval': 0.0,
            'combine_fpr95': 0.0,
            'combine_fpr95_interval': 0.0,
            'combine_auc_pr': 0.0,
            'combine_auc_pr_interval': 0.0,
            'combine_f1_score': 0.0,
            'combine_f1_score_interval': 0.0,
        }

        # validation statistics
        self.val_log = {
            'val_max_acc_epoch': 0,
            'val_max_auroc_epoch': 0,
            'val_max_acc': 0.0,
            'val_max_acc_interval': 0.0,
            'val_max_auroc': 0.0,
            'val_max_auroc_interval': 0.0,
            'val_total_loss': 0.0,
            'val_acc': 0.0,
            'val_acc_interval': 0.0,
            'val_energy_auroc': 0.0,
            'val_energy_auroc_interval': 0.0,
            'val_energy_fpr95': 0.0,
            'val_energy_fpr95_interval': 0.0,
            'val_energy_auc_pr': 0.0,
            'val_energy_auc_pr_interval': 0.0,
            'val_energy_f1_score': 0.0,
            'val_energy_f1_score_interval': 0.0,
            'val_logits_auroc': 0.0,
            'val_logits_auroc_interval': 0.0,
            'val_logits_fpr95': 0.0,
            'val_logits_fpr95_interval': 0.0,
            'val_logits_auc_pr': 0.0,
            'val_logits_auc_pr_interval': 0.0,
            'val_logits_f1_score': 0.0,
            'val_logits_f1_score_interval': 0.0,
            'val_combine_auroc': 0.0,
            'val_combine_auroc_interval': 0.0,
            'val_combine_fpr95': 0.0,
            'val_combine_fpr95_interval': 0.0,
            'val_combine_auc_pr': 0.0,
            'val_combine_auc_pr_interval': 0.0,
            'val_combine_f1_score': 0.0,
            'val_combine_f1_score_interval': 0.0,
        }

        # test statistics
        self.test_log = {
            'test_total_loss': 0.0,
            'test_acc': 0.0,
            'test_acc_interval': 0.0,
            'test_energy_auroc': 0.0,
            'test_energy_auroc_interval': 0.0,
            'test_energy_fpr95': 0.0,
            'test_energy_fpr95_interval': 0.0,
            'test_energy_auc_pr': 0.0,
            'test_energy_auc_pr_interval': 0.0,
            'test_energy_f1_score': 0.0,
            'test_energy_f1_score_interval': 0.0,
            'test_logits_auroc': 0.0,
            'test_logits_auroc_interval': 0.0,
            'test_logits_fpr95': 0.0,
            'test_logits_fpr95_interval': 0.0,
            'test_logits_auc_pr': 0.0,
            'test_logits_auc_pr_interval': 0.0,
            'test_logits_f1_score': 0.0,
            'test_logits_f1_score_interval': 0.0,
            'test_combine_auroc': 0.0,
            'test_combine_auroc_interval': 0.0,
            'test_combine_fpr95': 0.0,
            'test_combine_fpr95_interval': 0.0,
            'test_combine_auc_pr': 0.0,
            'test_combine_auc_pr_interval': 0.0,
            'test_combine_f1_score': 0.0,
            'test_combine_f1_score_interval': 0.0,
        }

    @abc.abstractmethod
    def train(self):
        pass

    @abc.abstractmethod
    def evaluate(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def evaluate_test(self, *args, **kwargs):
        pass

    def try_evaluate(self, epoch):
        args = self.args
        if self.train_epoch % args.eval_interval == 0:
            result_list = self.evaluate(self.val_loader)
            print("epoch {}, val, loss={:.4f} acc: {:.4f} + {:.4f} "
                  "Energy: auroc={:.4f}+{:.4f}, fpr95={:.4f}+{:.4f}, auc-pr={:.4f}+{:.4f}, f1 score={:.4f}+{:.4f} "
                  "Logits: auroc={:.4f}+{:.4f}, fpr95={:.4f}+{:.4f}, auc-pr={:.4f}+{:.4f}, f1 score={:.4f}+{:.4f} "
                  "Combine: auroc={:.4f}+{:.4f}, fpr95={:.4f}+{:.4f}, auc-pr={:.4f}+{:.4f}, f1 score={:.4f}+{:.4f} "
                  .format(epoch, *result_list))

            for idx, key in enumerate(list(self.val_log.keys())[6:]):
                self.val_log[key] = result_list[idx]

            if result_list[1] >= self.val_log['val_max_acc']:
                self.val_log['val_max_acc_epoch'] = self.train_epoch
                self.val_log["val_max_acc"] = result_list[1]
                self.val_log['val_max_acc_interval'] = result_list[2]
                self.save_model('max_acc')

            if result_list[3] >= self.val_log['val_max_auroc']:
                self.val_log['val_max_auroc_epoch'] = self.train_epoch
                self.val_log['val_max_auroc'] = result_list[3]
                self.val_log['val_max_auroc_interval'] = result_list[4]
                self.save_model('max_auroc')

            # wandb log
            wandb.log(self.val_log, step=self.train_step)

    def try_logging(self, l_few, l_open, l_energy, l_all,
                    acc, energy_auroc, logits_auroc, combine_auroc):
        args = self.args
        if self.train_step % args.log_interval == 0:
            print('epoch {}, train {:06g}/{:06g}, total loss={:.4f}, loss={:.4f} acc={:.4f}, '
                  "Energy: auroc={:.4f}, fpr95={:.4f}, auc-pr={:.4f} f1 score={:.4f}"
                  "Logits: auroc={:.4f}, fpr95={:.4f}, auc-pr={:.4f} f1 score={:.4f}"
                  "Combine: auroc={:.4f}, fpr95={:.4f}, auc-pr={:.4f} f1 score={:.4f}"
                  'lr={:.4g}'
                  .format(self.train_epoch, self.train_step, self.max_steps, l_all, l_few, acc,
                          energy_auroc[0], energy_auroc[1], energy_auroc[2], energy_auroc[3],
                          logits_auroc[0], logits_auroc[1], logits_auroc[2], logits_auroc[3],
                          combine_auroc[0], combine_auroc[1], combine_auroc[2], combine_auroc[3],
                          self.optimizer.param_groups[0]['lr']))
            print('data_timer: {:.2f} sec, forward_timer: {:.2f} sec, '
                  'backward_timer: {:.2f} sec, '
                  'optimize_timer: {:.2f} sec'
                  .format(self.data_time.item(), self.forward_time.item(),
                          self.backward_time.item(), self.optimizer_time.item()))

            self.train_log = {
                'few_shot_loss': l_few,
                'open_set_loss': l_open,
                'energy_loss': l_energy,
                'total_loss': l_all,
                'acc': acc,
                'energy_auroc': energy_auroc[0],
                'energy_fpr95': energy_auroc[1],
                'energy_auc_pr': energy_auroc[2],
                'energy_f1_score': energy_auroc[3],
                'logits_auroc': logits_auroc[0],
                'logits_fpr95': logits_auroc[1],
                'logits_auc_pr': logits_auroc[2],
                'logits_f1_score': logits_auroc[3],
                'combine_auroc': combine_auroc[0],
                'combine_fpr95': combine_auroc[1],
                'combine_auc_pr': combine_auroc[2],
                'combine_f1_score': combine_auroc[3],
            }

            # wandb log
            wandb.log(self.train_log, step=self.train_step)

    def save_model(self, name):
        torch.save(
            dict(params=self.model.state_dict()),
            osp.join(self.args.save_path, name + '.pth')
        )
