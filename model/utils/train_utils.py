import os
import shutil
import time

import torch
import numpy as np
from scipy import interpolate
from sklearn.metrics import roc_curve, roc_auc_score, average_precision_score, f1_score


def ensure_path(dir_path, scripts_to_save=None, debug=False):
    if os.path.exists(dir_path):
        if debug or input('{} exists, remove? ([y]/n)'.format(dir_path)) != 'n':
            shutil.rmtree(dir_path)
            os.mkdir(dir_path)
    else:
        os.mkdir(dir_path)

    print('Experiment dir : {}'.format(dir_path))
    if scripts_to_save is not None:
        script_path = os.path.join(dir_path, 'scripts')
        if not os.path.exists(script_path):
            os.makedirs(script_path)
        for src_file in scripts_to_save:
            dst_file = os.path.join(dir_path, 'scripts', os.path.basename(src_file))
            print('copy {} to {}'.format(src_file, dst_file))
            if os.path.isdir(src_file):
                shutil.copytree(src_file, dst_file)
            else:
                shutil.copyfile(src_file, dst_file)


class Averager:
    def __init__(self):
        self.n = 0
        self.v = 0

    def add(self, x):
        self.v = (self.v * self.n + x) / (self.n + 1)
        self.n += 1
        return self.v

    def item(self):
        return self.v


class ListAverager:
    def __init__(self, num=1):
        self.n = 0
        self.v = [0] * num

    def add(self, x):
        for idx, value in enumerate(x):
            self.v[idx] = (self.v[idx] * self.n + value) / (self.n + 1)
            self.n += 1
        return self.v

    def item(self):
        return self.v


def count_acc(logits, label):
    pred = torch.argmax(logits, dim=1)
    if torch.cuda.is_available():
        return (pred == label).type(torch.cuda.FloatTensor).mean().item()
    else:
        return (pred == label).type(torch.FloatTensor).mean().item()


def calc_auroc(known_scores, unknown_scores):
    y_true = np.array([0] * len(known_scores) + [1] * len(unknown_scores))
    y_score = np.concatenate([known_scores, unknown_scores])
    y_pred = np.where(y_score >= np.sort(y_score)[75], 1, 0)
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    fpr95 = float(interpolate.interp1d(tpr, fpr)(0.95))
    auc_pr = average_precision_score(y_true, y_score)
    auc_score = roc_auc_score(y_true, y_score)
    f_score = f1_score(y_true, y_pred, average="binary")

    return auc_score, fpr95, auc_pr, f_score


def euclidean_metric(a, b):
    n = a.shape[0]
    m = b.shape[0]
    a = a.unsqueeze(1).expand(n, m, -1)
    b = b.unsqueeze(0).expand(n, m, -1)
    logits = -((a - b) ** 2).sum(dim=2)
    return logits


class Timer:
    def __init__(self):
        self.o = time.time()

    def measure(self, p=1):
        x = (time.time() - self.o) / p
        x = int(x)
        if x >= 3600:
            return '{:.1f}h'.format(x / 3600)
        if x >= 60:
            return '{}m'.format(round(x / 60))
        return '{}s'.format(x)


def compute_confidence_interval(data):
    """
    Compute 95% confidence interval
    :param data: An array of mean accuracy (or mAP) across a number of sampled episodes.
    :return: the 95% confidence interval for this data.
    """
    a = 1.0 * np.array(data)
    m = np.mean(a)
    std = np.std(a)
    pm = 1.96 * (std / np.sqrt(len(a)))
    return m, pm
