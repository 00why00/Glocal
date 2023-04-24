import argparse
import os
import random
import sys

import numpy as np
import torch
import pprint
from model.utils.config import set_gpu, postprocess_args

sys.path.append(os.getcwd())

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_epoch', type=int, default=100)
    parser.add_argument('--episodes_per_epoch', type=int, default=600)
    parser.add_argument('--num_eval_episodes', type=int, default=600)
    parser.add_argument('--model_class', type=str, default='OpenNet', choices=['OpenNet', 'GEL'])
    parser.add_argument('--distance', type=str, default='euclidean', choices=['euclidean', 'pixel_sim'])
    parser.add_argument('--backbone_class', type=str, default='Res12', choices=['ConvNet', 'Res12', 'Res18', 'WRN'])
    parser.add_argument('--dataset', type=str, default='MiniImageNet', choices=['MiniImageNet', 'TieredImageNet',
                                                                                'CIFAR-FS', 'FC100'])
    parser.add_argument('--pretrain', type=bool, default=False)
    parser.add_argument('--closed_way', type=int, default=5)
    parser.add_argument('--closed_eval_way', type=int, default=5)
    parser.add_argument('--open_way', type=int, default=5)
    parser.add_argument('--open_eval_way', type=int, default=5)
    parser.add_argument('--shot', type=int, default=1)
    parser.add_argument('--query', type=int, default=15)
    parser.add_argument('--eval_query', type=int, default=15)
    parser.add_argument('--temperature', type=float, default=64)

    # optimization parameters
    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--lr_mul', type=float, default=10)
    parser.add_argument('--lr_scheduler', type=str, default='step', choices=['multistep', 'step', 'cosine'])
    parser.add_argument('--step_size', type=str, default='20')
    parser.add_argument('--gamma', type=float, default=0.5)
    parser.add_argument('--fix_BN', action='store_true', default=False)  # do not update the running mean/var in BN
    parser.add_argument('--augment', action='store_true', default=False)
    parser.add_argument('--multi_gpu', action='store_true', default=False)
    parser.add_argument('--gpu', default='1')
    parser.add_argument('--init_weights', type=str, default='./initialization/{}-{}.pth')

    # usually untouched parameters
    parser.add_argument('--mom', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=0.0005)  # we find this weight decay value works the best
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--log_interval', type=int, default=300)
    parser.add_argument('--eval_interval', type=int, default=1)
    parser.add_argument('--save_dir', type=str, default='./checkpoints')
    parser.add_argument('--freeze_cls', action='store_true', default=False)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--data_path', type=str, default=None)
    parser.add_argument('--test', action='store_true', default=False)
    parser.add_argument('--test_model_path', type=str, default=None)
    parser.add_argument('--debug', action='store_true', default=False)

    # model parameters
    parser.add_argument('--attention', action='store_true', default=False)
    parser.add_argument('--open_loss', action='store_false', default=True)
    parser.add_argument('--open_loss_scale', default=0.5, type=float)
    parser.add_argument('--energy', action='store_true', default=False)
    parser.add_argument('--energy_loss', action='store_true', default=False)
    parser.add_argument('--m_in', type=float, default=-1,
                        help='margin for in-distribution; above this value will be penalized')
    parser.add_argument('--m_out', type=float, default=1,
                        help='margin for out-distribution; below this value will be penalized')
    parser.add_argument('--energy_method', type=str, default="sum", choices=["sum", "min"])
    parser.add_argument('--energy_distance', type=float, default=2.)

    # parameters for pixel-wise module
    parser.add_argument('--pixel_wise', action='store_true', default=False)
    parser.add_argument('--pixel_conv', action='store_true', default=False)
    parser.add_argument('--top_method', type=str, default='query', choices=['que0ry', 'proto', 'all'])
    parser.add_argument('--top_k', type=int, default=1)
    parser.add_argument('--SnaTCHer', action='store_true', default=False)

    # parameters for ahead combine
    parser.add_argument('--ahead_combine', action='store_true', default=False)
    parser.add_argument('--learnable_margin', action='store_true', default=False)

    # parameters for new benchmark
    parser.add_argument('--new_benchmark', type=str, default=None, choices=[None, 'test', 'all'])

    # cross domain
    parser.add_argument('--cross', type=str, default=None, choices=['MiniImageNet', 'TieredImageNet', 'CIFAR-FS',
                                                                    'FC100', 'cub'])

    # method
    parser.add_argument('--method', type=str, default="GEL", choices=["GEL", "SnaTCHer"])

    args = parser.parse_args()

    if args.init_weights == './initialization/{}-{}.pth':
        args.init_weights = args.init_weights.format(args.dataset, args.shot)

    if args.pixel_wise:
        args.model_class = "GEL"

    if args.pixel_conv:
        args.m_in = -1 * args.energy_distance / 2
        args.m_out = 1 * args.energy_distance / 2

    args.way = args.closed_way + args.open_way
    args.eval_way = args.closed_eval_way + args.open_eval_way
    args.eval_shot = args.shot
    args.num_classes = args.way

    args = postprocess_args(args)
    args_printer = pprint.PrettyPrinter()
    args_printer.pprint(args)

    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    set_gpu(args.gpu)

    if args.method == "GEL":
        from model.trainer.fsor_trainer_GEL import FSORTrainerGEL
        trainer = FSORTrainerGEL(args)
        if not args.test:
            trainer.train()
        trainer.evaluate_test(path='max_acc.pth')
        trainer.evaluate_test(path='max_auroc.pth')
        trainer.evaluate_test(path='epoch-last.pth')
    elif args.method == "SnaTCHer":
        from model.trainer.fsor_trainer_snatcher_f import FSORTrainerSnaTCherF
        trainer = FSORTrainerSnaTCherF(args)
        trainer.evaluate_test()
    else:
        raise NotImplementedError

    print(args.save_path)
