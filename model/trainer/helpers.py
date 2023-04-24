import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader

from model.dataloader.samplers import FewShotOpenSetSampler


def get_dataloader(args):
    if args.dataset == 'MiniImageNet':
        from model.dataloader.mini_imagenet import MiniImageNet as Dataset
    elif args.dataset == 'TieredImageNet':
        from model.dataloader.tiered_imagenet import TieredImageNet as Dataset
    elif args.dataset == 'CIFAR-FS':
        from model.dataloader.cifar import CifarFs as Dataset
    elif args.dataset == 'FC100':
        from model.dataloader.fc100 import Fc100 as Dataset
    else:
        raise ValueError('Non-supported Dataset.')

    num_device = torch.cuda.device_count()
    num_episodes = args.episodes_per_epoch * num_device if args.multi_gpu else args.episodes_per_epoch
    num_workers = args.num_workers * num_device if args.multi_gpu else args.num_workers

    train_set = Dataset('train', args, augment=args.augment)
    args.num_class = train_set.num_class
    train_sampler = FewShotOpenSetSampler(train_set.label,
                                          num_episodes,
                                          max(args.way, args.num_classes),
                                          args.shot + args.query,
                                          random_open=True if args.new_benchmark == "all" else False)

    train_loader = DataLoader(dataset=train_set,
                              num_workers=num_workers,
                              batch_sampler=train_sampler,
                              pin_memory=True)

    val_set = Dataset('val', args)
    val_sampler = FewShotOpenSetSampler(val_set.label,
                                        args.num_eval_episodes,
                                        args.eval_way,
                                        args.eval_shot + args.eval_query,
                                        random_open=True if args.new_benchmark == "all" else False)
    val_loader = DataLoader(dataset=val_set,
                            batch_sampler=val_sampler,
                            num_workers=args.num_workers,
                            pin_memory=True)

    if args.cross is not None:
        if args.cross == 'MiniImageNet':
            from model.dataloader.mini_imagenet import MiniImageNet as CrossDataset
        elif args.cross == 'TieredImageNet':
            from model.dataloader.tiered_imagenet import TieredImageNet as CrossDataset
        elif args.cross == 'CIFAR-FS':
            from model.dataloader.cifar import CifarFs as CrossDataset
        elif args.cross == 'FC100':
            from model.dataloader.fc100 import Fc100 as CrossDataset
        elif args.cross == 'cub':
            from model.dataloader.cub import Cub as CrossDataset
        else:
            raise ValueError('Non-supported Dataset.')
        test_set = CrossDataset('test', args)
    else:
        test_set = Dataset('test', args)

    test_sampler = FewShotOpenSetSampler(test_set.label,
                                         600,  # args.num_eval_episodes,
                                         args.eval_way,
                                         args.eval_shot + args.eval_query,
                                         random_open=False if args.new_benchmark is None else True)
    test_loader = DataLoader(dataset=test_set,
                             batch_sampler=test_sampler,
                             num_workers=args.num_workers,
                             pin_memory=True)

    return train_loader, val_loader, test_loader


def prepare_model(args, model):
    # load pre-trained model (no FC weights)
    if args.init_weights is not None:
        model_dict = model.state_dict()
        if "mini-feat" not in args.init_weights:
            pretrained_dict = torch.load(args.init_weights)['params']
            if 'fc.weight' in pretrained_dict.keys():
                del pretrained_dict['fc.weight']
                del pretrained_dict['fc.bias']
        else:
            pretrained_dict = torch.load(args.init_weights)['params']
            if args.backbone_class == 'ConvNet':
                pretrained_dict = {'encoder.' + k: v for k, v in pretrained_dict.items()}
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

        if args.freeze_cls:
            for p in model.encoder.parameters():
                p.requires_grad = False
            for p in model.slf_attn.parameters():
                p.requires_grad = False

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    if args.multi_gpu:
        model.encoder = nn.DataParallel(model.encoder, dim=0)
        para_model = model.to(device)
    else:
        para_model = model.to(device)

    return para_model


def prepare_optimizer(model, args):
    top_para = [v for k, v in model.named_parameters() if 'encoder' not in k]

    # as in the literature, we use ADAM for ConvNet and SGD for other backbones
    if args.backbone_class == 'ConvNet':
        optimizer = optim.Adam(
            [{'params': model.encoder.parameters()},
             {'params': top_para, 'lr': args.lr * args.lr_mul}],
            lr=args.lr,
            # weight_decay=args.weight_decay, do not use weight_decay here
        )
    else:
        optimizer = optim.SGD(
            [{'params': model.encoder.parameters()},
             {'params': top_para, 'lr': args.lr * args.lr_mul}],
            lr=args.lr,
            momentum=args.mom,
            nesterov=True,
            weight_decay=args.weight_decay
        )

    if args.lr_scheduler == 'step':
        lr_scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=int(args.step_size),
            gamma=args.gamma
        )
    elif args.lr_scheduler == 'multistep':
        lr_scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[int(_) for _ in args.step_size.split(',')],
            gamma=args.gamma,
        )
    elif args.lr_scheduler == 'cosine':
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            args.max_epoch,
            eta_min=0  # a tuning parameter
        )
    else:
        raise ValueError('No Such Scheduler')

    return optimizer, lr_scheduler
