import os
import time
import wandb


def set_gpu(x):
    os.environ['CUDA_VISIBLE_DEVICES'] = x
    print('using gpu:', x)


def postprocess_args(args, make_path=True):
    save_path1 = '-'.join([args.dataset, args.model_class, args.backbone_class,
                           '{:02d}w{:02d}s{:02}q'.format(args.way, args.shot, args.query)])
    save_path2 = '_'.join([str('_'.join(args.step_size.split(','))), str(args.gamma),
                           'lr{:.2g}mul{:.2g}'.format(args.lr, args.lr_mul),
                           str(args.lr_scheduler),
                           'T{}'.format(args.temperature),
                           'bsz{:03d}'.format(max(args.way, args.num_classes) * (args.shot + args.query)),
                           str(time.strftime('%Y%m%d_%H%M%S'))
                           ])
    if args.init_weights is not None:
        if "mini-feat" not in args.init_weights:
            save_path1 += args.init_weights.split('/')[-1].split('.')[0]
        else:
            save_path1 += '-Pre'

    if args.distance == "euclidean":
        save_path1 += '-DIS'
    elif args.distance == "pixel_sim":
        save_path1 += '-PIM'
    else:
        raise NotImplementedError

    if args.dataset == "MiniImageNet":
        save_path1 += '-New'

    if args.energy:
        save_path1 += '-Energy'

        if not args.augment:
            save_path2 += '-NoAug'

        if args.attention:
            save_path2 += '-ATT'

        if not args.open_loss:
            save_path2 += "-woSML"

        if args.energy_loss:
            save_path2 += "-EL"
            if args.energy_method == "min":
                save_path2 += "-min"
            else:
                save_path2 += "-sum"

        if args.pixel_wise:
            save_path2 += "-Pixel"

            if args.pixel_conv:
                save_path2 += "-Conv"

            if args.freeze_cls:
                save_path2 += "=FrCls"

        if args.top_k is not None:
            save_path2 += '-top_{}_{}'.format(args.top_method, args.top_k)

        if args.SnaTCHer:
            save_path2 += '-STR'

        if args.energy_distance != 2:
            save_path2 += '-dis_{}'.format(args.energy_distance)

        if args.ahead_combine:
            save_path2 += '-AHC'

        if args.learnable_margin:
            save_path2 += '-LMA'

        if args.new_benchmark is not None:
            save_path2 += '-NB_{}'.format(args.new_benchmark)

    if args.debug:
        save_path2 = 'Debug'

    if args.method != "GEL":
        save_path1 = args.method
        save_path2 = str(time.strftime('%Y%m%d_%H%M%S'))

    if args.cross is not None:
        save_path3 = 'cross_{}'.format(args.cross)
        if args.test:
            args.cross_save_path = os.path.join(args.test_model_path, save_path3)
        else:
            args.cross_save_path = os.path.join(args.save_path, save_path1, save_path2, save_path3)
    else:
        args.cross_save_path = None

    if make_path:
        if not os.path.exists(os.path.join(args.save_dir, save_path1)):
            os.mkdir(os.path.join(args.save_dir, save_path1))
    args.save_path = os.path.join(args.save_dir, save_path1, save_path2)

    if args.method == "GEL":
        wandb.init(project="GEL",
                   entity="0why0",
                   group=save_path1,
                   name=save_path2,
                   tags=[args.dataset],
                   magic=True)
        wandb.config.update(args)

    return args
