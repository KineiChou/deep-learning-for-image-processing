import time
import os
import datetime

import torch
from torchvision.ops.misc import FrozenBatchNorm2d

import transforms
from my_dataset_coco import CocoDetection
from my_dataset_voc import VOCInstances
from backbone import resnext50_32x4d_fpn_backbone
from network_files import MaskRCNN
import train_utils.train_eval_utils as utils
from train_utils import GroupedBatchSampler, create_aspect_ratio_groups, init_distributed_mode, save_on_master, mkdir


def create_model(num_classes, load_pretrain_weights=True):
    # 如果GPU显存很小，batch_size不能设置很大，建议将norm_layer设置成FrozenBatchNorm2d(默认是nn.BatchNorm2d)
    # FrozenBatchNorm2d的功能与BatchNorm2d类似，但参数无法更新
    # trainable_layers包括['layer4', 'layer3', 'layer2', 'layer1', 'conv1']， 5代表全部训练
    # backbone = resnet50_fpn_backbone(norm_layer=FrozenBatchNorm2d,
    #                                  trainable_layers=3)
    # resnet50 imagenet weights url: https://download.pytorch.org/models/resnet50-0676ba61.pth
    backbone = resnext50_32x4d_fpn_backbone(weights="DEFAULT", trainable_layers=2)
    model = MaskRCNN(backbone, num_classes=num_classes)

    if load_pretrain_weights:
        # coco weights url: "https://download.pytorch.org/models/maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth"
        weights_dict = torch.load("./maskrcnn_resnet50_fpn_coco.pth", map_location="cpu")
        for k in list(weights_dict.keys()):
            if ("box_predictor" in k) or ("mask_fcn_logits" in k):
                del weights_dict[k]

        print(model.load_state_dict(weights_dict, strict=False))

    return model


def main(args):
    init_distributed_mode(args)
    print(args)

    device = torch.device(args.device)

    # 用来保存coco_info的文件
    now = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    det_results_file = f"det_results{now}.txt"
    seg_results_file = f"seg_results{now}.txt"

    # Data loading code
    print("Loading data")

    data_transform = {
        "train": transforms.Compose([transforms.ToTensor(),
                                     transforms.RandomHorizontalFlip(0.5)]),
        "val": transforms.Compose([transforms.ToTensor()])
    }

    COCO_root = args.data_path

    # load train data set
    # coco2017 -> annotations -> instances_train2017.json
    train_dataset = CocoDetection(COCO_root, "train", data_transform["train"])
    # VOCdevkit -> VOC2012 -> ImageSets -> Main -> train.txt
    # train_dataset = VOCInstances(data_root, year="2012", txt_name="train.txt")

    # load validation data set
    # coco2017 -> annotations -> instances_val2017.json
    val_dataset = CocoDetection(COCO_root, "val", data_transform["val"])
    # VOCdevkit -> VOC2012 -> ImageSets -> Main -> val.txt
    # val_dataset = VOCInstances(data_root, year="2012", txt_name="val.txt")

    print("Creating data loaders")
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        test_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
    else:
        train_sampler = torch.utils.data.RandomSampler(train_dataset)
        test_sampler = torch.utils.data.SequentialSampler(val_dataset)

    if args.aspect_ratio_group_factor >= 0:
        # 统计所有图像比例在bins区间中的位置索引
        group_ids = create_aspect_ratio_groups(train_dataset, k=args.aspect_ratio_group_factor)
        train_batch_sampler = GroupedBatchSampler(train_sampler, group_ids, args.batch_size)
    else:
        train_batch_sampler = torch.utils.data.BatchSampler(
            train_sampler, args.batch_size, drop_last=True)

    data_loader = torch.utils.data.DataLoader(
        train_dataset, batch_sampler=train_batch_sampler, num_workers=args.workers,
        collate_fn=train_dataset.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        val_dataset, batch_size=1,
        sampler=test_sampler, num_workers=args.workers,
        collate_fn=train_dataset.collate_fn)

    print("Creating model")
    # create model num_classes equal background + classes
    model = create_model(num_classes=args.num_classes + 1, load_pretrain_weights=args.pretrain)
    model.to(device)

    if args.distributed and args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_steps, gamma=args.lr_gamma)

    # 如果传入resume参数，即上次训练的权重地址，则接着上次的参数训练
    if args.resume:
        # If map_location is missing, torch.load will first load the module to CPU
        # and then copy each parameter to where it was saved,
        # which would result in all processes on the same machine using the same set of devices.
        checkpoint = torch.load(args.resume, map_location='cpu')  # 读取之前保存的权重文件(包括优化器以及学习率策略)
        model_without_ddp.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1
        if args.amp and "scaler" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler"])

    if args.test_only:
        utils.evaluate(model, data_loader_test, device=device)
        return

    train_loss = []
    learning_rate = []
    val_map = []

    print("Start training")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        mean_loss, lr = utils.train_one_epoch(model, optimizer, data_loader,
                                              device, epoch, args.print_freq,
                                              warmup=True, scaler=scaler)

        # update learning rate
        lr_scheduler.step()

        # evaluate after every epoch
        det_info, seg_info = utils.evaluate(model, data_loader_test, device=device)

        # 只在主进程上进行写操作
        if args.rank in [-1, 0]:
            train_loss.append(mean_loss.item())
            learning_rate.append(lr)
            val_map.append(det_info[1])  # pascal mAP

            # write into txt
            with open(det_results_file, "a") as f:
                # 写入的数据包括coco指标还有loss和learning rate
                result_info = [f"{i:.4f}" for i in det_info + [mean_loss.item()]] + [f"{lr:.6f}"]
                txt = "epoch:{} {}".format(epoch, '  '.join(result_info))
                f.write(txt + "\n")

            with open(seg_results_file, "a") as f:
                # 写入的数据包括coco指标还有loss和learning rate
                result_info = [f"{i:.4f}" for i in seg_info + [mean_loss.item()]] + [f"{lr:.6f}"]
                txt = "epoch:{} {}".format(epoch, '  '.join(result_info))
                f.write(txt + "\n")

        if args.output_dir:
            # 只在主进程上执行保存权重操作
            save_files = {'model': model_without_ddp.state_dict(),
                          'optimizer': optimizer.state_dict(),
                          'lr_scheduler': lr_scheduler.state_dict(),
                          'args': args,
                          'epoch': epoch}
            if args.amp:
                save_files["scaler"] = scaler.state_dict()
            save_on_master(save_files,
                           os.path.join(args.output_dir, f'model_{epoch}.pth'))

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

    if args.rank in [-1, 0]:
        # plot loss and lr curve
        if len(train_loss) != 0 and len(learning_rate) != 0:
            from plot_curve import plot_loss_and_lr
            plot_loss_and_lr(train_loss, learning_rate)

        # plot mAP curve
        if len(val_map) != 0:
            from plot_curve import plot_map
            plot_map(val_map)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=__doc__)

    # 训练文件的根目录(coco2017)
    parser.add_argument('--data-path', default='/data/coco2017', help='dataset')
    # 训练设备类型
    parser.add_argument('--device', default='cuda', help='device')
    # 检测目标类别数(不包含背景)
    parser.add_argument('--num-classes', default=90, type=int, help='num_classes')
    # 每块GPU上的batch_size
    parser.add_argument('-b', '--batch-size', default=4, type=int,
                        help='images per gpu, the total batch size is $NGPU x batch_size')
    # 指定接着从哪个epoch数开始训练
    parser.add_argument('--start_epoch', default=0, type=int, help='start epoch')
    # 训练的总epoch数
    parser.add_argument('--epochs', default=26, type=int, metavar='N',
                        help='number of total epochs to run')
    # 数据加载以及预处理的线程数
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    # 学习率，这个需要根据gpu的数量以及batch_size进行设置0.02 / bs * num_GPU
    parser.add_argument('--lr', default=0.005, type=float,
                        help='initial learning rate, 0.02 is the default value for training '
                             'on 8 gpus and 2 images_per_gpu')
    # SGD的momentum参数
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    # SGD的weight_decay参数
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    # 针对torch.optim.lr_scheduler.StepLR的参数
    parser.add_argument('--lr-step-size', default=8, type=int, help='decrease lr every step-size epochs')
    # 针对torch.optim.lr_scheduler.MultiStepLR的参数
    parser.add_argument('--lr-steps', default=[16, 22], nargs='+', type=int,
                        help='decrease lr every step-size epochs')
    # 针对torch.optim.lr_scheduler.MultiStepLR的参数
    parser.add_argument('--lr-gamma', default=0.1, type=float, help='decrease lr by a factor of lr-gamma')
    # 训练过程打印信息的频率
    parser.add_argument('--print-freq', default=50, type=int, help='print frequency')
    # 文件保存地址
    parser.add_argument('--output-dir', default='./multi_train', help='path where to save')
    # 基于上次的训练结果接着训练
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--aspect-ratio-group-factor', default=3, type=int)
    parser.add_argument('--test-only', action="store_true", help="test only")

    # 开启的进程数(注意不是线程)
    parser.add_argument('--world-size', default=4, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')
    parser.add_argument("--sync-bn", dest="sync_bn", help="Use sync batch norm", type=bool, default=False)
    parser.add_argument("--pretrain", type=bool, default=True, help="load COCO pretrain weights.")
    # 是否使用混合精度训练(需要GPU支持混合精度)
    parser.add_argument("--amp", default=False, help="Use torch.cuda.amp for mixed precision training")

    args = parser.parse_args()

    # 如果指定了保存文件地址，检查文件夹是否存在，若不存在，则创建
    if args.output_dir:
        mkdir(args.output_dir)

    main(args)
