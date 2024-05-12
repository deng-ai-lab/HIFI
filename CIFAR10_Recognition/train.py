import datetime
import os
import time
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.cuda import amp
from models import spiking_resnet
from modules import neuron
import argparse
from spikingjelly.clock_driven import functional
from spikingjelly.clock_driven import surrogate as surrogate_sj
from modules import surrogate as surrogate_self
from utils import Bar, Logger, AverageMeter, accuracy
from utils.augmentation import RandomMixup, RandomCutmix, ClassificationPresetTrain, InterpolationMode
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data.dataloader import default_collate
import collections
import random
import numpy as np
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def main():

    parser = argparse.ArgumentParser(description='SNN training')
    parser.add_argument('-seed', default=3407, type=int)
    parser.add_argument('-name', default='', type=str, help='specify a name for the checkpoint and log files')
    parser.add_argument('-T', default=6, type=int, help='simulating time-steps')
    parser.add_argument('-tau', default=1.1, type=float, help='a hyperparameter for the LIF model')
    parser.add_argument('-vth', default=1.1, type=float, help='a hyperparameter for the LIF model')
    parser.add_argument('-b', default=128, type=int, help='batch size')
    parser.add_argument('-epochs', default=300, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('-j', default=4, type=int, metavar='N', help='number of data loading workers (default: 4)')
    parser.add_argument('-data_dir', type=str, default='./data', help='directory of the used dataset')
    parser.add_argument('-dataset', default='cifar10', type=str, help='should be either cifar10 or cifar100')
    parser.add_argument('-out_dir', type=str, default='./logs', help='root dir for saving logs and checkpoint')
    parser.add_argument('-surrogate', default='triangle', type=str, help='used surrogate function. should be sigmoid, rectangle, or triangle')
    parser.add_argument('-resume', type=str, help='resume from the checkpoint path')
    parser.add_argument('-pre_train', type=str, help='load a pretrained model. used for imagenet')
    parser.add_argument('-amp', action='store_true', help='automatic mixed precision training')
    parser.add_argument('-opt', type=str, help='use which optimizer. SGD or AdamW', default='SGD')
    parser.add_argument('-lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('-momentum', default=0.9, type=float, help='momentum for SGD')
    parser.add_argument('-lr_scheduler', default='CosALR', type=str, help='use which schedule. StepLR or CosALR')
    parser.add_argument('-step_size', default=100, type=float, help='step_size for StepLR')
    parser.add_argument('-gamma', default=0.1, type=float, help='gamma for StepLR')
    parser.add_argument('-T_max', default=300, type=int, help='T_max for CosineAnnealingLR')
    parser.add_argument('-model', type=str, default='spiking_vgg11_bn', help='use which SNN model')
    parser.add_argument('-drop_rate', type=float, default=0.0, help='dropout rate')
    parser.add_argument('-weight_decay', type=float, default=0)
    parser.add_argument('-loss_lambda', type=float, default=0.05, help='the scaling factor for the MSE term in the loss')
    parser.add_argument('-mse_n_reg', action='store_true', help='loss function setting')
    parser.add_argument('-loss_means', type=float, default=1.0, help='used in the loss function when mse_n_reg=False')
    parser.add_argument('-save_init', action='store_true', help='save the initialization of parameters')

    args = parser.parse_args()
    print(args)

    _seed_ = args.seed
    random.seed(_seed_)
    torch.manual_seed(_seed_)  # use torch.manual_seed() to seed the RNG for all devices (both CPU and CUDA)
    torch.cuda.manual_seed_all(_seed_)
    np.random.seed(_seed_)


    ########################################################
    # data preparing
    ########################################################
    if args.dataset == 'cifar10' or args.dataset == 'cifar100':
        c_in = 3
        if args.dataset == 'cifar10':
            dataloader = datasets.CIFAR10
            num_classes = 10
            normalization_mean = (0.4914, 0.4822, 0.4465)
            normalization_std = (0.2023, 0.1994, 0.2010)

        elif args.dataset == 'cifar100':
            dataloader = datasets.CIFAR100
            num_classes = 100
            normalization_mean = (0.5071, 0.4867, 0.4408)
            normalization_std = (0.2675, 0.2565, 0.2761)

        mixup_transforms = []
        mixup_transforms.append(RandomMixup(num_classes, p=1.0, alpha=0.2))
        mixup_transforms.append(RandomCutmix(num_classes, p=1.0, alpha=1.))
        mixupcutmix = transforms.RandomChoice(mixup_transforms)
        collate_fn = lambda batch: mixupcutmix(*default_collate(batch))  # noqa: E731

        transform_train = ClassificationPresetTrain(mean=normalization_mean,
                                                    std=normalization_std, 
                                                    interpolation=InterpolationMode('bilinear'),
                                                    auto_augment_policy='ta_wide',
                                                    random_erase_prob=0.1)

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(normalization_mean, normalization_std),
        ])

        train_set = dataloader(root=args.data_dir, train=True, transform=transform_train,download=True)
        test_set = dataloader(root=args.data_dir, train=False, transform=transform_test, download=True)

        train_data_loader = DataLoader(
            dataset=train_set,
            batch_size=args.b,
            collate_fn=collate_fn,
            shuffle=True,
            drop_last=True,
            num_workers=args.j,
            pin_memory=True
        )
        test_data_loader = DataLoader(
            dataset=test_set,
            batch_size=args.b,
            shuffle=False,
            drop_last=False,
            num_workers=args.j,
            pin_memory=True
        )

    else:
        raise NotImplementedError


    ##########################################################
    # model preparing
    ##########################################################
    if args.surrogate == 'sigmoid':
        surrogate_function = surrogate_sj.Sigmoid()
    elif args.surrogate == 'rectangle':
        surrogate_function = surrogate_self.Rectangle()
    elif args.surrogate == 'triangle':
        surrogate_function = surrogate_sj.PiecewiseQuadratic()

    neuron_model = neuron.HIFINeuron

    # if args.dataset == 'cifar10' or args.dataset == 'cifar100':
    if args.dataset.startswith('cifar'):
        net = spiking_resnet.__dict__[args.model](neuron=neuron_model, num_classes=num_classes, neuron_dropout=args.drop_rate,
                                                  tau=args.tau, v_threshold=args.vth, surrogate_function=surrogate_function, c_in=c_in, fc_hw=1)
        print('using Resnet model.')

    else:
        raise NotImplementedError
    print('Total Parameters: %.2fM' % (sum(p.numel() for p in net.parameters()) / 1000000.0))
    net.cuda()


    ##########################################################
    # optimizer preparing
    ##########################################################
    inner_params = []
    outer_params = []
    for name, param in net.named_parameters():
        if 'attri' in name:
            inner_params.append(param)
        else:
            outer_params.append(param)

    if args.opt == 'SGD':
        optimizer = torch.optim.SGD([
                {'params': inner_params, 'lr': args.lr * 0.01},
                {'params': outer_params}
            ],lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.opt == 'AdamW':
        optimizer = torch.optim.AdamW([
                {'params': inner_params, 'lr': args.lr * 0.01},
                {'params': outer_params}
            ], lr=args.lr, weight_decay=args.weight_decay)
    else:
        raise NotImplementedError(args.opt)

    if args.lr_scheduler == 'StepLR':
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    elif args.lr_scheduler == 'CosALR':
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.T_max)
    else:
        raise NotImplementedError(args.lr_scheduler)

    scaler = None
    if args.amp:
        scaler = amp.GradScaler()


    ##########################################################
    # loading models from checkpoint
    ##########################################################
    start_epoch = 0
    max_test_acc = 0

    if args.resume:
        print('resuming...')
        checkpoint = torch.load(args.resume, map_location='cpu')
        net.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        start_epoch = checkpoint['epoch'] + 1
        max_test_acc = checkpoint['max_test_acc']
        print('start epoch:', start_epoch, ', max test acc:', max_test_acc)

    if args.pre_train:
        checkpoint = torch.load(args.pre_train, map_location='cpu')
        state_dict2 = collections.OrderedDict([(k, v) for k, v in checkpoint['net'].items()])
        net.load_state_dict(state_dict2)
        print('use pre-trained model, max test acc:', checkpoint['max_test_acc'])


    ##########################################################
    # output setting
    ##########################################################
    out_dir = os.path.join(args.out_dir, f'ARCHIVED_{args.dataset}_{args.model}_{args.name}_T{args.T}_tau{args.tau}_vth{args.vth}_e{args.epochs}_bs{args.b}_{args.opt}_lr{args.lr}_wd{args.weight_decay}_SG_{args.surrogate}_drop{args.drop_rate}_losslamb{args.loss_lambda}_')

    if args.lr_scheduler == 'CosALR':
        out_dir += f'CosALR_{args.T_max}'
    elif args.lr_scheduler == 'StepLR':
        out_dir += f'StepLR_{args.step_size}_{args.gamma}'
    else:
        raise NotImplementedError(args.lr_scheduler)

    if args.amp:
        out_dir += '_amp'

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        print(f'Mkdir {out_dir}.')
    else:
        print('out dir already exists:', out_dir)

    # save the initialization of parameters
    if args.save_init:
        checkpoint = {
            'net': net.state_dict(),
            'epoch': 0,
            'max_test_acc': 0.0
        }
        torch.save(checkpoint, os.path.join(out_dir, 'checkpoint_0.pth'))

    with open(os.path.join(out_dir, 'args.txt'), 'w', encoding='utf-8') as args_txt:
        args_txt.write(str(args))

    writer = SummaryWriter(os.path.join(out_dir, 'logs'), purge_step=start_epoch)


    ##########################################################
    # training and testing
    ##########################################################
    criterion_mse = nn.MSELoss()

    for epoch in range(start_epoch, args.epochs):
        ############### training ###############
        start_time = time.time()
        net.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        end = time.time()

        bar = Bar('Processing', max=len(train_data_loader))

        train_loss = 0
        train_acc = 0
        train_samples = 0
        batch_idx = 0

        for frame, label in train_data_loader:
            batch_idx += 1
            t_step = args.T

            label = label.cuda()

            batch_loss = 0
            optimizer.zero_grad()
            for t in range(t_step):
                input_frame = frame.cuda()
                if args.amp:
                    with amp.autocast():
                        if t == 0:
                            out_fr = net(input_frame)
                            total_fr = out_fr.clone().detach()
                        else:
                            out_fr = net(input_frame)
                            total_fr += out_fr.clone().detach()
                        # Calculate the loss
                        if args.loss_lambda > 0.0:  # the loss is a cross entropy term plus a mse term
                            if args.mse_n_reg:  # the mse term is not treated as a regularizer
                                label_one_hot = F.one_hot(label, num_classes).float()
                            else:
                                label_one_hot = torch.zeros_like(out_fr).fill_(args.loss_means).to(out_fr.device)
                            mse_loss = criterion_mse(out_fr, label_one_hot)
                            # get current learning rate
                            current_lr = optimizer.param_groups[0]['lr']
                            loss = ((1 - args.loss_lambda) * F.cross_entropy(out_fr, label, label_smoothing=current_lr) + args.loss_lambda * mse_loss) / t_step
                        else:  # the loss is just a cross entropy term
                            loss = F.cross_entropy(out_fr, label) / t_step
                    scaler.scale(loss).backward()

                else:
                    if t == 0:
                        out_fr = net(input_frame)
                        total_fr = out_fr.clone().detach()
                    else:
                        out_fr = net(input_frame)
                        total_fr += out_fr.clone().detach()
                    if args.loss_lambda > 0.0:
                        label_one_hot = torch.zeros_like(out_fr).fill_(args.loss_means).to(out_fr.device)
                        if args.mse_n_reg:
                            label_one_hot = F.one_hot(label, num_classes).float()
                        mse_loss = criterion_mse(out_fr, label_one_hot)
                        loss = ((1 - args.loss_lambda) * F.cross_entropy(out_fr, label) + args.loss_lambda * mse_loss) / t_step
                    else:
                        loss = F.cross_entropy(out_fr, label) / t_step

                    loss.backward()


                batch_loss += loss.item()
                train_loss += loss.item() * label.numel()

            if args.amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            losses.update(batch_loss, input_frame.size(0))

            train_samples += label.numel()

            functional.reset_net(net)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # plot progress
            bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f}'.format(
                        batch=batch_idx,
                        size=len(train_data_loader),
                        data=data_time.avg,
                        bt=batch_time.avg,
                        total=bar.elapsed_td,
                        eta=bar.eta_td,
                        loss=losses.avg,
                        )
            bar.next()
        bar.finish()

        train_loss /= train_samples
        writer.add_scalar('train_loss', train_loss, epoch)

        lr_scheduler.step()


        ############### testing ###############
        net.eval()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        end = time.time()
        bar = Bar('Processing', max=len(test_data_loader))

        test_loss = 0
        test_acc = 0
        test_samples = 0
        batch_idx = 0
        with torch.no_grad():
            for frame, label in test_data_loader:
                batch_idx += 1
                label = label.cuda()
                t_step = args.T
                total_loss = 0

                for t in range(t_step):
                    input_frame = frame.cuda()
                    if t == 0:
                        out_fr = net(input_frame)
                        total_fr = out_fr.clone().detach()
                    else:
                        out_fr = net(input_frame)
                        total_fr += out_fr.clone().detach()
                    # Calculate the loss
                    if args.loss_lambda > 0.0: # the loss is a cross entropy term plus a mse term
                        if args.mse_n_reg:  # the mse term is not treated as a regularizer
                            label_one_hot = F.one_hot(label, num_classes).float()
                        else:
                            label_one_hot = torch.zeros_like(out_fr).fill_(args.loss_means).to(out_fr.device)
                        mse_loss = criterion_mse(out_fr, label_one_hot)
                        loss = ((1 - args.loss_lambda) * F.cross_entropy(out_fr, label) + args.loss_lambda * mse_loss) / t_step
                    else: # the loss is just a cross entropy term
                        loss = F.cross_entropy(out_fr, label) / t_step
                    total_loss += loss

                test_samples += label.numel()
                test_loss += total_loss.item() * label.numel() # type: ignore
                test_acc += (total_fr.argmax(1) == label).float().sum().item()
                functional.reset_net(net)

                # measure accuracy and record loss
                prec1, prec5 = accuracy(total_fr.data, label.data, topk=(1, 5))
                losses.update(total_loss, input_frame.size(0))
                top1.update(prec1.item(), input_frame.size(0))
                top5.update(prec5.item(), input_frame.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                # plot progress
                bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                            batch=batch_idx,
                            size=len(test_data_loader),
                            data=data_time.avg,
                            bt=batch_time.avg,
                            total=bar.elapsed_td,
                            eta=bar.eta_td,
                            loss=losses.avg,
                            top1=top1.avg,
                            top5=top5.avg,
                            )
                bar.next()
        bar.finish()

        test_loss /= test_samples
        test_acc /= test_samples
        writer.add_scalar('test_loss', test_loss, epoch)
        writer.add_scalar('test_acc', test_acc, epoch)


        ############### saving checkpoint ###############
        save_max = False
        if test_acc > max_test_acc:
            max_test_acc = test_acc
            save_max = True

        checkpoint = {
            'net': net.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'epoch': epoch,
            'max_test_acc': max_test_acc
        }

        if save_max:
            torch.save(checkpoint, os.path.join(out_dir, 'checkpoint_max.pth'))

        torch.save(checkpoint, os.path.join(out_dir, 'checkpoint_latest.pth'))

        total_time = time.time() - start_time
        
        result_str = f'epoch={epoch}, train_loss={train_loss}, train_acc={train_acc}, test_loss={test_loss}, test_acc={test_acc}, max_test_acc={max_test_acc}, total_time={total_time}, escape_time={(datetime.datetime.now()+datetime.timedelta(seconds=total_time * (args.epochs - epoch))).strftime("%Y-%m-%d %H:%M:%S")}'

        print(result_str)

        with open(os.path.join(out_dir, 'result.txt'), 'a', encoding='utf-8') as result_txt:
            result_txt.write(result_str + '\n')

        print("after one epoch: %fGB" % (torch.cuda.max_memory_reserved(0) / 1024 / 1024 / 1024))

if __name__ == '__main__':
    main()
