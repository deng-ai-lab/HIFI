import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from spikingjelly.clock_driven import functional
from spikingjelly.clock_driven import surrogate as surrogate_sj
from modules import neuron, surrogate
from models import spiking_resnet, spiking_vgg_bn
from utils import Bar, AverageMeter, accuracy
import argparse
import collections
import os
import time


def main():
    parser = argparse.ArgumentParser(description='SNN testing')
    parser.add_argument('-T', default=6, type=int, help='Simulation time-steps')
    parser.add_argument('-tau', default=1.1, type=float, help='Membrane time constant')
    parser.add_argument('-data_dir', default='./data', type=str, help='Directory to store data')
    parser.add_argument('-dataset', default='cifar10', type=str, help='Should be either cifa10 or cifar100')
    parser.add_argument('-ckpt', default=None, type=str, help='Path to the checkpoint')
    parser.add_argument('-out_dir', default='./logs', type=str, help='Root dir for saving results')
    parser.add_argument('-surrogate', default='triangle', type=str, help='Used surrogate function. should be sigmoid, rectangle, or triangle')
    parser.add_argument('-model', default='spiking_resnet18', type=str, help='Model to use')

    args = parser.parse_args()
    print(args)


    ########################################################
    # data preparing
    ########################################################
    if args.dataset == 'cifar10' or args.dataset == 'cifar100':
        c_in = 3
        if args.dataset == 'cifar10':
            dataloader = datasets.CIFAR10
            num_class = 10
            normalization_mean = (0.4914, 0.4822, 0.4465)
            normalization_std = (0.2023, 0.1994, 0.2010)

        else:
            dataloader = datasets.CIFAR100
            num_class = 100
            normalization_mean = (0.5071, 0.4867, 0.4409)
            normalization_std = (0.2675, 0.2565, 0.2761)

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(normalization_mean, normalization_std)
        ])

        test_set = dataloader(root=args.data_dir, train=False, download=True, transform=transform_test)

        test_data_loader = torch.utils.data.DataLoader(test_set, batch_size=100, shuffle=False, num_workers=4, drop_last=False, pin_memory=True)

    else:
        raise NotImplementedError
    

    ##########################################################
    # model preparing
    ##########################################################
    if args.surrogate == 'sigmoid':
        surrogate_function = surrogate_sj.Sigmoid()
    elif args.surrogate == 'rectangle':
        surrogate_function = surrogate.Rectangle()
    elif args.surrogate == 'triangle':
        surrogate_function = surrogate_sj.PiecewiseQuadratic()

    neuron_model = neuron.HIFINeuron

    if args.dataset.startswith('cifar'):
        net = spiking_resnet.__dict__[args.model](neuron=neuron_model, num_classes=num_class, tau=args.tau, surrogate_function=surrogate_function, c_in=c_in, fc_hw=1)
        print('using Resnet model')
    else:
        raise NotImplementedError
    
    print('Total Parameters: %.2fM' % (sum(p.numel() for p in net.parameters()) / 1e6))
    net.cuda()


    ##########################################################
    # loading models from checkpoint
    ##########################################################
    ckpt = torch.load(args.ckpt, map_location='cpu')
    state_dict = collections.OrderedDict([(k, v) for k, v in ckpt['net'].items()])
    net.load_state_dict(state_dict)
    print(f'Using checkpoint from {args.ckpt}, max accuracy: {ckpt["max_test_acc"]:.4f}')


    ##########################################################
    # output setting
    ##########################################################
    out_dir = os.path.join(args.out_dir, f'TEST_{args.dataset}_{args.model}_T{args.T}_tau{args.tau}_SG_{args.surrogate}')

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        print(f'Mkidr {out_dir}.')
    else:
        print(f'{out_dir} already exists.')

    with open(os.path.join(out_dir, 'args.txt'), 'w', encoding='utf-8') as args_txt:
        args_txt.write(str(args))   


    ##########################################################
    # testing
    ##########################################################
    start_time = time.time()
    net.eval()
    
    batch_time = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end_time = time.time()

    bar = Bar('Testing', max=len(test_data_loader))

    test_acc = 0
    test_samples = 0
    batch_idx = 0
    layer_fr = []
    
    with torch.no_grad():
        for frame, label in test_data_loader:
            batch_idx += 1
            label = label.cuda()
            t_step = args.T

            for t in range(t_step):
                input_frame = frame

                out_fr, spike_fr = net.test(input_frame)

                layer_fr.append(spike_fr)

                if t == 0:
                    total_fr = out_fr.clone().detach()
                else:
                    total_fr += out_fr.clone().detach()

            test_samples = label.numel()
            test_acc += (total_fr.argmax(dim=1) == label).float().sum().item()
            functional.reset_net(net)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(total_fr.data, label.data, topk=(1, 5))
            top1.update(prec1.item(), input_frame.size(0))
            top5.update(prec5.item(), input_frame.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end_time)
            end_time = time.time()

            # plot progress
            # plot progress
            bar.suffix  = '({batch}/{size}) | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                batch=batch_idx,
                size=len(test_data_loader),
                bt=batch_time.avg,
                total=bar.elapsed_td,
                eta=bar.eta_td,
                top1=top1.avg,
                top5=top5.avg,
                )
            bar.next()
    bar.finish()

    layer_fr = torch.mean(torch.tensor(layer_fr), dim=0) * args.T
    layer_flops = torch.tensor(net.calc_flops())

    OP_ANN = 1.
    OP_SNN = (layer_flops[1:] * layer_fr).sum() / layer_flops[1:].sum()
    OP_LAYER1 = (layer_flops[0]) / layer_flops.sum()
    ENERGY = (OP_ANN * 4.6 / (OP_LAYER1 * 4.6 + (1 - OP_LAYER1) * OP_SNN * 0.9))

    NUM_PARAM = sum(p.numel() for p in net.parameters()) / 1e6

    test_acc /= test_samples

    total_time = time.time() - start_time

    result_str = f'test_acc: {test_acc:.4f}, top1: {top1.avg:.4f}, top5: {top5.avg:.4f}, total_time: {total_time:.2f}, energy: {ENERGY:.4f} , params: {NUM_PARAM:.4f}M\n'

    print(result_str)
    with open(os.path.join(out_dir, 'result.txt'), 'a', encoding='utf-8') as result_txt:
        result_txt.write(result_str)

if __name__ == '__main__':
    main()
