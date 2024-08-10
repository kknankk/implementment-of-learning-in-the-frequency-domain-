from __future__ import print_function

import argparse
import os
import sys
import warnings
import shutil
import time
import random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.utils.data.distributed
import torch.optim as optim
import torch.utils.data as data
from datasets.dataset_imagenet_dct import ImageFolderDCT
import datasets.cvtransforms as transforms
from models.imagenet.resnet import ResNetDCT_Upscaled_Static
from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig
from datasets.dataloader_imagenet_dct import valloader_upscaled_static
from tensorboardX import SummaryWriter
from datasets.train_val_lader import val_loader,train_loader

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')

# Parse arguments
parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

# Datasets
parser.add_argument('-d', '--data', default='path to dataset', type=str)
parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
# Optimization options
parser.add_argument('--train-batch', default=256, type=int, metavar='N',
                    help='train batchsize (default: 256)')
parser.add_argument('--test-batch', default=256, type=int, metavar='N',
                    help='test batchsize (default: 200)')
# parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
#                     help='manual epoch number (useful on restarts)')

parser.add_argument('-c', '--checkpoint', default='checkpoints', type=str, metavar='PATH',
                    help='path to save checkpoint (default: checkpoints)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--lr', '--learning-rate', default=1e-1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--schedule', type=int, nargs='+', default=[150, 225],
                        help='Decrease learning rate at these epochs.')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
# Architecture
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet50dct',
                    help='model architecture: (default: resnet50dct)')
# Miscs
parser.add_argument('--world-size', default=1, type=int,
                    help='number of distributed processes')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='gloo', type=str,
                    help='distributed backend')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--subset', default='192', type=str, help='subset of y, cb, cr')
#Device options
parser.add_argument('--gpu-id', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--pretrained', default='True', type=str2bool,
                    help='load pretrained model or not')

args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}

# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
use_cuda = torch.cuda.is_available()
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Random seed
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
if use_cuda:
    torch.cuda.manual_seed_all(args.manualSeed)

best_prec1 = 0  # best test accuracy

def main():
    global args, best_prec1
    start_epoch = args.start_epoch  # start from epoch 0 or last checkpoint epoch
    batch_logger = Logger(os.path.join(args.checkpoint, 'every_epoch_record.txt'), title='Batch Metrics')
    batch_logger.set_names(['Batch', 'Loss', 'Top1', 'Top5'])

    epoch_logger = Logger(os.path.join(args.checkpoint, 'epoches.txt'), title='Batch Metrics')
    epoch_logger.set_names(['epoch', 'train_loss', 'train_acc1', 'val_loss','val_acc1'])

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    args.distributed = args.world_size > 1

    if args.distributed:
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size)
#--------dataloader-------------
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    train_loader1 = train_loader(args,traindir)
    val_loader1 = val_loader(args,valdir)
#---------------TODO:model--------
    model = ResNetDCT_Upscaled_Static(channels=int(args.subset), pretrained=args.pretrained)
    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = optim.SGD(model.parameters(), lr=args.lr,  momentum=args.momentum, weight_decay=args.weight_decay)
    # optimizer=torch.optim.Adam(model.parameters(), lr=0.01, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, amsgrad=False)

    # Resume
    title = 'ImageNet-' + args.arch
    if not os.path.isdir(args.checkpoint):
        mkdir_p(args.checkpoint)

    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        checkpoint = torch.load(args.resume)
        args.start_epoch = checkpoint['epoch']
        best_prec1 = checkpoint['best_prec1']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title, resume=True)

        print("=> loaded checkpoint '{}' (epoch {})"
              .format(args.resume, checkpoint['epoch']))
        args.checkpoint = os.path.dirname(args.resume)
    # else:
    #     logger = Logger(os.path.join(args.checkpoint, 'train_model.txt'), title=title)
    #     logger.set_names(['Learning Rate', 'Train Loss Top 1', 'Train Loss Top 5','Valid Loss Top 1', 'Valid Loss Top 5','Train Acc.', 'Valid Acc.'])


    if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
        model.features = torch.nn.DataParallel(model.features)
        model = model.cuda()
    else:
        model = torch.nn.DataParallel(model).cuda()

    cudnn.benchmark = True
    print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))

    # Data loading code


    if args.evaluate:
        print('\nEvaluation only')
        test_loss, test_acc_top1, test_acc_top5 = test(val_loader, model, criterion)
        print(' Test Loss:  %.8f, Test Acc Top1:  %.2f, Test Acc Top5:  %.2f' % (test_loss, test_acc_top1, test_acc_top5))
        return
    
#-----------main's:for epoch in range(start_epoch, args.epochs):
    for epoch in range(start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)

        print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, state['lr']))
        # train_loss,top1_acc, train_acc = train(train_loader1, model, criterion, optimizer, epoch, use_cuda)
        # train_top1_acc,train_top5_acc, train_loss = train(train_loader1, model, criterion, optimizer, epoch, use_cuda, batch_logger)
        # val_top1_acc,val_top5_acc, val_loss = test(val_loader1, model, criterion, optimizer, epoch, use_cuda, batch_logger)
        
        train_loss,train_top1_acc,train_top5_acc = train(train_loader1, model, criterion, optimizer, epoch, use_cuda, batch_logger)
        val_loss,val_top1_acc,val_top5_acc = test(val_loader1, model, criterion, optimizer, epoch, use_cuda, batch_logger)


        # train_loss_top1,train_loss_top5, train_acc = train(train_loader1, model, criterion, optimizer, epoch, use_cuda)
        # test_loss_top1,test_loss_top5, test_acc = test(val_loader1, model, criterion, epoch, use_cuda)

        # append logger file
        # logger.append([state['lr'], train_top1_,train_loss_top5, test_loss_top1,test_loss_top5, train_acc, test_acc])
        # logger.append([state['lr'], train_top1_acc,train_top5_acc, train_loss,val_top1_acc,val_top5_acc, val_loss])
        epoch_logger.append([f'epoch {epoch + 1}', train_loss, train_top1_acc, val_loss,val_top1_acc])
        best_acc=0
        # save model
        is_best = val_top1_acc > best_acc
        best_acc = max(val_top1_acc, best_acc)
        save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'acc': val_top1_acc,
                'best_acc': best_acc,
                'optimizer' : optimizer.state_dict(),
            }, is_best, checkpoint=args.checkpoint)

    logger.close()
    logger.plot()
    savefig(os.path.join(args.checkpoint, 'log.eps'))

    print('Best acc:')
    print(best_acc)

import logging

# # 设置日志记录配置
# logging.basicConfig(
#     filename='every_epoch_record.log',  # 日志文件名
#     filemode='a',                       # 追加模式
#     format='%(asctime)s - %(message)s', # 日志格式，包含时间戳
#     level=logging.INFO                  # 记录信息级别
# )

# # 初始化日志记录器
# logger = logging.getLogger()

def train(train_loader1, model, criterion, optimizer, epoch, use_cuda,batch_logger):
    # switch to train mode
    model.train()
    # batch_logger = Logger(os.path.join(args.checkpoint, 'every_epoch_record.txt'), title='Batch Metrics')
    # batch_logger.set_names(['Batch', 'Loss', 'Top1', 'Top5'])


    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    bar = Bar('Processing', max=len(train_loader1))
    for batch_idx, (inputs, targets) in enumerate(train_loader1):
        # measure data loading time
        data_time.update(time.time() - end)

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)
        # print(f'traget{targets.shape}')
        # compute output
        outputs = model(inputs)
        # print(f'output{outputs.shape}')
        # breakpoint()
        loss = criterion(outputs, targets)
        # optimizer = optim.SGD(model.parameters(), lr=args.lr,  momentum=args.momentum, weight_decay=args.weight_decay)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))

        losses.update(loss.data.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

        batch_logger.append([f'train Batch {batch_idx + 1}', losses.avg, top1.avg, top5.avg])

    # Close the batch logger at the end of the epoch

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                    batch=batch_idx + 1,
                    size=len(train_loader1),
                    data=data_time.val,
                    bt=batch_time.val,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=losses.avg,
                    top1=top1.avg,
                    top5=top5.avg,
                    )
        bar.next()
        # logger.info(f'Batch {batch_idx + 1}/{len(train_loader1)}: Loss: {losses.avg:.4f}, Top1: {top1.avg:.4f}, Top5: {top5.avg:.4f}')
    # batch_logger.close()
    bar.finish()
    # with open(os.path.join(args.checkpoint, 'every_epoch_record.txt'), 'a') as f:
    #     f.write(f'Epoch {epoch+1}, Train Loss: {losses.avg:.4f}, Train Top1: {top1.avg:.4f}, Train Top5: {top5.avg:.4f}\n')

    return (losses.avg, top1.avg,top5.avg)



def test(val_loader1, model, criterion,optimizer,epoch, use_cuda,batch_logger):
    bar = Bar('Processing', max=len(val_loader1))

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for batch_idx, (image, targets) in enumerate(val_loader1):
            # measure data loading time
            data_time.update(time.time() - end)

            # image, target = image.cuda(non_blocking=True), target.cuda(non_blocking=True)
            if use_cuda:
                image, targets = image.cuda(), targets.cuda()
            image, targets = torch.autograd.Variable(image, volatile=True), torch.autograd.Variable(targets)

            # compute output
            output = model(image)
            loss = criterion(output, targets)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, targets.data, topk=(1, 5))
            losses.update(loss.item(), image.size(0))
            top1.update(prec1.item(), image.size(0))
            top5.update(prec5.item(), image.size(0))

            batch_logger.append([f'Val Batch {batch_idx + 1}', losses.avg, top1.avg, top5.avg])


            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # plot progress
            bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                        batch=batch_idx + 1,
                        size=len(val_loader1),
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
    # with open(os.path.join(args.checkpoint, 'every_epoch_record.txt'), 'a') as f:
    #     f.write(f'Epoch {epoch+1}, Val Loss: {losses.avg:.4f}, Val Top1: {top1.avg:.4f}, Val Top5: {top5.avg:.4f}\n')

    return (losses.avg, top1.avg, top5.avg)

def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))

def adjust_learning_rate(optimizer, epoch):
    global state
    if epoch in args.schedule:
        state['lr'] *= args.gamma
        for param_group in optimizer.param_groups:
            param_group['lr'] = state['lr']


if __name__ == '__main__':
    main()

#python main/train_val.py --gpu-id 2,3  --arch ResNetDCT_Upscaled_Static -d /data/ke/tiny-imagenet-200
