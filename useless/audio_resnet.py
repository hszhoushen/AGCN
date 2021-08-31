import sys

sys.path.append('../')
sys.path.append('./')
sys.path.append("../model/")
sys.path.append("../resnet-audio/")
sys.path.append("../resnet-image/")
sys.path.append("./data/")

import torch
import time
import shutil
import torch.nn as nn

from torch.utils.data import DataLoader
from model.CVS_dataset import CVS_Audio
from model.data_partition import esc50_construction

from model.network import GCN_audio
from resnet_audio import resnet50

from tensorboardX import SummaryWriter
from utils.arguments import arguments_parse


import torchvision.models as models
import numpy as np
import os

best_prec1 = 0

def main():

    global best_prec1

    # argsPaser
    argsPaser = arguments_parse()
    args = argsPaser.argsParser()


    # audio construction
    # data_sample[0] train data/    data_sample[1] label
    # data_sample[2] val data/      data_sample[3] label
    # data_sample[4] test data/     data_sample[5] label

    data_sample = esc50_construction(args.csv_file, test_set_id=args.test_set_id)


    # training set
    audio_dataset = CVS_Audio(args.data_dir, data_sample, data_type='train')
    # for idx, (data, image) in enumerate(audio_dataset):
    #     print(idx)
    audio_dataloader = DataLoader(dataset=audio_dataset, batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.num_threads)

    # test set
    test_audio_dataset = CVS_Audio(args.data_dir, data_sample, data_type='test')
    # for idx, (data, image) in enumerate(test_audio_dataset):
    #     print(idx)
    test_audio_dataloader = DataLoader(dataset=test_audio_dataset, batch_size=args.batch_size, shuffle=True,
                                       num_workers=args.num_threads)


    # create the model for classification
    audio_net = resnet50(num_classes=527)
    state = torch.load(args.audio_net_weights)['model']
    audio_net.load_state_dict(state)
    audio_net.fc = nn.Linear(2048, args.cls_num)
    audio_net = audio_net.cuda()

    # audio_gcn_model = GCN_audio(args.cls_num).cuda()


    # configure gradient
    for param in audio_net.parameters():
        param.requires_grad = True


    # best model name and latest model name
    latest_model_name =  '../weights/' + args.model_type + '_latest' + '.pth.tar'
    best_model_name = '../weights/' + args.model_type + '_best' + '.pth.tar'

    # optimizer
    audio_net_params = list(audio_net.parameters())

    optimizer = torch.optim.SGD(audio_net_params,
                                lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # writer
    writer = SummaryWriter(comment='audio_resnet')

    for epoch in range(args.start_epoch, args.epochs):

        # adjust learning rate before training
        adjust_learning_rate(args, optimizer, epoch)

        # network training
        train_losses, train_acc1, train_acc5 = net_train(args, audio_net,
                                                         audio_dataloader, optimizer, epoch)

        # test test
        test_losses, test_acc1, test_acc5 = net_validate(args, audio_net,
                                                         test_audio_dataloader, optimizer, epoch)

        # remember best prec@1 and save checkpoint
        is_best = test_acc1 > best_prec1

        best_prec1 = max(test_acc1, best_prec1)

        print("The best accuracy obtained during training is = {}".format(best_prec1))

        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'audio_net_state_dict': audio_net.state_dict(),
            'best_prec1': best_prec1,
        }, is_best, best_model_name, latest_model_name)


        # TensorboardX writer
        writer.add_scalar('LR/Train', args.lr, epoch)

        writer.add_scalar('Acc1/Train', train_acc1, epoch)
        writer.add_scalar('Acc1/Test', test_acc1, epoch)


        writer.add_scalar('Acc5/Train', train_acc5, epoch)
        writer.add_scalar('Acc5/Test', test_acc5, epoch)

        writer.add_scalar('Loss/Train', train_losses, epoch)
        writer.add_scalar('Loss/Test', test_losses, epoch)


    # close the SummaryWriter
    writer.close()


def save_checkpoint(state, is_best, best_model_name, latest_model_name):
    torch.save(state, latest_model_name)
    if is_best:
        shutil.copyfile(latest_model_name, best_model_name)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def net_train(args, audio_net, data_loader, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # resnet for feature extraction
    audio_net.train()

    criterion = nn.CrossEntropyLoss().cuda()

    end = time.time()
    for i, data in enumerate(data_loader, 0):
        # measure data loading time
        data_time.update(time.time() - end)

        # clear optimizer
        optimizer.zero_grad()
        aud, label = data

        # to cuda
        label = label.cuda()
        aud = aud.type(torch.FloatTensor).cuda()

        if(aud.shape[0] != 16):
            continue

        # output = audio_net(aud)
        # output[0].shape, 32x2048, output[1].shape, 32x527
        # print('output:', len(output), output[0].shape, output[1].shape)

        resnet_output = audio_net(aud)[1]            # here for extracting the event predictions
        # print('resnet_output:', resnet_output.shape)

        # compute gradient and loss for SGD step
        loss = criterion(resnet_output, label)
        loss.backward()                 # loss propagation
        optimizer.step()                # optimizing

        # measure accuracy and record loss
        prec1, prec5 = accuracy(resnet_output, label, topk=(1, 5))


        losses.update(loss, aud.size(0))
        top1.update(prec1, aud.size(0))
        top5.update(prec5, aud.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  #'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  #'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, i, len(data_loader), loss=losses, top1=top1, top5=top5))

    return losses.avg, top1.avg, top5.avg


def net_validate(args, audio_net, data_loader, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    audio_net.eval()

    criterion = nn.CrossEntropyLoss().cuda()

    end = time.time()

    with torch.no_grad():

        for i, data in enumerate(data_loader, 0):
            # measure data loading time
            data_time.update(time.time() - end)

            aud, label = data
            label = label.cuda()
            aud = aud.type(torch.FloatTensor).cuda()

            if (aud.shape[0] != 16):
                continue

            resnet_output = audio_net(aud)[1]            # here for extracting the event predictions
            # print('resnet_output:', resnet_output.shape)

            # compute gradient and loss for SGD step
            loss = criterion(resnet_output, label)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(resnet_output, label, topk=(1, 5))

            losses.update(loss, aud.size(0))
            top1.update(prec1, aud.size(0))
            top5.update(prec5, aud.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                      #'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      #'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                       epoch, i, len(data_loader), loss=losses, top1=top1, top5=top5))

    return losses.avg, top1.avg, top5.avg


def adjust_learning_rate(args, optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 10 epochs"""
    lr = args.lr * (0.1 ** (epoch // args.lri))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    print('args.lr after adjustment:', param_group['lr'])


def my_forward(Faudio):

    # print('Faudio:', Faudio.shape)

    N = Faudio.shape[0]
    C = Faudio.shape[1]
    W = Faudio.shape[2]
    H = Faudio.shape[3]

    K = 16
    if(N < K):
        print('N:', N)
        return False

    Fre = torch.empty(N, W, H).cuda()

    # acquire Fre (N,H,W) from Frgb (N,C,H,W)
    for channel in range(C):
        Fre = torch.add(Faudio[:,channel,:,:], Fre)
    # print('Fre:', Fre.shape, Fre[0,:,:])

    # Acquire the top K index of Frgb
    feature = Fre.view(N, H*W)
    # print('feature:', feature.shape)
    sorted, indices = torch.sort(feature, descending=True)
    # print('sorted:', sorted.shape, sorted)
    # print('indices:', indices.shape, indices)

    # feature = feature.view(x.size(0), -1)
    # print('feature:', feature.shape)


    topK_index = indices[:, 0:K]
    # print('topK_index:', topK_index.shape, topK_index)

    nodes = []
    rows = []
    columns = []
    cnt = 0



    # shape of topK_index is (N, nodes)
    for i in range(K):      # lop from one image to other
        for j in range(K):  # loop from one node to other
        # for index in topK_index[0, :]:
            # print('i:', i, 'j:', j)
            index = topK_index[i, j]
            rows.append(index / H - 1)

            if index % H == 0:      # e.g., 36 (8,3)
                columns.append(H-1)
            else:
                columns.append(index % H - 1)

            # print('index:', index, 'row:', rows[cnt], 'column:', columns[cnt])

            # switch the feature from 1x1024 to 1024x1
            nodes.append(Faudio[i, :, rows[cnt], columns[cnt]].reshape(-1,1))

            cnt = cnt + 1

        imgs = torch.stack(nodes)
        # nodes = []
    # print('imgs:', imgs.shape)


    #y_predicts = gcn_model(imgs)
    #y_targets = torch.randint(1,10, [K]).cuda()
    #print('shape of y_predicts:', y_predicts.shape)

    #crossentropyloss_output = crossentropyloss(y_predicts, y_targets)
    #print('crossentropyloss_output:', crossentropyloss_output)

    return imgs


# main function
if __name__ == '__main__':
    main()

def my_forward_test(Faudio):

    #gcn_model = GCN_audio().cuda()
    #gcn_model.eval()
    #print('gcn_model:', gcn_model)
    crossentropyloss = nn.CrossEntropyLoss().cuda()

    print('Faudio:', Faudio.shape)

    N = Faudio.shape[0]
    C = Faudio.shape[1]
    W = Faudio.shape[2]
    H = Faudio.shape[3]

    Fre = torch.empty(N, W, H).cuda()

    # acquire Fre (N,H,W) from Frgb (N,C,H,W)
    for channel in range(C):
        Fre = torch.add(Faudio[:,channel,:,:], Fre)
    print('Fre:', Fre.shape, Fre[0,:,:])

    # Acquire the top K index of Frgb
    feature = Fre.view(N, H*W)
    print('feature:', feature.shape)
    sorted, indices = torch.sort(feature, descending=True)
    print('sorted:', sorted.shape, sorted)
    print('indices:', indices.shape, indices)

    # feature = feature.view(x.size(0), -1)
    # print('feature:', feature.shape)

    K = 16
    topK_index = indices[:, 0:K]
    print('topK_index:', topK_index.shape, topK_index)

    nodes = []
    rows = []
    columns = []
    cnt = 0

    # shape of topK_index is (N, nodes)
    for i in range(K):      # lop from one image to other
        for j in range(K):  # loop from one node to other
        # for index in topK_index[0, :]:

            index = topK_index[i, j]
            rows.append(index / H - 1)
            columns.append(index % H - 1)
            print('row:', rows[cnt], 'column:', columns[cnt])

            # switch the feature from 1x1024 to 1024x1
            nodes.append(Faudio[i, :, rows[cnt], columns[cnt]].reshape(-1,1))

            cnt = cnt + 1

        imgs = torch.stack(nodes)
        # nodes = []
    print('imgs:', imgs.shape)


    return feature


