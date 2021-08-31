import sys

sys.path.append('../')
sys.path.append('./')
sys.path.append("../model/")
sys.path.append("../resnet-audio/")
sys.path.append("../resnet-image/")
sys.path.append("./data/")

import os
import torch
import time
import shutil
from torch.utils.data import DataLoader
import argparse
from model.CVS_dataset_noise import CVS_Audio
from model.network import GCN_audio_top_med
from resnet_audio import resnet50
import torch.nn as nn
from model.network import Audio_Classifier
from tensorboardX import SummaryWriter
from utils.arguments import arguments_parse
from utils.arguments import dataset_selection

import torchvision.models as models


best_prec1 = 0

def main():

    global best_prec1

    # argsPaser
    argsPaser = arguments_parse()
    args = argsPaser.argsParser()
    print('args:', args)

    # esc50 construction
    print('args.dataset_name:', args.dataset_name)
    data_selection = dataset_selection()
    data_dir, data_sample = data_selection.datsetSelection(args)

    # training set
    # audio_dataset = CVS_Audio(args, data_dir, data_sample, data_type='train')
    # audio_dataloader = DataLoader(dataset=audio_dataset, batch_size=args.batch_size, shuffle=True,
    #                              num_workers=args.num_threads)

    # test set
    test_audio_dataset = CVS_Audio(args, data_dir, data_sample, data_type='test')
    test_audio_dataloader = DataLoader(dataset=test_audio_dataset, batch_size=args.batch_size, shuffle=True,
                                       num_workers=args.num_threads)

    # best model name and latest model name
    best_model_name = '/data/lgzhou/esc/weights/esc50/audio_gcn_med_5_16_epoch_10_best.pth.tar'

    # load the pretrained model
    audio_gcn_state = torch.load(best_model_name)

    # create the model for classification
    audio_net = resnet50(num_classes=527)
    state = audio_gcn_state['audio_net_state_dict']
    audio_net.load_state_dict(state)
    audio_net = audio_net.cuda()

    # audio gcn model
    audio_gcn_model = GCN_audio_top_med(16).cuda()
    state = audio_gcn_state['audio_gcn_model_state_dict']
    audio_gcn_model.load_state_dict(state)
    audio_gcn_model = audio_gcn_model.cuda()

    # for testing
    audio_net.eval()
    audio_gcn_model.eval()
   
    # create the model for classification
    ''' 
    audio_net = resnet50(num_classes=527)
    state = torch.load(args.audio_net_weights)['model']
    audio_net.load_state_dict(state)
    audio_net = audio_net.cuda()

    # audio gcn model
    audio_gcn_model = GCN_audio_top_med(16).cuda()

    # configure gradient
    for param in audio_net.parameters():
        param.requires_grad = True
    for param in audio_gcn_model.parameters():
        param.requires_grad = True

    # optimizer
    audio_net_params = list(audio_net.parameters())
    audio_gcn_model_params = list(audio_gcn_model.parameters())

    optimizer = torch.optim.SGD([{'params':audio_net_params},
                                 {'params':audio_gcn_model_params, 'lr':0.01}],
                                lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # writer
    # writer = SummaryWriter(comment=args.model_type)

    # training
    for epoch in range(args.start_epoch, args.epochs):

        # adjust learning rate before training
        adjust_learning_rate(args, optimizer, epoch)

        # network training
        train_losses, train_acc1, train_acc5 = net_train(args, audio_net, audio_gcn_model,
                                                         audio_dataloader, optimizer, epoch)

        # test test
        test_losses, test_acc1, test_acc5 = net_validate(args, audio_net, audio_gcn_model,
                                                         test_audio_dataloader, optimizer, epoch)

        # remember best prec@1 and save checkpoint
        is_best = test_acc1 > best_prec1
        best_prec1 = max(test_acc1, best_prec1)
        print("The best test accuracy obtained during training is = {}".format(best_prec1))


        # best model name and latest model name
        model_dir = os.path.join('../weights', args.dataset_name)
        last_model_name = args.model_type + '_' + str(args.test_set_id) + \
                          '_epoch_' + str(epoch) + '.pth.tar'
        best_model_name = args.model_type + '_' + str(args.test_set_id) + \
                          '_epoch_' + str(epoch) + '_best' + '.pth.tar'

        last_model_path = os.path.join(model_dir,  last_model_name)
        best_model_path = os.path.join(model_dir, best_model_name)

        print('last:', last_model_path)
        print('best:', best_model_path)

        if (epoch % 20 == 0 and epoch > 0):
            # save checkpoints
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'audio_net_state_dict': audio_net.state_dict(),
                'audio_gcn_model_state_dict':audio_gcn_model.state_dict(),
                'best_prec1': best_prec1,
            }, is_best, best_model_path, last_model_path)
'''

        # TensorboardX writer
        # writer.add_scalar('LR/Train', args.lr, epoch)
        # writer.add_scalar('Acc1/Train', train_acc1, epoch)
        # writer.add_scalar('Acc1/Test', test_acc1, epoch)
        # writer.add_scalar('Acc5/Train', train_acc5, epoch)
        # writer.add_scalar('Acc5/Test', test_acc5, epoch)
        # writer.add_scalar('Loss/Train', train_losses, epoch)
        # writer.add_scalar('Loss/Test', test_losses, epoch)

    # close the SummaryWriter
    # writer.close()

def net_train(args, audio_net, audio_gcn_model, data_loader, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    audio_net.train()
    audio_gcn_model.train()
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

        # output = audio_net(aud)
        # output[0].shape, bs x 2048
        # output[1].shape, bs x args.num_classes

        output = audio_net(aud)

        Faudio = output[2]
        # print('Faudio.shape:', Faudio.shape)

        gcn_output = audio_gcn_model(Faudio)

        # compute gradient and loss for SGD step
        loss = criterion(gcn_output, label)
        loss.backward()                 # loss propagation
        optimizer.step()                # optimizing

        # measure accuracy and record loss
        prec1, prec5 = accuracy(gcn_output, label, topk=(1, 5))

        # print('aud.size(0):', aud.size(0))
        # print('loss:', loss.shape, loss)
        # print('prec1:', prec1.shape, prec1)

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


def net_validate(args, audio_net, audio_gcn_model, data_loader, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    audio_net.eval()
    audio_gcn_model.eval()

    criterion = nn.CrossEntropyLoss().cuda()

    end = time.time()

    with torch.no_grad():

        for i, data in enumerate(data_loader, 0):
            # measure data loading time
            data_time.update(time.time() - end)

            aud, label = data
            label = label.cuda()
            aud = aud.type(torch.FloatTensor).cuda()

            output = audio_net(aud)
            # Faudio = output[2]
            Faudio = output[4]

            gcn_output = audio_gcn_model(Faudio)

            # compute gradient and loss for SGD step
            loss = criterion(gcn_output, label)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(gcn_output, label, topk=(1, 5))

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

def save_checkpoint(state, is_best, best_model_path, last_model_path):
    torch.save(state, last_model_path)
    if is_best:
        shutil.copyfile(last_model_path, best_model_path)


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
        correct_k = correct[:k].contiguous().view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res





def adjust_learning_rate(args, optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 10 epochs"""
    lr = args.lr * (0.1 ** (epoch // args.lri))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr



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


