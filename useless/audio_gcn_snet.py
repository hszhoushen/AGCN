import sys

sys.path.append('../')
sys.path.append('./')
sys.path.append("../model/")
sys.path.append("../resnet-audio/")
sys.path.append("../resnet-image/")
sys.path.append("./data/")
sys.path.append('../utils/')

import torch
import time
import shutil
import torch.nn as nn
from torch.utils.data import DataLoader
from model.CVS_dataset import CVS_Audio

from model.network import GCN_audio
from model.network import SNet

from utils.arguments import arguments_parse
from resnet_audio import resnet50
from tensorboardX import SummaryWriter

from utils.arguments import dataset_selection

best_prec1 = 0

def main():

    global best_prec1

    # argsPaser
    argsPaser = arguments_parse()
    args = argsPaser.argsParser()
    print('args:', args)

    # data construction
    print('args.dataset_name:', args.dataset_name)
    data_selection = dataset_selection()
    data_dir, data_sample = data_selection.datsetSelection(args)


    # training set
    audio_dataset = CVS_Audio(args, data_dir, data_sample, data_type='train')
    audio_dataloader = DataLoader(dataset=audio_dataset, batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.num_threads)

    # test set
    test_audio_dataset = CVS_Audio(args, data_dir, data_sample, data_type='test')
    test_audio_dataloader = DataLoader(dataset=test_audio_dataset, batch_size=args.batch_size, shuffle=True,
                                       num_workers=args.num_threads)

    # create the model for classification
    audio_net = SNet(num_classes=args.num_classes)
    # state = torch.load(args.audio_net_weights)['model']
    # audio_net.load_state_dict(state)
    audio_net.fc3 = nn.Linear(2048, args.num_classes)
    audio_net = audio_net.cuda()

    # create gcn model
    audio_gcn_model = GCN_audio(args.num_classes).cuda()


    # configure gradient
    for param in audio_net.parameters():
        param.requires_grad = True
    for param in audio_gcn_model.parameters():
        param.requires_grad = True

    # best model name and latest model name
    latest_model_name =  '../weights/' + args.model_type + '_' + args.dataset_name + '_' + \
                         str(args.test_set_id) +'_latest' + '.pth.tar'

    best_model_name = '../weights/' + args.model_type + '_'+ args.dataset_name + '_' + \
                      str(args.test_set_id) + '_best' + '.pth.tar'

    # optimizer
    audio_net_params = list(audio_net.parameters())
    audio_gcn_model_params = list(audio_gcn_model.parameters())

    optimizer = torch.optim.SGD([{'params':audio_net_params},
                                 {'params':audio_gcn_model_params, 'lr':0.01}],
                                lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # writer
    writer = SummaryWriter(comment='audio_gcn_snet')

    # start training
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
        print("The best accuracy obtained during training is = {}".format(best_prec1))

        # save checkpoint
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'audio_net_state_dict': audio_net.state_dict(),
            'audio_gcn_model_state_dict': audio_gcn_model.state_dict(),
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

def net_train(args, audio_net, audio_gcn_model, data_loader, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # resnet for feature extraction
    audio_net.train()
    # gcn model for graph feature refinement
    audio_gcn_model.train()

    criterion = nn.CrossEntropyLoss().cuda()

    end = time.time()
    for i, data in enumerate(data_loader, 0):
        # measure data loading time
        data_time.update(time.time() - end)

        # clear optimizer
        optimizer.zero_grad()
        audio, label = data

        # to cuda
        label = label.cuda()
        audio = audio.type(torch.FloatTensor).cuda()

        output = audio_net(audio)
        # compute gradient and loss for SGD step
        loss = criterion(output, label)
        loss.backward()                 # loss propagation
        optimizer.step()                # optimizing

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output, label, topk=(1, 5))

        # print('aud.size(0):', aud.size(0))
        # print('loss:', loss.shape, loss)
        # print('prec1:', prec1.shape, prec1)

        losses.update(loss, audio.size(0))
        top1.update(prec1, audio.size(0))
        top5.update(prec5, audio.size(0))

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

            audio, label = data
            label = label.cuda()
            audio = audio.type(torch.FloatTensor).cuda()

            output = audio_net(audio)

            # compute gradient and loss for SGD step
            loss = criterion(output, label)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output, label, topk=(1, 5))

            losses.update(loss, audio.size(0))
            top1.update(prec1, audio.size(0))
            top5.update(prec5, audio.size(0))

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



# main function
if __name__ == '__main__':
    main()




