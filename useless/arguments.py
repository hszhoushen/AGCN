# Developed by Liguang Zhou, 2020.9.30

import argparse
from model.data_partition import esc_dataset_construction


Learning_rate = 0.01
best_prec1 = 0
Learning_rate_interval = 20

class arguments_parse(object):

    def argsParser(self):

        parser = argparse.ArgumentParser(description='AUDIO GCN')
        parser.add_argument('--csv_file', type=str, default='/data/lgzhou/dataset/ESC-50/meta/esc50.csv',
                            help='the path of the csv_file')
        parser.add_argument('--data_dir', type=str, default='/data/lgzhou/dataset/ESC-50/audio/',
                            help='image net weights')

        parser.add_argument('--bs', type=int, default=16, help='training batch size')
        parser.add_argument('--epochs', type=int, default=40, help='training epoch')
        parser.add_argument('--start-epoch', type=int, default=0, metavar='N',
                            help='manual epoch number (useful on restarts)')

        parser.add_argument('--gpu_ids', type=str, default='[0,1,2,3]', help='USING GPU IDS e.g.\'[0,4]\'')
        parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum (default: 0.9)')
        parser.add_argument('--audio_net_weights', type=str, default='../weights/audioset_audio_pretrain.pt',
                            help='audio net weights')

        parser.add_argument('--test_set_id', type=int, default=5, help='test set id 5')
        parser.add_argument('--dataset_name', type=str, default='esc50', help='dataset name')
        parser.add_argument('--num_threads', type=int, default=4, help='number of threads')
        parser.add_argument('--seed', type=int, default=1)

        parser.add_argument('--lr', '--learning-rate', type=float, default=Learning_rate,
                            metavar='LR', help='initial learning rate')
        parser.add_argument('--lri', '--learning-rate-interval', type=int, default=Learning_rate_interval,
                            metavar='LRI', help='learning rate interval')
        parser.add_argument('--print-freq', '-p', type=int, default=500,
                            metavar='N', help='print frequency (default: 10)')
        parser.add_argument('--weight-decay', '--wd', type=float, default=1e-4,
                            metavar='W', help='weight decay (default: 1e-4)')
        parser.add_argument('--num_classes', type=int, default=50,
                            help='class number (default: 50)')
        parser.add_argument('--model_type', type=str, default='audio_gcn',
                            help='choose the type of object model')
        parser.add_argument('--arch', '-a', type=str, default='resnet50',
                            metavar='ARCH', help='model architecture')
        parser.add_argument('--nodes_num', type=int, default=16,
                            help='num of nodes in sound texture graph')
        parser.add_argument('--fusion', type=bool, default=False,
                            help='the model is the  single model or fusion model')
        parser.add_argument('--mixup', type=bool,default=False,
                            help='the model use the mixup or not')
        parser.add_argument('--alpha', type=float, default=1.0,
                            help='the alpha parameter of mixup')

        args = parser.parse_args()

        return args

class dataset_selection(object):

    def datsetSelection(self, args):
        if args.dataset_name == 'US8K':
            csv_file = '/data/lgzhou/dataset/UrbanSound8K/metadata/UrbanSound8K.csv'
            data_dir = '/data/lgzhou/dataset/UrbanSound8K/audio/'

        elif args.dataset_name == 'esc10' or 'esc50':
            csv_file = '/data/lgzhou/dataset/ESC-50/meta/esc50.csv'
            data_dir = '/data/lgzhou/dataset/ESC-50/audio/'

        data_sample = esc_dataset_construction(csv_file, args.test_set_id, args.dataset_name)

        return data_dir, data_sample

