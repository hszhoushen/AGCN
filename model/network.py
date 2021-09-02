# by Liguang Zhou, 2020.9.30
from model.graph_init import Graph_Init

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn import BatchNorm2d

from resnet_audio.resnet_audio import resnet50, resnet18

import torch.utils.model_zoo as model_zoo
import torchvision.models as models

# 默认的resnet网络，已预训练
model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

def num_flat_features(self, x):
    size = x.size()[1:]  # all dimensions except the batch dimension
    num_features = 1
    for s in size:
        num_features *= s
    return num_features

def audio_forward(K, Faudio):

    N = Faudio.shape[0]
    C = Faudio.shape[1]
    W = Faudio.shape[2]
    H = Faudio.shape[3]

    Fre = torch.empty(N, W, H).cuda()

    # acquire Fre (N,H,W) from Frgb (N,C,H,W)
    for channel in range(C):
        Fre = torch.add(Faudio[:, channel, :, :], Fre)
    # print('Fre:', Fre.shape, Fre[0,:,:])

    # Acquire the top K index of Frgb
    feature = Fre.view(N, H * W)
    # print('feature:', feature.shape)

    sorted, indices = torch.sort(feature, descending=True)
    # print('sorted:', sorted.shape, sorted)
    # print('indices:', indices.shape, indices)

    topK_index = indices[:, 0:K]
    # print('topK_index:', topK_index.shape, topK_index)

    nodes = []
    sound_batch = []
    rows = []
    columns = []
    cnt = 0

    # shape of topK_index is (N, nodes)
    for i in range(N):  # lop from one image to other based on batch size

        for j in range(K):  # loop from one node to other
            # for index in topK_index[0, :]:
            index = topK_index[i, j]
            if index < H:
                row = index // H
            else:
                row = index // H - 1
            rows.append(row)

            if index % H == 0:  # e.g., 36 (8,3)
                columns.append(torch.tensor(H - 1).cuda())
            else:
                columns.append(index % H - 1)


            # switch the feature from 1x1024 to 1024x1
            node = Faudio[i, :, rows[cnt], columns[cnt]].reshape(-1, 1)
            nodes.append(node)
            cnt = cnt + 1

        sound = torch.stack(nodes)
        nodes = []
        sound_batch.append(sound)


    # Adjacency matrix calculation
    sound_graph_construction = Graph_Init(K, N, rows, columns)
    Lnormtop, Lnormmed = sound_graph_construction.Dynamic_Lnorm()

    sound_batch = torch.stack(sound_batch)

    return sound_batch, Lnormtop.cuda(), Lnormmed.cuda()


def audio_med_forward(nodes_num, Faudio):
    N = Faudio.shape[0]  # batch size
    C = Faudio.shape[1]
    W = Faudio.shape[2]
    H = Faudio.shape[3]

    #This line causes nonreproducibility....
    #refer to https://pytorch.org/docs/1.4.0/notes/randomness.html
    # There are some PyTorch functions that use CUDA functions that can be a
    # source of non-determinism. One class of such CUDA functions are atomic
    # operations, in particular atomicAdd, where the order of parallel additions
    # to the same value is undetermined and, for floating-point variables, a source
    #  of variance in the result. PyTorch functions that use atomicAdd in the forward
    #  include torch.Tensor.index_add_(), torch.Tensor.scatter_add_(), torch.bincount().
    #Maybe in torch.add, there is some above-mentioned operations!

    # acquire Fre (N,H,W) from Frgb (N,C,H,W)
    Fre = torch.empty(N, W, H).cuda()
    for channel in range(C):
        Fre = torch.add(Faudio[:, channel, :, :], Fre)
    # Fre = torch.sum(Faudio, dim=1)

    # Acquire the top K index of Frgb
    feature = Fre.view(N, H * W)
    # print('feature:', feature.shape)
    sorted, indices = torch.sort(feature, descending=True)
    # print('sorted:', sorted.shape, sorted)
    # print('indices:', indices.shape, indices)

    # feature = feature.view(x.size(0), -1)
    # print('feature:', feature.shape)

    topK_index = indices[:, 0:nodes_num]
    # print("topK_index",topK_index)
    # print('topK_index:', topK_index.shape, topK_index)
    avg_index = indices[:, (H * W // 2 - nodes_num // 2 - 1):(H * W // 2 - nodes_num // 2 - 1 + nodes_num)]
    # print('avg_index:', avg_index.shape, avg_index)

    nodes = []
    sound_batch = []
    rows = []
    columns = []
    cnt = 0

    # shape of topK_index is (N, nodes)
    for i in range(N):  # lop from one image to other

        # top K
        for j in range(int(nodes_num/2)):  # loop from one node to other

            index = topK_index[i, j]
            # print('top K:', index) #debug
            if index < H:
                row = index // H
            else:
                # row = torch.tensor(index // H - 1).cuda()
                row = torch.tensor((index-1)//H).cuda()
            rows.append(row)

            if index % H == 0:  # e.g., 36 (8,3)
                column = torch.tensor(H-1).cuda()
            else:
                column = index % H - 1
            columns.append(column)

            # switch the feature from 1x1024 to 1024x1
            node = Faudio[i, :, rows[cnt], columns[cnt]].reshape(-1, 1)
            nodes.append(node)
            cnt = cnt + 1

        # median K
        for k in range(int(nodes_num/2)):
            index_avg = avg_index[i, k]
            # print('median K:', index_avg) #debug
            if index_avg < H:
                row = index_avg // H
            else:
                # row = torch.tensor(index_avg // H - 1).cuda()
                row = torch.tensor((index_avg-1)//H).cuda()
            rows.append(row)

            if index_avg % H == 0:
                column = torch.tensor(H-1).cuda()
            else:
                column = index_avg % H - 1
            columns.append(column)

            node = Faudio[i, :, rows[cnt], columns[cnt]].reshape(-1, 1)
            nodes.append(node)
            cnt = cnt + 1

        # top K
        for j in range(int(nodes_num/2), nodes_num):  # loop from one node to other

            index = topK_index[i, j]
            # print('top K:', index) #debug
            if index < H:
                row = index // H
            else:
                # row = torch.tensor(index // H - 1).cuda()
                row = torch.tensor((index-1)//H).cuda()
            rows.append(row)

            if index % H == 0:  # e.g., 36 (8,3)
                column = torch.tensor(H-1).cuda()
            else:
                column = index % H - 1
            columns.append(column)

            # switch the feature from 1x1024 to 1024x1
            node = Faudio[i, :, rows[cnt], columns[cnt]].reshape(-1, 1)
            nodes.append(node)

            cnt = cnt + 1

        # median K
        for k in range(int(nodes_num/2), nodes_num):
            index_avg = avg_index[i, k]
            # print('median K:', index_avg) #debug
            if index_avg < H:
                row = index_avg // H
            else:
                # row = torch.tensor(index_avg // H - 1).cuda()
                row = torch.tensor((index_avg-1)//H).cuda()
            rows.append(row)

            if index_avg % H == 0:
                column = torch.tensor(H - 1).cuda()
            else:
                column = index_avg % H - 1
            columns.append(column)

            node = Faudio[i, :, rows[cnt], columns[cnt]].reshape(-1, 1)
            nodes.append(node)
            cnt = cnt + 1

        sound = torch.stack(nodes)
        nodes = []
        sound_batch.append(sound)

    for row in rows:
        if row < 0:
            print(row.data)
    for column in columns:
        if column < 0:
            print(column.data)

    # testing return data
    # for (row,column) in zip(rows,columns):
    #     print(row.data, column.data)

    print('sound.shape:', sound.shape)
    sound_batch = torch.stack(sound_batch)
    print('sound_batch.shape:', sound_batch.shape)
    print('rows.shape:', rows.shape)
    print('columns.shape:', columns.shape)

    # Adjacency matrix calculation
    sound_graph_construction = Graph_Init(nodes_num, N, rows, columns)
    Lnormtop, Lnormmed = sound_graph_construction.Dynamic_Lnorm()

    return sound_batch, Lnormtop.cuda(), Lnormmed.cuda(), rows, columns

def img_med_forward(nodes_num, Faudio, sound_graph_construction):
    N = Faudio.shape[0]  # batch size
    W = Faudio.shape[2]
    H = Faudio.shape[3]

    # acquire Fre (N,H,W) from Frgb (N,C,H,W)
    Fre = torch.sum(Faudio, dim=1)

    # Acquire the top K index of Fre
    feature = Fre.view(N, H * W)
    # print('feature:', feature.shape, feature)

    sorted, indices = torch.sort(feature, descending=True)
    # print('sorted:', sorted.shape, sorted)
    # print('indices:', indices.shape, indices)

    topK_index = indices[:, 0:nodes_num]
    avgK_index = indices[:, (H * W // 2 - nodes_num // 2 - 1):(H * W // 2 - nodes_num // 2 - 1 + nodes_num)]
    # print('topK_index:', topK_index.shape)
    # print('avg_index:', avgK_index.shape)
    scene_index = torch.cat([topK_index, avgK_index], dim=1)
    # print('scene_index:', scene_index.shape)

    rows = scene_index // H
    columns = scene_index % H

    nodes = []
    sound_batch = []

    # shape of topK_index is (N, nodes_num*2)
    for i in range(N):                          # lop from one image to other
        for j in range(int(rows.shape[1])):     # loop from one node to other
            # switch the feature from 1x1024 to 1024x1
            node = Faudio[i, :, rows[i, j], columns[i, j]].reshape(-1, 1)
            nodes.append(node)

        # convert the list to torch tensor
        sound = torch.stack(nodes)
        # clear nodes
        nodes = []
        # save the torch tensor to list
        sound_batch.append(sound)


    sound_batch = torch.stack(sound_batch)
    # Adjacency matrix calculation
    Lnormtop, Lnormmed = sound_graph_construction.Dynamic_Lnorm(rows, columns)

    return sound_batch, Lnormtop.cuda(), Lnormmed.cuda(), rows, columns


class ConvBNReLU(nn.Module):
    def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1, *args, **kwargs):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_chan,
                              out_chan,
                              kernel_size=ks,
                              stride=stride,
                              padding=padding,
                              bias=False)
        self.bn = BatchNorm2d(out_chan)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class MultiheadAttention(nn.Module):

    def __init__(self, input_dim, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be 0 modulo number of heads."

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Stack all weight matrices 1...h together for efficiency
        # Note that in many implementations you see "bias=False" which is optional
        self.qkv_proj = nn.Linear(input_dim, 3*embed_dim)
        self.o_proj = nn.Linear(embed_dim, embed_dim)
        # self.conv = ConvBNReLU(input_dim, 3*embed_dim, ks=1)

        self._reset_parameters()

    def scaled_dot_product(q, k, v, mask=None):
        d_k = q.size()[-1]
        attn_logits = torch.matmul(q, k.transpose(-2, -1))
        attn_logits = attn_logits / math.sqrt(d_k)
        if mask is not None:
            attn_logits = attn_logits.masked_fill(mask == 0, -9e15)
        attention = F.softmax(attn_logits, dim=-1)
        values = torch.matmul(attention, v)
        return values, attention

    def _reset_parameters(self):
        # Original Transformer initialization, see PyTorch documentation
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        self.qkv_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.o_proj.weight)
        self.o_proj.bias.data.fill_(0)


    def forward(self, x, mask=None, return_attention=False):
        print('x.size():', x.size())
        batch_size, seq_length, embed_dim, _ = x.size()
        qkv = self.qkv_proj(x)
        print('qkv.shape:', qkv.shape)
        # Separate Q, K, V from linear output
        qkv = qkv.reshape(batch_size, seq_length, self.num_heads, 3*self.head_dim)
        qkv = qkv.permute(0, 2, 1, 3) # [Batch, Head, SeqLen, Dims]
        q, k, v = qkv.chunk(3, dim=-1)
        print('q.shape:', q.shape)
        print('k.shape:', k.shape)
        print('v.shape:', v.shape)

        # Determine value outputs
        # values, attention = self.scaled_dot_product(q, k, v, mask=mask)
        # values = values.permute(0, 2, 1, 3) # [Batch, SeqLen, Head, Dims]
        # values = values.reshape(batch_size, seq_length, embed_dim)
        # o = self.o_proj(values)
        #
        # if return_attention:
        #     return o, attention
        # else:
        #     return o

class AttentionFusionModule(nn.Module):
    def __init__(self, in_chan, out_chan, *args, **kwargs):
        super(AttentionFusionModule, self).__init__()
        self.conv = ConvBNReLU(in_chan, out_chan, ks=1, stride=1, padding=0)
        self.conv_atten = nn.Conv2d(out_chan, out_chan, kernel_size=1, bias=False)
        self.bn_atten = BatchNorm2d(out_chan)
        self.sigmoid_atten = nn.Sigmoid()

    def forward(self, feat16, feat32):
        feat32_up = F.interpolate(feat32, feat16.size()[2:], mode='nearest')
        #print('feat32_up:', feat32_up.shape)
        fcat = torch.cat([feat16, feat32_up], dim=1)
        #print('fcat.shape:', fcat.shape)
        feat = self.conv(fcat)
        #print('feat.shape:', feat.shape)

        atten = F.avg_pool2d(feat, feat.size()[2:])
        #print('atten.shape:', atten.shape)
        atten = self.conv_atten(atten)
        #print('atten.shape:', atten.shape)

        atten = self.bn_atten(atten)
        #print('atten.shape:', atten.shape)

        atten = self.sigmoid_atten(atten)
        #print('atten.shape:', atten.shape)

        return atten

class AFM(nn.Module):
    def __init__(self, args, pretrain=True):
        super(AFM, self).__init__()

        if (args.dataset_name == 'Places365-7'):
            # no pretrained model
            # img_resnet50 = resnet50(num_classes=1000)
            # model_arch = 'resnet50'
            # print('Loading the pretrained model from the weights {%s}!' % model_urls[model_arch])
            # img_resnet50.load_state_dict(model_zoo.load_url(model_urls[model_arch]))

            img_resnet50 = resnet50(num_classes=14)
            model_file = './weights/resnet50_best_res50.pth.tar'
            print('Loading the pretrained model from the weights {%s}!' % model_file)
            checkpoint = torch.load(model_file)
            state_dict = {str.replace(k, 'module.', ''): v for k, v in checkpoint['state_dict'].items()}
            img_resnet50.load_state_dict(state_dict)

            img_resnet50.cuda()
            for param in img_resnet50.parameters():
                param.requires_grad = True

        if (args.dataset_name == 'Places365-14'):
            # pretrained model
            img_resnet50 = resnet50(num_classes=14)
            model_file = './weights/resnet50_best_res50.pth.tar'
            print('Loading the pretrained model from the weights {%s}!' % model_file)
            checkpoint = torch.load(model_file)
            state_dict = {str.replace(k, 'module.', ''): v for k, v in checkpoint['state_dict'].items()}
            img_resnet50.load_state_dict(state_dict)
            img_resnet50.cuda()
            for param in img_resnet50.parameters():
                param.requires_grad = True

        elif(args.dataset_name=='MIT67'):
            # no pretrained model
            img_resnet50 = resnet50(num_classes=1000)
            model_arch = 'resnet50'
            print('Loading the pretrained model from the weights {%s}!' % model_urls[model_arch])
            img_resnet50.load_state_dict(model_zoo.load_url(model_urls[model_arch]))
            img_resnet50.cuda()

            for param in img_resnet50.parameters():
                param.requires_grad = True


        # img_resnet50.fc = nn.Linear(2048, args.num_classes)
        # img_resnet50.cuda()

        self.img_resnet50 = img_resnet50
        self.afm = AttentionFusionModule(3072, 1024)
        self.conv_head32 = ConvBNReLU(2048, 1024, ks=3, stride=1, padding=1)
        self.conv_head16 = ConvBNReLU(1024, 1024, ks=3, stride=1, padding=1)

        # self.conv_head1 = ConvBNReLU(512, 512, ks=3, stride=1, padding=1)
        # self.sam = StripAttentionModule(512, 512)
        # self.conv_head2 = ConvBNReLU(512, 512, ks=3, stride=1, padding=1)

    def forward(self, x):
        resnet_output, feat8, feat16, feat32 = self.img_resnet50(x)
        # print('feat8.shape:', feat8.shape)
        # print('feat16.shape:', feat16.shape)
        # print('feat32.shape:', feat32.shape)

        h8, w8 = feat8.size()[2:]
        h16, w16 = feat16.size()[2:]

        # Attention Fusion Module
        feat16 = self.conv_head16(feat16)       # C: 1024->1024
        # print('feat16.shape:', feat16.shape)

        atten = self.afm(feat16, feat32)        # C: 1024, 2048->3072->1024
        # print('atten.shape:', atten.shape)

        feat32 = self.conv_head32(feat32)       # C: 2048->1024
        # print('feat32.shape:', feat32.shape)

        feat32 = torch.mul(feat32, atten)       # C: 1024->1024
        # print('feat32.shape:', feat32.shape)

        feat32_up = F.interpolate(feat32, (h16, w16), mode='nearest')
        # print('feat32_up.shape:', feat32_up.shape)

        feat16 = torch.mul(feat16, (1 - atten))
        # print('feat16.shape:', feat16.shape)

        feat16_sum = feat16 + feat32_up

        # feature smoothness
        # feat16_sum = self.conv_head1(feat16_sum)
        # print('feat16_sum.shape:', feat16_sum.shape)
        #
        # # Strip Attention Module
        # feat16_sum = self.sam(feat16_sum)
        # print('feat16_sum.shape:', feat16_sum.shape)
        # feat16_up = F.interpolate(feat16_sum, (h8, w8), mode='nearest')
        # print('feat16_up.shape:', feat16_up.shape)
        # feat16_up = self.conv_head2(feat16_up)
        # print('feat16_up.shape:', feat16_up.shape)
        return feat16_sum, resnet_output


class PyramidAttentionFusionModule(nn.Module):
    def __init__(self, in_chan, out_chan, *args, **kwargs):
        super(PyramidAttentionFusionModule, self).__init__()
        self.conv1 = ConvBNReLU(in_chan, int(out_chan/2), ks=1, stride=1, padding=0)
        self.conv2 = ConvBNReLU(in_chan, int(out_chan/2), ks=3, padding=1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.maxpool = nn.AdaptiveMaxPool2d(1)

        self.conv3 = nn.Conv2d(out_chan, out_chan, kernel_size=1, bias=False)
        self.conv4 = nn.Conv2d(out_chan, out_chan, kernel_size=3, padding=1, bias=False)

        self.bn3 = BatchNorm2d(out_chan)
        self.sigmoid_atten = nn.Sigmoid()

    def forward(self, Fm1, Fm2, Fm3):
        Fm1_down = F.interpolate(Fm1, scale_factor=0.5)
        Fm3_up = F.interpolate(Fm3, Fm2.size()[2:], mode='nearest')


        fcat = torch.cat([Fm2, Fm3_up, Fm1_down], dim=1)
        # dimension reduction
        # print('fcat.shape:', fcat.shape)
        feat_3_3 = self.conv1(fcat)
        feat_1_1 = self.conv2(fcat)
        # print('feat_3_3.shape:', feat_3_3.size(), feat_3_3.size()[2:])

        feat = torch.cat([feat_1_1, feat_3_3], dim=1)
        # print('feat.shape:', feat.shape)

        atten = F.avg_pool2d(feat, feat.size()[2:])
        # print('atten.shape:', atten.shape)

        atten_max = self.maxpool(atten)
        # print('atten_max.shape:', atten.shape)

        atten_avg = self.avgpool(atten)
        # print('atten_avg.shape:', atten.shape)
        atten = torch.add(atten_avg, atten_max)
        # print('atten.shape:', atten.shape)

        atten = self.bn3(atten)
        # print('atten.shape:', atten.shape)

        atten = self.sigmoid_atten(atten)
        # print('atten.shape:', atten.shape)

        return atten

class SpatialAttention(nn.Module):
    def __init__(self, in_channels=4, out_channels=1, kernel_size=3):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(1024, 1, 3, padding=1, bias=False)     # 3x3 conv, padding=1
        self.conv2 = nn.Conv2d(1024, 1, 1, padding=0, bias=False)     # 1x1 conv, padding=0
        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # print('x.shape:', x.shape)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        # print('avg_out.shape:', avg_out.shape)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        # print('max_out.shape:', max_out.shape)
        out_1_1 = self.conv1(x)
        # print('out_1_1.shape:', out_1_1.shape)
        out_3_3 = self.conv2(x)
        # print('out_3_3.shape:', out_3_3.shape)

        x = torch.cat([avg_out, max_out, out_1_1, out_3_3], dim=1)
        # print('x.shape:', x.shape)

        x = self.conv3(x)
        # print('x.shape:', x.shape)

        return self.sigmoid(x)

class PAFM(nn.Module):
    def __init__(self, args, pretrain=True):
        super(PAFM, self).__init__()

        if (args.dataset_name == 'Places365-7' or args.dataset_name == 'SUNRGBD'):
            # no pretrained model
            # img_resnet50 = resnet50(num_classes=1000)
            # model_arch = 'resnet50'
            # print('Loading the pretrained model from the weights {%s}!' % model_urls[model_arch])
            # img_resnet50.load_state_dict(model_zoo.load_url(model_urls[model_arch]))

            img_resnet50 = resnet50(num_classes=14)
            model_file = './weights/resnet50_best_res50.pth.tar'
            print('Loading the pretrained model from the weights {%s}!' % model_file)
            checkpoint = torch.load(model_file)
            state_dict = {str.replace(k, 'module.', ''): v for k, v in checkpoint['state_dict'].items()}
            img_resnet50.load_state_dict(state_dict)

            img_resnet50.cuda()
            for param in img_resnet50.parameters():
                param.requires_grad = True

        if (args.dataset_name == 'Places365-14'):
            # pretrained model
            img_resnet50 = resnet50(num_classes=14)
            model_file = './weights/resnet50_best_res50.pth.tar'
            print('Loading the pretrained model from the weights {%s}!' % model_file)
            checkpoint = torch.load(model_file)
            state_dict = {str.replace(k, 'module.', ''): v for k, v in checkpoint['state_dict'].items()}
            img_resnet50.load_state_dict(state_dict)
            img_resnet50.cuda()
            for param in img_resnet50.parameters():
                param.requires_grad = True

        elif(args.dataset_name=='MIT67'):
            # no pretrained model
            img_resnet50 = resnet50(num_classes=1000)
            model_arch = 'resnet50'
            print('Loading the pretrained model from the weights {%s}!' % model_urls[model_arch])
            img_resnet50.load_state_dict(model_zoo.load_url(model_urls[model_arch]))
            img_resnet50.cuda()

            for param in img_resnet50.parameters():
                param.requires_grad = True

        self.img_resnet50 = img_resnet50
        self.pafm = PyramidAttentionFusionModule(3072, 1024)

        self.conv_Fm1 = ConvBNReLU(512, 1024, ks=3, stride=1, padding=1)
        self.conv_Fm2 = ConvBNReLU(1024, 1024, ks=3, stride=1, padding=1)
        self.conv_Fm3 = ConvBNReLU(2048, 1024, ks=3, stride=1, padding=1)

        self.sa1 = SpatialAttention()
        self.sa2 = SpatialAttention()
        self.sa3 = SpatialAttention()


    def forward(self, x):
        resnet_output, Fm1, Fm2, Fm3 = self.img_resnet50(x)

        hFm2, wFm2 = Fm2.size()[2:]

        Fm1 = self.conv_Fm1(Fm1)
        Fms1 = self.sa1(Fm1)
        Fm1 = torch.mul(Fm1, Fms1)
        # print('Fm1.shape:', Fm1.shape)

        Fm2 = self.conv_Fm2(Fm2)       # C: 1024->1024
        Fms2 = self.sa2(Fm2)
        Fm2 = torch.mul(Fm2, Fms2)
        # print('Fm2.shape:', Fm2.shape)

        Fm3 = self.conv_Fm3(Fm3)       # C: 2048->1024
        Fms3 = self.sa3(Fm3)
        Fm3 = torch.mul(Fm3, Fms3)

        atten = self.pafm(Fm1, Fm2, Fm3)        # C: 1024, 1024, 1024 -> 3072 -> 1024

        Fm3_up = F.interpolate(Fm3, (hFm2, wFm2), mode='nearest')
        Fm3_up = torch.mul(Fm3_up, atten/3)

        Fm1_down = F.interpolate(Fm1, scale_factor=0.5)
        Fm1_down = torch.mul(Fm1_down, atten/3)

        Fm2 = torch.mul(Fm2, atten/3)

        Fm2_sum = Fm2 + Fm1_down + Fm3_up

        # print("Fm2_sum[:,:10,:10,:1]",Fm2_sum[:,:10,:10,:1])
        # print("resnet_output[:,:10,:10,:1]",resnet_output[:,:10,:10,:1])

        return Fm2_sum, resnet_output


class GCN_audio_fea(nn.Module):

    def __init__(self):
        super(GCN_audio_fea, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=2048, out_channels=256, kernel_size=1, bias=True)
        self.relu = nn.ReLU()
        # self.conv2 = nn.Conv2d(in_channels=256, out_channels=64, kernel_size=1, bias=True)
        self.K = 16

        self.fc1 = nn.Linear(4096, 512)

    def forward(self, sounds):

        sound_graph = audio_forward(self.K, sounds)

        outs = []
        for i in range(sound_graph.shape[0]):
            sound = sound_graph[i]

            # img->16x1024x1 to x->16x1024x1x1
            x = sound.view(sound.shape[0], sound.shape[1], 1, 1)

            # after conv1 and relu, x-> 16x256x1x1
            x = self.relu(self.conv1(x))
            x = x.view(x.shape[0], x.shape[1])
            # print('x:', x.shape)
            # print('self.norm:', self.Lnorm.shape)

            y = torch.mm(self.Lnorm, x)
            # print('y:', y.shape)

            y = y.view(y.shape[0] * y.shape[1])
            # print('y.shape:', y.shape)

            out = self.fc1(y)

            outs.append(out)
        results = torch.stack(outs)
        # print('results:', results.shape)

        return results

class GCN_audio(nn.Module):

    def __init__(self, num_classes=50):
        super(GCN_audio, self).__init__()
        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(in_channels=2048, out_channels=256, kernel_size=1, bias=True)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        # self.conv2 = nn.Conv2d(in_channels=256, out_channels=64, kernel_size=1, bias=True)
        self.K = 16

        self.fc1 = nn.Linear(4096, 512)
        self.fc2 = nn.Linear(512, self.num_classes)
        self.softmax = nn.Softmax()

    def forward(self, sounds):
        outs = []

        # print('sounds:', len(sounds))
        sound_graph = audio_forward(self.K, sounds)
        # print('sound_graph:', sound_graph.shape)

        for i in range(sound_graph.shape[0]):

            sound = sound_graph[i]

            # img->16x1024x1 to x->16x1024x1x1
            x = sound.view(sound.shape[0], sound.shape[1], 1, 1)
            # print('x:', x.shape)

            # after conv1 and relu, x-> 16x256x1x1
            x = self.relu(self.conv1(x))
            x = x.view(x.shape[0], x.shape[1])
            # print('x:', x.shape)
            # print('self.norm:', self.Lnorm.shape)

            y = torch.mm(self.Lnorm, x)
            # print('y:', y.shape)

            y = y.view(y.shape[0]*y.shape[1])
            # print('y.shape:', y.shape)
            out = self.fc1(y)
            out = self.relu(out)
            out = self.dropout(out)

            out = self.fc2(out)
            outs.append(out)

        results = torch.stack(outs)
        # print('results:', results.shape)

        return results

# audio_gcn_med.py
class GCN_audio_top_med(nn.Module):

    def __init__(self, args):
        super(GCN_audio_top_med, self).__init__()
        self.nodes_num = args.nodes_num
        self.batch_size = args.bs
        self.num_classes = args.num_classes

        if(args.dataset_name == 'Places365-7'):
            self.conv1 = nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=1, bias=True)
            self.conv2 = nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=1, bias=True)

        elif(args.dataset_name == 'Places365-14'):
            self.conv1 = nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=1, bias=True)
            self.conv2 = nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=1, bias=True)

        elif(args.dataset_name == 'MIT67'):
            self.conv1 = nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=1, bias=True)
            self.conv2 = nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=1, bias=True)

        # self.conv2 = nn.Conv2d(in_channels=256, out_channels=64, kernel_size=1, bias=True)

        if (self.nodes_num==16):
            self.fc1 = nn.Linear(8192, 1024)
        elif (self.nodes_num==4):
            self.fc1 = nn.Linear(2048, 1024)
        elif (self.nodes_num==8):
            self.fc1 = nn.Linear(4096, 1024)
        elif (self.nodes_num==12):
            self.fc1 = nn.Linear(6144, 1024)
        elif (self.nodes_num==20):
            self.fc1 = nn.Linear(10240, 1024)
        elif (self.nodes_num==24):
            self.fc1 = nn.Linear(12288, 1024)

        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, self.num_classes)
        self.bn1 = nn.BatchNorm1d(1024)
        self.bn2 = nn.BatchNorm1d(512)
        self.bn3 = nn.BatchNorm1d(self.num_classes)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.25)
        self.softmax = nn.Softmax()

        self.graph_construction = Graph_Init(self.nodes_num, self.batch_size)


    def forward(self, images):
        # images->16,1024,28,28
        sound_graph, Lnormtop, Lnormmed, rows, columns = img_med_forward(self.nodes_num, images, self.graph_construction)
        # sound_graph-> 16,32,1024,1
        # print('sound_graph.shape:', sound_graph.shape)
        # print('Lnormtop.shape:', Lnormtop.shape)
        # print('Lnormmed.shape:', Lnormmed.shape)

        # sound_graph = self.MHA(sound_graph)
        # print('sound_graph.shape after MHA:', sound_graph.shape)


        # for each image, construct a graph
        sound = sound_graph[:,0:self.nodes_num,:,:]
        # print('sound.shape:', sound.shape)
        sound_m = sound_graph[:, self.nodes_num:2*self.nodes_num, :, :]
        # print('sound_m.shape:', sound_m.shape)

        # img->16 x nodes_num x 1024x1 to x-> (16xnodes_num) x 1024 x 1 x 1
        x = sound.reshape(sound.shape[0]*sound.shape[1], sound.shape[2], 1, 1)
        x_m = sound_m.reshape(sound_m.shape[0]*sound_m.shape[1], sound.shape[2], 1, 1)
        # print('x:', x.shape)
        # print('x_m:', x_m.shape)

        # after conv1 and relu, x-> 16x256x1x1
        x = self.relu((self.conv1(x)))
        # print('x:', x.shape)
        # remove 1x1, obtain x-> 16x256
        x = x.view(images.shape[0], int(x.shape[0]/images.shape[0]), x.shape[1])
        # print('x:', x.shape)

        x_m = self.relu((self.conv2(x_m)))
        # print('x_m:', x_m.shape)
        x_m = x_m.view(images.shape[0], int(x_m.shape[0]/images.shape[0]), x_m.shape[1])
        # print('x_m:', x_m.shape)

        graph_fusion_lst = []
        for i in range(sound_graph.shape[0]):
            y = torch.mm(Lnormtop[i], x[i])
            y_m = torch.mm(Lnormmed[i], x_m[i])
            # print('y.shape:', y[i].shape)
            # print('y_m.shape:', y_m[i].shape)
            y = y.view(y.shape[0]*y.shape[1])
            y_m = y_m.view(y_m.shape[0] * y_m.shape[1])
            # print('y.shape:', y[i].shape)
            # print('y_m.shape:', y_m[i].shape)
            graph_fusion = torch.cat((y, y_m), 0)
            # print('graph_fusion.shape:', graph_fusion.shape)
            graph_fusion_lst.append(graph_fusion)

        out = torch.stack(graph_fusion_lst)

        # print('out.shape:', out.shape)
        out = self.dropout(self.relu(self.bn1(self.fc1(out))))
        out = self.dropout(self.relu(self.bn2(self.fc2(out))))

        out = self.fc3(out)
        out = self.bn3(out)
        out = self.softmax(out)
        # print('out.shape:', out.shape)

        return out, rows, columns

        # for i in range(sound_graph.shape[0]):
        #     sound = sound_graph[i][0:self.K]
        #     print('sound.shape:', sound.shape)
        #     sound_m = sound_graph[i][self.K:2*self.K]
        #     print('sound_m.shape:', sound_m.shape)
        #
        #     # img->16x1024x1 to x->16x1024x1x1
        #     x = sound.view(sound.shape[0], sound.shape[1], 1, 1)
        #     x_m = sound_m.view(sound_m.shape[0], sound_m.shape[1], 1, 1)
        #
        #     print('x:', x.shape)
        #     print('x_m:', x_m.shape)
        #     # x = self.MHA(x)
        #     # x_m = self.MHA(x_m)
        #
        #     # after conv1 and relu, x-> 16x256x1x1
        #     x = self.relu(self.conv1(x))
        #     # remove 1x1, obtain x-> 16x256
        #     x = x.view(x.shape[0], x.shape[1])
        #     print('x:', x.shape)
        #
        #     x_m = self.relu(self.conv1(x_m))
        #     x_m = x_m.view(x_m.shape[0], x_m.shape[1])
        #     print('x_m:', x_m.shape)
        #
        #     y = torch.mm(Lnormtop[i], x)
        #     y = y.view(y.shape[0]*y.shape[1])
        #     print('y.shape:', y.shape)
        #
        #     y_m = torch.mm(Lnormmed[i], x_m)
        #     y_m = y_m.view(y_m.shape[0]*y_m.shape[1])
        #     print('y_m.shape:', y_m.shape)
        #
        #     y = torch.cat((y,y_m),0)
        #     print('y.cat shape:', y.shape)
        #
        #     out = self.fc1(y)
        #     print('fc1:', out.shape)
        #     out = self.relu(out)
        #     out = self.dropout(out)
        #     out = self.fc2(out)
        #     print('fc2:', out.shape)
        #
        #     out = self.relu(out)
        #     out = self.dropout(out)
        #     out = self.fc3(out)
        #     print('fc3:', out.shape)
        #
        #     outs.append(out)

# audio_fusion_med.py
class GCN_audio_top_med_fea(nn.Module):

    def __init__(self, args):
        super(GCN_audio_top_med_fea, self).__init__()
        self.nodes_num = args.nodes_num
        self.batch_size = args.bs
        self.num_classes = args.num_classes

        self.conv1 = nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=1, bias=True)
        self.convbn1 = nn.BatchNorm2d(256)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.25)
        self.K = args.nodes_num

        # if (self.K==16):
        #     self.fc1 = nn.Linear(8192, 1024)
        # elif (self.K==8):
        #     self.fc1 = nn.Linear(4096, 1024)
        # elif (self.K==12):
        #     self.fc1 = nn.Linear(6144, 1024)
        # elif (self.K==4):
        #     self.fc1 = nn.Linear(2048, 1024)
        # elif (self.K==20):
        #     self.fc1 = nn.Linear(10240, 1024)
        # elif (self.K==24):
        #     self.fc1 = nn.Linear(12288, 1024)

        if self.K == 4:
            self.fc1 = nn.Linear(1024, 1024)
        elif (self.K == 8):
            self.fc1 = nn.Linear(2048, 1024)
        elif (self.K == 12):
            self.fc1 = nn.Linear(3072, 1024)
        elif (self.K == 16):
            self.fc1 = nn.Linear(4096, 1024)
        elif (self.K == 20):
            self.fc1 = nn.Linear(5120, 1024)
        elif (self.K == 24):
            self.fc1 = nn.Linear(6144, 1024)
        self.bn1 = nn.BatchNorm1d(1024)

        self.fc2 = nn.Linear(1024, 512)
        self.bn2 = nn.BatchNorm1d(512)

        self.fc3 = nn.Linear(512, self.num_classes)
        self.softmax = nn.Softmax()
        self.graph_construction = Graph_Init(self.nodes_num, self.batch_size)


    def forward(self, sounds):
        # print('imgs:', len(imgs)/self.K)
        sound_graph, Lnormtop, Lnormmed, rows, columns = img_med_forward(self.K, sounds, self.graph_construction)
        # print('sound_graph.shape:', sound_graph.shape)

        # for i in range(self .K):
        sound = sound_graph[:, 0:self.K]
        # print('sound.shape:', sound.shape)
        sound_m = sound_graph[:, self.K:2 * self.K]
        # print('sound_m.shape:', sound_m.shape)

        # img->16x1024x1 to x->16x1024x1x1
        x = sound.reshape(sound.shape[0]*sound.shape[1], sound.shape[2], 1, 1)
        x_m = sound_m.reshape(sound_m.shape[0]*sound_m.shape[1], sound_m.shape[2], 1, 1)
        # print('x:', x.shape)
        # print('x_m:', x_m.shape)

        # after conv1 and relu, x-> 16x256x1x1
        x = self.relu(self.convbn1(self.conv1(x)))
        x = x.view(sounds.shape[0], int(x.shape[0]/sounds.shape[0]), x.shape[1])
        # print('x:', x.shape)

        x_m = self.relu(self.convbn1(self.conv1(x_m)))
        x_m = x_m.view(sounds.shape[0], int(x_m.shape[0]/sounds.shape[0]), x_m.shape[1])
        # print('x_m:', x_m.shape)


        graph_fusion_lst = []
        for i in range(sound_graph.shape[0]):
            y = torch.mm(Lnormtop[i], x[i])
            y_m = torch.mm(Lnormmed[i], x_m[i])
            # print('y.shape:', y.shape)
            # print('y_m.shape:', y_m.shape)
            y = y.view(y.shape[0]*y.shape[1])
            y_m = y_m.view(y_m.shape[0] * y_m.shape[1])
            # print('y.shape:', y.shape)
            # print('y_m.shape:', y_m.shape)
            # graph_fusion = torch.add(y, y_m)
            graph_fusion = (y + y_m) / 2
            # print('graph_sum.shape:', graph_fusion.shape)
            graph_fusion_lst.append(graph_fusion)


        out = torch.stack(graph_fusion_lst)
        # print('out.shape:', out.shape, out)
        out = self.dropout(self.relu(self.bn1(self.fc1(out))))
        # print('out.shape:', out.shape, out)
        out = self.dropout(self.relu(self.bn2(self.fc2(out))))
        # print('out.shape:', out.shape, out)

        return out, rows, columns

        # # for i in range(self .K):
        # for i in range(sound_graph.shape[0]):
        #     sound = sound_graph[i][0:self.K]
        #     # print('sound.shape:', sound.shape)
        #     sound_m = sound_graph[i][self.K:2*self.K]
        #
        #     # img->16x1024x1 to x->16x1024x1x1
        #     x = sound.view(sound.shape[0], sound.shape[1], 1, 1)
        #     x_m = sound_m.view(sound_m.shape[0], sound_m.shape[1], 1, 1)
        #     print('x:', x.shape)
        #     print('x_m:', x_m.shape)
        #
        #     # after conv1 and relu, x-> 16x256x1x1
        #     x = self.relu(self.bn1(self.conv1(x)))
        #     x = x.view(x.shape[0], x.shape[1])
        #     print('x:', x.shape)
        #
        #     x_m = self.relu(self.bn1(self.conv1(x_m)))
        #     x_m = x_m.view(x_m.shape[0], x_m.shape[1])
        #     print('x_m:', x_m.shape)
        #
        #
        #     y = torch.mm(Lnormtop[i], x)
        #     y = y.view(y.shape[0]*y.shape[1])
        #     # print('y:', y.shape)
        #     # print('y.shape:', y.shape)
        #
        #     y_m = torch.mm(Lnormmed[i], x_m)
        #     y_m = y_m.view(y_m.shape[0]*y_m.shape[1])
        #
        #     y = torch.cat((y,y_m),0)
        #     # print('y.con shape:', y.shape)
        #     out = self.fc1(y)
        #     out = self.relu(out)
        #     out = self.dropout(out)
        #     out = self.fc2(out)
        #     out = self.relu(out)
        #     outs.append(out)
        #
        # results = torch.stack(outs)
        #
        # return results, rows, columns


class Audio_Fusion_Classifier(nn.Module):

    def __init__(self, args):
        super(Audio_Fusion_Classifier, self).__init__()
        self.num_classes = args.num_classes
        # gcn feature 512, resnet feature, 2048

        self.fc1 = nn.Linear(2560, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, self.num_classes)
        self.bn2 = nn.BatchNorm1d(self.num_classes)

        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, gcn_predicts, model_predicts):
        # model_predicts = model_predicts.view(model_predicts.shape[0], model_predicts.shape[1])
        # print('shape:', gcn_predicts.shape, model_predicts.shape)

        out = torch.cat((gcn_predicts, model_predicts), dim=1)
        # print('out:', out.shape)

        out = self.fc1(out)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.fc2(out)
        out = self.bn2(out)

        return out


