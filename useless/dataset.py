# Developed by Liguang Zhou, 2020.9.17

import json
import torch
import torchvision.datasets as datasets
import torchvision.models as models
import os
import numpy as np

class DatasetSelection(object):
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name

    def datasetSelection(self):
        if(self.dataset_name == 'avasrd'):
            train_data_dir = '/data/guest/avasr/avasrd/train_img'
            val_data_dir = '/data/guest/avasr/avasrd/val_img'
            test_data_dir = '/data/guest/avasr/avasrd/test_img'

        return train_data_dir, val_data_dir, test_data_dir

    def load_dict(self, filename):
        with open(filename,"r") as json_file:
            dic = json.load(json_file)
        return dic

    def discriminative_matrix_estimation(self):
        # create p_o_c matrix


        if(self.dataset_name == 'Places365-7'):
            fileName = './object_information/150obj_result_Places365_7.npy'
            self.num_sp = np.load(fileName)
            fileName = './object_information/150obj_number_Places365_7.npy'
            self.num_total = np.load(fileName)
            self.cls_num = 7
            self.obj_num = 150

        elif(self.dataset_name == 'Places365-14'):
            fileName = './object_information/150obj_result_Places365_14.npy'
            self.num_sp = np.load(fileName)
            fileName = './object_information/150obj_number_Places365_14.npy'
            self.num_total=np.load(fileName)
            self.cls_num = 14
            self.obj_num = 150

        matrix_p_o_c = np.zeros(shape=(self.cls_num,self.obj_num,self.obj_num))

        for i in range(self.cls_num):
            X=[]
            Y=[]
            Z=[]
            p_o_c = self.num_sp[i] / self.num_total[i]
            p_o_c = p_o_c.reshape(1,p_o_c.shape[0])
        #    print(p_o_c)
            p_o_c_tran=p_o_c.T
        #    print(p_o_c_tran)
            matrix_p_o_c[i]=np.dot(p_o_c_tran,p_o_c)
            
            
        matrix_p_c_o = np.zeros(shape=(self.cls_num,self.obj_num,self.obj_num))
        discriminative_matrix = np.zeros(shape=(self.obj_num,self.obj_num))
        temp=np.zeros(shape=self.cls_num)
        for i in range(self.obj_num):
            for j in range(self.obj_num):
                sum=0
                for k in range(self.cls_num):
                    sum += matrix_p_o_c[k][i][j]*1/self.cls_num
                if sum == 0:
                    matrix_p_c_o[k][i][j]=0
                    continue
                for k in range(self.cls_num):
                    matrix_p_c_o[k][i][j]=matrix_p_o_c[k][i][j]*1/self.cls_num/sum
                    temp[k]=matrix_p_c_o[k][i][j]
                discriminative_matrix[i][j]=temp.std()

        # print('discriminative_matrix:', discriminative_matrix.shape, discriminative_matrix)
        return discriminative_matrix



class ImageFolderWithPaths(datasets.ImageFolder):
        """Custom dataset that includes image file paths. Extends
        torchvision.datasets.ImageFolder
        """

        # override the __getitem__ method. this is the method dataloader calls
        def __getitem__(self, index):
            # this is what ImageFolder normally returns 
            original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
            # the image file path
            path = self.imgs[index][0]
            # make a new tuple that includes original and the path
            tuple_with_path = (original_tuple + (path,))
            return tuple_with_path