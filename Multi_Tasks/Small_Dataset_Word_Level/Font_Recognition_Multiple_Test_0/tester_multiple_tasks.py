import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import time
import sys
import os
import copy


class Tester:
    def __init__(self, model, optimizer, lr_schedule, is_use_cuda, test_data_loader, start_epoch=0, num_epochs=25, logger=None, model_name=None, train_me_where=None):
        self.model = model
        self.model_name = model_name
        self.optimizer = optimizer
        self.lr_schedule = lr_schedule
        self.is_use_cuda = is_use_cuda

        self.test_data_loader = test_data_loader

        self.start_epoch = start_epoch
        self.num_epochs = num_epochs

        self.cur_epoch = start_epoch
        self.best_acc = 100.00
        self.best_loss = sys.float_info.max
        self.criterion = [nn.CrossEntropyLoss(), nn.CrossEntropyLoss(), nn.CrossEntropyLoss(), nn.CrossEntropyLoss()]

        self.dataloaders_dict = {
            'test': self.test_data_loader,
        }
        n_len_test = len(self.test_data_loader.dataset)

        self.dataset_sizes = {'test': n_len_test}

        self.best_model_wts = copy.deepcopy(self.model.state_dict())
        self.logger = logger

        if train_me_where == "from_middle":
            epoch_num = "67"
            path_to_load = '/home/tmondal/Python_Projects/Font_Recognition/Word_Level/Backup_Data/Font_Recognition_Multiple_Word_Test/checkpoint/resnet_multi_task/'+'/Models_epoch_' + epoch_num + '.ckpt'
            checkpoint = torch.load(path_to_load)
            self.model.load_state_dict(checkpoint['state_dict'])
            self.best_model_wts = copy.deepcopy(self.model.state_dict())
            self.best_acc = checkpoint['best_acc']
            self.cur_epoch = checkpoint['cur_epoch']
            self.num_epochs = checkpoint['num_epochs']
            self.start_epoch = self.cur_epoch
            print("I am inside train me where and the path is :", path_to_load)

    def train(self):
        since = time.time()

        # Each epoch has a training and validation phase
        for phase in ['test']:

            self.model.eval()  # Set model to evaluate mode

            scan_corrects = 0.0
            size_corrects = 0.0
            type_corrects = 0.0
            emphas_corrects = 0.0

            for inputs, single_image_label_scans, single_image_label_sizes, single_image_label_types, \
                single_image_label_emphas in self.dataloaders_dict[phase]:

                if self.is_use_cuda:
                    inputs, single_image_label_scans, single_image_label_sizes, single_image_label_types, \
                    single_image_label_emphas = inputs.cuda(), single_image_label_scans.cuda(), \
                                                single_image_label_sizes.cuda(), single_image_label_types.cuda(), \
                                                single_image_label_emphas.cuda()

                    single_image_label_scans = single_image_label_scans.squeeze()
                    single_image_label_sizes = single_image_label_sizes.squeeze()
                    single_image_label_types = single_image_label_types.squeeze()
                    single_image_label_emphas = single_image_label_emphas.squeeze()
                else:
                    single_image_label_scans = single_image_label_scans.squeeze()
                    single_image_label_sizes = single_image_label_sizes.squeeze()
                    single_image_label_types = single_image_label_types.squeeze()
                    single_image_label_emphas = single_image_label_emphas.squeeze()

                self.optimizer.zero_grad()  # for zero initializing the gradients, which is necessary in pytorch
                outputs = self.model(inputs)

                scan_corrects += torch.sum(torch.max(outputs[0], 1)[1] == torch.max(single_image_label_scans, 1)[1])
                size_corrects += torch.sum(torch.max(outputs[1], 1)[1] == torch.max(single_image_label_sizes, 1)[1])
                type_corrects += torch.sum(torch.max(outputs[2], 1)[1] == torch.max(single_image_label_types, 1)[1])
                emphas_corrects += torch.sum(torch.max(outputs[3], 1)[1] == torch.max(single_image_label_emphas, 1)[1])

            scan_acc = scan_corrects / self.dataset_sizes[phase]
            size_acc = size_corrects / self.dataset_sizes[phase]
            type_acc = type_corrects / self.dataset_sizes[phase]
            emphas_acc = emphas_corrects / self.dataset_sizes[phase]

            print('{} scanning_Acc: {:.4f}  font size_acc: {:.4f}  font type_acc: {:.4f}'
                  'font emphasis_acc: {:.4f}'.format(phase, scan_acc, size_acc, type_acc, emphas_acc))

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
