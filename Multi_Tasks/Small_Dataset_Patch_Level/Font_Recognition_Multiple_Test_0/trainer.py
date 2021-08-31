import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import time
import sys
import os
import copy


class Trainer:
    def __init__(self, model, optimizer, lr_schedule, is_use_cuda, train_data_loader,
                 valid_data_loader=None, start_epoch=0, num_epochs=25, logger=None, model_name=None, train_me_where = None):
        self.model = model
        self.model_name = model_name
        self.optimizer = optimizer
        self.lr_schedule = lr_schedule
        self.is_use_cuda = is_use_cuda
        self.train_data_loader = train_data_loader
        self.valid_data_loader = valid_data_loader
        self.start_epoch = start_epoch
        self.num_epochs = num_epochs

        self.cur_epoch = start_epoch
        self.best_acc = 100.00
        self.best_loss = sys.float_info.max
        self.criterion = [nn.CrossEntropyLoss(), nn.CrossEntropyLoss(), nn.CrossEntropyLoss(), nn.CrossEntropyLoss()]

        self.dataloaders_dict = {
            'train': self.train_data_loader,
            'val': self.valid_data_loader
        }
        n_len_train = len(self.train_data_loader.dataset)
        n_len_valid = len(self.valid_data_loader.dataset)

        self.dataset_sizes = {'train': n_len_train,
                              'val': n_len_valid}

        self.best_model_wts = copy.deepcopy(self.model.state_dict())
        self.logger = logger

        if train_me_where == "from_middle":
            epoch_num = "52"
            checkpoint = torch.load('/home/tmondal/Python_Projects/Font_Recognition/Patch_Level/Backup_Data/Font_Recognition_Multiple_Test_0/checkpoint/' + model_name + '/Models_epoch_' + epoch_num + '.ckpt')
            self.model.load_state_dict(checkpoint['state_dict'])
            self.best_model_wts = copy.deepcopy(self.model.state_dict())
            self.best_acc = checkpoint['best_acc']
            self.cur_epoch = checkpoint['cur_epoch']
            self.num_epochs = checkpoint['num_epochs']
            self.start_epoch = self.cur_epoch

    def _save_best_model(self):
        # Save Model
        self.logger.append('Saving Model...')
        state = {
            'state_dict': self.model.state_dict(),
            'best_acc': self.best_acc,
            'cur_epoch': self.cur_epoch,
            'num_epochs': self.num_epochs
        }

        if not os.path.isdir('/home/tmondal/Python_Projects/Font_Recognition/Patch_Level/Backup_Data/Font_Recognition_Multiple_Test_0/checkpoint/' + self.model_name):
            os.makedirs('/home/tmondal/Python_Projects/Font_Recognition/Patch_Level/Backup_Data/Font_Recognition_Multiple_Test_0/checkpoint/' + self.model_name)

        # if not os.path.isdir('./checkpoint_best_weights/' + self.model_name):
            # os.makedirs('./checkpoint_best_weights/' + self.model_name)

        torch.save(state,
                   '/home/tmondal/Python_Projects/Font_Recognition/Patch_Level/Backup_Data/Font_Recognition_Multiple_Test_0/checkpoint/' + self.model_name + '/Models' + '_epoch_%d' % self.cur_epoch + '.ckpt')  # Notice

        # torch.save(self.best_model_wts,
                   # './checkpoint_best_weights/' + self.model_name + '/Models' + '_epoch_%d' % self.cur_epoch + '.pth')

    def train(self):
        since = time.time()
        for epoch in range(self.start_epoch, self.num_epochs):
            print('Epoch {}/{}'.format(epoch, self.num_epochs - 1))
            print('-' * 10)

            self.cur_epoch = epoch
            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    self.lr_schedule.step()
                    self.model.train()  # Set model to training mode
                else:
                    self.model.eval()  # Set model to evaluate mode

                losses = []

                running_loss = 0.0
                running_loss0 = 0.0
                running_loss1 = 0.0
                running_loss2 = 0.0
                running_loss3 = 0.0

                scan_corrects = 0.0
                size_corrects = 0.0
                type_corrects = 0.0
                emphas_corrects = 0.0

                # for i, (inputs, single_image_label_scans, single_image_label_sizes, single_image_label_types,
                #         single_image_label_emphas, single_image_whole_label) in enumerate(self.train_data_loader):

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

                    # forward; track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):

                        outputs = self.model(inputs)
                        loss0 = self.criterion[0](outputs[0], torch.max(single_image_label_scans.float(), 1)[1])
                        loss1 = self.criterion[1](outputs[1], torch.max(single_image_label_sizes.float(), 1)[1])
                        loss2 = self.criterion[2](outputs[2], torch.max(single_image_label_types.float(), 1)[1])
                        loss3 = self.criterion[3](outputs[3], torch.max(single_image_label_emphas.float(), 1)[1])

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss = loss0 + loss1 + loss2 + loss3
                            loss.backward()
                            self.optimizer.step()

                    # statistics of all the images in a batch ; remember that this is not the stats for a single image
                    running_loss += loss.item() * inputs.size(0)
                    running_loss0 += loss0.item() * inputs.size(0)
                    running_loss1 += loss1.item() * inputs.size(0)
                    running_loss2 += loss2.item() * inputs.size(0)
                    running_loss3 += loss3.item() * inputs.size(0)

                    losses.append(loss.item())  # get the loss of all the images in this batch

                    scan_corrects += torch.sum(torch.max(outputs[0], 1)[1] == torch.max(single_image_label_scans, 1)[1])
                    size_corrects += torch.sum(torch.max(outputs[1], 1)[1] == torch.max(single_image_label_sizes, 1)[1])
                    type_corrects += torch.sum(torch.max(outputs[2], 1)[1] == torch.max(single_image_label_types, 1)[1])
                    emphas_corrects += torch.sum(torch.max(outputs[3], 1)[1] == torch.max(single_image_label_emphas, 1)[1])

                # getting the result of all the batch (for each epoch)
                epoch_loss = running_loss / self.dataset_sizes[phase]  # epoch error
                epoch_loss0 = running_loss0 / self.dataset_sizes[phase]  # epoch error
                epoch_loss1 = running_loss1 / self.dataset_sizes[phase]  # epoch error
                epoch_loss2 = running_loss2 / self.dataset_sizes[phase]  # epoch error
                epoch_loss3 = running_loss3 / self.dataset_sizes[phase]  # epoch error

                scan_acc = scan_corrects / self.dataset_sizes[phase]
                size_acc = size_corrects / self.dataset_sizes[phase]
                type_acc = type_corrects / self.dataset_sizes[phase]
                emphas_acc = emphas_corrects / self.dataset_sizes[phase]

                batch_mean_loss = (np.mean(losses))
                print('{} total loss: {:.4f} scanning loss: {:.4f} font size loss: {:.4f} font type loss: {:.4f}  font '
                      'emphasis loss {:.4f}'.format(phase, epoch_loss, epoch_loss0, epoch_loss1, epoch_loss2,
                                                    epoch_loss3))

                print('{} mean_loss of all the batches: {:.4f}'.format(phase, batch_mean_loss))

                print('{} scanning_Acc: {:.4f}  font size_acc: {:.4f}  font type_acc: {:.4f}'
                      'font emphasis_acc: {:.4f}'.format(phase, scan_acc, size_acc, type_acc, emphas_acc))

                # deep copy the model
                print("The epoch_loss is :", epoch_loss)
                print("The current best accuracy is : ", self.best_acc )

                if phase == 'val' and epoch_loss < self.best_acc:
                # if phase == 'val':
                    print('saving with loss of {}'.format(epoch_loss),
                          'improved over previous {}'.format(self.best_acc))
                    self.best_acc = epoch_loss
                    self.best_model_wts = copy.deepcopy(self.model.state_dict())
                    self._save_best_model()
            print()
        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f}'.format(float(self.best_acc)))

        # load best model weights
        self.model.load_state_dict(self.best_model_wts)
        torch.save(self.best_model_wts, 'full_data_224_multi_task.pth')



