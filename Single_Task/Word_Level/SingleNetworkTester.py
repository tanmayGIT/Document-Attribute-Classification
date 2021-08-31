from __future__ import print_function, division
import numpy as np
import torch
import torch.nn as nn
import time
import sys
import os
import copy


class SingleNetworkTester:

    def __init__(self, model, optimizer, lr_schedule, is_use_cuda, test_data_loader,
                 start_epoch=0, num_epochs=25, logger=None, model_name=None, network_type=None,
                 folder_name=None, train_me_where=None, epoch_load=None):
        self.model = model
        self.model_name = model_name
        self.network_type = network_type

        self.optimizer = optimizer
        self.lr_schedule = lr_schedule
        self.is_use_cuda = is_use_cuda
        self.test_data_loader = test_data_loader

        print(" I am inside init function of test file ")

        self.start_epoch = start_epoch
        self.num_epochs = num_epochs

        self.cur_epoch = start_epoch
        self.best_acc = 0.
        self.best_loss = sys.float_info.max
        self.criterion = [nn.CrossEntropyLoss()]

        self.dataloaders_dict = {
            'test': self.test_data_loader
        }
        n_len_test = len(self.test_data_loader.dataset)
        self.dataset_sizes = {'test': n_len_test}

        self.best_model_wts = copy.deepcopy(self.model.state_dict())
        self.best_acc = 100
        self.logger = logger

        print("The model name is {0} and the network type is {1} and the folder name is {2} and the train me where is "
              "{3} and the epoch load is {4}".format(model_name, network_type, folder_name, train_me_where, epoch_load))

        if train_me_where == "from_middle":
            path_to_load = '/home/tmondal/Python_Projects/Font_Recognition/checkpoint/' \
                           + folder_name + '/Models_epoch_' + epoch_load + '.ckpt'

            checkpoint = torch.load(path_to_load)

            self.model.load_state_dict(checkpoint['state_dict'])
            self.best_model_wts = copy.deepcopy(self.model.state_dict())
            self.best_acc = checkpoint['best_acc']
            self.cur_epoch = checkpoint['cur_epoch']
            self.num_epochs = checkpoint['num_epochs']
            self.start_epoch = self.cur_epoch

            print("I have loaded the path : ", path_to_load)

    def _save_best_model(self):
        # Save Model
        self.logger.append('Saving Model...')
        state = {
            'state_dict': self.model.state_dict(),
            'best_acc': self.best_acc,
            'cur_epoch': self.cur_epoch,
            'num_epochs': self.num_epochs
        }

        if not os.path.isdir('/home/tmondal/Python_Projects/Font_Recognition/checkpoint/' + self.model_name +
                             self.network_type):
            os.makedirs('/home/tmondal/Python_Projects/Font_Recognition/checkpoint/' + self.model_name +
                        self.network_type)

        torch.save(state,
                   '/home/tmondal/Python_Projects/Font_Recognition/checkpoint/' + self.model_name + self.network_type +
                   '/Models' + '_epoch_%d' % self.cur_epoch + '.ckpt')  # Notice

    def train(self):
        since = time.time()

        # Each epoch has a training and validation phase
        for phase in ['test']:
            self.model.eval()  # Set model to evaluate mode

            losses = []
            running_loss = 0.0
            get_corrects = 0.0

            for inputs, single_image_label_scans, single_image_label_sizes, single_image_label_types, \
                    single_image_label_emphas, single_image_whole_label in self.dataloaders_dict[phase]:

                # print("The i is {0} and the input size is {1}".format(i, len(inputs)))
                if self.network_type == "scanning":
                    needed_var = single_image_label_scans
                elif self.network_type == "font_size":
                    needed_var = single_image_label_sizes
                elif self.network_type == "font_type":
                    needed_var = single_image_label_types
                elif self.network_type == "font_emphasis":
                    needed_var = single_image_label_emphas
                elif self.network_type == "all_labels_together":
                    needed_var = single_image_whole_label
                else:
                    raise Exception("Sorry, no criterion is matching")

                if self.is_use_cuda:
                    needed_var = needed_var.cuda()
                    inputs = inputs.cuda()
                    needed_var = needed_var.squeeze()
                else:
                    needed_var = needed_var.squeeze()

                self.optimizer.zero_grad()  # for zero initializing the gradients, which is necessary in pytorch

                # forward; track history if only in train
                with torch.set_grad_enabled(phase == 'train'):

                    outputs = self.model(inputs)
                    loss = self.criterion[0](outputs, torch.max(needed_var.float(), 1)[1])

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        self.optimizer.step()

                # statistics of all the images in a batch ; remember that this is not the stats for a single image
                running_loss += loss.item() * inputs.size(0)

                losses.append(loss.item())  # get the loss of all the images in this batch

                get_corrects += torch.sum(torch.max(outputs, 1)[1] == torch.max(needed_var, 1)[1])

                # getting the result of all the batch (for each epoch)
            epoch_loss = running_loss / self.dataset_sizes[phase]  # epoch error

            variable_acc = get_corrects / self.dataset_sizes[phase]

            batch_mean_loss = (np.mean(losses))
            print('{} total loss: {:.4f}'.format(phase, epoch_loss))
            print('{} mean_loss of all the batches: {:.4f}'.format(phase, batch_mean_loss))
            print('{} variable_Acc: {:.4f}'.format(phase, variable_acc))

        time_elapsed = time.time() - since
        print('Testing complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f}'.format(float(self.best_acc)))
    
        # load best model weights
        self.model.load_state_dict(self.best_model_wts)
