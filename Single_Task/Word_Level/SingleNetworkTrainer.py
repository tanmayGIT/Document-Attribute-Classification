from __future__ import print_function, division
import numpy as np
import torch
import torch.nn as nn
import time
import sys
import os
import copy


class SingleNetworkTrainer:

    def __init__(self, model, optimizer, lr_schedule, is_use_cuda, train_data_loader,
                 valid_data_loader=None, start_epoch=0, num_epochs=25, logger=None, model_name=None, network_type=None):
        self.model = model
        self.model_name = model_name
        self.network_type = network_type

        self.optimizer = optimizer
        self.lr_schedule = lr_schedule
        self.is_use_cuda = is_use_cuda
        self.train_data_loader = train_data_loader
        self.valid_data_loader = valid_data_loader
        self.start_epoch = start_epoch
        self.num_epochs = num_epochs

        self.cur_epoch = start_epoch
        self.best_acc = 0.
        self.best_loss = sys.float_info.max
        self.criterion = [nn.CrossEntropyLoss()]

        self.dataloaders_dict = {
            'train': self.train_data_loader,
            'val': self.valid_data_loader
        }
        n_len_train = len(self.train_data_loader.dataset)
        n_len_valid = len(self.valid_data_loader.dataset)

        self.dataset_sizes = {'train': n_len_train,
                              'val': n_len_valid}

        self.best_model_wts = copy.deepcopy(self.model.state_dict())
        self.best_acc = 100
        self.logger = logger

        if train_me_where == "from_middle":
            epoch_num = "52"
            checkpoint = torch.load('/home/tmondal/Python_Projects/Font_Recognition/checkpoint/' + model_name +
                                    '/Models_epoch_' + epoch_num + '.ckpt')

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

        if not os.path.isdir('/home/tmondal/Python_Projects/Font_Recognition/checkpoint/' + self.model_name +
                             self.network_type):
            os.makedirs('/home/tmondal/Python_Projects/Font_Recognition/checkpoint/' + self.model_name +
                        self.network_type)

        torch.save(state,
                   '/home/tmondal/Python_Projects/Font_Recognition/checkpoint/' + self.model_name + self.network_type +
                   '/Models' + '_epoch_%d' % self.cur_epoch + '.ckpt')  # Notice

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
                get_corrects = 0.0
                # print('The Length {}'.format(len(self.train_data_loader)))

                # for i, (inputs, single_image_label_scans, single_image_label_sizes, single_image_label_types,
                #         single_image_label_emphas, single_image_whole_label) in enumerate(self.train_data_loader):
                
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
                    # print(get_corrects)
    
                    # getting the result of all the batch (for each epoch)
                epoch_loss = running_loss / self.dataset_sizes[phase]  # epoch error

                variable_acc = get_corrects / self.dataset_sizes[phase]

                batch_mean_loss = (np.mean(losses))
                print('{} total loss: {:.4f}'.format(phase, epoch_loss))

                print('{} mean_loss of all the batches: {:.4f}'.format(phase, batch_mean_loss))

                print('{} variable_Acc: {:.4f}'.format(phase, variable_acc))

                # deep copy the model
                if phase == 'val' and variable_acc > self.best_acc:
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
    
        str_save = 'full_data_224_single_task' + self.network_type + '.pth'
        torch.save(self.best_model_wts, str_save)
