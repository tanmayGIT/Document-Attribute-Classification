from collections import OrderedDict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from trainer import Trainer
from torchvision import transforms
import numpy as np
from multi_image_data_loader_new import WordImageDS as multiloaderDataset
from network_model import VariousModels, CombineMultiOutputModel, CombineMultiOutputModelConcat
from logger import Logger
import os
import os.path
from os import path
import shutil


def load_state_dict(model_dir, is_multi_gpu):
    print('In the function load_state_dict')
    state_dict = torch.load(model_dir, map_location=lambda storage, loc: storage)['state_dict']
    if is_multi_gpu:
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v
        return new_state_dict
    else:
        return state_dict


def create_save_multi_dataset_together():
    # This following part is for generating the data by using the dataloader and to save it

    training_imag_paths_patch = "/data/zenith/user/tmondal/Font_Data/Train_Data_Patch/"
    training_imag_paths_words = "/data/zenith/user/tmondal/Font_Data/Train_Data_Server/"

    validation_imag_paths_patch = "/data/zenith/user/tmondal/Font_Data/Validation_Data_Patch/"
    validation_imag_paths_words = "/data/zenith/user/tmondal/Font_Data/Validation_Data_Server/"

    if path.isdir(training_imag_paths_patch):
        print("Training path patch exists")
    else:
        raise SystemExit('Training path patch doesnt exists')

    if path.isdir(training_imag_paths_words):
        print("Training path words exists")
    else:
        raise SystemExit('Training path words doesnt exists')

    if path.isdir(validation_imag_paths_patch):
        print("Validation path patch exists")
    else:
        raise Exception('Validation path patch doesnt exists')

    if path.isdir(validation_imag_paths_words):
        print("Validation path words exists")
    else:
        raise Exception('Validation path words doesnt exists')

    data_transforms = {
        'train': transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                     ]),
        'val': transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                   ]),
        'test': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

    train_imgs_obj = multiloaderDataset(training_imag_paths_words, training_imag_paths_patch, 400,
                                        transform=data_transforms['train'])  # if we don't pass the 3rd parameter then
    #  it takes default value which is -1; that means take all the images

    validation_imgs_obj = multiloaderDataset(validation_imag_paths_words, validation_imag_paths_patch, 100,
                                             transform=data_transforms['val'])  # if we don't pass the 3rd parameter
    #  then it takes default value which is -1; that means take all the images

    mn_dataset_loader_train = torch.utils.data.DataLoader(dataset=train_imgs_obj,
                                                          batch_size=200, shuffle=True, num_workers=6, drop_last=True)
    mn_dataset_loader_validation = torch.utils.data.DataLoader(dataset=validation_imgs_obj,
                                                               batch_size=200, shuffle=True, num_workers=6, drop_last=True)
    print("The size of train dataset:", len(train_imgs_obj)*2)
    print("The size of validation dataset:", len(validation_imgs_obj)*2)
    print(train_imgs_obj[0])

    return mn_dataset_loader_train, mn_dataset_loader_validation


def main():
    logger = Logger('/home/tmondal/Python_Projects/Font_Recognition/Combined_Patch_Word/Backup/Font_Recognition_Multiple_4/logs/' +
                    "save_training_params_crossnet_224" + '.log', True)

    model_name = "resnet_multi_task"
    train_me_where = "from_begining"  # "from_middle"
    print('I am in concat equal')

    mn_dataset_loader_multi_train, mn_dataset_loader_multi_valid = create_save_multi_dataset_together()

    num_classes = 2
    feature_extract = True

    is_use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if is_use_cuda else "cpu")

    my_model = VariousModels(model_name, num_classes, feature_extract)
    model_ft, input_size = my_model.initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, 512)

    dd = .1
    model_1 = CombineMultiOutputModelConcat(model_ft, dd)

    lrlast = .001
    lrmain = .0001

    optim1 = optim.Adam(
        [
            {"params": model_1.resnet_model.parameters(), "lr": lrmain},
            {"params": model_1.x11.parameters(), "lr": lrlast},

            {"params": model_1.x1c.parameters(), "lr": lrlast},

            {"params": model_1.brk1.parameters(), "lr": lrlast},
            {"params": model_1.brk2.parameters(), "lr": lrlast},
            {"params": model_1.brk3.parameters(), "lr": lrlast},
            {"params": model_1.brk4.parameters(), "lr": lrlast},

            {"params": model_1.x1.parameters(), "lr": lrlast},
            {"params": model_1.x2.parameters(), "lr": lrlast},

            {"params": model_1.y1o.parameters(), "lr": lrlast},
            {"params": model_1.y2o.parameters(), "lr": lrlast},
            {"params": model_1.y3o.parameters(), "lr": lrlast},
            {"params": model_1.y4o.parameters(), "lr": lrlast},
        ])

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model_1 = nn.DataParallel(model_1.cuda())
        print('model and cuda mixing done')
    model_1 = model_1.to(device)

    optimizer_ft = optim1  # Observe that all parameters are being optimized
    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=5, gamma=0.1)

    # Observe that all parameters are being optimized
    start_epoch = 0
    num_epochs = 90

    # Train and evaluate
    my_trainer = Trainer(model_1, optimizer_ft, exp_lr_scheduler, is_use_cuda,
                         mn_dataset_loader_multi_train, mn_dataset_loader_multi_valid, start_epoch, num_epochs, logger,
                         model_name, train_me_where)
    my_trainer.train()


if __name__ == '__main__':
    main()
