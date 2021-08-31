from collections import OrderedDict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from trainer import Trainer
from torchvision import transforms
from word_image_datasets import WordImageDS
from word_image_dataset_small import WordImageDS as wordSmallDataset
from network_model import VariousModels, MultiOutputModel
from logger import Logger
import os
import os.path
from os import path
import shutil


def load_state_dict(model_dir, is_multi_gpu):
    print ('In the function load_state_dict')
    state_dict = torch.load(model_dir, map_location=lambda storage, loc: storage)['state_dict']
    if is_multi_gpu:
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v
        return new_state_dict
    else:
        return state_dict


def create_save_dataset_small():
    # This following part is for generating the data by using the dataloader and to save it
    training_imag_paths = "Train_Data_Comp_Small/"
    validation_imag_paths = "Validation_Data_Comp_Small/"

    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

    train_imgs_obj = wordSmallDataset(training_imag_paths, 800, transform=data_transforms['train'])
    validation_imgs_obj = wordSmallDataset(validation_imag_paths, 800, transform=data_transforms['val'])
    # test_imgs_obj = wordSmallDataset(validation_imag_paths, 50, transform=data_transforms['val'])

    mn_dataset_loader_train_scanning = torch.utils.data.DataLoader(dataset=train_imgs_obj,
                                                                   batch_size=20, shuffle=True, num_workers=0)
    mn_dataset_loader_validation_scanning = torch.utils.data.DataLoader(dataset=validation_imgs_obj,
                                                                        batch_size=10, shuffle=True, num_workers=0)
    # mn_dataset_loader_testing = torch.utils.data.DataLoader(dataset=test_imgs_obj, batch_size=10, shuffle=True,
    #                                                         num_workers=5)
    print(train_imgs_obj[0])

    # torch.save(mn_dataset_loader_train_scanning, 'mn_dataset_loader_train_scanning_small.pth')
    # torch.save(mn_dataset_loader_validation_scanning, 'mn_dataset_loader_validation_scanning_small.pth')
    # torch.save(mn_dataset_loader_testing, 'mn_dataset_loader_testings_small.pth')

    return mn_dataset_loader_train_scanning, mn_dataset_loader_validation_scanning


def create_save_dataset():
    # This following part is for generating the data by using the dataloader and to save it
    training_imag_paths = "/data/zenith/user/tmondal/Font_Data/Train_Data_Server/"
    validation_imag_paths = "/data/zenith/user/tmondal/Font_Data/Validation_Data_Server/"
    # testing_imag_paths = "/home/mondal/Documents/Dataset/L3i_Text_Copies/Test_Data_Comp/"

    if(path.isdir(training_imag_paths)):
        print("Training path exists")
    else:
        raise SystemExit("Training path doesn't exists")

    if(path.isdir(validation_imag_paths)):
        print("Validation path exists")
    else:
        raise SystemExit("Validation path doesn't exists")

    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }
    print ("entering in wordimageds")
    train_imgs_obj = WordImageDS(training_imag_paths, transform=data_transforms['train'])
    print("exit from wordimageds")

    mn_dataset_loader_train = torch.utils.data.DataLoader(dataset=train_imgs_obj,batch_size=800, shuffle=True, num_workers=0)
    # torch.save(mn_dataset_loader_train, 'mn_dataset_loader_train.pth')

    validation_imgs_obj = WordImageDS(validation_imag_paths, transform=data_transforms['val'])
    mn_dataset_loader_validation = torch.utils.data.DataLoader(dataset=validation_imgs_obj,batch_size=800, shuffle=True, num_workers=0)
    # torch.save(mn_dataset_loader_validation, 'mn_dataset_loader_validation.pth')
    print("The size of train dataset:", len(train_imgs_obj))
    print("The size of validation dataset:", len(validation_imgs_obj))
    print(train_imgs_obj[1010])

    # test_imgs_obj = WordImageDS(testing_imag_paths, transform=data_transforms['test'])
    # mn_dataset_loader_testing = torch.utils.data.DataLoader(dataset=test_imgs_obj, batch_size=100, shuffle=True,
    #                                                         num_workers=5)
    # torch.save(mn_dataset_loader_testing, 'mn_dataset_loader_testing.pth')

    return mn_dataset_loader_train, mn_dataset_loader_validation


def main():
    logger = Logger('/home/tmondal/Python_Projects/Font_Recognition/logs/'+"save_training_params_crossnet_224"+'.log', True)
    model_name = "resnet_multi_task"
    train_me_where = "from_begining" # "from_middle" 
    # create_save_dataset()
    mn_dataset_loader_train, mn_dataset_loader_valid = create_save_dataset()

    num_classes = 2
    # batch_size = 8
    feature_extract = True

    # mn_dataset_loader_train = torch.load('mn_dataset_loader_train_scanning_small.pth')
    # mn_dataset_loader_valid = torch.load('mn_dataset_loader_validation_scanning_small.pth')
    # mn_dataset_loader_testing = torch.load('mn_dataset_loader_testing_small.pth')

    is_use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if is_use_cuda else "cpu")

    my_model = VariousModels(model_name, num_classes, feature_extract)
    model_ft, input_size = my_model.initialize_model(model_name, num_classes, feature_extract, use_pretrained = True)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, 512)

    dd = .1
    model_1 = MultiOutputModel(model_ft, dd)

    lrlast = .001
    lrmain = .0001

    optim1 = optim.Adam(
        [
            {"params": model_1.resnet_model.parameters(), "lr": lrmain},
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

    # print(model_1)
    # print(model_1.parameters())

    optimizer_ft = optim1  # Observe that all parameters are being optimized
    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=10, gamma=0.1)

    # Observe that all parameters are being optimized
    start_epoch = 0
    num_epochs = 90

    # Train and evaluate
    my_trainer = Trainer(model_1, optimizer_ft, exp_lr_scheduler, is_use_cuda,
                         mn_dataset_loader_train, mn_dataset_loader_valid, start_epoch, num_epochs, logger, model_name, train_me_where)
    my_trainer.train()


if __name__ == '__main__':
    main()
