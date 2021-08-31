from collections import OrderedDict
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from trainer import Trainer
from torchvision import transforms
from word_image_datasets import WordImageDS

from network_model import VariousModels, MultiOutputModel
from logger import Logger
import os
import os.path
from os import path
import shutil


def main():
    test_imag_paths = "/data/zenith/user/tmondal/Font_Data/Test_Data_Server/"

    saved_model_path = "/home/tmondal/Python_Projects/Font_Recognition/checkpoint/weights_word_level_old/" \
                       "resnet_multi_task_old/Models_epoch_70.ckpt"
    torch.cuda.empty_cache()

    data_transforms = {
        'test': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

    is_use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if is_use_cuda else "cpu")

    test_imgs_obj = WordImageDS(test_imag_paths, transform=data_transforms['test'])
    mn_dataset_loader_test = torch.utils.data.DataLoader(dataset=test_imgs_obj, batch_size=500,
                                                               shuffle=True, num_workers=0, drop_last=True)

    n_len_test = len(mn_dataset_loader_test.dataset)
    print('Total number of test images :', n_len_test)

    # The model initialization
    model_name = "resnet_multi_task"
    num_classes = 2
    feature_extract = True

    my_model = VariousModels(model_name, num_classes, feature_extract)
    model_ft, input_size = my_model.initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, 512)

    dd = .1
    model = MultiOutputModel(model_ft, dd)
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model.cuda())
        print('model and cuda mixing done')

    model = model.to(device)
    # end of model initialization

    checkpoint = torch.load(saved_model_path)  # change here
    model.load_state_dict(checkpoint['state_dict'])

    model.eval()  # Set model to evaluate mode
    scan_corrects = 0.0
    size_corrects = 0.0
    type_corrects = 0.0
    emphas_corrects = 0.0

    for inputs, single_image_label_scans, single_image_label_sizes, single_image_label_types, \
        single_image_label_emphas in mn_dataset_loader_test:

        if is_use_cuda:
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

        outputs = model(inputs)

        scan_corrects += torch.sum(torch.max(outputs[0], 1)[1] == torch.max(single_image_label_scans, 1)[1])
        size_corrects += torch.sum(torch.max(outputs[1], 1)[1] == torch.max(single_image_label_sizes, 1)[1])
        type_corrects += torch.sum(torch.max(outputs[2], 1)[1] == torch.max(single_image_label_types, 1)[1])
        emphas_corrects += torch.sum(torch.max(outputs[3], 1)[1] == torch.max(single_image_label_emphas, 1)[1])

    variable_acc_scan = scan_corrects / n_len_test
    variable_acc_size = size_corrects / n_len_test
    variable_acc_type = type_corrects / n_len_test
    variable_acc_emphas = emphas_corrects / n_len_test

    print('Testing Accuracy Scan: {:4f}'.format(float(variable_acc_scan)))
    print('Testing Accuracy Size: {:4f}'.format(float(variable_acc_size)))
    print('Testing Accuracy Type: {:4f}'.format(float(variable_acc_type)))
    print('Testing Accuracy Emphasis: {:4f}'.format(float(variable_acc_emphas)))


if __name__ == '__main__':
    main()
